from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        # ---------- 投机解码 ----------
        #草稿候选token。★ 刻意【不】放进 token_ids：
        #   token_ids 参与 block_manager.hash_blocks() 的前缀缓存哈希，
        #   如果候选混进去，被拒后 token_ids 变了而旧哈希还在
        #   → 下次相同前缀会命中【错误缓存】→ 静默输出错误 token。
        #   所以草稿只存在这里，验证通过后才写进 token_ids。
        self.draft_tokens: list[int] = []
        # draft 路线：候选位置在 draft 模型下的【原始 logits】(k, vocab)。
        # ★ 只有 draft model 路线才有 —— n-gram 没有分布，用不了。
        #   没有它就退化成单点分布，接受率会掉到 p[draft]。
        # ★★ 存的是 logits 不是概率：verify_batch 会自己做 softmax(logits/T)。
        #    存概率会被再 softmax 一次、压成均匀分布，无损性直接破裂
        #    （字段名原本叫 draft_probs，正是这个名字诱导出了那个 bug）。
        self.draft_logits = None
        # 本步实际落地的 token 数（accepted + bonus），用于统计接受率
        self.last_accepted = 0
        # ---------- P6 任务 D：draft 侧 KV 的有效水位 ----------
        # 「从位置 0 起连续有效的 draft KV 个数」。draft 提议只能从这个水位
        # 之后继续；水位之前有缺口（如 gate 关闭若干步时 target 自己生成的 token）
        # 就必须先补齐，只喂最后一个 token 恢复不了缺失前缀。
        # 维护点见 model_runner.prepare_prefill / run() / scheduler.postprocess_spec。
        self.draft_valid_len = 0
        # ---------- B 步：draft 自己的滑窗块表 ----------
        # 窗口关闭时恒为空（draft 用 target 的 block_table，行为与 P6 一致）。
        # 窗口打开时是 M = draft_window/block_size 个物理块的【环形缓冲】：
        # 绝对块号 b 永远落在 ring[b % M]，于是任何时刻环里存的就是「最近 M 个块」。
        # 由 Scheduler 在 prefill 时分配（BlockManager.draft_acquire）、抢占时释放。
        # ★ 必须一起过进程边界：TP>1 时每条 rank 都要按同一套块表跑 draft。
        self.draft_block_table: list[int] = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def append_tokens(self, token_ids: list[int]):
        """一次落地多个 token（投机解码用）。

        ★ 关键：被拒绝的候选不能进 token_ids，所以调用方必须只传
          「已接受的候选 + 可能的 bonus」，而不是全部候选。
        """
        if not token_ids:
            return
        self.token_ids.extend(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens += len(token_ids)
        self.last_accepted = len(token_ids)

    def advance_draft_watermark(self, start: int, end: int):
        """prefill 了一个 [start, end) 的 chunk 之后，推进 draft KV 的有效水位。

        规则：**只有与已有有效前缀连续**的新写入才延长水位。

            · 新请求（start == 0 == draft_valid_len）      → 水位 = end
            · 分块 prefill 的后续 chunk（start == 水位）   → 水位 = end
            · 前缀缓存命中（start > 水位）                 → 水位【不动】
              draft 的 KV 在另一套物理缓冲里，target 命中了不代表 draft 也有；
              这里保守留成缺口，让 proposer 补齐。真正能从缓存复用的部分，
              由 Scheduler 在 allocate 之后按 Block.draft_hash 先推到
              num_cached_tokens —— 于是 start == 水位，走进上面第二条分支。

        放在 Sequence 上是因为它同时被 ModelRunner（prepare_prefill）与
        CPU 测试调用，规则只能有一份。
        """
        if start <= self.draft_valid_len:
            self.draft_valid_len = max(self.draft_valid_len, end)

    def tokens_in(self, start: int, end: int) -> list[int]:
        """取 [start, end) 区间的 token，超出部分用草稿候选补。

        验证阶段要送进 forward 的序列是：
            已确认的 token[start:] + 草稿候选
        这个方法让调用方不用到处判断「哪些是已确认的、哪些是草稿」。
        """
        out = self.token_ids[start:end]
        if self.draft_tokens:
            # end 超出已确认范围的部分用草稿补
            need = end - len(out)
            if need > 0:
                out = out + self.draft_tokens[:need]
        return out

    def __getstate__(self):
        # ★ draft_valid_len 必须一起过进程边界：TP>1 时每条 rank 都会执行
        #   propose，补齐位置要一致，否则各 rank 的 draft KV 水位不同步。
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state,
                self.draft_valid_len, self.draft_block_table)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.block_table, last_state,
         self.draft_valid_len, self.draft_block_table) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
