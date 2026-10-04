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
        # 本步实际落地的 token 数（accepted + bonus），用于统计接受率
        self.last_accepted = 0

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
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
