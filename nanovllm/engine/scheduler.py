from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # ---------- 投机解码 ----------
        self.spec_k = config.spec_k
        self.spec_batch_threshold = config.spec_batch_threshold
        # n-gram 提议器。由 ModelRunner 注入（引擎建好后才有），
        # 这样 token_ids 更新后能立刻把新 token 加进索引。
        self.spec_proposer = None

    def set_spec_proposer(self, proposer):
        self.spec_proposer = proposer

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # ---------- 投机解码：这一步要给 k+1 个 token 分配槽位 ----------
            # 之所以是 k+1：1 个是本该decode 的 token，k 个是待验证的候选。
            # 但只有「本该 decode 的那个」的 KV 是最终有效的，
            # 候选的 KV 写入后靠 slot 覆写回收（见 model_runner.prepare_verify）
            need = 1 + (self.spec_k if self.spec_enabled() else 0)
            while not self.block_manager.can_append(seq, need):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = need
                seq.is_prefill = False
                self.block_manager.may_append(seq, need)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def spec_enabled(self) -> bool:
        """本步是否启用投机解码（含 batch 门控）。"""
        if self.spec_k <= 0:
            return False
        if self.spec_batch_threshold and len(self.running) > self.spec_batch_threshold:
            return False
        return True

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list, is_prefill: bool):
        """token_ids 的形态：
           普通 decode   -> [int]          每序列 1 个
           投机验证       -> list[list[int]] 每序列若干个（已接受的 + bonus）
        """
        if not is_prefill and token_ids and isinstance(token_ids[0], list):
            self.postprocess_spec(seqs, token_ids)
            return
        for seq, token_id in zip(seqs, token_ids):
            # ★ 顺序：必须先推进 num_cached_tokens，再做 hash_blocks。
            #   hash_blocks 靠 (num_cached_tokens, num_scheduled_tokens) 算出
            #   哪些 block 已完成；而它内部读到的必须是【更新后】的值，
            #   否则登记进前缀缓存的哈希对应的是「旧的完成量」——
            #   下次相同前缀会命中错误的缓存，静默输出错误 token。
            #   （投机路径对此更敏感：验证阶段的 start 直接取自num_cached_tokens。）
            seq.num_cached_tokens += seq.num_scheduled_tokens
            self.block_manager.hash_blocks(seq)
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # ★ 新生成的 token 必须入 n-gram 索引，否则下一步提不出候选。
            #   普通 decode 路径也要做 —— 大部分 step 其实走的是这条（候选为空时退回），
            #   漏了它会让索引永远追不上，导致投机占比恒为 0。
            if self.spec_proposer is not None:
                self.spec_proposer.observe(seq.token_ids)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess_spec(self, seqs: list[Sequence], accepted_tokens: list[list[int]]):
        """投机验证后的收尾：一次落地多个 token。

        ★ 三件事必须做对，漏了任何一条都不会报错，只会静默出错：
          ① num_cached_tokens 只推进【真正确认的】部分，不能把被拒候选算进去
          ② hash_blocks 必须在 token_ids 更新【之后】调用，且草稿不在 token_ids 里
          ③ EOS 和 max_tokens 要在【每个】落地的 token 上检查，而不是只看最后一个
        """
        for seq, toks in zip(seqs, accepted_tokens):
            seq.draft_tokens = []# 草稿用完即弃，绝不进 token_ids
            seq.draft_probs = None         # ★ 同理，draft 分布也必须清，否则会串用上一轮

            if not toks:
                # 一个都没接受：把本该 decode 的那个位置也退掉，num_computed 回退
                seq.num_cached_tokens = max(0, seq.num_cached_tokens - seq.num_scheduled_tokens)
                seq.num_scheduled_tokens = 0
                continue

            # ① 先记已确认的 token（此刻token_ids 还没变）
            confirmed = seq.num_tokens
            seq.append_tokens(toks)
            # num_scheduled_tokens 是 1+k，只有被接受的那部分对应真实位置
            seq.num_cached_tokens += len(toks)

            # ② token_ids 已更新，现在才能安全地做前缀缓存哈希
            self.block_manager.hash_blocks(seq)
            # ★ 把新生成的 token 增量加入 n-gram 索引，
            #   否则下一步提不出以这些新 token 结尾的候选
            #   （必须放在 append_tokens 之后 —— 那之前 token_ids 还没这些 token）
            if self.spec_proposer is not None:
                self.spec_proposer.observe(seq.token_ids)

            seq.num_scheduled_tokens = 0

            # ③逐个检查终止条件（中途撞 EOS 就截断，后面的候选全部丢弃）
            newly = toks
            for i, t in enumerate(newly):
                if (not seq.ignore_eos and t == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    # 截断：把这一步多写的 token 砍掉
                    extra = len(newly) - i - 1
                    if extra > 0:
                        del seq.token_ids[len(seq.token_ids) - extra:]
                        seq.num_tokens -= extra
                        seq.last_token = seq.token_ids[-1]
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    break
