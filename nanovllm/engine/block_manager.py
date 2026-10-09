from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        # ---------- draft 侧 KV 的有效性标记（与 target 的前缀缓存无关）----------
        # 物理块里【draft 模型算过的内容】的块哈希；-1 = draft 没算过 / 内容已变。
        # ★ 为什么必须单独一个字段：前缀缓存 hash_to_block_id 存的是 target 的
        #   KV 块；draft 的 KV 写在【另一套】物理缓冲里（每层 k/v 是独立张量）。
        #   两者共用同一份 block_table（逻辑块 i → 物理块 i），但 cache 张量分开，
        #   所以「target 命中前缀缓存」推不出「draft 侧也有这段前缀」——
        #   draft 那个物理块里可能是上一次谁留下的内容。
        #   只有 draft 真的按正确前缀算过这一整块，才在这里盖上对应的哈希。
        self.draft_hash = -1

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.draft_hash = -1


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def _blocks_needed(self, seq: Sequence, n_tokens: int) -> int:
        """要容纳本次写入的 n_tokens 个 token，block_table 至少要有几块。

        ★★ 这里踩过一个会【触发 CUDA 非法访存】的坑，务必看清：
          本步真正被写的 token 位于
              position:  len-1, len, ..., len+n-2
          （prepare_decode 和 prepare_verify 的 slot_mapping 都是从 len-1 起算的，
            因为前一个 token 的 KV 要重算一遍）。
          所以最后一个位置是 len+n-2，不是 len+n-1。

          原来的写法是按「len .. len+n-1」算空位的：
              remaining_in_block = block_size - (len % block_size)
          当 len 正好是 block_size 的整数倍时，它给出 remaining = block_size
          （以为当前 block 空着），其实当前 block 已经【正好写满】，
          下一个 token 必须落到新 block —— 于是少分配一块，
          block_table 里那一位是填充值 -1。
          draft/verify 的 attention 拿到 cache_seqlens 要跨两块时，
          flash-attn 就去读第 -1 页 → illegal memory access。

          （64 个 token 的短生成永远碰不到 256 的边界，所以一直没暴露；
            换成 256 token 立刻崩。）
        """
        if n_tokens <= 0:
            return len(seq.block_table)
        last_pos = len(seq) + n_tokens - 2
        return last_pos // self.block_size + 1

    def can_append(self, seq: Sequence, n_tokens: int = 1) -> bool:
        """判断能否再追加 n_tokens 个 token（投机解码时 n_tokens = 1+k）。"""
        need = self._blocks_needed(seq, n_tokens) - len(seq.block_table)
        return len(self.free_block_ids) >= max(0, need)

    def may_append(self, seq: Sequence, n_tokens: int = 1):
        """追加 n_tokens 个 token 所需的 block。

        ★ 投机解码下多分配的 block 是给【待验证候选】用的，它们的 KV 之后会被
          覆写回收（不需要真正的回滚，见 model_runner 的注释）。
          所以这里只管分配，不负责回收——ref_count 的回收在 deallocate 时统一做。
        """
        need = self._blocks_needed(seq, n_tokens) - len(seq.block_table)
        for _ in range(max(0, need)):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, num_new_tokens: int):
        """把本步【刚填满】的 block 登记进前缀缓存哈希表。

        契约（所有调用点必须一致 —— 这里正是之前矛盾的来源）
        ----------------------------------------------------
        入参 num_new_tokens = 本次真正被【确认】并推进的 token 数：
            普通 / 分块 prefill -> 本步处理的 prompt token 数
            普通 decode        -> 1（投机没捞到候选、退回普通 decode 时也是 1，
                                  不能拿 num_scheduled_tokens 顶替，那是 1+k）
            投机验证           -> len(toks)（被接受的候选 + bonus），同样 ≠ 1+k
        调用时 seq.num_cached_tokens 必须【已经】推进到新值（旧值 + num_new_tokens），
        登记区间取「旧完成块数 → 新完成块数」：

            start = (num_cached_tokens - num_new_tokens) // block_size
            end   =  num_cached_tokens                   // block_size

        ★ 为什么 num_cached_tokens 就是「有效缓存量」：
          它数的是「已确认 + KV 已落地」的 token。刚采出来的那个 token 自己
          那一格 KV 还没写 —— 要等下一次前向按 position = len-1 重算
          （见 model_runner.prepare_decode 的 slot_mapping），所以
          num_cached_tokens 恰好比 num_tokens 少 1。
          于是「整块落在 num_cached_tokens 之内」⇔「这一块的 KV 全部有效」，
          边界严丝合缝，不会登记到 KV 还没落地的块。

        ★★ 投机解码的正确性红线（错了不报错，只会静默输出错误 token）：
          只能登记【已确认 + 整块填满 + block_table 里确实存在】的块。
          草稿候选存在 seq.draft_tokens，刻意不进 token_ids，就是为了让
          这里的 seq.block(i) 永远碰不到被拒候选。越界块、半满块、
          以及「前一块还没登记过」的断链块，一律不登记：
          登记错了 → 下次相同前缀命中错误缓存 → 静默输出错误 token。
        """
        if num_new_tokens <= 0:
            return
        bs = self.block_size
        start = (seq.num_cached_tokens - num_new_tokens) // bs
        end = min(seq.num_cached_tokens // bs, len(seq.block_table))
        if start >= end:
            return
        if start > 0:
            h = self.blocks[seq.block_table[start - 1]].hash
            if h == -1:
                # 前一块从来没登记过（窗口不连续），拿不到正确的链前缀。
                # 宁可这一步不登记，也不能用断链的哈希污染前缀缓存。
                return
        else:
            h = -1
        for i in range(start, end):
            token_ids = seq.block(i)
            if len(token_ids) < bs:
                # 半满块：token_ids 里还没攒满 bs 个已确认 token（越界时切片更短）。
                # 它还没有资格代表一个完整前缀，等填满再说。
                break
            block = self.blocks[seq.block_table[i]]
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    # ==================================================================
    #  draft 侧 KV 的有效性追踪（前缀缓存命中的「全量补齐」修复）
    # ==================================================================
    def mark_draft_valid(self, seq: Sequence, draft_valid_len: int):
        """把「draft 已经按正确前缀算过」的完整块盖上一个 draft_hash。

        契约
        ----
        draft_valid_len = `Sequence.draft_valid_len`，即「从位置 0 起连续有效的
        draft KV 个数」。它由 prepare_prefill / propose / postprocess_spec 维护：
          · prefill 的每个 chunk 成功后推进到 chunk 末尾；
          · 每轮 propose 后 = 最后已确认位置 + k，再由 postprocess_spec 夹到
            num_tokens - 1（最后一个 bonus 位置没有 draft KV，是个真缺口）；
          · gate 关闭期间【不推进】——target 自己生成的 token draft 没看过；
          · 抢占归零。

        所以「整块落在水位之内」⇔「这一块的每个位置都由 draft 在正确的已确认
        前缀下算过」。此时才把 block.draft_hash 置成 block.hash（内容哈希）。
        未登记进前缀缓存（hash == -1）的块跳过：它本来就不会被命中复用。

        ★ 不做任何假设、只做盖章：这个方法永远不会让 draft 以为某块有效，
          除非水位真的覆盖到了它。
        """
        if draft_valid_len <= 0:
            return
        bs = self.block_size
        n = min(int(draft_valid_len) // bs, len(seq.block_table))
        for i in range(n):
            block = self.blocks[seq.block_table[i]]
            if block.hash != -1:
                block.draft_hash = block.hash

    def draft_valid_cached_blocks(self, seq: Sequence, num_cached_blocks: int) -> int:
        """前缀缓存命中的块里，draft 侧也确认有效的【连续前缀块数】。

        调用时机：can_allocate 返回 num_cached_blocks 之后、allocate 之后。
        can_allocate 已经保证这 num_cached_blocks 个块的内容就是 seq 的前缀
        （链式哈希相等 + token_ids 逐个相等），而块内容是不可变的
        （被命中的块 ref_count>0，不会被 _allocate_block 回收重写），
        所以这里只需确认 draft 侧在【同一个物理块】上算过同样的内容：

            block.draft_hash == block.hash

        相等即有效。任何一块断了（-1 或哈希对不上）就到此为止 —— 只返回
        连续有效的那一段，剩下的交给 proposer 用 catchup 补齐。

        ★ 这就是「target 命中 ≠ draft 有效」这条红线的落点：draft_hash 只由
          mark_draft_valid 盖，绝不从前缀缓存事件推断。
        """
        n = 0
        for i in range(min(int(num_cached_blocks), len(seq.block_table))):
            block = self.blocks[seq.block_table[i]]
            if block.draft_hash == -1 or block.draft_hash != block.hash:
                break
            n += 1
        return n
