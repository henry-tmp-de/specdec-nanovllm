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

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


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

    def hash_blocks(self, seq: Sequence):
        """把「已确认」的 block 登记进前缀缓存哈希表。

        ★★ 投机解码的正确性红线：
          这里必须只用【已确认的 token】。草稿候选存在 seq.draft_tokens 里，
          刻意不进 token_ids，就是为了让这个函数永远碰不到被拒的 token。
          如果候选混进了 token_ids，被拒后 token_ids 变了而旧哈希还在
          hash_to_block_id 里 → 下次相同前缀会命中错误缓存 → 静默输出错误 token。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
