"""回归测试（纯 CPU）：投机一步要写 n 个 token 时，block 必须分配够。

真踩过的 bug（会引发 CUDA illegal memory access）
-----------------------------------------------
本步真正被写的 token 位于

    position:  len-1, len, ..., len+n-2

（`prepare_decode` / `prepare_verify` 的 slot_mapping 都是从 len-1 起算的，
  因为上一个 token 的 KV 要重算一遍。）
所以最后写入的位置是 len+n-2，不是 len+n-1。

旧实现按「len .. len+n-1」算当前 block 还剩多少空位：

    remaining_in_block = block_size - (len % block_size)

当 len 正好是 block_size 的整数倍时，它给出 remaining = block_size
（以为当前 block 空着），**其实当前 block 已经正好写满** ——
于是少分配一块，block_table 里那一位是填充值 -1。
attention 拿到的 cache_seqlens 要求跨两块时，flash-attn 就去读第 -1 页：

* eager 路径：读到别处的垃圾内存，不报错，但 KV 是错的；
* 拍成 CUDA graph 后：直接 illegal memory access 崩溃。

64 个 token 的短生成永远碰不到 256 的边界，所以这个 bug 藏了很久；
把 max_tokens 从 64 改成 256 立刻崩。

跑：python tests/test_block_alloc.py
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)

from nanovllm.engine.block_manager import BlockManager      # noqa: E402
from nanovllm.engine.sequence import Sequence               # noqa: E402
from nanovllm.sampling_params import SamplingParams         # noqa: E402

BS = 256
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def fresh(seq_len: int, num_blocks: int = 32):
    bm = BlockManager(num_blocks, BS)
    seq = Sequence([0] * seq_len, SamplingParams(temperature=1.0, max_tokens=8))
    bm.allocate(seq, 0)
    return bm, seq


print("=" * 70)
print("测试：may_append / can_append 的 block 数")
print("=" * 70)

# ---- 关键用例：len 正好是 block_size 整数倍 ----
bm, seq = fresh(256)
check("allocate 后是 1 块", len(seq.block_table) == 1, f"{len(seq.block_table)}")
check("len=256, n=1 时不用补块（只写位置 255）",
      bm._blocks_needed(seq, 1) == 1, f"{bm._blocks_needed(seq, 1)}")
check("len=256, n=3（投机 k=2）时补 1 块（要写 255/256/257）",
      bm._blocks_needed(seq, 3) == 2, f"{bm._blocks_needed(seq, 3)}")
bm.may_append(seq, 3)
check("may_append 后确实是 2 块", len(seq.block_table) == 2, f"{len(seq.block_table)}")

# ---- 普通 decode 跨边界：len=257 -> 要写位置 256 -> 补 1 块 ----
bm, seq = fresh(257)
bm.may_append(seq, 1)
check("普通 decode：len=257 时补 1 块（要写位置 256）",
      len(seq.block_table) == 2, f"{len(seq.block_table)}")

# ---- 边界内的情形不该多分配 ----
for seq_len, n, want in [(1, 2, 1), (254, 3, 1), (255, 1, 1), (300, 1, 2), (300, 3, 2)]:
    bm, seq = fresh(seq_len)
    got = bm._blocks_needed(seq, n)
    check(f"len={seq_len}, n={n} -> 需要 {want} 块", got == want, f"得到 {got}")

# ---- can_append 与实际分配一致：判 true 就必须真的分配得出来 ----
ok = True
for seq_len in [1, 200, 255, 256, 257, 511, 512, 513]:
    for n in [1, 2, 3, 4, 7]:
        bm, seq = fresh(seq_len, num_blocks=8)
        if bm.can_append(seq, n):
            before = len(seq.block_table)
            bm.may_append(seq, n)
            need = bm._blocks_needed(seq, n)
            if len(seq.block_table) < need:
                ok = False
                print(f"    ✗ len={seq_len} n={n}: can_append 说可以，"
                      f"但只分到 {len(seq.block_table)} 块（需 {need}）")
check("can_append 说可以 -> may_append 一定分得够", ok)

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
