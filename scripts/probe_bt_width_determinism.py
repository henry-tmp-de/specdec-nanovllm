"""微探针：flash_attn_with_kvcache 的输出对【块表宽度 / 物理块地址】是否 bitwise 不变？

动机（P7-C 逐位对拍发现的残留差异）
------------------------------------
W=0 与 W=2048（L=1024，窗口盖住全部上下文）的逐次对拍里：
  · 前 13 次 draft 前向逐一 bitwise 相同（这批走 CUDA graph，图的块表宽度
    在 capture 时固定 18，两边同宽、只有表值不同）；
  · 第 14 次起不同 —— 第 14 次正是【第一个走 eager 的前向】（batch=3 不在
    图桶 {1,2,4} 里），两边 tokens / positions / context_lens 完全相同，
    唯一差别是 eager 路径块表宽度 = max(len(bt))：5（全上下文）vs 8（滑窗环）：
        A(W=0)   : [[5, 6, 7, 8, 21], ...]                 宽 5
        B(W=2048): [[8, 9, 10, 11, 12, 13, 14, 15], ...]   宽 8

本脚本判三件事（只用 flash_attn，★ 不加载模型、不启引擎、不占 2333 端口）：
  1. 宽度：前 5 块同一份内容、额外表项随意 → 加宽是否改变输出
  2. 物理块地址：两块表指向【不相交】的物理块、但逻辑内容逐位相同 → 是否改变输出
  3. 同一份 cache 只加宽表

用法: python scripts/probe_bt_width_determinism.py [device]
"""
import sys

import torch
from flash_attn import flash_attn_with_kvcache

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
BLOCK, KVH, HD, NH = 256, 8, 128, 32
SEQ = 1038                        # 实机：需要逻辑块 0..4（1038 = 4*256+14）
SCALE = HD ** -0.5
NB = 64


def run(q, k, v, lens, table):
    bt = torch.tensor([table], dtype=torch.int32, device=q.device)
    return flash_attn_with_kvcache(q, k, v, cache_seqlens=lens, block_table=bt,
                                   softmax_scale=SCALE, causal=True).clone()


def d(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    dev = torch.device(DEV)
    torch.manual_seed(0)
    q = torch.randn(1, 1, NH, HD, dtype=torch.bfloat16, device=dev)
    lens = torch.tensor([SEQ], dtype=torch.int32, device=dev)
    KT = torch.randn(8, BLOCK, KVH, HD, dtype=torch.bfloat16, device=dev)
    VT = torch.randn_like(KT)      # 8 个"逻辑块"的内容

    def build(pairs, extra_phys=()):
        """把逻辑块 lb 的内容放到物理块 p；extra_phys 填随机。"""
        k = torch.empty(NB, BLOCK, KVH, HD, dtype=torch.bfloat16, device=dev)
        v = torch.empty_like(k)
        for lb, p in pairs:
            k[p].copy_(KT[lb]); v[p].copy_(VT[lb])
        for p in extra_phys:
            k[p].copy_(torch.randn(BLOCK, KVH, HD, dtype=torch.bfloat16, device=dev))
            v[p].copy_(torch.randn(BLOCK, KVH, HD, dtype=torch.bfloat16, device=dev))
        return k, v

    print("== 1. 宽度：前 5 块同一份内容，额外表项=有效的另一批块 ==")
    k1, v1 = build([(i, i) for i in range(8)], extra_phys=range(8, 40))
    ref = run(q, k1, v1, lens, [0, 1, 2, 3, 4])
    for w in (5, 6, 8, 12, 16, 18, 24, 32):
        o = run(q, k1, v1, lens, list(range(w)))
        print(f"   width={w:>2} → max|Δ| vs width=5 = {d(ref, o):.8f}"
              + ("  （逐位相同）" if d(ref, o) == 0 else "  ★ 不同"))

    print("\n== 2. 物理块地址：两块表指向【不相交】物理块，逻辑内容逐位相同 ==")
    tblA = [5, 6, 7, 8, 21]
    kA, vA = build([(0, 5), (1, 6), (2, 7), (3, 8), (4, 21)]
                   + [(i, 40 + i) for i in (5, 6, 7)], extra_phys=[43, 44])
    tblB = [32, 33, 34, 35, 36, 37, 38, 39]
    kB, vB = build([(i, 32 + i) for i in range(8)], extra_phys=[49, 50])
    oA, oB = run(q, kA, vA, lens, tblA), run(q, kB, vB, lens, tblB)
    print(f"   同内容、不同物理块、宽 5 vs 宽 8 → max|Δ| = {d(oA, oB):.8f}"
          + ("  （逐位相同）" if d(oA, oB) == 0 else "  ★ 不同"))
    kC, vC = build([(i, 48 + i) for i in range(5)] + [(i, 56 + i) for i in (5, 6, 7)],
                   extra_phys=[60, 61])
    kD, vD = build([(i, i) for i in range(5)] + [(i, 16 + i) for i in (5, 6, 7)],
                   extra_phys=[20, 21])
    oC = run(q, kC, vC, lens, [48, 49, 50, 51, 52])
    oD = run(q, kD, vD, lens, [0, 1, 2, 3, 4])
    print(f"   同宽 5、同内容、物理地址不同 → max|Δ| = {d(oC, oD):.8f}"
          + ("  （逐位相同：地址无关）" if d(oC, oD) == 0 else "  ★ 不同"))

    print("\n== 3. 同一份 cache 只加宽表（内容不动）==")
    kE, vE = build([(i, i) for i in range(8)], extra_phys=[40, 41])
    o5 = run(q, kE, vE, lens, [0, 1, 2, 3, 4])
    for w in (6, 8, 12, 18, 32):
        print(f"   宽 5 → 宽 {w:>2} = {d(o5, run(q, kE, vE, lens, list(range(w)))):.8f}")

    print("\n== 4. 额外表项 = -1（引擎图路径的填充值）==")
    k4, v4 = build([(i, i) for i in range(8)], extra_phys=[40, 41])
    for extra in ([-1], [0], [3]):
        tbl = [0, 1, 2, 3, 4] + extra
        try:
            o = run(q, k4, v4, lens, tbl)
            nan = bool(torch.isnan(o).any())
            print(f"   表 [0..4]+{extra} → max|Δ| vs 宽5 = {d(o5, o):.8f}  NaN={nan}")
        except Exception as e:
            print(f"   表 [0..4]+{extra} → 异常 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
