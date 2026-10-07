"""Paged decode attention：正确性对拍 + 性能 + roofline。

三件事
------
1. **正确性**：自己写的 kernel（v1/v2）和纯 torch 参照实现逐元素对拍。
2. **性能**：v1 / v2 / flash-attn 三方对比，扫不同 context 长度。
3. **roofline**：算出「必须搬多少字节、必须算多少浮点」，
   得出算术密度，和 3090 的拐点比 —— 判定它到底是不是访存瓶颈，
   再看实测带宽离 936 GB/s 峰值还差多远。

跑：CUDA_VISIBLE_DEVICES=7 python scripts/bench_paged_decode.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from nanovllm.kernels.paged_decode_attn import (
    paged_decode_attention, paged_decode_attention_ref)

# ---- Qwen3-4B 的 attention 配置 ----
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 32, 8, 128
BLOCK_SIZE = 256
SCALE = HEAD_DIM ** -0.5

# ---- 3090 硬件参数 ----
PEAK_GBPS = 936.0          # HBM 带宽
PEAK_TFLOPS = 35.6         # fp16（不含 sparsity 的理论峰值）
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GBPS * 1e9)   # 拐点 FLOP/Byte

DTYPE = torch.bfloat16
dev = "cuda"


def make_case(ctx_len, num_seqs=1, seed=0):
    """造一组随机的分页 KV cache + q。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max_blocks * num_seqs + 4          # 多留几块
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          generator=g, dtype=torch.float32).to(DTYPE).to(dev)
    v_cache = torch.randn_like(k_cache)
    # 每条序列给一段连续且互不重叠的物理块
    bt = torch.arange(num_blocks, dtype=torch.int32)[:max_blocks * num_seqs]
    block_table = bt.reshape(num_seqs, max_blocks).to(dev)
    context_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device=dev)
    q = torch.randn(num_seqs, NUM_Q_HEADS, HEAD_DIM,
                    generator=g, dtype=torch.float32).to(DTYPE).to(dev)
    return q, k_cache, v_cache, block_table, context_lens


def bytes_moved(ctx_len, num_seqs, kv_reads):
    """这个 kernel 必须从显存搬多少字节。

    KV cache 是大头：每个位置 K、V 各 num_kv_heads×head_dim 个元素。
    kv_reads = K/V 相对「理想只读一遍」的倍数（v1 的 GQA 重复读算 4 倍）。
    """
    per_pos = 2 * NUM_KV_HEADS * HEAD_DIM * torch.tensor([], dtype=DTYPE).element_size()
    return ctx_len * num_seqs * per_pos * kv_reads


def flops(ctx_len, num_seqs):
    """QK^T 与 PV 各 2×，共 4 × num_q_heads × head_dim × L 次浮点运算。"""
    return 4.0 * NUM_Q_HEADS * HEAD_DIM * ctx_len * num_seqs


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0        # ms -> us


print("=" * 78)
print("Paged decode attention —— 正确性 / 性能 / roofline")
print("=" * 78)
print(f"配置：{NUM_Q_HEADS} q头 / {NUM_KV_HEADS} kv头 / head_dim {HEAD_DIM} / "
      f"block_size {BLOCK_SIZE} / {DTYPE}")
print(f"3090：带宽 {PEAK_GBPS} GB/s，fp16 {PEAK_TFLOPS} TFLOPS，"
      f"roofline 拐点 {RIDGE:.1f} FLOP/Byte")
print()

# ---------------- 1. roofline 判定 ----------------
print("-" * 78)
print("【1】它到底是计算瓶颈还是访存瓶颈？")
print("-" * 78)
print(f"{'context':>8} {'KV字节(v2)':>12} {'FLOPs':>12} {'算术密度':>10} {'判定':>14}")
for L in [256, 1024, 4096]:
    b = bytes_moved(L, 1, kv_reads=1)
    f = flops(L, 1)
    ai = f / b
    verdict = "访存瓶颈" if ai < RIDGE / 4 else ("计算瓶颈" if ai > RIDGE * 4 else "两者之间")
    print(f"{L:>8} {b/1e6:>10.2f}MB {f/1e6:>10.2f}M {ai:>9.2f} {verdict:>14}")
print(f"\n  => 算术密度只有拐点的 1/{RIDGE/4:.0f} 量级 —— 大量时间花在搬 KV 上，")
print(f"     所以唯一有意义的优化是【少搬字节】和【把带宽用满】。")
print()

# ---------------- 2. 正确性 ----------------
print("-" * 78)
print("【2】正确性：两个参照")
print("-" * 78)
print("  口径说明（很重要，不然会误判）：")
print("    * v1 走 fp32 累加，v2/v3 走 tl.dot（bf16 tensor core）——")
print("      bf16 的天然精度下限就在 1e-2 绝对误差量级，拿它去卡 5e-3 是过严。")
print("    * 所以对 fp32 参照用 2e-2 的绝对口径；")
print("    * 和 flash-attn 对拍用 1e-2（同为 bf16 口径）。")
print("      注：v2/v3 走 tl.dot，和最 flash-attn 其实吻合到 2e-3 以内；")
print("          反而是 fp32 的 v1 因为「算得更准」而离 bf16 的 flash-attn 更远。")
print()

try:
    from flash_attn import flash_attn_with_kvcache
    has_fa = True
except Exception as ex:                                   # noqa: BLE001
    print(f"  （flash-attn 不可用：{ex}）")
    has_fa = False

ok = True
for L in [1, 7, 256, 300, 1024, 4096]:
    q, kc, vc, bt, cl = make_case(L, num_seqs=2, seed=L)
    ref = paged_decode_attention_ref(q, kc, vc, bt, cl, SCALE).float()
    fa = (flash_attn_with_kvcache(q.unsqueeze(1), kc, vc, cache_seqlens=cl,
                                  block_table=bt, softmax_scale=SCALE, causal=True
                                  ).squeeze(1).float() if has_fa else None)
    for ver in (1, 2, 3):
        out = paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=ver).float()
        d_ref = (out - ref).abs().max().item()
        line = f"  ctx={L:<5} v{ver}  vs fp32: {d_ref:.5f}"
        if d_ref > 2e-2:
            ok = False
            line += " [FAIL]"
        if fa is not None:
            d_fa = (out - fa).abs().max().item()
            line += f"   vs flash-attn: {d_fa:.5f}"
            if d_fa > 1e-2:
                ok = False
                line += " [FAIL]"
        print(line)
print()

# ---------------- 3. 性能 ----------------
print("-" * 78)
print("【3】性能：v1 / v2 / flash-attn")
print("-" * 78)
print(f"{'ctx':>6} {'v1(us)':>9} {'v2(us)':>9} {'v3(us)':>9} {'flash(us)':>10} "
      f"{'v2带宽':>9} {'v3带宽':>9} {'fa带宽':>9} {'v3/峰值':>8}")
rows = []
for L in [256, 512, 1024, 2048, 4096]:
    q, kc, vc, bt, cl = make_case(L, num_seqs=1, seed=1)
    q4 = q.unsqueeze(1)                                   # flash-attn 要 (b, s, h, d)

    t1 = timeit(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=1))
    t2 = timeit(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=2))
    t3 = timeit(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=3))
    tf = timeit(lambda: flash_attn_with_kvcache(
        q4, kc, vc, cache_seqlens=cl, block_table=bt, softmax_scale=SCALE,
        causal=True)) if has_fa else float("nan")

    b_ideal = bytes_moved(L, 1, kv_reads=1)
    bw2 = b_ideal / (t2 * 1e-6) / 1e9
    bw3 = b_ideal / (t3 * 1e-6) / 1e9
    bwf = b_ideal / (tf * 1e-6) / 1e9 if has_fa else float("nan")
    print(f"{L:>6} {t1:>9.2f} {t2:>9.2f} {t3:>9.2f} {tf:>10.2f} "
          f"{bw2:>8.1f}G {bw3:>8.1f}G {bwf:>8.1f}G {bw3/PEAK_GBPS*100:>7.1f}%")
    rows.append(dict(ctx=L, v1_us=round(t1, 2), v2_us=round(t2, 2), v3_us=round(t3, 2),
                     fa_us=round(tf, 2) if has_fa else None,
                     v3_bw_gbps=round(bw3, 1)))

# ---- split 数扫一遍：并行度是 v3 的命门，要看它什么时候够用 ----
print()
print("v3 的 split 数扫描（ctx=4096，看并行度什么时候喂饱 GPU）：")
print(f"{'splits':>7} {'程序数':>7} {'时间(us)':>10} {'带宽':>10} {'占峰值':>8}")
for sp in [1, 2, 4, 8, 16, 32]:
    q, kc, vc, bt, cl = make_case(4096, num_seqs=1, seed=1)
    t = timeit(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE,
                                              version=3, splits=sp))
    bw = bytes_moved(4096, 1, 1) / (t * 1e-6) / 1e9
    print(f"{sp:>7} {sp * NUM_KV_HEADS:>7} {t:>10.2f} {bw:>9.1f}G {bw/PEAK_GBPS*100:>7.1f}%")

print()
print("  （v1 的「带宽」按它实际搬的字节算：GQA 下 K/V 被 4 个 query 头各读一遍）")
print()
print("  ★ 读法：")
print("    - 如果 v2 明显快于 v1 -> GQA 重复读确实是瓶颈（访存假设成立）")
print("    - v2/峰值 的百分比 = 显存带宽被用掉多少，这是这个 kernel 唯一要紧的效率指标")
print("    - 和 flash-attn 的差距 = 还有多少可挖")

# ---------------- 4. 实测带宽天花板 ----------------
# ncu 在这台机器上被禁了（ERR_NVGPUCTRPERM，拿不到显存计数器），
# 所以用一个纯流式的 copy 来测「这台卡实际能跑出多少带宽」。
# 拿它当 roofline 的天花板，比用 936 GB/s 的规格值硬得多。
print()
print("-" * 78)
print("【4】这台卡实际能跑出多少带宽（roofline 的天花板参照）")
print("-" * 78)
best_bw = 0.0
for mb in [32, 128, 512]:
    n = mb * 1024 * 1024 // 4
    a = torch.empty(n, dtype=torch.float32, device=dev).normal_()
    b = torch.empty_like(a)
    t = timeit(lambda: b.copy_(a), warmup=5, iters=20)
    gbps = 2 * n * 4 / (t * 1e-6) / 1e9        # 读一份 + 写一份
    best_bw = max(best_bw, gbps)
    print(f"  纯 copy {mb:>4}MB：{t:>8.1f} us  ->  {gbps:>7.1f} GB/s")
print(f"\n  实测天花板 ≈ {best_bw:.1f} GB/s（规格值是 {PEAK_GBPS:.0f} GB/s）")
print(f"  下面所有 kernel 的「占峰值」都应该改用它来算才算公允。")
print()
print("  ★ 结论口径：")
for r in rows:
    bw = r["v3_bw_gbps"]
    print(f"    ctx={r['ctx']:<5} v3 实测 {bw:>6.1f} GB/s = 实测天花板的 {bw/best_bw*100:>5.1f}%")

# ---------------- 5. 调参扫描：诊断「为什么比 flash-attn 慢」 ----------------
print()
print("-" * 78)
print("【5】v3 调参扫描（ctx=4096）—— 验证「warp 不够 / 没有流水线」这两个假设")
print("-" * 78)
print("  背景：v3 只有 num_seqs×num_kv_heads×splits = 64 个 CTA，")
print("        Triton 默认 num_warps=4 -> 256 warp 摊到 82 个 SM ≈ 3 warp/SM，")
print("        而 SM 支持 48~64 个。线程太少就藏不住 DRAM 延迟，带宽自然上不去。")
print()
q, kc, vc, bt, cl = make_case(4096, num_seqs=1, seed=1)
b_ideal = bytes_moved(4096, 1, 1)

print(f"{'block_n':>8} {'warps':>6} {'stages':>7} {'时间(us)':>10} {'带宽':>9} {'占实测天花板':>12}")
best = (None, 1e9)
for bn in [64, 128, 256]:
    for nw in [4, 8]:
        for ns in [1, 3]:
            try:
                t = timeit(lambda: paged_decode_attention(
                    q, kc, vc, bt, cl, SCALE, version=3, block_n=bn,
                    num_warps=nw, num_stages=ns))
            except Exception as ex:                        # noqa: BLE001
                print(f"{bn:>8} {nw:>6} {ns:>7}   跳过: {str(ex)[:44]}")
                continue
            bw = b_ideal / (t * 1e-6) / 1e9
            if t < best[1]:
                best = ((bn, nw, ns), t)
            print(f"{bn:>8} {nw:>6} {ns:>7} {t:>10.2f} {bw:>8.1f}G {bw/best_bw*100:>11.1f}%")
if best[0]:
    print(f"\n  最好：block_n={best[0][0]} num_warps={best[0][1]} num_stages={best[0][2]}"
          f" -> {best[1]:.1f} us（对照：flash-attn 42 us，v3 默认 65.5 us）")

# ---------------- 6. KV cache 布局实验 ----------------
print()
print("-" * 78)
print("【6】KV cache 布局实验（kernel 一行没改，只换 tensor 和它的 stride）")
print("-" * 78)
print("  默认布局 (block, offset, kv_head, head_dim)：")
print("    一个 program 只读 128 个元素（256B），然后跳过其余 7 个头（1792B）——")
print("    跨步 2048B 的碎读，DRAM 的行缓冲命中率差。")
print("  head-major (block, kv_head, offset, head_dim)：")
print("    同一个头的 512 个位置连成一片，是 128KB 的连续区间。")
print()
q, kc, vc, bt, cl = make_case(4096, num_seqs=1, seed=1)
kc_h = kc.permute(0, 2, 1, 3).contiguous()
vc_h = vc.permute(0, 2, 1, 3).contiguous()
b4096 = bytes_moved(4096, 1, 1)

t_a = timeit(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE,
                                            version=3, layout="interleaved"))
t_b = timeit(lambda: paged_decode_attention(q, kc_h, vc_h, bt, cl, SCALE,
                                            version=3, layout="head_major"))
oa = paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=3).float()
ob = paged_decode_attention(q, kc_h, vc_h, bt, cl, SCALE, version=3,
                            layout="head_major").float()
for name, t in [("默认 (block, offset, kvhead, d)", t_a),
                ("head-major (block, kvhead, offset, d)", t_b)]:
    bw = b4096 / (t * 1e-6) / 1e9
    print(f"  {name:<38} {t:>8.2f} us  {bw:>7.1f} GB/s  {bw/best_bw*100:>5.1f}%")
print(f"\n  两种布局的输出差异 {float((oa - ob).abs().max()):.6f}（应该 ~0）")

print()
print("@@PD@@" + json.dumps({"ridge": round(RIDGE, 1), "correct": ok,
                            "measured_peak_gbps": round(best_bw, 1),
                            "layout": {"interleaved_us": round(t_a, 2),
                                       "head_major_us": round(t_b, 2)},
                            "best_v3": {"cfg": best[0], "us": round(best[1], 2)} if best[0] else None,
                            "rows": rows}))
