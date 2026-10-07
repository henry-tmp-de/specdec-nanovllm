"""用 torch.profiler 读【纯 kernel GPU 时间】，和墙钟时间对比。

为什么要做这个
--------------
之前用「50 次连发 + CUDA 事件包住整段」测墙钟时间，得到 v3 = 67us。
但改了 block_n / num_warps / num_stages / 布局【全都不动】——
一个真正的访存瓶颈不该这么无动于衷。

可疑点：v3 是【两个 kernel 启动】（partial + combine），
Triton 每次启动的 Python 侧开销 5~15us，两次就是 10~30us。
如果墙钟时间里混着 CPU 的排队时间，那「29% 带宽」这个结论就是错的。

本脚本直接读 profiler 给的 cuda_time，区分开：
    墙钟时间 = kernel GPU 时间 + CPU 启动/排队时间
如果不是访存瓶颈，那么 kernel 时间应该明显小于墙钟时间。

跑：CUDA_VISIBLE_DEVICES=7 python scripts/prof_paged_decode.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.profiler import profile, ProfilerActivity

from nanovllm.kernels.paged_decode_attn import paged_decode_attention
from bench_paged_decode import make_case, bytes_moved, NUM_KV_HEADS, SCALE

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_FA = True
except Exception:                                          # noqa: BLE001
    HAS_FA = False

CTX = 4096
ITERS = 30
q, kc, vc, bt, cl = make_case(CTX, num_seqs=1, seed=1)
q4 = q.unsqueeze(1)
b_ideal = bytes_moved(CTX, 1, 1)

VARIANTS = {
    "v1": lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=1),
    "v2": lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=2),
    "v3": lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=3),
}
if HAS_FA:
    VARIANTS["flash-attn"] = lambda: flash_attn_with_kvcache(
        q4, kc, vc, cache_seqlens=cl, block_table=bt, softmax_scale=SCALE, causal=True)


def wall_us(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def gpu_us(fn, iters=ITERS):
    """用 profiler 读纯 kernel 时间（每次调用所有 kernel 的 cuda_time 之和）。"""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA or ev.self_device_time_total > 0:
            total += ev.self_device_time_total
    return total / iters          # profiler 的单位已经是 us


print("=" * 78)
print(f"纯 kernel GPU 时间 vs 墙钟时间（ctx={CTX}）")
print("=" * 78)
print(f"{'变体':<12}{'墙钟(us)':>10}{'kernel(us)':>12}{'CPU开销':>10}"
      f"{'kernel带宽':>12}{'占天花板':>10}")
print("-" * 78)

rows = []
for name, fn in VARIANTS.items():
    w = wall_us(fn)
    g = gpu_us(fn)
    bw_gpu = b_ideal / (g * 1e-6) / 1e9
    rows.append((name, w, g))
    print(f"{name:<12}{w:>10.2f}{g:>12.2f}{w - g:>10.2f}"
          f"{bw_gpu:>10.1f}G{bw_gpu/842.0*100:>9.1f}%")

print()
print("★ 读法：")
print("  - 如果『kernel 时间』明显小于『墙钟时间』，说明墙钟里混了 CPU 排队时间，")
print("    之前用墙钟算出来的带宽是【低估】的。")
print("  - 比『kernel 带宽』才公平 —— 那才是 kernel 真正把带宽用到了多少。")
print("  - flash-attn 同样处理：它的墙钟里也有一次 Python 启动开销。")

# ---------------- 决定性验证：用 CUDA Graph 把启动开销消掉 ----------------
print()
print("=" * 78)
print("★ 决定性验证：把 attention 整个用 CUDA Graph 捕获，看墙钟掉不掉")
print("=" * 78)
print("  如果 v3 的墙钟从 77us 掉到 ~33us，就证明那 45us 确实是启动开销，")
print("  而不是 kernel 慢 —— 也就是说，我前面的结论（『kernel 只到 29%』）是错的。")
print()


def cuda_graph_us(fn, iters=ITERS):
    """把 fn 捕获成一张 CUDA Graph，再测 replay 的墙钟时间。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                 # 必须先热够，让分配都发生在 capture 之前
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters * 1000.0


print(f"{'变体':<12}{'墙钟(us)':>10}{'kernel(us)':>12}{'+CUDA Graph':>14}"
      f"{'Graph后带宽':>13}")
print("-" * 78)
for name, fn in VARIANTS.items():
    w = wall_us(fn)
    g = gpu_us(fn)
    try:
        gw = cuda_graph_us(fn)
        bw = b_ideal / (gw * 1e-6) / 1e9
        print(f"{name:<12}{w:>10.2f}{g:>12.2f}{gw:>14.2f}{bw:>11.1f}G")
    except Exception as ex:                              # noqa: BLE001
        print(f"{name:<12}{w:>10.2f}{g:>12.2f}   capture 失败: {str(ex)[:28]}")

print()
print("  注：flash-attn 的墙钟只有 43us 是因为它只启动 1 次；")
print("      v3 要启动 2 次（partial + combine），Python 侧开销翻倍还多。")
