"""CUDA 版配置扫描：主 kernel 单独时间 + 完整时间（含 combine）。

用 torch.profiler 读纯 kernel GPU 时间；分两个口径：
  * main  = 只跑 paged_decode_mma_kernel 的时间
  * total = main + combine（这才是「一个算子调用的代价」）

跑：CUDA_VISIBLE_DEVICES=7 python scripts/sweep_cuda_paged_decode.py
"""
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.profiler import profile, ProfilerActivity

from bench_paged_decode import make_case, bytes_moved, SCALE
from nanovllm.kernels.paged_decode_attn import paged_decode_attention
from nanovllm.kernels.paged_decode_attn_cuda import (
    paged_decode_attention_cuda, paged_decode_attention_cuda_partial)

PEAK = 841.8
CTX = 4096


def gpu_us(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(ev.self_device_time_total for ev in prof.key_averages()
               if ev.self_device_time_total > 0) / iters


CFGS = [
    (32, 2, 2), (32, 2, 3), (32, 2, 4),
    (64, 2, 2), (64, 2, 3),
    (64, 4, 2), (64, 4, 3),
    (128, 4, 1), (128, 8, 1),
]
SPLITS = [4, 8, 16, 32]

q, kc, vc, bt, cl = make_case(CTX, num_seqs=1, seed=1)
b = bytes_moved(CTX, 1, 1)

print(f"GPU {torch.cuda.get_device_name(0)}  ctx={CTX}  KV={b/1e6:.2f}MB  "
      f"实测天花板 {PEAK} GB/s")
print(f"Triton v3 参照：")
t3 = gpu_us(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=3))
print(f"  Triton v3 完整 = {t3:.2f} us")

allrows = []
print()
print(f"{'block_n':>8}{'warps':>6}{'stg':>4}{'splits':>7}{'main(us)':>10}"
      f"{'total(us)':>10}{'main带宽':>10}{'占天花板':>9}{'vs Triton':>10}")
print("-" * 84)
for bn, w, st in CFGS:
    for sp in SPLITS:
        cfg = dict(splits=sp, block_n=bn, warps=w, stages=st)
        try:
            tm = gpu_us(lambda c=cfg: paged_decode_attention_cuda_partial(
                q, kc, vc, bt, cl, SCALE, **c))
            tt = gpu_us(lambda c=cfg: paged_decode_attention_cuda(
                q, kc, vc, bt, cl, SCALE, **c))
        except Exception as ex:                       # noqa: BLE001
            print(f"{bn:>8}{w:>6}{st:>4}{sp:>7}   跳过: {str(ex)[:44]}")
            continue
        bw = b / (tm * 1e-6) / 1e9
        print(f"{bn:>8}{w:>6}{st:>4}{sp:>7}{tm:>10.2f}{tt:>10.2f}"
              f"{bw:>9.1f}G{bw/PEAK*100:>8.1f}%{t3/tt:>9.2f}x")
        allrows.append(dict(block_n=bn, warps=w, stages=st, splits=sp,
                            main=tm, total=tt, bw=bw))

allrows.sort(key=lambda r: r["total"])
print()
print("  ★ 完整时间最好的 5 个配置：")
for r in allrows[:5]:
    print(f"    block_n={r['block_n']:>3} warps={r['warps']} stages={r['stages']} "
          f"splits={r['splits']:>2}  main={r['main']:6.2f}us total={r['total']:6.2f}us "
          f"({r['bw']:5.1f}GB/s = 天花板的 {r['bw']/PEAK*100:.1f}%)")

allrows.sort(key=lambda r: r["main"])
print("  ★ 主 kernel 单独最快的 5 个配置：")
for r in allrows[:5]:
    print(f"    block_n={r['block_n']:>3} warps={r['warps']} stages={r['stages']} "
          f"splits={r['splits']:>2}  main={r['main']:6.2f}us total={r['total']:6.2f}us "
          f"({r['bw']:5.1f}GB/s = 天花板的 {r['bw']/PEAK*100:.1f}%)")
