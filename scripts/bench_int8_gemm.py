#!/usr/bin/env python
"""bench_int8_gemm.py —— 融合反量化 INT8 GEMM 的微基准（Phase 2 步骤 b）

回答的问题：**字节省下来了，时间也省下来了吗？**

对照：cuBLAS bf16（`F.linear`，引擎现在跑的就是它） vs 我们的 Triton 融合反量化
kernel（读 int8 + scale，在寄存器里反量化后进 bf16 MMA）。

形状 = Qwen3-4B 的真实 decode 线性层（tp=1）：
    qkv_proj[6144,2560]  o_proj[2560,4096]  gate_up[19456,2560]  down[2560,9728]
    lm_head[151936,2560]（bf16，进「整步」合计但【不量化】）

口径：时间是 torch.profiler 的 device_time 之和，**不是墙钟**。
用法：
  CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_int8_gemm.py            # 默认配置
  CUDA_VISIBLE_DEVICES=0 SWEEP=1 python -u scripts/bench_int8_gemm.py    # 扫配置
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from nanovllm.layers.quant import quantize_weight, dequantize_weight, int8_linear, _BLOCK

DEV = "cuda"
# ★ 范围变更后主战场是 **draft（Qwen3-0.6B）的 FFN**，target 形状保留做参照。
DRAFT_FFN = [("draft_gate_up", 6144, 1024), ("draft_down", 1024, 3072)]
DRAFT_ATTN = [("draft_qkv", 4096, 1024), ("draft_o", 1024, 2048)]
TARGET_ALL = [("qkv_proj", 6144, 2560), ("o_proj", 2560, 4096),
              ("gate_up_proj", 19456, 2560), ("down_proj", 2560, 9728)]
WHICH = os.environ.get("SHAPESET", "draft_ffn")
if WHICH == "draft_ffn":
    SHAPES = DRAFT_FFN
elif WHICH == "draft_all":
    SHAPES = DRAFT_FFN + DRAFT_ATTN
else:
    SHAPES = TARGET_ALL
LM_HEAD = None if WHICH.startswith("draft") else ("lm_head", 151936, 2560)
N_LAYERS = 28 if WHICH.startswith("draft") else 36
MS = [int(x) for x in os.environ.get("MS", "1,4,7,16,32").split(",")]
NITER = int(os.environ.get("NITER", "50"))
GRAN = os.environ.get("GRAN", "per_channel")
GS = int(os.environ.get("GS", "128"))
SWEEP = os.environ.get("SWEEP", "0") == "1"

CANDS = [
    dict(BM=16, BN=64, BK=64, warps=4, stages=3, split=1),
    dict(BM=16, BN=128, BK=64, warps=4, stages=3, split=1),
    dict(BM=16, BN=128, BK=64, warps=4, stages=4, split=1),
    dict(BM=16, BN=256, BK=64, warps=8, stages=3, split=1),
    dict(BM=16, BN=64, BK=128, warps=4, stages=3, split=1),
    dict(BM=16, BN=128, BK=64, warps=4, stages=3, split=2),
    dict(BM=16, BN=128, BK=64, warps=4, stages=3, split=4),
    dict(BM=16, BN=128, BK=64, warps=4, stages=3, split=8),
    dict(BM=16, BN=64, BK=64, warps=4, stages=4, split=4),
    dict(BM=32, BN=128, BK=64, warps=8, stages=3, split=1),
]


def cfg_str(c):
    return (f"BM{c['BM']}xBN{c['BN']}xBK{c['BK']}w{c['warps']}s{c['stages']}"
            f"k{c['split']}")


def dev_time_us(fn, iters):
    prof = profile(activities=[ProfilerActivity.CUDA])
    with prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for ev in prof.events():
        if getattr(ev, "device_type", None) == torch.autograd.DeviceType.CUDA:
            tot += float(getattr(ev, "device_time", 0) or 0)
    del prof
    return tot / iters


def main():
    print(f"=== bench_int8_gemm set={WHICH} gran={GRAN} gs={GS} ms={MS} iters={NITER} sweep={SWEEP} "
          f"dev={torch.cuda.get_device_name(0)} ===")
    torch.manual_seed(0)
    allnames = SHAPES + ([LM_HEAD] if LM_HEAD else [])
    weights = {}
    for name, N, K in allnames:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) * 0.02
        if LM_HEAD and name == LM_HEAD[0]:
            weights[name] = (w, None, None)
            continue
        q, s = quantize_weight(w, GRAN, GS)
        weights[name] = (w, q.to(DEV), s.to(DEV))

    if SWEEP:
        for M in [1, 16]:
            print(f"\n### SWEEP M={M}")
            for name, N, K in SHAPES:
                w, q, s = weights[name]
                x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.5
                tb = dev_time_us(lambda: F.linear(x, w), NITER)
                best = None
                ref = F.linear(x, dequantize_weight(q.cpu(), s.cpu(), GRAN, GS)
                               .to(torch.bfloat16).to(DEV))
                for c in CANDS:
                    try:
                        y = int8_linear(x, q, s, None, GRAN, GS, cfg_=c)
                        errmax = (y.float() - ref.float()).abs().max().item() /                             max(ref.float().abs().max().item(), 1e-9)
                        t = dev_time_us(lambda: int8_linear(x, q, s, None, GRAN, GS, cfg_=c),
                                        NITER)
                    except Exception as e:
                        print(f"    {name:>12} {cfg_str(c):>22} FAIL {type(e).__name__}: {e}")
                        continue
                    bw = (q.numel() + s.numel() * 2) / t / 1e3  # GB/s
                    tag = ""
                    if best is None or t < best[1]:
                        best = (cfg_str(c), t); tag = " <-"
                    print(f"    {name:>12} {cfg_str(c):>22} {t:8.2f}us {t/tb:6.3f}x "
                          f"{bw:6.0f}GB/s err={errmax*100:5.2f}%{tag}")
                print(f"    {name:>12} {'cublas bf16':>22} {tb:8.2f}us 1.000x "
                      f"{(w.numel()*2)/tb/1e3:6.0f}GB/s   BEST={best[0]} {tb/best[1]:.3f}x")
        return

    results = []
    cfgs = [None] if not SWEEP else CANDS
    for M in MS:
        row = {"M": M, "layers": {}}
        tot_bf16 = tot_int8 = 0.0
        for name, N, K in SHAPES:
            w, q, s = weights[name]
            x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.5
            ref = F.linear(x, dequantize_weight(q.cpu(), s.cpu(), GRAN, GS)
                           .to(torch.bfloat16).to(DEV))
            y = int8_linear(x, q, s, None, GRAN, GS)
            err = (y.float() - ref.float()).abs().max().item()
            t_bf16 = dev_time_us(lambda: F.linear(x, w), NITER)
            t_int8 = dev_time_us(lambda: int8_linear(x, q, s, None, GRAN, GS), NITER)
            nb = q.numel() + s.numel() * 2
            row["layers"][name] = {
                "M": M, "bf16_us": t_bf16, "int8_us": t_int8,
                "speedup": t_bf16 / t_int8, "max_abs_err": err,
                "ref_absmax": ref.float().abs().max().item(),
                "int8_bytes": nb, "bf16_bytes": w.numel() * 2,
                "int8_GBs": nb / t_int8 / 1e3, "bf16_GBs": w.numel() * 2 / t_bf16 / 1e3,
            }
            tot_bf16 += t_bf16 * N_LAYERS
            tot_int8 += t_int8 * N_LAYERS
        t_lm = 0.0
        if LM_HEAD:
            name, N, K = LM_HEAD
            w, _, _ = weights[name]
            x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.5
            t_lm = dev_time_us(lambda: F.linear(x, w), NITER)
        tot_bf16 += t_lm
        tot_int8 += t_lm
        row.update(full_step_bf16_us=tot_bf16, full_step_int8_us=tot_int8,
                   full_step_speedup=tot_bf16 / tot_int8,
                   int8_weight_bytes=sum(x["int8_bytes"] * N_LAYERS
                                         for x in row["layers"].values()) + (w.numel() * 2 if LM_HEAD else 0),
                   bf16_weight_bytes=sum(x["bf16_bytes"] * N_LAYERS
                                         for x in row["layers"].values()) + (w.numel() * 2 if LM_HEAD else 0))
        results.append(row)
        print("@@R@@" + json.dumps(row, ensure_ascii=False))

    print("\n=== SUMMARY (device time) ===")
    hdr = " | ".join(f"{n:>17}" for n, _, _ in SHAPES)
    print(f"{'M':>4} | {hdr} | full_step (us)")
    for r in results:
        cells = []
        for name, _, _ in SHAPES:
            d = r["layers"][name]
            cells.append(f"{d['bf16_us']:6.1f}->{d['int8_us']:6.1f} {d['speedup']:.2f}x"
                         f" {d['int8_GBs']:4.0f}GB/s")
        print(f"{r['M']:>4} | " + " | ".join(f"{c:>17}" for c in cells) +
              f" | {r['full_step_bf16_us']:8.1f} -> {r['full_step_int8_us']:7.1f}"
              f"  ({r['full_step_speedup']:.3f}x)")
    for r in results:
        rel = max(d["max_abs_err"] / max(d["ref_absmax"], 1e-9) for d in r["layers"].values())
        print(f"  M={r['M']:>3} 最大相对误差 ~{rel*100:.2f}%  权重 "
              f"{r['bf16_weight_bytes']/2**30:.2f} -> {r['int8_weight_bytes']/2**30:.2f} GiB "
              f"({r['int8_weight_bytes']/r['bf16_weight_bytes']:.3f}x)")
    print("@@DONE@@")


if __name__ == "__main__":
    main()
