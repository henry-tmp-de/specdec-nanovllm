"""CUDA 版 paged decode attention：正确性对拍 + 纯 kernel 时间三方对比。

口径（和 scripts/prof_paged_decode.py 完全一致，不然没法比）：
  * 纯 kernel GPU 时间 = torch.profiler 的 self_device_time_total 累加，
    不是墙钟 —— Triton 每次启动的 Python 开销 ~20us 会把 kernel 盖住。
  * 带宽分母 = 实测天花板 842 GB/s（纯 copy 流式测出来的），不是规格值 936。

跑：
  CUDA_VISIBLE_DEVICES=7 python scripts/bench_cuda_paged_decode.py            # 全量
  CUDA_VISIBLE_DEVICES=7 python scripts/bench_cuda_paged_decode.py --quick    # 只跑 ctx=4096
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.profiler import profile, ProfilerActivity

from bench_paged_decode import make_case, bytes_moved, NUM_Q_HEADS, NUM_KV_HEADS, SCALE
from nanovllm.kernels.paged_decode_attn import (
    paged_decode_attention, paged_decode_attention_ref)
from nanovllm.kernels.paged_decode_attn_cuda import (
    paged_decode_attention_cuda, paged_decode_attention_cuda_partial, get_ext)

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_FA = True
except Exception:                                          # noqa: BLE001
    HAS_FA = False

PEAK = 841.8          # 实测天花板 GB/s
CTX_LIST = [256, 512, 1024, 2048, 4096]

DEFAULT_CFG = dict(splits=8, block_n=64, warps=4, stages=3, fused=True)


def gpu_us(fn, iters=50, warmup=10):
    """纯 kernel GPU 时间（us），profiler 单位就是 us。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            total += ev.self_device_time_total
    return total / iters


def kernel_breakdown(fn, iters=50):
    """把每个 kernel 的时间单独列出来。"""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = []
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            out.append((ev.key[:60], ev.self_device_time_total / iters))
    out.sort(key=lambda x: -x[1])
    return out


def correctness(cfgs):
    print("=" * 96)
    print("【1】正确性：对 fp32 torch 参照 + 对 flash-attn（同为 bf16 口径）")
    print("=" * 96)
    print(f"{'ctx':>6} {'cfg':>28} {'vs fp32':>12} {'vs flash-attn':>14} {'ref|max|':>10}  verdict")
    allok = True
    results = []
    for L in [1, 7, 256, 300, 1024, 4096]:
        q, kc, vc, bt, cl = make_case(L, num_seqs=2, seed=L)
        ref = paged_decode_attention_ref(q, kc, vc, bt, cl, SCALE).float()
        if HAS_FA:
            fa = flash_attn_with_kvcache(q.unsqueeze(1), kc, vc, cache_seqlens=cl,
                                         block_table=bt, softmax_scale=SCALE,
                                         causal=True).squeeze(1).float()
        refmax = ref.abs().max().item()
        for name, cfg in cfgs.items():
            out = paged_decode_attention_cuda(q, kc, vc, bt, cl, SCALE, **cfg).float()
            d_ref = (out - ref).abs().max().item()
            d_fa = (out - fa).abs().max().item() if HAS_FA else float("nan")
            # bf16 天然精度下限就在 1e-2 绝对误差量级；对 flash-attn 同为 bf16 要求更严
            ok = (d_ref < 2e-2) and (not HAS_FA or d_fa < 1e-2)
            if not ok:
                allok = False
            print(f"{L:>6} {name:>28} {d_ref:>12.5f} {d_fa:>14.5f} {refmax:>10.3f}  "
                  f"{'OK' if ok else 'FAIL'}")
            results.append(dict(ctx=L, cfg=name, d_ref=d_ref, d_fa=d_fa, ok=ok))
    print()
    print(f"  → 全部通过：{allok}")
    return allok, results


def perf(cfgs, ctx_list):
    print()
    print("=" * 96)
    print("【2】纯 kernel GPU 时间（torch.profiler）：CUDA vs Triton v3 vs flash-attn")
    print("=" * 96)
    print(f"{'ctx':>6} {'Triton v3':>11} {'flash-attn':>11} " +
          " ".join(f"{n:>14}" for n in cfgs) + f" {'best CUDA':>10} {'vs v3':>8} {'带宽':>9}")
    rows = []
    for L in ctx_list:
        q, kc, vc, bt, cl = make_case(L, num_seqs=1, seed=1)
        q4 = q.unsqueeze(1)
        t3 = gpu_us(lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=3))
        tf = gpu_us(lambda: flash_attn_with_kvcache(
            q4, kc, vc, cache_seqlens=cl, block_table=bt, softmax_scale=SCALE,
            causal=True)) if HAS_FA else float("nan")
        tc = {}
        for name, cfg in cfgs.items():
            tc[name] = gpu_us(lambda c=cfg: paged_decode_attention_cuda(
                q, kc, vc, bt, cl, SCALE, **c))
        best = min(tc.items(), key=lambda kv: kv[1])
        bw = bytes_moved(L, 1, 1) / (best[1] * 1e-6) / 1e9
        print(f"{L:>6} {t3:>11.2f} {tf:>11.2f} " +
              " ".join(f"{v:>14.2f}" for v in tc.values()) +
              f" {best[0]:>10} {t3/best[1]:>7.2f}x {bw:>7.1f}G")
        rows.append(dict(ctx=L, triton_v3=t3, fa=tf, cuda=tc,
                         best_cfg=best[0], best_us=best[1],
                         speedup_vs_v3=t3 / best[1], bw_gbps=bw))
    return rows


def breakdown(ctx, cfg):
    print()
    print("=" * 96)
    print(f"【3】逐 kernel 拆解（ctx={ctx}）—— CUDA 版 vs Triton v3")
    print("=" * 96)
    q, kc, vc, bt, cl = make_case(ctx, num_seqs=1, seed=1)

    for tag, fn in [("Triton v3", lambda: paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=3)),
                    ("CUDA", lambda: paged_decode_attention_cuda(q, kc, vc, bt, cl, SCALE, **cfg))]:
        rows = kernel_breakdown(fn)
        print(f"  {tag}:  合计 {sum(v for _, v in rows):.2f} us")
        for k, v in rows:
            print(f"      {v:>9.2f} us   {k}")

    # 单独把 combine 拎出来反复跑，看它自己的稳态时间（排除主 kernel 的干扰）
    pm = paged_decode_attention_cuda_partial(q, kc, vc, bt, cl, SCALE, **cfg)
    pl = torch.empty_like(pm)
    pacc = torch.empty(pm.shape[0], pm.shape[1], pm.shape[2], 4, 128,
                       dtype=torch.float32, device=pm.device)
    ext = get_ext()
    tc = gpu_us(lambda: ext.combine_only(pm, pl, pacc, 32, 8, cfg["splits"]))
    print(f"  单独反复跑 combine kernel：{tc:.2f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs", type=str, default="")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    if args.configs:
        cfgs = {}
        for item in args.configs.split(";"):
            name, kv = item.split("=")
            d = {}
            for p in kv.split(","):
                k, v = p.split(":")
                d[k] = int(v)
            cfgs[name] = d
    else:
        cfgs = {
            "s8/n64/w4/st3": dict(splits=8, block_n=64, warps=4, stages=3, fused=True),
            "s8/n64/w4/st1": dict(splits=8, block_n=64, warps=4, stages=1, fused=True),
        }
        if args.quick:
            cfgs = {"s8/n64/w4/st3": cfgs["s8/n64/w4/st3"]}

    ctx_list = [4096] if args.quick else CTX_LIST

    print(f"GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability()}")
    ok_cfgs = {"s8/n64/w4/st3": dict(splits=8, block_n=64, warps=4, stages=3, fused=True)}
    ok, cons = correctness(ok_cfgs)
    rows = perf(cfgs, ctx_list) if ok else []
    breakdown(4096, list(cfgs.values())[0])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(correct=ok, correctness=cons, rows=rows), f,
                      ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
