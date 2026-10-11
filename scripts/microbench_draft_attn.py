"""microbench_draft_attn.py — draft decode attention 的形状/块表宽度敏感性

不加载模型、不启引擎、不占 2333 端口。可与其它实验并行跑（换张卡）。

问题：
  1. draft 的 attention（1 query, 16 q heads, 8 kv heads, hd 128）在 ctx=4096 下
     到底要多久？块表宽度（pad 到 32 列 vs 正好 17 列）影响多大？
  2. 在 CUDA graph 里被 capture 时（capture 那刻 cache_seqlens=0）kernel 时间是否
     与 eager 不同 —— 验证「num_splits 被冻结」的猜想。
"""
import os
import json
import torch
from flash_attn import flash_attn_with_kvcache

BS = 256
HD = 128
DT = torch.bfloat16
DEV = "cuda"


def mk(nh, kvh, nb, ctx_kv_max=8192):
    nb = max(nb, (ctx_kv_max + BS - 1) // BS)
    k = torch.randn(nb, BS, kvh, HD, device=DEV, dtype=DT) * 0.1
    v = torch.randn(nb, BS, kvh, HD, device=DEV, dtype=DT) * 0.1
    q = torch.randn(1, 1, nh, HD, device=DEV, dtype=DT) * 0.1
    return q, k, v


def bench(fn, iters=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def graph_bench(fn, iters=200):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return bench(lambda: g.replay(), iters, warm=20)


def run_case(name, nh, kvh, ctx, widths, scale):
    q, k, v = mk(nh, kvh, 64)
    lens = torch.tensor([ctx], device=DEV, dtype=torch.int32)
    out = {}
    need = (ctx + BS - 1) // BS
    for w in widths:
        if w < need:
            continue
        bt = torch.full((1, w), -1, device=DEV, dtype=torch.int32)
        bt[0, :need] = torch.arange(need, device=DEV, dtype=torch.int32)

        def call():
            return flash_attn_with_kvcache(q, k, v, cache_seqlens=lens,
                                            block_table=bt, softmax_scale=scale,
                                            causal=True)
        t_eager = bench(call)
        # graph captured with cache_seqlens=0 (就像引擎 capture_draft_cudagraph 那样)
        lens0 = torch.zeros_like(lens)
        bt0 = torch.zeros_like(bt)
        call0 = lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens0,
                                                block_table=bt0, softmax_scale=scale,
                                                causal=True)
        call0()
        torch.cuda.synchronize()
        g0 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g0):
            call0()
        torch.cuda.synchronize()
        # 把真实值写进静态缓冲再 replay
        with torch.inference_mode():
            lens0.copy_(lens)
            bt0.copy_(bt)
        t_g0 = bench(lambda: g0.replay(), iters=200, warm=20)
        # graph captured with real seqlens
        t_greal = graph_bench(call, iters=200)
        out[w] = {"eager": round(t_eager, 2), "graph_seqlens0": round(t_g0, 2),
                  "graph_real": round(t_greal, 2)}
        print("  %-10s ctx=%5d width=%3d | eager=%7.2f  graph(cap0)=%7.2f  graph(real)=%7.2f us"
              % (name, ctx, w, t_eager, t_g0, t_greal))
    return out


def main():
    scale = HD ** -0.5
    res = {}
    print("=== microbench_draft_attn gpu=%s ===" % os.environ.get("CUDA_VISIBLE_DEVICES"))
    for ctx in [1, 1024, 4096]:
        res["draft_ctx%d" % ctx] = run_case("draft", 16, 8, ctx, [17, 20, 24, 32, 64], scale)
    res["target_ctx4096"] = run_case("target", 32, 8, 4096, [16, 20, 32, 64], scale)
    # num_splits 显式控制（flash-attn 支持 num_splits kwarg）
    print("\n[explicit num_splits]")
    q, k, v = mk(16, 8, 64)
    lens = torch.tensor([4096], device=DEV, dtype=torch.int32)
    bt = torch.full((1, 32), -1, device=DEV, dtype=torch.int32)
    bt[0, :16] = torch.arange(16, device=DEV, dtype=torch.int32)
    for ns in [0, 1, 2, 4, 8]:
        try:
            f = lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens,
                                                block_table=bt, softmax_scale=scale,
                                                causal=True, num_splits=ns)
            t = bench(f)
            res["draft_ns%d" % ns] = round(t, 2)
            print("  num_splits=%2d -> %.2f us" % (ns, t))
        except Exception as ex:
            print("  num_splits=%2d 不支持: %s" % (ns, ex))
    print("@@ATTN@" + json.dumps(res))


if __name__ == "__main__":
    main()
