"""microbench_graph_attn.py

验证一个猜想：nano-vllm 在 capture_cudagraph() 时用的是
    context_lens = torch.zeros(bs)      # 全 0
    block_tables = torch.zeros(bs, max_num_blocks)
也就是说 flash_attn_with_kvcache 是在 cache_seqlens=0 的状态下被【capture】进图的。
flash-attn 的 num_splits 是【host 侧】按 max(cache_seqlens) 或 block_table 宽度
算出来的启发式 —— 它一旦在图 capture 时定下来，就【冻结】在整张图里，
之后每次 replay 都按那个值跑，永远不会随真实的 ctx 变化。

所以猜想：graph 模式下 attention kernel 的时间与 ctx 弱相关、且固定偏大。

三个对照（同一个 KV cache、同一个真实 KV 长度）：
  A eager          : 直接调 kernel，cache_seqlens = 真实值
  B graph(cap0)    : 在 cache_seqlens=0 下拍图，replay 前把真实值写进静态缓冲
  C graph(capReal) : 在 cache_seqlens=真实值 下拍图，直接 replay
如果 B 明显慢于 C 且 C ≈ A -> 猜想成立（capture 时的 0 把 num_splits 冻坏了）。

不加载模型、不启引擎、不占 2333 端口。
"""
import torch
import flash_attn
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda"
BS, KVH, HD, NH = 256, 8, 128, 32
DT = torch.bfloat16
MAX_BLOCKS = 64

k_cache = torch.randn(MAX_BLOCKS, BS, KVH, HD, device=dev, dtype=DT)
v_cache = torch.randn(MAX_BLOCKS, BS, KVH, HD, device=dev, dtype=DT)
q = torch.randn(1, 1, NH, HD, device=dev, dtype=DT)   # (batch, seqlen_q=1, nheads, hd)
scale = HD ** -0.5


def mk_bt(ctx, width):
    need = (ctx + BS - 1) // BS
    assert width >= need, (ctx, width, need)
    bt = torch.full((1, width), -1, device=dev, dtype=torch.int32)
    bt[0, :need] = torch.arange(need, device=dev, dtype=torch.int32)
    return bt


def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def eager(ctx, width):
    bt = mk_bt(ctx, width)
    lens = torch.tensor([ctx], device=dev, dtype=torch.int32)
    return timeit(lambda: flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=lens, block_table=bt,
        softmax_scale=scale, causal=True))


def graph_case(ctx, width, capture_lens):
    """capture_lens: 拍图时 cache_seqlens 缓冲里的值（0 或 ctx）。"""
    bt = mk_bt(ctx, width)
    lens = torch.zeros(1, device=dev, dtype=torch.int32)
    if capture_lens:
        lens.fill_(ctx)
    for _ in range(5):      # warmup（图外）
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=lens,
                                block_table=bt, softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=lens,
                                block_table=bt, softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    if not capture_lens:
        lens.fill_(ctx)     # 模拟 run_model 每步写 context_lens
    return timeit(lambda: g.replay(), iters=200)


print(f"flash_attn={flash_attn.__version__} dev={torch.cuda.get_device_name(0)}")
print(f"{'ctx':>6} {'btW':>5} {'A_eager':>9} {'B_graph_cap0':>13} {'C_graph_capReal':>16}")
for ctx, width in ((1024, 32), (4096, 32), (8192, 64)):
    a = eager(ctx, width)
    b = graph_case(ctx, width, capture_lens=False)
    c = graph_case(ctx, width, capture_lens=True)
    print(f"{ctx:>6} {width:>5} {a:>9.2f} {b:>13.2f} {c:>16.2f}")
