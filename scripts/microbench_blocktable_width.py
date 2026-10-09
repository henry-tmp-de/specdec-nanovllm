"""microbench_blocktable_width.py

只测 flash_attn_with_kvcache 一个 kernel，用来验证一个猜想：

  引擎在 CUDA graph 模式下，attention kernel 的 block_table 宽度被固定成
  max_num_blocks = max_model_len / 256（nano-vllm capture 时就按这个宽度分配静态缓冲），
  eager 模式下 block_table 宽度 = 实际用到的块数（+1）。
  flash-attn 的 split 启发式若按 block_table 宽度 × block_size 估计 KV 长度，
  graph 模式就会【多做无用的 attention 工作】—— 表现为 attention kernel 时间
  几乎不随 ctx 变化、且显著高于 eager。

本脚本不加载任何模型、不启引擎、不占 2333 端口。可与其他实验并行（用别的卡）。
"""
import os
import torch
import flash_attn
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda"
BS = 256          # nano-vllm kvcache_block_size
KVH = 8           # Qwen3-4B num_key_value_heads
HD = 128          # head_dim
NH = 32           # num_heads
DT = torch.bfloat16

MAX_BLOCKS = 64
k_cache = torch.randn(MAX_BLOCKS, BS, KVH, HD, device=dev, dtype=DT)
v_cache = torch.randn(MAX_BLOCKS, BS, KVH, HD, device=dev, dtype=DT)
q = torch.randn(1, 1, NH, HD, device=dev, dtype=DT)   # (batch, seqlen_q=1, nheads, hd)
scale = HD ** -0.5


def bench(ctx, width, iters=200):
    """ctx = 真实 KV 长度；width = block_table 的列数（多余列填 -1）。"""
    need = (ctx + BS - 1) // BS
    assert width >= need, (ctx, width, need)
    bt = torch.full((1, width), -1, device=dev, dtype=torch.int32)
    bt[0, :need] = torch.arange(need, device=dev, dtype=torch.int32)
    lens = torch.tensor([ctx], device=dev, dtype=torch.int32)
    for _ in range(20):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=lens,
                                block_table=bt, softmax_scale=scale, causal=True)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=lens,
                                block_table=bt, softmax_scale=scale, causal=True)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0   # us


print(f"flash_attn={flash_attn.__version__} dev={torch.cuda.get_device_name(0)}")
print(f"{'ctx':>6} {'bt_width':>9} {'us/call':>9}   note")
for ctx, widths in ((256, [1, 32, 64]), (1024, [4, 5, 16, 32, 64]),
                    (4096, [16, 17, 32, 64]), (8192, [32, 33, 64])):
    for w in widths:
        try:
            t = bench(ctx, w)
            note = "engine-eager-like" if w <= (ctx // BS + 1) else "engine-graph-like(pad)"
            print(f"{ctx:>6} {w:>9} {t:>9.2f}   {note}")
        except Exception as ex:
            print(f"{ctx:>6} {w:>9} {'ERR':>9}   {type(ex).__name__}: {ex}")
    print()
