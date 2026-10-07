"""判断实验：KV cache 的访存模式值多少带宽？（决定要不要写 CUDA 版）

严格对照：两组都读满整个 cache、program 数相同、每个 program 字节数相同，
只有【访存模式】不同。

  A 切片读：program p 读【第 p 个头】，扫全部位置
            -> 每个位置读 256B，跳 1792B
  B 整行读：program p 读【自己那 1/NPROG 位置区间】的整行 2048B
            -> 完全连续

cache 取 33.6MB，远超 6MB L2，两组都是真读显存。

跑：CUDA_VISIBLE_DEVICES=7 python scripts/probe_kv_access.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl

NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM = 64, 256, 8, 128
ROW = NUM_KV_HEADS * HEAD_DIM                 # 1024
N_POS = NUM_BLOCKS * BLOCK_SIZE               # 16384
CACHE_BYTES = N_POS * ROW * 2                 # bf16 -> 33.6 MB
NPROG, BLOCK_N = 64, 128
dev = "cuda"


@triton.jit
def _read_slice(cache_ptr, out_ptr, n_pos,
                BLOCK_SIZE: tl.constexpr, ROW: tl.constexpr, D: tl.constexpr,
                BLOCK_N: tl.constexpr, NKVH: tl.constexpr):
    """A：只读【自己那个头】的 D 个元素，扫全部位置。"""
    pid = tl.program_id(0)
    kvh = pid % NKVH                       # ★ 不同 program 读不同头，合起来才是全量
    # ★★ 关键：位置方向也要按 program 数切分，否则 8 个 program 每个都扫全部位置，
    #    那就是 8 倍冗余读，和 B 组根本不可比（前两版都栽在这里）
    lane = pid // NKVH
    nlanes = tl.num_programs(0) // NKVH
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    total = 0.0
    for base in range(lane * BLOCK_N, n_pos, BLOCK_N * nlanes):
        pos = base + offs_n
        ptr = (cache_ptr + (pos // BLOCK_SIZE)[:, None] * (BLOCK_SIZE * ROW)
               + (pos % BLOCK_SIZE)[:, None] * ROW + kvh * D + offs_d[None, :])
        total += tl.sum(tl.load(ptr).to(tl.float32))
    tl.store(out_ptr + pid, total)


@triton.jit
def _read_full(cache_ptr, out_ptr, n_pos,
               BLOCK_SIZE: tl.constexpr, ROW: tl.constexpr,
               BLOCK_N: tl.constexpr, NPROG: tl.constexpr):
    """B：读满整行 ROW，只扫自己那 1/NPROG 的位置区间。"""
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, ROW)
    per = n_pos // NPROG
    total = 0.0
    for base in range(pid * per, (pid + 1) * per, BLOCK_N):
        pos = base + offs_n
        ptr = (cache_ptr + (pos // BLOCK_SIZE)[:, None] * (BLOCK_SIZE * ROW)
               + (pos % BLOCK_SIZE)[:, None] * ROW + offs_d[None, :])
        total += tl.sum(tl.load(ptr).to(tl.float32))
    tl.store(out_ptr + pid, total)


def timeit(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                    device=dev, dtype=torch.bfloat16)
out = torch.zeros(NPROG, device=dev)

print(f"cache {CACHE_BYTES/1e6:.1f} MB（远超 6MB L2，真读显存）；"
      f"{NPROG} 个 program，两组各读满 {CACHE_BYTES/1e6:.1f} MB")
print()
print(f"{'读法':<38}{'时间(us)':>10}{'带宽(GB/s)':>13}{'占 842 天花板':>14}")
print("-" * 76)

tA = timeit(lambda: _read_slice[(NPROG,)](cache, out, N_POS, BLOCK_SIZE=BLOCK_SIZE,
                                          ROW=ROW, D=HEAD_DIM, BLOCK_N=BLOCK_N,
                                          NKVH=NUM_KV_HEADS))
tB = timeit(lambda: _read_full[(NPROG,)](cache, out, N_POS, BLOCK_SIZE=BLOCK_SIZE,
                                         ROW=ROW, BLOCK_N=BLOCK_N, NPROG=NPROG))
for name, t in [("A 切片读（256B / 跳 1792B）", tA),
                ("B 整行读（2048B 连续）", tB)]:
    bw = CACHE_BYTES / (t * 1e-6) / 1e9
    print(f"{name:<38}{t:>10.2f}{bw:>13.1f}{bw/842*100:>13.1f}%")

print()
print(f"  -> 两种访存模式的带宽比 = {tA/tB:.2f}x")
print("     （>1.5 说明访存模式是个大杠杆，值得用 CUDA 试；≈1 说明这条路不用走）")
