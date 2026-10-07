"""给 ncu 用的最小驱动：每个 kernel 只跑一次，方便采集。

roofline 不能只靠理论估算 —— 必须看真实计数器：
  * dram__bytes_read.sum      实际从显存读了多少字节（对比理论值，验证「重复读」假设）
  * dram__throughput.avg.pct_of_peak_sustained_elapsed   实测带宽占峰值百分比
  * gpu__time_duration.sum    真正花在 kernel 上的时间
  * sm__throughput...         计算单元忙不忙（判定是不是在等内存）
  * launch__registers_per_thread / launch__occupancy_limit_registers   寄存器压力

用法（在服务器上）：
  bash scripts/run_ncu.sh
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from nanovllm.kernels.paged_decode_attn import paged_decode_attention
from bench_paged_decode import make_case, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SCALE

CTX = int(os.environ.get("CTX", "1024"))
VER = int(os.environ.get("VER", "2"))

torch.manual_seed(0)
q, kc, vc, bt, cl = make_case(CTX, num_seqs=1, seed=1)
print(f"ctx={CTX} version={VER} q_heads={NUM_Q_HEADS} kv_heads={NUM_KV_HEADS} d={HEAD_DIM}")

# 预热（让 Triton 编译完），ncu 只关心真正那次 launch
for _ in range(3):
    paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=VER)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push(f"paged_decode_v{VER}")
out = paged_decode_attention(q, kc, vc, bt, cl, SCALE, version=VER)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("done", out.shape)
