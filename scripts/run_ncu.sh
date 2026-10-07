#!/bin/bash
# 用 ncu 采真实计数器，验证 roofline 的假设。
#
# 为什么必须采：理论估算只能给「应该搬多少字节」，
# 而「实际搬了多少 / 带宽用了几成 / 算力单元忙不忙」只能从计数器读。
# 判定一个 kernel 是不是访存瓶颈，靠的就是：
#     dram 吞吐接近峰值 且 sm 吞吐远低于峰值  -> 在等内存
#
# 用法（服务器上）：
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_ncu.sh 1024 2      # ctx=1024, v2
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_ncu.sh 4096 1
#
# ⚠️ 共享机器上 ncu 常因为没有 perf counter 权限而失败（ERR_NVGPUCTRPERM）。
#    这里用 --clock-control none 避免锁频（锁频通常需要 root）。
set -u

CTX=${1:-1024}
VER=${2:-2}
PY=/home/ziru/nano-vllm/venv/bin/python
NCU=/usr/local/cuda/bin/ncu

cd "$(dirname "$0")/.."

echo "=== ncu: ctx=$CTX version=$VER ==="
CTX=$CTX VER=$VER $NCU \
  --target-processes all \
  --kernel-name regex:paged_decode \
  --launch-skip 3 --launch-count 1 \
  --clock-control none \
  --metrics \
gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum,\
launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  $PY scripts/ncu_paged_decode.py 2>&1 | tail -40
