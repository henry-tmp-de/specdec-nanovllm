#!/bin/bash
# 串行跑（nano-vllm 硬编码 tcp://localhost:2333，不能并发）
# ★ 每个配置【单独写文件】并保留完整 stderr —— 用管道 + grep 会把结果缓冲住，
#   中途 kill 就全丢了（已经丢过一次）。
cd /home/ziru/nano-vllm/repo
export CUDA_VISIBLE_DEVICES=7
PY=/home/ziru/nano-vllm/venv/bin/python
OUT=/home/ziru/nano-vllm/repo/bench_results.txt
: > $OUT

run() {
  echo "===== bench_spec.py $* =====" >> $OUT
  t0=$(date +%s)
  timeout 1200 $PY -u bench_spec.py "$@" >> $OUT 2>&1
  rc=$?
  echo "(exit $rc, $(( $(date +%s) - t0 ))s)" >> $OUT
}

run draft 2 64 1
run base  0 64 1
run base  0 64 0
run draft 1 64 1
run draft 4 64 1
run draft 6 64 1
run draft 2 64 0
echo "ALL_DONE" >> $OUT
