#!/bin/bash
# 最终数据：更长的生成（256 token）把接受率/加速比收稳，并做归因（同一份修复后代码，
# 只差 CUDA graph 开关）。
cd /home/ziru/nano-vllm/repo
export CUDA_VISIBLE_DEVICES=7
PY=/home/ziru/nano-vllm/venv/bin/python
OUT=/home/ziru/nano-vllm/repo/final_results.txt
: > $OUT

run() {
  echo "===== bench_spec.py $* =====" >> $OUT
  t0=$(date +%s)
  timeout 1800 $PY -u bench_spec.py "$@" >> $OUT 2>&1
  echo "(exit $?, $(( $(date +%s) - t0 ))s)" >> $OUT
}

run base  0 256 1
run draft 2 256 1
run draft 6 256 1
run draft 6 256 0
echo "ALL_DONE" >> $OUT
