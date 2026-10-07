#!/bin/bash
# 引擎级无损性：base / base2（基线的第二次独立运行，当噪声地板）/ draft
cd /home/ziru/nano-vllm/repo
export CUDA_VISIBLE_DEVICES=7
PY=/home/ziru/nano-vllm/venv/bin/python
OUT=/home/ziru/nano-vllm/repo/lossless_results.txt
: > $OUT
for m in base base2 draft; do
  echo "===== $m =====" >> $OUT
  timeout 1800 $PY -u check_lossless_engine.py $m >> $OUT 2>&1
done
echo ALL_DONE >> $OUT
