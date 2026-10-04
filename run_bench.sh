#!/bin/bash
# 每个配置独立进程跑，避免 dist 重复初始化
cd /home/ziru/nano-vllm/repo
export CUDA_VISIBLE_DEVICES=7
PY=/home/ziru/nano-vllm/venv/bin/python
for k in 0 2 3 4; do
  echo "--- spec_k=$k ---"
  timeout 300 $PY bench_spec.py $k 20 2>&1 | grep -E "@@JSON@@|Error|Traceback|assert" | head -5
done
