#!/bin/bash
# Phase 0 代理实验驱动：串行跑多个权重规模的模型（引擎硬编码 2333 端口，必须串行）
set -x
cd /home/ziru/nano-vllm/p1-work
P=/home/ziru/nano-vllm/venv/bin/python
ROOT=/home/ziru/nano-vllm
for spec in "Qwen3-4B models/Qwen3-4B" "Qwen3-1.7B models/Qwen3-1.7B" "Qwen3-0.6B models/Qwen3-0.6B"; do
  TAG=$(echo "$spec" | awk '{print $1}')
  MP=$(echo "$spec" | awk '{print $2}')
  if [ ! -f "$ROOT/$MP/config.json" ]; then
    echo "##### SKIP $TAG (no config.json) #####"
    continue
  fi
  echo "##### $TAG #####"
  CUDA_VISIBLE_DEVICES=0 MODEL=$ROOT/$MP TAG=$TAG CTXS=1024,4096 BATCHES=1,4 \
    OUTLEN=256 RUNS=5 GRAPHS=1 PROFILE=1 \
    $P -u scripts/p0_weight_bw.py > "$ROOT/p0_$TAG.log" 2>&1
  echo "##### $TAG exit=$? #####"
done
echo P0_ALL_DONE
