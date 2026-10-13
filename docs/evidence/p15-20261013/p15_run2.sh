#!/bin/bash
cd /home/ziru/nano-vllm/p15-work
export PY=/home/ziru/nano-vllm/venv/bin/python
export CUDA_VISIBLE_DEVICES=4
mkdir -p logs
echo "=== START graphprobe $(date -Is) ==="
NANOVLLM_PORT=2431 timeout 1200 $PY -u scripts/p15_graphprobe.py > logs/graphprobe.log 2>&1
echo "=== graphprobe rc=$? $(date -Is) ==="
echo "=== START micro-multidepth $(date -Is) ==="
CTX=16384 CHUNK=1024 REPS=15 DEPTHS=0,5,10,14 NANOVLLM_PORT=2432 \
  timeout 1800 $PY -u scripts/p15_overlap_micro.py > logs/micro_depth.log 2>&1
echo "=== micro rc=$? $(date -Is) ==="
echo "=== ALL DONE $(date -Is) ==="
