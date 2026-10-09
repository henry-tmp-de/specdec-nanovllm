#!/bin/bash
# A 步的批量跑法（每组独立进程 + 全新引擎 + 每个 rep 重新冷启动）。
#
#   A7_KIND=quant  → scripts/a7_quant.py <group> 4 4096 64 64 <rep> 6
#                   组：HOT / HOTX / COLD
#   A7_KIND=verify → scripts/a7_verify.py <mode> <rep>
#                   模式：REF / NEW / OLD / NEG / NEGX
#
# 用法: A7_TAG=<tag> A7_KIND=quant A7_GROUPS="HOT HOTX COLD" bash run_a7.sh
# ★ 一律走环境变量，不用位置参数 —— 通过 ssh 传位置参数踩过坑。
set -u
ROOT=/home/ziru/nano-vllm/p1-work
PY=/home/ziru/nano-vllm/venv/bin/python
export NV_ROOT=$ROOT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$ROOT"

KIND=${A7_KIND:-quant}
TAG=${A7_TAG:-base}
REPS=${A7_REPS:-5}
if [ "$KIND" = "verify" ]; then
  GLIST=${A7_GROUPS:-"REF NEW OLD NEG NEGX"}
  OUTDIR=${A7_OUT:-$ROOT/a7-runs/verify-$TAG}
else
  GLIST=${A7_GROUPS:-"HOT HOTX COLD"}
  OUTDIR=${A7_OUT:-$ROOT/a7-runs/$TAG}
fi
mkdir -p "$OUTDIR"
echo "== kind=$KIND tag=$TAG groups=[$GLIST] reps=$REPS gpu=$CUDA_VISIBLE_DEVICES out=$OUTDIR =="
for rep in $(seq 1 "$REPS"); do
  for g in $GLIST; do
    f="$OUTDIR/${TAG}_${g}_r${rep}.log"
    echo "[$(date +%H:%M:%S)] $g rep$rep -> $f"
    if [ "$KIND" = "verify" ]; then
      timeout 2400 $PY scripts/a7_verify.py "$g" "$rep" > "$f" 2>&1
    else
      timeout 2400 $PY scripts/a7_quant.py "$g" 4 4096 64 64 "$rep" 6 > "$f" 2>&1
    fi
    echo "    exit=$?"
  done
done
echo "== done =="
