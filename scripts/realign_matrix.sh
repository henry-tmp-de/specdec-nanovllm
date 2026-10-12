#!/bin/bash
# realign_matrix.sh —— 修复后重定位：ALIGN × P × M × ctx 的引擎内确定性 α 网格。
# 一个配置一个进程；最多 8 个并发铺在 GPU 0..7 上（每 GPU 一个进程）。
# 每个 cell 跑 RUNS 遍，供 run-to-run SE。
cd /home/ziru/nano-vllm/realign-work || exit 1
PY=/home/ziru/nano-vllm/venv/bin/python
mkdir -p logs/matrix out/matrix
RUNS=${RUNS:-3}
CONC=${CONC:-8}

JOBS=()
for ctx in 8192 16384; do
  for align in 0 1; do
    for p in 0 1 2 4; do
      for m in 5 12; do
        for r in $(seq 1 "$RUNS"); do
          JOBS+=("a${align}_p${p}m${m}_c${ctx}_r${r}|${ctx}|$((m*256))|${p}|${align}")
        done
      done
    done
  done
  for r in $(seq 1 "$RUNS"); do
    JOBS+=("w0_c${ctx}_r${r}|${ctx}|0|0|0")
  done
done

echo "TOTAL JOBS: ${#JOBS[@]}"
i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag ctx window head align <<< "$job"
  gpu=$(( i % 8 ))
  port=$(( 2800 + i ))
  while [ "$(jobs -r | wc -l)" -ge "$CONC" ]; do sleep 1; done
  (
    CTX=$ctx WINDOW=$window HEAD=$head ALIGN=$align B=4 OUTLEN=256 K=6 \
      MAXLEN=16640 MAX_SEQS=4 TAG=$tag OUT=out/matrix/$tag.json \
      CUDA_VISIBLE_DEVICES=$gpu NANOVLLM_PORT=$port \
      $PY -u scripts/engine_alpha.py > logs/matrix/$tag.log 2>&1
    echo "DONE $tag rc=$? gpu=$gpu"
  ) &
  i=$(( i + 1 ))
done
wait
echo ALLDONE
