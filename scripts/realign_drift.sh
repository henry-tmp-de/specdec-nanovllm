#!/bin/bash
# realign_drift.sh —— 量化「第二次漂移」：把生成拉到 512 token，让它跨进下一个块。
#   ctx=8192  : 漂移起点 pos>=8448 (块33)
#   ctx=16384 : 漂移起点 pos>=16640 (块65)
# MAXLEN 统一 17152（= ceil((16384+512)/256)*256）保证同口径。
cd /home/ziru/nano-vllm/realign-work || exit 1
PY=/home/ziru/nano-vllm/venv/bin/python
mkdir -p logs/drift out/drift
RUNS=${RUNS:-3}
CONC=${CONC:-8}
MAXLEN=17152

JOBS=()   # tag|ctx|window|head|align|outlen|script
for ctx in 8192 16384; do
  for outlen in 256 512; do
    for r in $(seq 1 "$RUNS"); do
      JOBS+=("w0_c${ctx}_o${outlen}_r${r}|${ctx}|0|0|0|${outlen}|engine_alpha.py")
    done
    for align in 0 1; do
      for pm in "0 5" "2 5"; do
        set -- $pm; p=$1; m=$2
        for r in $(seq 1 "$RUNS"); do
          JOBS+=("a${align}_p${p}m${m}_c${ctx}_o${outlen}_r${r}|${ctx}|$((m*256))|${p}|${align}|${outlen}|engine_alpha.py")
        done
      done
    done
  done
done
# 机制级探针（B=1）：逐轮 TV(q_eng, q_ideal) + 位置
for align in 0 1; do
  JOBS+=("probe_a${align}_c8192|8192|1280|0|${align}|512|slide_probe.py")
done
JOBS+=("probe_a1_p2m5_c8192|8192|1280|2|1|512|slide_probe.py")

echo "TOTAL JOBS: ${#JOBS[@]}"
i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag ctx window head align outlen script <<< "$job"
  gpu=$(( i % 8 ))
  port=$(( 3000 + i ))
  while [ "$(jobs -r | wc -l)" -ge "$CONC" ]; do sleep 1; done
  (
    if [ "$script" = "slide_probe.py" ]; then
      CTX=$ctx WINDOW=$window HEAD=$head ALIGN=$align B=1 OUTLEN=$outlen K=6 \
        MAXLEN=$MAXLEN MAX_SEQS=1 PROBE_STRIDE=1 CONTROL=1 TAG=$tag OUT=out/drift/$tag.json \
        CUDA_VISIBLE_DEVICES=$gpu NANOVLLM_PORT=$port \
        $PY -u scripts/$script > logs/drift/$tag.log 2>&1
    else
      CTX=$ctx WINDOW=$window HEAD=$head ALIGN=$align B=4 OUTLEN=$outlen K=6 \
        MAXLEN=$MAXLEN MAX_SEQS=4 TAG=$tag OUT=out/drift/$tag.json \
        CUDA_VISIBLE_DEVICES=$gpu NANOVLLM_PORT=$port \
        $PY -u scripts/$script > logs/drift/$tag.log 2>&1
    fi
    echo "DONE $tag rc=$? gpu=$gpu"
  ) &
  i=$(( i + 1 ))
done
wait
echo ALLDONE
