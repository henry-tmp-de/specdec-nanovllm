#!/bin/bash
# realign_trun.sh —— 通用单/多 job 运行器，【每张 GPU 严格串行】（等上一条彻底退出再发下一条，
# 避免 22GB 进程 teardown 未完成导致 OOM）。
# job 行格式：tag|ctx|window|head|align|outlen|script   （从 $1 指定的文件读）
cd /home/ziru/nano-vllm/realign-work || exit 1
PY=/home/ziru/nano-vllm/venv/bin/python
LIST=${1:-/tmp/jobs.txt}
OUTDIR=${OUTDIR:-out/matrix}
LOGDIR=${LOGDIR:-logs/matrix}
MAXLEN=${MAXLEN:-16640}
mkdir -p "$OUTDIR" "$LOGDIR"

declare -a GPUPID
n=0
while IFS='|' read -r tag ctx window head align outlen script rlopt; do
  [ -z "$tag" ] && continue
  RL=${rlopt:-${REALIGN:-0}}
  gpu=$(( n % 8 ))
  if [ -n "${GPUPID[$gpu]}" ]; then
    wait "${GPUPID[$gpu]}" 2>/dev/null
    sleep 3
  fi
  (
    if [ "$script" = "slide_probe.py" ]; then
      CTX=$ctx WINDOW=$window HEAD=$head ALIGN=$align REALIGN=$RL B=1 OUTLEN=$outlen K=6 \
        MAXLEN=$MAXLEN MAX_SEQS=1 PROBE_STRIDE=${PROBE_STRIDE:-1} CONTROL=${CONTROL:-1} \
        TAG=$tag OUT=$OUTDIR/$tag.json \
        CUDA_VISIBLE_DEVICES=$gpu NANOVLLM_PORT=$(( 3000 + n )) \
        $PY -u scripts/$script > $LOGDIR/$tag.log 2>&1
    elif [ "$script" = "realign_ttft.py" ]; then
      CTX=$ctx WINDOW=$window HEAD=$head ALIGN=$align REALIGN=$RL \
        B=${B:-1} REPS=${REPS:-3} MAXLEN=$MAXLEN MAX_SEQS=${MAX_SEQS:-4} \
        TAG=$tag OUT=$OUTDIR/$tag.json \
        CUDA_VISIBLE_DEVICES=$gpu NANOVLLM_PORT=$(( 3000 + n )) \
        $PY -u scripts/$script > $LOGDIR/$tag.log 2>&1
    else
      CTX=$ctx WINDOW=$window HEAD=$head ALIGN=$align REALIGN=$RL \
        B=${B:-4} OUTLEN=$outlen K=6 MAXLEN=$MAXLEN MAX_SEQS=${MAX_SEQS:-4} \
        TAG=$tag OUT=$OUTDIR/$tag.json \
        CUDA_VISIBLE_DEVICES=$gpu NANOVLLM_PORT=$(( 3000 + n )) \
        $PY -u scripts/$script > $LOGDIR/$tag.log 2>&1
    fi
    echo "DONE $tag rc=$? gpu=$gpu"
  ) &
  GPUPID[$gpu]=$!
  n=$(( n + 1 ))
done < "$LIST"
wait
echo ALLDONE n=$n
