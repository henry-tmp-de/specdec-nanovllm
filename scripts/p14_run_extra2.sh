#!/usr/bin/env bash
# p14_run_extra2.sh —— 收尾两测：① 把池子放大到装得下 3 条 16K 再看场景 B；
#                                ② 场景 C 加"批有多宽"的归因打点。
set -u
PY=/home/ziru/nano-vllm/venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs out

run() {
    local log="$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "=== $(date -Is) START $log : ${envs[*]} $*"
    env "${envs[@]}" "$PY" -u "$@" > "logs/$log.log" 2>&1
    echo "=== $(date -Is) DONE  $log (rc=$?)"
}

# ① 16K × 3 条 + 草稿滑窗（target 池 174 → 253 块，N_max 2 → 3）
#    ⇒ 若轮转在这里兑现了 §d.5 的预言，就说明 16K 那次的"没效果"确实来自 KV 准入。
run p14_ab_B_16k_win CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2583 \
    SCEN=B CHUNK=2048 RUNS=3 WIN=4608 -- scripts/p14_ab_probe.py

# ② 场景 C 重跑（探针新加了 mean_decode_batch / mean_prefill_batch 打点）
run p14_thr_c2048_attr CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2584 \
    NREQ=8 LONG_N=4 LONG_CTX=16384 SHORT_CTX=2048 OUT=256 CHUNK=2048 RUNS=3 \
    -- scripts/p14_throughput.py

echo "=== EXTRA2 DONE $(date -Is)"
