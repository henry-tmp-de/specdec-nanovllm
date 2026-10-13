#!/usr/bin/env bash
# p14_run_extra.sh —— 补测：把场景 C 的吞吐差钉实 + 归因 + 边界线索。串行跑，只占 GPU 0。
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

# ① 场景 C 加跑数到 5 —— 12% 那个差到底是不是真的
run p14_thr_c2048_r5 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2580 \
    NREQ=8 LONG_N=4 LONG_CTX=16384 SHORT_CTX=2048 OUT=256 CHUNK=2048 RUNS=5 \
    -- scripts/p14_throughput.py

# ② 归因：全是短 prompt（池子永远装得下）⇒ 若差异消失，说明差异来自 KV 准入而不是轮转本身
run p14_thr_allshort CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2581 \
    NREQ=8 LONG_N=0 LONG_CTX=16384 SHORT_CTX=2048 OUT=256 CHUNK=2048 RUNS=3 \
    -- scripts/p14_throughput.py

# ③ 归因：全开滑窗（草稿小池子）看 target 池子能涨到多少块 —— 只读 HEAD 行
run p14_pool_withwindow CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2582 \
    SCEN=B CTX=16384 CHUNK=2048 RUNS=1 ROTS=off WIN=4608 -- scripts/p14_ab_probe.py

echo "=== EXTRA DONE $(date -Is)"
