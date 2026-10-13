#!/bin/bash
cd /home/ziru/nano-vllm/p15-work
export PY=/home/ziru/nano-vllm/venv/bin/python
export CUDA_VISIBLE_DEVICES=4
mkdir -p logs
run() {   # run <tag> <env-assignments...> -- <script> <args...>
  local tag=$1; shift
  echo "=== START $tag $(date -Is) ==="
  env "$@" NANOVLLM_PORT=$PORT timeout 3000 $PY -u $SCRIPT $ARGS > logs/$tag.log 2>&1
  echo "=== END $tag rc=$? $(date -Is) ==="
}
# ---------- 1) 重叠微基准（决定性）----------
PORT=2421 SCRIPT=scripts/p15_overlap_micro.py ARGS=""
run micro CTX=16384 CHUNK=1024 REPS=15
# ---------- 2) 场景 A：decode 饥饿 ----------
PORT=2422 SCRIPT=scripts/p15_scenarios.py
run scenA RUNS=3 SCEN=A CTX=16384 CHUNK=1024 OUT=256
# ---------- 3) 场景 B：prefill 公平 ----------
PORT=2423 SCRIPT=scripts/p15_scenarios.py
run scenB RUNS=3 SCEN=B CTX=16384 CHUNK=1024 OUT=256
# ---------- 4) 场景 C：稳态吞吐 ----------
PORT=2424 SCRIPT=scripts/p15_scenarios.py
run scenC RUNS=3 SCEN=C CTX=16384 CHUNK=1024 OUT=256
echo "=== ALL DONE $(date -Is) ==="
