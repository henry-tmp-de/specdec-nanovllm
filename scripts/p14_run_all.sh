#!/usr/bin/env bash
# p14_run_all.sh —— 把 P14 的全部探针**串行**跑完（一次只占 GPU 0，绝不并发）。
#
# ★ 为什么串行：本项目实测过（4 卡并行跑端到端验收，eager 档差 ±4% > 0.5% 噪声地板），
#   而 P14 的主指标是 ITL / TTFT 这种毫秒级延迟 —— 多进程会抢主机 CPU（eager 路径的
#   Python dispatch）、抢磁盘 I/O（每个进程都要载 8 GB target + 1.2 GB draft）、
#   抢主机内存带宽。所以这里**一个个来**。
#
# 用法（服务器）:
#   cd /home/ziru/nano-vllm/p14-work && setsid nohup bash scripts/p14_run_all.sh \
#       > logs/p14_run_all.log 2>&1 < /dev/null &
set -u
PY=/home/ziru/nano-vllm/venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs out

run() {                     # run <日志名> <环境变量...> -- <脚本> <参数...>
    local log="$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "=== $(date -Is) START $log : ${envs[*]} $*"
    env "${envs[@]}" "$PY" -u "$@" > "logs/$log.log" 2>&1
    echo "=== $(date -Is) DONE  $log (rc=$?)"
}

# ---------- 场景 B：prefill 公平（P14 主场）----------
# ① 共用口径：ctx=16384 / out=256 / k=6 / max_num_seqs=4
run p14_ab_B_16k_c2048 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2571 \
    SCEN=B CHUNK=2048 RUNS=3 -- scripts/p14_ab_probe.py
# ② 同场景、不同 chunk（看轮转对块大小的敏感性）
run p14_ab_B_16k_c1024 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2572 \
    SCEN=B CHUNK=1024 RUNS=3 -- scripts/p14_ab_probe.py
# ③ 池子装得下的 ctx（8K × 3 = 96 块 < 174）⇒ 隔离出【纯调度】的收益
run p14_ab_B_8k_c2048 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2573 \
    SCEN=B CTX=8192 CHUNK=2048 RUNS=3 -- scripts/p14_ab_probe.py
# ④ 更小：4K × 3
run p14_ab_B_4k_c1024 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2574 \
    SCEN=B CTX=4096 CHUNK=1024 RUNS=3 -- scripts/p14_ab_probe.py

# ---------- 场景 A：decode 饥饿（对照组，P14 不该让它变差）----------
run p14_ab_A_16k_c2048 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2575 \
    SCEN=A CHUNK=2048 RUNS=3 -- scripts/p14_ab_probe.py

# ---------- 场景 C：稳态吞吐 ----------
run p14_thr_c2048 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2576 \
    NREQ=8 LONG_N=4 LONG_CTX=16384 SHORT_CTX=2048 OUT=256 CHUNK=2048 RUNS=3 \
    -- scripts/p14_throughput.py

# ---------- KV 代价：N=1..4（再跑一遍 chunk=1024，确认边界与块大小无关）----------
run p14_kv_c1024 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2577 \
    NS=1,2,3,4 CHUNK=1024 CTX=16384 -- scripts/p14_kv_probe.py

echo "=== ALL DONE $(date -Is)"
