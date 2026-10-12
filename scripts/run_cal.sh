#!/bin/bash
# run_cal.sh —— (P=2, M=5) 收尾定标【全部复算命令】（一个配置一个进程，铺在 8 张卡上）
# 服务器工作区：/home/ziru/nano-vllm/cal-work   （代码 = 主分支 75729b0 的原样拷贝）
# 详见 docs/优化台账.md §2.7.1 / §2.7.2。
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/ziru/nano-vllm/venv/bin/python}
mkdir -p out/det out/eng logs

# ======================================================================
# ① 离线确定性仪器（固定 wikitext 上下文，按引擎的窗口语义重建 draft 上下文）
#    α = 1 − TV(p, q)；16 窗口 × 2048 位置（其中 256 个"生成位"）
# ======================================================================
det() { local g=$1 tag=$2; shift 2
  env "$@" CUDA_VISIBLE_DEVICES=$g NANOVLLM_PORT=$((2600+g)) OUT=out/det/$tag.json \
      $PY -u scripts/det_alpha.py > logs/$tag.log 2>&1; }

det 0 w0_8k     L=8192  CFG=w0
det 1 m5_8k     L=8192  CFG=m5
det 2 p2m5_8k   L=8192  CFG=p2m5
det 3 w0_16k    L=16384 CFG=w0
det 4 m5_16k    L=16384 CFG=m5
det 5 p2m5_16k  L=16384 CFG=p2m5
det 6 p2m12_16k L=16384 CFG=p2m12
det 7 p2m18_16k L=16384 CFG=p2m18
det 0 p2m33_16k L=16384 CFG=p2m33
det 1 p2m48_16k L=16384 CFG=p2m48
det 2 p2m63_16k L=16384 CFG=p2m63
det 3 w0_8k_b   L=8192  CFG=w0        # 复现性重跑
det 4 p2m5_8k_b L=8192  CFG=p2m5
det 5 self_8k   L=8192  CFG=p2m5 SELFTEST=1

# ======================================================================
# ② 引擎内确定性仪器（★ 主口径）：挂 verify_batch，拿引擎自己的 p/q 算 1 − TV
#    同一次运行同时得到「确定性 α」与「项目原有 accept_rate」
# ======================================================================
eng() { local g=$1 tag=$2; shift 2
  env "$@" CUDA_VISIBLE_DEVICES=$g NANOVLLM_PORT=$((2740+g)) \
      $PY -u scripts/engine_alpha.py > logs/$tag.log 2>&1; }

eng 0 e_w0_8k      CTX=8192  WINDOW=0    HEAD=0 TAG=w0_8k
eng 1 e_m5_8k      CTX=8192  WINDOW=1280 HEAD=0 TAG=m5_8k
eng 2 e_p1m5_8k    CTX=8192  WINDOW=1280 HEAD=1 TAG=p1m5_8k
eng 3 e_p2m5_8k    CTX=8192  WINDOW=1280 HEAD=2 TAG=p2m5_8k
eng 4 e_p4m5_8k    CTX=8192  WINDOW=1280 HEAD=4 TAG=p4m5_8k
eng 5 e_p0m12_8k   CTX=8192  WINDOW=3072 HEAD=0 TAG=p0m12_8k
eng 6 e_p0m18_8k   CTX=8192  WINDOW=4608 HEAD=0 TAG=p0m18_8k
eng 7 e_p0m33_8k   CTX=8192  WINDOW=8448 HEAD=0 TAG=p0m33_8k   # ★ 不滑动对照
eng 0 e_p2m33_8k   CTX=8192  WINDOW=8448 HEAD=2 TAG=p2m33_8k
eng 0 e_w0_16k     CTX=16384 WINDOW=0    HEAD=0 TAG=w0_16k
eng 1 e_m5_16k     CTX=16384 WINDOW=1280 HEAD=0 TAG=m5_16k
eng 2 e_p2m5_16k   CTX=16384 WINDOW=1280 HEAD=2 TAG=p2m5_16k
eng 3 e_p2m12_16k  CTX=16384 WINDOW=3072 HEAD=2 TAG=p2m12_16k
eng 4 e_p2m18_16k  CTX=16384 WINDOW=4608 HEAD=2 TAG=p2m18_16k
eng 5 e_p2m33_16k  CTX=16384 WINDOW=8448 HEAD=2 TAG=p2m33_16k

# ======================================================================
# ③ 项目原有的自由生成口径（对照：证明两个口径本来就是同一个量）
# ======================================================================
for g in lc_w0 mw_m5 mw_p2m5; do
  CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2790 \
    $PY -u scripts/acceptance_matrix.py --group $g --ctxs 8192 --bs 4 --runs 3 \
        --out out/eng/harness_$g.json > logs/harness_$g.log 2>&1
done

# ======================================================================
# ④ 诊断（§2.7.2 的两条排除证据）—— 各一个配置一个进程
# ======================================================================
# 4a. draft 每次前向真实用的 (positions, context_len, block_table)
WINDOW=1280 HEAD=0 B=1 TAG=p0m5 CUDA_VISIBLE_DEVICES=0 $PY -u scripts/dbg_win.py
WINDOW=1280 HEAD=2 B=1 TAG=p2m5 CUDA_VISIBLE_DEVICES=0 $PY -u scripts/dbg_win.py
# 4b. 第一次投机轮：引擎的 draft logits 倒出来，与离线几种"候选上下文"比 TV
WINDOW=1280 HEAD=0 TAG=p0m5 OUT=out/eng/dump_p0m5.pt CUDA_VISIBLE_DEVICES=0 $PY -u scripts/eng_dump_q.py
WINDOW=1280 HEAD=2 TAG=p2m5 OUT=out/eng/dump_p2m5.pt CUDA_VISIBLE_DEVICES=0 $PY -u scripts/eng_dump_q.py
DUMP=out/eng/dump_p0m5.pt $PY -u scripts/cmp_dump_q.py
DUMP=out/eng/dump_p2m5.pt $PY -u scripts/cmp_dump_q.py
# 4c. ★ 窗口 prefill 写进环里的 KV 对不对（排除"写坏 KV"）
WINDOW=1280 HEAD=0 TAG=p0m5 OUT=out/eng/kv_p0m5.pt CUDA_VISIBLE_DEVICES=0 $PY -u scripts/dump_kv.py
WINDOW=1280 HEAD=2 TAG=p2m5 OUT=out/eng/kv_p2m5.pt CUDA_VISIBLE_DEVICES=1 $PY -u scripts/dump_kv.py
DUMP=out/eng/kv_p0m5.pt $PY -u scripts/cmp_kv.py
DUMP=out/eng/kv_p2m5.pt $PY -u scripts/cmp_kv.py
