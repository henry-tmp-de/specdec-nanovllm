#!/bin/bash
# lc_run_rest.sh —— 长上下文补测的收尾三件事（都在 GPU 0 上【串行】跑）：
#   (a) W=16640 (M=65) 的块数：16K 下"不截断"的窗口到底是省还是亏
#   (b)(c) ctx=4096 下 max_model_len 8192 vs 16640 的诊断（解释 lc_w0 的 α 偏移）
#   (d)(e) 并行 vs 串行抽查：lc_w8448 / lc_w0 各单进程重跑一遍
cd /home/ziru/nano-vllm/p1-work
PY=/home/ziru/nano-vllm/venv/bin/python
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/tmp/ti_d03 TRITON_CACHE_DIR=/tmp/tr_d03
mkdir -p out/lc_diag out/lc_serial

echo "=== (a) M=65 block count ==="
NANOVLLM_PORT=2395 WINDOW=16640 SEQS=4 $PY -u /tmp/lc_blocks.py 2>&1 | grep RESULT

echo "=== (b)(c) mml diagnostic ==="
for w in 0 8448; do
  NANOVLLM_PORT=$((2396 + w / 1000)) WINDOW=$w MAXLEN=8192 CTX=4096 B=1 RUNS=5 \
    $PY -u scripts/lc_mml_diag.py 2>&1 | grep "@@DIAG@@" >> out/lc_diag/diag.log
done

echo "=== (d) serial lc_w8448 ==="
NANOVLLM_PORT=2401 $PY -u scripts/acceptance_matrix.py --group lc_w8448 \
  --ctxs 4096,8192 --bs 1,4 --runs 5 --out out/lc_serial/lc_w8448.json > out/lc_serial/lc_w8448.log 2>&1

echo "=== (e) serial lc_w0 ==="
NANOVLLM_PORT=2402 $PY -u scripts/acceptance_matrix.py --group lc_w0 \
  --ctxs 8192 --bs 1,4 --runs 5 --out out/lc_serial/lc_w0.json > out/lc_serial/lc_w0.log 2>&1

echo REST_DONE
