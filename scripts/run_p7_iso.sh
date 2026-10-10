#!/bin/bash
# P7-C 隔离实验：那 ~0.8× 的每轮耗时到底归「滑窗截断上下文」还是「draft 有自己的紧凑私有池」？
#
# ★ 本实验【零代码改动】—— 纯粹是 spec_draft_window 的取值，所以 spec_draft_window=0
#   的主路径一行都没被碰过（逐字节等价另用 p7_equiv.py 对拍 p1-work-base/ 验证）。
#
# 设计（L=1024, OUT=128, k=6, B=4，序列最长 ~1156 token → 需要 5 个块）：
#
#   W=0      baseline：draft 借用 target 的物理块表（实机 id = [5,6,7,8,21]，第 5 块离得很远），
#            draft 池 177 块；draft 看全部上下文。
#   W=1280   M=5：draft 有【自己的】环形块表，块数正好 = W=0 实际需要的 5 块
#            （实机 id = [5,6,7,8,9]，完全连续）；M*bs = 1280 >= 1156 → 【不截断】，
#            逐条 ctx 与 W=0 完全相同、块表实参宽度也相同（都是 5）。
#            → 与 W=0 唯一的差别就是「draft 的 KV 读的是哪几个物理块」。
#            → 实测这一档与 W=0 的 226 次前向 logits 全部逐位相同（0/226 不同）、
#              接受率逐组完全相同 —— 语义上是彻底的空操作，只换了池布局。
#   W=2048   M=8：私有环、不截断、表宽 8（比 W=0 宽 3 → 有 4.9e-4 的数值扰动）
#   W=4608   M=18：私有环、不截断、池 144 块（≈W=0 的 4.0 GB）→ 用来排除「池大小」这个解释
#
# 判据：`wall_s / rounds`（每轮耗时，吞吐单次只有 2-5 s 噪声大）；逐 rep 配对比值 +
#       中位数与组间范围；同时报接受率（变了就说明变体引入了别的差异，要查）。
#
# 结论（见 commit 报告）：W=1280 复现了 ~0.83×/轮 的加速且接受率逐位不变 →
#   这个收益归「draft 拥有自己的紧凑私有池/块表」（任务 B 的"解耦 draft 分配"），
#   **与窗口截断上下文无关**，而且可以在【零接受率代价】下拿到、与 W 解耦。
#
# 用法: bash scripts/run_p7_iso.sh   （服务器、GPU 需空闲、2333 需空闲；引擎串行）
set -u
ROOT=${NV_ROOT:-/home/ziru/nano-vllm/p1-work}
PY=/home/ziru/nano-vllm/venv/bin/python
export NV_ROOT=$ROOT
cd $ROOT
mkdir -p p7-runs/iso3
for wl in rep nat; do
  n=5; [ "$wl" = nat ] && n=3
  for rep in $(seq 1 $n); do
    for W in 1280 4608; do
      f=$ROOT/p7-runs/iso3/W${W}_${wl}_r${rep}.log
      timeout 900 $PY scripts/p7_window.py $W $wl $rep 4 1024 128 6 > $f 2>&1
      echo "  W=$W $wl r$rep exit=$?"
    done
  done
done
# 语义核查：W=1280 与 W=0 必须逐位等价（并证明 ctx 逐条相同 = 真的没截断）
for cfg in "ISO_W1280 1280" "ISO_W0 0"; do
  set -- $cfg
  timeout 1800 $PY scripts/p7_equiv.py $2 1024 128 4 p7-runs/iso3/$1.json
  echo "  equiv $1 exit=$?"
done
echo ISODONE
