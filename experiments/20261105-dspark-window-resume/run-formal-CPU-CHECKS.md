# 固定批量基准：命令与口径

`bench_formal.py` 每次只加载一个 arm/B/上下文，完整预热后默认测三个 panel × 三次，逐次重置 seed 与显存峰值并保留逐请求交付、逐轮、真实图重放、现有序列状态、候选/接受、显存和失败。主计时不插入额外 GPU events、窗口探针或正确性前向；保留交付 hook、CPU 轮记账和 target graph replay 计数开销。`--diagnostic-cost` 单独打开 draft/verify/commit GPU event 与窗口槽探针，标为诊断，不进入正式吞吐。正式数据必须在 GPU correctness smoke 审核通过后采集。模型权重不在本目录。当前只是 CPU 工具检查，不是 GPU 结果；测试见 [CPU-FORMAL-TESTS.txt](CPU-FORMAL-TESTS.txt)。

所有 arm 均使用 0.70 显存预算、TP1、T=1、禁用 target prefix lookup、统一 `max_model_len=ctx+output+8`、graph-on 和图补桶；AR k=6，DSpark 七候选。AR window 为块对齐最近块，开启 align/realign；DSpark window 为 anchor 之前最近 W 个 token，二者逐位置窗口含义不同。`phase=capacity` 只用于单列容量诊断；当前引擎不能显式固定相同 target KV 块数，因此只记录固定总预算及实际块数，不将它称为同 target 池实测。全矩阵将按 B/ctx 轮换各 arm 的启动顺序，同一 arm 的三次重复在同一进程中完成，须保留全部原值和时间漂移这一限制。

下列命令只示例参数：`$RUN_ROOT` 是已提交且干净的代码 clone；`$RUN_OUT` 是源码目录外的新结果目录；`$RUN_GPU` 和 `$RUN_UUID` 由主协调现场核实；`$RUN_PY`、`$RUN_TARGET` 和 `$RUN_DSPARK` 见 [smoke CLI](run-tools-CPU-CHECKS.md)，并设置：

```bash
RUN_AR=/home/ziru/nano-vllm/models/Qwen3-0.6B
RUN_DATA_4K="$RUN_OUT/workload-4k.json"
cd "$RUN_ROOT"

# 仅生成计划，绝不启动模型或 GPU。
"$RUN_PY" -B experiments/20261105-dspark-window-resume/run_plan.py \
  --phase pilot --window 1024 --out "$RUN_OUT/pilot-plan.json"
"$RUN_PY" -B experiments/20261105-dspark-window-resume/run_plan.py \
  --phase formal --window 1024 --out "$RUN_OUT/formal-plan.json"

# 一个 pilot job；同一 ledger 延续 smoke 已消耗预算。
"$RUN_PY" -u experiments/20261105-dspark-window-resume/gpu_control.py \
  --gpu "$RUN_GPU" --uuid "$RUN_UUID" --stage pilot-b2-w512-ring --kind pilot \
  --ledger "$RUN_OUT/gpu-budget.jsonl" --log "$RUN_OUT/pilot-b2-w512-ring.log" \
  --stage-timeout-s 600 -- "$RUN_PY" -u experiments/20261105-dspark-window-resume/bench_formal.py \
  --phase pilot --arm dspark_ring --batch 2 --ctx 4096 --window 512 \
  --panel code --repeat 0 --target "$RUN_TARGET" --ar-model "$RUN_AR" \
  --dspark-model "$RUN_DSPARK" --manifest "$RUN_DATA_4K" \
  --out "$RUN_OUT/pilot-b2-w512-ring.jsonl"

# 全部计划完成后用精确模式批量审计；1K/4K manifest 分别传入。
"$RUN_PY" -B experiments/20261105-dspark-window-resume/audit_results.py \
  --mode formal --input-glob "$RUN_OUT/formal-*.jsonl" \
  --ledger "$RUN_OUT/gpu-budget.jsonl" --manifest "$RUN_MANIFEST" \
  --manifest "$RUN_DATA_4K" --plan "$RUN_OUT/formal-plan.json" \
  --min-repeats 3 --out "$RUN_OUT/formal-audit.json"
```

完整计划默认五 arm、B1/2/4/8/16、1K/4K、三个 panel × 三次；`run_plan.py` 仅输出顺序，不代替现场授权或自动执行。Pilot 默认 code 单次、B2/B8、W512/1024/2048 的 full_cache/ring 加 W0 对照，可调整面板/重复。容量默认单列 16K/B4、prefill chunk 4096 的 full/ring，保留初始化、实际 prefill 或 decode 失败的阶段，禁止缩小负载后混比。
