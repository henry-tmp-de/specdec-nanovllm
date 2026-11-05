# DSpark 滑窗工具：CPU 检查与单卡 CLI

这些脚本只提供实验入口，**当前没有本轮 GPU 执行结果**。模型是现场已有的 Qwen3-4B、官方 DSpark block-7；T=1，七候选。`bench_smoke.py` 的额外数值检查改变了执行开销，其吞吐不能作为正式性能结论。强制接受仅用于数值诊断，不是采样分布测试。

本地 CPU 验证（14 个测试；原始输出见 [CPU-TOOL-TESTS.txt](CPU-TOOL-TESTS.txt)）：

```powershell
D:\Anaconda\python.exe -B experiments\20261105-dspark-window-resume\tests_gpu_control.py -v
D:\Anaconda\python.exe -B experiments\20261105-dspark-window-resume\tests_cpu_tools.py -v
```

下面的 Linux 命令仅是可审阅模板；现场 UUID、卡空闲状态、启动顺序、预算和是否执行由主协调决定。所有阶段必须共用唯一 `ledger`。`$RUN_ROOT` 指已冻结且干净的源码克隆根目录；`$RUN_OUT` 指源码树外的新结果目录；`$RUN_GPU` 是已核对的 nvidia-smi index，`$RUN_UUID` 为该卡完整 UUID。每阶段的日志和原始结果均必须使用新文件。守护器只支持有 `/proc` 与 pidfd 的 Linux，收到终止或挂断信号时仅清理本批已识别身份。其轮询无法阻止他人恰在两个检查之间启动；观察到外部计算进程即停止。

```bash
RUN_PY=/home/ziru/nano-vllm/venv/bin/python
RUN_TARGET=/home/ziru/nano-vllm/models/Qwen3-4B
RUN_DSPARK=/home/ziru/nano-vllm/dspark-integration-20261101/models/dspark-4b
RUN_MANIFEST=/home/ziru/nano-vllm/dspark-integration-20261101/results/workload-manifest.json
cd "$RUN_ROOT"

# 一个真实模型 B3/ring/padded 的诊断，不包含跨模式数值合同。
"$RUN_PY" -u experiments/20261105-dspark-window-resume/gpu_control.py \
  --gpu "$RUN_GPU" --uuid "$RUN_UUID" --stage smoke-b3-ring --kind smoke \
  --ledger "$RUN_OUT/gpu-budget.jsonl" --log "$RUN_OUT/smoke-b3-ring.log" \
  --stage-timeout-s 300 -- "$RUN_PY" -u experiments/20261105-dspark-window-resume/bench_smoke.py \
  --target "$RUN_TARGET" --draft "$RUN_DSPARK" --manifest "$RUN_MANIFEST" \
  --out "$RUN_OUT/smoke-b3-ring.jsonl" --batch 3 --max-seqs 4 --ctx 252 \
  --output-tokens 32 --mode ring --window 256 --graph padded

# 固定历史 W0/full_cache/ring 对拍：eager 或 graph 各运行一个新阶段。
"$RUN_PY" -u experiments/20261105-dspark-window-resume/gpu_control.py \
  --gpu "$RUN_GPU" --uuid "$RUN_UUID" --stage fixed-replay-eager --kind smoke \
  --ledger "$RUN_OUT/gpu-budget.jsonl" --log "$RUN_OUT/fixed-replay-eager.log" \
  --stage-timeout-s 900 -- "$RUN_PY" -u experiments/20261105-dspark-window-resume/gpu_fixed_replay.py \
  --target "$RUN_TARGET" --draft "$RUN_DSPARK" --manifest "$RUN_MANIFEST" \
  --out "$RUN_OUT/fixed-replay-eager.jsonl" --graph eager

# 多 owner 的补桶、ragged 与 commit；按 B=3、11、15 各用一个新阶段。
"$RUN_PY" -u experiments/20261105-dspark-window-resume/gpu_control.py \
  --gpu "$RUN_GPU" --uuid "$RUN_UUID" --stage padding-b3 --kind smoke \
  --ledger "$RUN_OUT/gpu-budget.jsonl" --log "$RUN_OUT/padding-b3.log" \
  --stage-timeout-s 300 -- "$RUN_PY" -u experiments/20261105-dspark-window-resume/gpu_padding_contract.py \
  --target "$RUN_TARGET" --draft "$RUN_DSPARK" --manifest "$RUN_MANIFEST" \
  --out "$RUN_OUT/padding-b3.jsonl" --batch 3
```

固定对拍的阈值在运行前规定：相同前驱 q / target logits 的最大 KL < 0.02、KV 相对 RMS < 0.05，参考旧 [GPU-CHECKS.md](../20261101-dspark-verify-opt/GPU-CHECKS.md)。最大 logit 绝对误差只作为诊断值报告。目标 KV 在固定对拍中抽查若干绝对位置，不宣称所有位置逐项对拍。多 owner 合同验证真实 verify scratch 对应各 owner 的独立环槽，以及应保留的旧槽没有被覆盖。

构造 4K 固定长度 workload，先在服务器 CPU 执行，使用本地 tokenizer、DeepSpec 任务与不重叠 Wiki 片段，不请求 GPU：

```bash
"$RUN_PY" -B experiments/20261105-dspark-window-resume/run_workload.py \
  --datasets /home/ziru/ugdd/code/DeepSpec-005e03b81cec/eval_datasets \
  --wiki /home/ziru/nano-vllm/p17-work/data/wikitext2_test.txt \
  --tokenizer "$RUN_TARGET" --ctx 4096 --rows 16 --out "$RUN_OUT/workload-4k.json"
```

这是补齐的合成上下文，文件逐行保存填充比例和截断标记。`--source` 的旧 1K token 拼接模式仅用于 CPU 或 smoke，不作为正式 4K workload。`audit_results.py --mode diagnostic` 只复核诊断原始记录；`--mode formal` 要求显式期望矩阵、原始输入 manifest、共享守护 ledger、完整配置及预热记录，拒绝将 instrumented smoke 作为性能结果。正式基准入口另行实现后才能使用该正式审计模式。
