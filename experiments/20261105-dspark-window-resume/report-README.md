# DSpark 结果报告工具

`report_results.py` 只读指定 JSONL、台账、manifest 和计划，在 CPU 生成新的 JSON 与 Markdown；不会启动模型、GPU 或下载。输出文件必须不存在，不改写 raw。测试用的合成 fixture 仅验证算术和拒绝错误归组，不是实验结果。

报告只覆盖本轮 nano-vLLM 分支，不是 vLLM/SGLang 引擎横测。DSpark 滑窗比较关闭 target prefix lookup，在线 mixed 不支持；不能外推到前缀复用或线上混批。4K 由任务后缀加 Wiki 填充构成，逐请求比例以 manifest 为准；报告会在实测记录均超过 95% 时明确标出，不能将吞吐面板当作自然长任务或任务质量评估。历史 prefill、量化、kernel 与混批结果不重算也不套用于本轮 DSpark。

```powershell
D:\Anaconda\python.exe -B experiments\20261105-dspark-window-resume\tests_report_results.py -v
```

本次 15 项 CPU fixture 和对既有 B11/UUID 诊断 raw 的只读解析见 [CPU-REPORT-CHECKS.txt](CPU-REPORT-CHECKS.txt)，不表示新增 GPU 结果。

Linux 示例（仅传已经存在的输入、manifest 和计划；`--input` / `--input-glob` / `--manifest` / `--plan` 均可重复）：

```bash
"$RUN_PY" -B experiments/20261105-dspark-window-resume/report_results.py \
  --input "$RUN_OUT/results/smoke-b3-ring-padded-01.jsonl" \
  --input "$RUN_OUT/results/smoke-b3-ring-padded-02.jsonl" \
  --input "$RUN_OUT/results/padding-b11-01.jsonl" \
  --input-glob "$RUN_OUT/results/pilot-*.jsonl" \
  --ledger "$RUN_OUT/results/gpu-budget.jsonl" \
  --manifest "$RUN_MANIFEST_1K" --manifest "$RUN_MANIFEST_4K" \
  --plan "$RUN_OUT/pilot-plan.json" --min-repeats 3 \
  --out-json "$RUN_OUT/report.json" --out-md "$RUN_OUT/report.md"
```

完整实验有 16K 容量或正式矩阵时，分别增加对应 raw、16K manifest 和 capacity/formal plan。未提供计划时只能显示已观察单元，不能判断矩阵齐全；没提供匹配 manifest、完整台账或预热记录的主实验标为未验证，不能算成配对结果。允许有失败的台账阶段，但总 GPU 时长超过 14400 秒、smoke 超过 1800 秒、ledger 未闭合或身份不一致时，报告阻止全部 matched 和曲线数值。ledger 只统计已结束阶段；有运行中阶段时不会将其当成零耗时。

未确认完整正式矩阵时，Markdown 标为 Interim；因此单独导入旧 0.70 pilot 不会呈现成完整正式结果。

- 完整请求吞吐：真实交付 token 数 / 从首个请求入队前至全部结束的墙钟。decode span：第一个 decode 轮的 `start_s` 至最后一个 decode 轮的 `end_s`，包含轮间 host 间隙；旧逐轮耗时之和另列，不能冒充 span。满 B 前缀仅指最初连续满批 decode 轮。
- TTFT、TPOT 及真实交付批次间隔从每个请求的交付记录复算，报告每次 P50/P95/P99 和样本数；批次间隔不插值、不称 SSE。accepted/proposed 与真实交付 tau 分列。
- 同 ctx/B/panel、输入 hash、种子、源码提交、GPU、目标模型元数据及运行条件匹配时，至少三个相同 repeat 才给全部配对比值、中位数和范围；不生成置信区间。跨 arm 的输出 hash 差异只作观测，随机 T=1 时不是正确性判断。
- 内存直接保留 raw 的 target/draft KV、scratch、权重、allocator allocated/reserved/peak、图捕获 allocated 增量和具名图池 reserved；预热前后 allocated/reserved 差值是全进程 allocator 变化，不冒称每张图的独立分配。这些数值可能重叠，不能相加。feature tensor bytes 按实际张量形状计数，projection bytes 是形状公式而非 allocator 峰值；W0 没有 feature 行/commit 计数插桩，显示未采集而非零。draft 输入构造与 commit enqueue 是 host 墙钟，不称 GPU kernel 时间。
- `curves` 中未测、失败、少于三次或条件不符的点为 null；`segments` 在缺口、输入前缀不一致或源码/GPU/target/draft 变化时断开。旧 pilot 的 .70、新 pilot 的 .85、正式计划以及容量阶段按各自 plan、预算与源码提交分别列出，不跨版本拼接；full/ring 容量只有同条件的实际成功记录才配对，target 池块数不同时不能声称同 target 池测量。
- 旧 B11 padded KL 失败、UUID 预检失败和其他失败按原文件/行/guard 原样保留。诊断开销、强制接受及历史 prefill/量化/kernel/混批数据不进入本轮主性能表。W0 是原路径，不将其结果归因于新滑窗收益。
