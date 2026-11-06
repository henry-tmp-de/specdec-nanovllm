# 可选分层prefill：单次小测结果

已在独立分支实现并接入正常LLM执行路径，默认关闭。用户指定的“块A进入第二层时启动块B第一层”可运行；本次单卡小测未获得吞吐收益，保留实验分支，不合main，不追加调参或重复。

## 功能

- 新开关`enable_layer_prefill=False`保持旧路径；True时按`layer_prefill_chunk_tokens`拆内部task，以同chunk前层/同序列前chunk同层的已record CUDA events表达依赖。
- 两条worker stream推进流水，默认chunk128/maxinflight2；serial模式使用相同对角DAG但仅一条worker stream，不能称为旧chunk-major或外层scheduler分块。
- 每个task保留h/residual、绝对位置、paged KV与独立Context；汇合并恢复原Context后只运行一次原LM head/采样，外部scheduler才推进水位。
- 首版仅Qwen3 BF16、TP1、target-only，关闭prefix cache、mixed、draft/spec与targetquant；不支持的组合提前报错。DSpark路径没有联合启用。支持范围见[使用配置](USAGE.md)。

## 按最新要求仅少量测试

单张RTX3090、Qwen3-4B BF16、同一0.70显存预算、prefix关闭、普通decode精确Graph。只有一个性能工作点：B4、每请求4096输入/128输出、code面板、T=1、ignore EOS、chunk128、inflight2；每臂一次预热（输出16，用于编译/stream初始化）与一次primary计时，不求均值/置信区间。输入含约96.34%–97.29%的Wiki填充，不称自然4K任务质量或生产压测。

| 模式 | Prefill单轮墙钟 | 完整请求墙钟 | 实际输出吞吐 |
|---|---:|---:|---:|
| 关闭功能，整段full | 2.595 s | 4.860 s | 105.346 tok/s |
| 同DAG单流serial | 4.815 s | 7.081 s | 72.309 tok/s |
| 同DAG双流pipeline | 4.946 s | 7.212 s | 70.991 tok/s |

该单点pipeline/full为0.673881倍（吞吐下降32.61%），pipeline/serial为0.981774倍（下降1.82%）。decode时间接近，退化主要落在prefill这一轮的墙钟中；本轮没有做GPU算子breakdown，不将其唯一归因为某个硬件瓶颈。单卡共享SM/带宽，小块GEMM、重复weight访问、Python发射与events成本均为待验证解释。

实际serial/pipeline各完成128个task、4608次layer调用，fallback为0；pipeline两条不同stream各64task。三臂均127次B4精确decode Graph、无padding/fallback，target KV池均207块。这证明功能实际执行，没有用回退冒充流水。未采NSYS，不据stream计数宣称已测得GPU kernel重叠或确定没有重叠。

共三次primary、12请求、1536个真实输出；三臂本次输出hash均相同，但不据此认证任务质量或大范围数值等价。[逐请求审计与表](analysis/final-01/report.md)、[计时细项](analysis/final-01/TIMING-DETAILS.md)、[CSV](analysis/final-01/performance.csv)。

## 最小正确性检查

只跑一组混长B2（257/513），三臂用相同固定历史续算16步；诊断替换sampler不参与速度计时。各臂独立模型/KV，page padding置零、有效逻辑位置预置NaN，完成fence后按实际block table抽全部层的边界K/V。

102条logits检查及2592条逐层K/V检查全部通过预声明门槛；serial/pipeline记录的最大KL和KV相对RMS均为0。full与分块的最大KL为0.00047045、KV相对RMS为0.04660（接近0.05门槛），argmax一致；不外推到全部长度/模型。完整logits/KV向量未落盘，raw后处理复核已记录指标和覆盖，不冒充重新计算向量差。

核心CPU/fake测试36项、旧DSpark/量化96项及4个回归脚本通过；这些单列，不作为GPU成功依据。主协调和GPT-6.1 Sol Medium亲审代码及真实raw/报告，数值与单次结果均完成独立复核。

## 版本、失败记录及收尾

核心提交6a5c61b；实际执行源b98fc8f。最初19cf工具pre-parser把`--stage`误作`--stage-timeout-s`缩写，启动在CUDA/ledger前失败、GPU成本0；原log/receipt保留。b98禁用缩写、仅解析outer参数并完整转发argv，真实回归后才启动实际GPU检查。没有重复primary计时。

同进程数值smoke出现TorchDynamo重编译上限警告；三个独立primary日志均未出现同类警告，不能拿它解释表中的primary退化。本轮按用户要求不补NSYS、扫描或重复测量。

完整4个GPU阶段全部ok/exit0，累计107.142822秒（约1.79单卡分钟，含加载/编译/预热/清理），未用满600秒预算。136个自有PID/starttick身份和三个controller均退出，现场GPU0为15MiB、0%、无compute；没有后续GPU任务。

- [资源释放](analysis/RESOURCE-FINAL.json)、[主协调独立复算](analysis/COORDINATOR-AUDIT.json)
- 完整raw/log/启动门禁保留于`evidence/final-01`；归档在`D:/tools/codex-runs/layer-prefill-20261106/final-evidence-01.tar.gz`及服务器同批根，约505KB，不含模型权重或凭据。
- 分支`feat/layer-pipelined-prefill-20261106`，默认关闭，未合main。
