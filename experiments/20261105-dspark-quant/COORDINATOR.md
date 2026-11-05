# DSpark W8A16：协调记录

## 本批目标与授权

2026-11-05 用户要求对现有 DSpark 草稿模型尝试分层量化，并验证有没有加速。允许本批必要实现、CPU审核和单卡推理评估；不训练，不改target精度，不追加W4/AWQ/GPTQ。原DSpark滑窗实验已经结束，旧结果和失败原件不动。

- 起点：155cd51c33f33c43dde1e5353c523be95f3c97db。
- 新工作区：D:/学习/specdec-dspark-quant，feat/dspark-quant-20261105。
- Terra Max编码，GPT-6.1 Sol Medium独立审核，主协调审核实际diff与原始结果并操作GPU。两Agent不得启动GPU。
- 只使用现场确认无compute进程、低显存、利用率0的一张3090；预计优先GPU0，启动前重新核UUID与占用。不得终止他人或用户其他作业。
- 累计GPU阶段墙钟上限3600秒，smoke累计上限600秒，加载、失败、清理均计入同一账。不是旧实验预算延长。无需sudo、安装依赖或下载模型。

## 固定设置

target Qwen3-4B BF16，官方DSpark block7，draft最近512位置独立ring KV，CUDA Graph精确batch桶，TP1，prefix关闭、mixed overlap关闭。target/shared embedding/shared LM head、fc、Markov、norm、confidence及KV保持BF16。专用量化开关默认none，不借用原AR的draft_quant开关。

首轮preset：none、FFN（5层gate/up/down共15线性层）、attention（5层q/k/v/o共20线性层）、all（35）。W8A16按输出通道对称PTQ，融合INT8权重读取与计算，禁止完整BF16权重在前向中反量化写回显存。CPU量化/加载开销单列，量化模型真实q必须传入拒绝采样。

## 分阶段评估

1. CPU契约与独立代码审核：默认路径、层名单、实际替换、共享权重、dtype、非法配置、guard实际预算。
2. 单卡真实权重micro/Graph数值诊断：量化算子对相同量化权重数学参照，BF16原权重误差另列。B=1/8/16对应draft有效行7/56/112；kernel与端到端分开。
3. 引擎先B1/1K/code/输出128，比较none/FFN/attention/all。若通过正确性且有收益迹象，再固定同输入测试B8/B16和code/math/general，每条件3次输出128。新源码none作基线，不与旧256输出吞吐直接配比。
4. 端到端吞吐为主要收益；记录TTFT、TPOT、接受候选比例、实际每轮交付、Graph命中/回退、实际参数与scale字节、target KV池块数。必要时另做有计时诊断，不把额外CUDA Events测量当正式吞吐。

若数值不符、引擎断言、OOM、外来GPU进程、预算不足则保存raw停当前阶段，由主协调判断。负收益也交付，不自动调参直到变快；没有GPU结果不能声称已加速。所有对照同模型/窗口/精度/输入/Graph策略/物理卡，输入填充性质与计时口径沿用旧方法并明示。

## 当前状态

本批已完成，无后续GPU任务，最终结论见FINAL_REPORT.md。历史初始快照：2026-11-05 10:34Z现场GPU0至5空闲，GPU6/7已有compute进程；不得用该历史快照重新启动作业。

## 代码与CPU检查

实现commit为6fea041。主协调已阅读新增DSpark专用量化模块、config/model_runner差异、bench wrapper与guard入口；调用顺序是严格加载官方权重后、全部warmup/capture前替换。Attention的k/v在write_context及proposal中调用同一模块，fc与KV精度仍为BF16，拒绝采样生产代码没有替换q。

主协调独立执行 `D:/Anaconda/python.exe -B -c "import numpy; import pytest; raise SystemExit(pytest.main(['-q','tests/test_dspark_quant.py']))"`，25项通过（3.42秒）。编码者另交joint90项与既有工具13项CPU日志；这些不是GPU数值或速度证据。GPU数学micro和独立Sol代码审核尚待完成。

原正式审计的分组字段不含量化preset，本批必须先按raw的`dspark_quant.preset`分组再复用compute_run，不能把四方案混合汇总。

## GPU阶段验收

冻结执行源码2ac595ee1565e1218bfabad8c827c32784a130b5；引擎与微基准分别通过Sol Medium独立代码审核。远端隔离根为`/home/ziru/nano-vllm/dspark-quant-20261105`，只用GPU0（f1619d21-c909-7c21-34f1-05c39a7b0c91），没有改其他模型/环境或使用sudo。

- micro-01：15个真实layer0权重形状/batch条件、30条eager/Graph记录，量化数学与Graph检查全通过。最大同量化参照relative RMS为0.001678953，Graph/eager输出完全一致。B1 Graph代表层加速1.33–2.54倍，B16五个代表形状全部变慢。guard正常退出，用时20.879秒，显卡释放；Sol独立复核通过。
- B1预实验：四preset各三面板、三重复，共36个实际测量run，全经compute_run及Sol独立复核有效。all对none逐重复吞吐比中位数：代码1.022438、数学1.021733、通用0.996971。量化已改变部分通用面板的接受比例，不能把kernel收益直接当生成收益。实际target KV块数none/ffn/attention/all为308/291/309/290；各方案同0.85预算，不是相同池容量。host PTQ约0/61.57/24.34/85.92秒，已包含在load_initial_graph_s，不能相加。
- 上述五个阶段累计298.169608秒，均ok并完成清理。未改变实现，按原计划进入B8/B16各四preset对照，仍同一3600秒总账。首次启动记录`BATCH-LAUNCH-01.json`，PID1136568/start_ticks907129166；仅作本次记录，不可未来凭旧PID操作进程。

本地证据先存`evidence/micro-01`与`evidence/prelim-01`，每份gpu-budget是各阶段快照，不能拼接重复计费；最终以单一完整账为准。CPU汇总工具1679c06及展示补丁2510042也经Sol审核；raw请求/计时复算与GPU身份/账目审计各有范围，不互相替代。

## 最终验收与停止

原计划B8/B16八组也已全部完成。完整108次测量、900条请求、115200个输出token，36组各三重复、27组共81对比较。主协调独立逐raw核验compute_run、实际模块数量/字节、保护参数、输入配对、源码、图计数与账，结果位于`analysis/final-01/COORDINATOR-AUDIT.json`；Sol Medium独立重算全部CSV字段、配对表与微基准，并审核最终人工报告，通过feature交付门禁。

权重可选择性压缩，但仅部分小幅正收益：B1代码/数学all约+2.2%；B8 Attention数学约+2.8%，其余多为负；B16只有FFN数学约+0.4%，最大负收益为Attention通用约-7.9%。默认none，不合main，不继续扫参数直到出现正收益。

13个GPU阶段全部ok，累计1129.839238秒，唯一smoke20.879470秒；410个自有PID/starttick身份及三个controller全部退出。FINAL-RELEASE的现场GPU0为15MiB/0%/无compute。完整原始记录在`evidence/final-01`，压缩包在D:/tools/codex-runs/dspark-quant-20261105及服务器本批根下，已核同一包，不含模型权重或凭据。没有后续GPU任务；旧PD/其他研究任务未启动。
