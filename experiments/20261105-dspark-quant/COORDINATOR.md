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

实现准备中，尚未启动任何GPU任务。2026-11-05 10:34Z现场GPU0至5空闲；GPU6/7已有compute进程。该快照不代表后续启动许可。
