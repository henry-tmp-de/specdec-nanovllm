# DSpark 量化条件质量诊断

## 测量含义

DSpark/DFlash 草稿依赖 target 的 prefix 特征，不是可以单独计算普通语言模型困惑度的自回归语言模型。此处固定 target 条件与真实前驱，计算每个位置标签的 `-log q`；先对所有 token 求 NLL 总和除以位置数，再取指数，称为**条件 NLL 指数分数**。`exp(mean_NLL_quant - mean_NLL_BF16)` 衡量相对 BF16 的分数变化，不能直接叫标准语料 PPL；负差值只表示此固定样本的分数较低，不推断泛化提升。

另逐位置计算 `KL(q_BF16 || q_quant)`、`KL(p_target || q_quant)` 与 `sum(min(p_target,q_quant)) = 1 - TV`。最后一项是给定相同真实历史的**单步分布重叠**，不是实际多步接受率、alpha 或 tau。T=1，target BF16，所有条件复用相同 prefix 与同一批 teacher-forced 真前驱。

## 数据与条件

- 固定此前 target-only Graph、T=1 的 24 条真实生成轨迹（code/math/general 各 8），来自同一原始 raw run 与 1K workload，经 token hash 对齐。每条使用 offset 0/64/128/192，得到 96 个 anchor、每组 672 个预测位置；这仍是 24 条轨迹，不是 96 个独立样本。不是人类 gold 或标准 WikiText PPL。
- 以 `prefix=prompt+output[:offset]`、`anchor=output[offset]`、`labels=output[offset+1:offset+8]` 构造一例。target prefix 长度 `C=1024+offset`；先仅前向 prefix，选择 HF `hidden_states[layer_id+1]` 的层 1/9/17/25/33 的最后 512 行，拼接特征。之后另用完整 prefix cache 前向 `[anchor]+7 labels`，前 7 行得到 target 对应条件分布。prefix 特征不可包含 anchor 或 label；target prefix cache 不裁到 512。
- DSpark 使用 W512、连续时间顺序 slots `[0,512)`、对应真实绝对 positions `[C-512,C)`。每个量化组重新写完整上下文 KV。此连续 slots 是窗口的数学诊断，不测 native ring 池生命周期。backbone 输入 anchor 与 6 个 mask；每个 Markov 前驱依次是 anchor 和 `labels[:6]`，先以 BF16 相加，再转 FP32 softmax。
- 完整 BF16 和全部 35 个 INT8 模块同时驻留，仅用于切换诊断，不代表部署显存或吞吐。15 组：none、q/k/v/o/gate/up/down 单类、q_o/kv/gate_up/q_o_down、ffn/attention/all。每个单类跨 5 层。q_o_down 依据旧 micro 中 q/o/down 部分形状的速度潜力预先加入；该组质量本身不证明端到端加速。fc、共享 embedding/LM head、Markov、confidence、norm、target、KV 保持 BF16。单独离线选择，不改生产 preset，不训练、不跑吞吐矩阵。

## 门禁与证据

1. CPU 合同验证切分、label shift、组名单、NLL 汇总及 KL 方向、模块还原和每组重建上下文；主协调和独立 Sol Medium 代码审核通过后再开 GPU。
2. GPU 执行只由主协调现场核空卡/UUID后启动。所有阶段共享一个 ledger，新 guard 实际限总 1200 秒、smoke 300 秒，包含加载失败与清理。
3. 先 2 例 × 15 组 smoke，核 prefix features hash 在 target suffix、各变体、首末 BF16 重放后不变；核 cache 范围、7 个位置、概率有限性与归一化、首末 BF16 NLL/KL 稳定。smoke 独立记录，不与正式结果混合。
4. 通过后另跑 full 96 例 × 15 组；原始逐位置数值、case/panel/offset/轨迹身份、来源/模型/软件/设备和组名单写 JSONL。按 overall、panel、位置汇总，报告样本覆盖、轨迹数及 `exp(mean ΔNLL)`。无需新生成样本。
5. HF `sdpa` 是本轮 target reference；旧 native/HF 两条短提示只验证过近似一致，不代表本次 1K 场景逐位一致。不能把条件质量分数等同于实际端到端接受率、准确率或推理速度。
