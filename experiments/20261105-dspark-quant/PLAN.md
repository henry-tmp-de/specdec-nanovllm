# DSpark W512 W8A16 试验计划

## 范围与实现

- 基线是本分支同一份源码、官方 DSpark/Qwen3-4B checkpoint、target BF16、DSpark ring W512、split CUDA Graph；none 为默认关闭的量化预设。
- `ffn` 精确选择 5 层的 gate/up/down 共 15 个线性层，`attention` 选择 q/k/v/o 共 20 个，`all` 共 35 个。按输出通道对称 INT8，BF16 scale、BF16 激活和 FP32 累加，重用 `int8_linear` 融合 kernel。fc、共享 embedding/LM head、Markov、confidence、norm 和 KV 保持原精度。
- 模型按原路径加载完整 BF16 权重后，对单个选中层按 256 输出行复制到 CPU、量化，再逐块复制 INT8 权重和 scale 到设备。不会在设备上生成完整 FP32 权重，也不缓存完整反量化 BF16 权重。加载峰值和 allocator 的影响仍须现场测量；目标池不能假定相同。
- 加载统计列出实际选中名称、形状和前后字节，正式记录还列全部 35 个候选、已排除参数、scale 缓冲、target 共享权重身份、实际 KV 块/缓存字节与图内存。不要把共享权重重复计入 DSpark。

## 执行门禁

1. CPU 合同通过后由主协调与独立 Sol Medium 审核；本地 CPU 模拟不构成 GPU 证据。
2. GPU 前由主协调重新确认卡、UUID、其他进程、模型路径和环境。所有阶段使用本目录 `gpu_control.py`、同一个 ledger；总上限 3600 秒、其中 smoke 上限 600 秒，含失败与清理。工作中若发现外部占用则按 guard 停止。
3. 真实权重 micro：layer 0 的 q/k/o/gate/down 五种形状（v 与 k 同形、up 与 gate 同形），B=1/8/16、每个 B 的行数为 7B。eager 和 CUDA Graph replay 分别比较 INT8 kernel 与原 BF16 linear 时间；按原 INT8 q/scale 的数学次序做输出对拍，graph/eager 也对拍。量化相对 BF16 的误差是诊断而非 kernel 正确性裁判。
4. 真引擎先 B1、1K、code、输出 128 比较 none/ffn/attention/all；有希望再测 B8/16 与 code/math/general，计划每格 3 次。相同模型、输入、0.85 预算和图设置，保留全部原始结果、失败、图命中/fallback、alpha/tau、实际 KV 容量。正式测量不增加逐步 CPU 同步或独立 event；旧工具的 diagnostic-cost 只用于单独诊断。
5. 拒绝采样必须消费量化 DSpark 实际 logits q。量化可能改变候选，不能以与 BF16 的 token 一致作为唯一正确性验收。micro 数学和图一致性先过，真实引擎再验收；未覆盖的真实前向/拒采边界如实标注。

## 解释边界

- micro 延迟、完整请求速度、模型加载时间和显存各自独立报告；净加速以实际完整请求为主。
- 只在固定负载、同图设置、预算、序列长度和 output cap 下比较，不推广到未测 batch/场景。每个 preset 的 KV 池可因加载峰值改变，单独记录，不将池变大等同速度。
- 不引入训练、W4/AWQ/GPTQ、target/共享权重量化、额外旧 full-cache 对照或 150 格全量跑数。
