# DSpark 接入与批量曲线：执行计划

## 问题与固定范围

在同一 nano-vLLM 分支、单张现场空闲 RTX 3090、TP=1 和 BF16 条件下，比较 Qwen3-4B 目标模型的完整请求成本：无投机、Qwen3-0.6B 自回归 draft、DeepSeek 官方训练的 DSpark draft。不开量化、滑窗或混流；不训练，不更换目标模型。B 为固定批中的请求数，不是开放到达率 HTTP 压测。

官方 4B 权重为 `deepseek-ai/dspark_qwen3_4b_block7`，版本 `3457dff1417cb84927f6098a5fcb7cee85c934b7`；目标为机器已有的 Qwen3-4B。DeepSpec 的配置指定 5 层、target layers `[1,9,17,25,33]`、mask token 151669、vanilla Markov rank 256。官方实现中七个输入是 anchor 加六个 mask，输出是 anchor 之后**七个候选**；AR 对照固定 k=6，DSpark 固定 block=7，候选长度差异必须在所有结果中写明。距输出长度上限不足 8 个 token 时只验证能容纳的前缀，仍保留一个 rejection/bonus 位置。

## 分层验收

1. CPU 契约：批量/逐条 backbone 等价；可见历史、拒绝后缀、padding sentinel、7 个依赖真实前驱的原始 logits、温度采样及拒绝采样分布。用同一合成权重分别运行当前实现和只读官方 DeepSpec，比较不同上下文长度的 backbone 与条件 logits；记录 FP32/BF16 误差。现有路径回归测试单列。
2. GPU smoke：每次启动前核实设备型号、显存及进程；只用一张完全空闲的 3090。分别检查 B=1、B=2，实际加载官方 checkpoint，核验共享 embedding/LM head 与目标模型的权重，比较官方参考的目标特征、draft logits、非因果注意力及缓存水位；覆盖全接、首位拒绝、部分接受、尾部和前缀缓存。T=1 用真实 q 的分布验收；如做 greedy 对拍，使用独立诊断入口，不改变正常采样器。任何正确性异常停止正式测量。
3. 正式：ctx=1024、输出 256、B=1/2/4/8/16、三种方法、代码/数学/一般文本 panel，各条件重复 3 次，arm 间使用相同 prompt IDs 和 seed。既有本地 DeepSpec 公开任务作问题源；若样本不足 1024 token，以不重复的 WikiText-2 自然文本片段补足上下文，问题仍在末尾，明确报告这种上下文构造。每次完整请求前统一清除未使用的 prefix 命中，单批内部如发生共享须记录。
4. 图公平：各 arm 不启用 `enforce_eager`；原有 AR 和 target verification 配置 B=1/2/4/8/16 图桶。DSpark 单独按 `(B, context bucket)` 抓图，统计捕获、命中和回退；预热同形状且正式测量包含实际 fallback。无法捕图或形状不覆盖则保留结果作为工程诊断，不据此否定算法。所有时间包含实际 prefill、请求排队、draft、verify、采样和 postprocess；预热和初始化单独列出。
5. 每个原始 JSONL 记录 arm、B、实际 active B 分布、任务/输入 ID 哈希、repeat、seed、k、输出长度、KV 容量/抢占、图命中与回退、TTFT、TPOT、整请求吞吐、稳态 decode 吞吐、draft/verify/postprocess 粗成本、候选/接受前缀/交付量、峰值显存和失败。以真实交付 token 为分母，详细定义在脚本中；不逐 kernel 同步计时。

## 资源与结果

整个批次累计不超过 2 GPU 小时，按本批所有启动、失败和重试累计；串行执行，进程只能清理本批自己的进程。容量不足保留具体 B 的失败，不换卡数或把较小活动批冒充该 B。结果须列每次原始值与离散程度，区分算法、实现/后端/图路径和设备容量。只有正确性通过、原默认路径无回退，并且 B=8/16 相对无投机与旧 AR 均有超过实测波动的完整请求净收益，才提交主协调考虑合并；其余情况保留分支。代码及小型报告可提交，权重、缓存和大量输出不提交。

执行中，统一 `gpu_memory_utilization=0.85` 的 B16 DSpark 在正式 prefill OOM。保留该失败；随后允许一次独立的 B16 三 arm 对照，统一把 KV 池预算设为 0.70，仍使用 B16、1024 输入、256 输出及相同图桶和测试样本。单列原配置失败与新配置结果，任何请求未实际达到 B16 都不算 B16。该重试不扩大任务或更换设备。

参考：<https://github.com/deepseek-ai/DeepSpec>，<https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7>。
