# DSpark 验证效率：padding 消融计划

## 先纠正上一轮配置

上一轮 `experiments/20261101-dspark-integration/bench.py` 显式使用 `spec_graph_pad=False`；现有 P16 target verify 和 0.6B AR draft 的补桶能力因此关闭。DSpark proposer 此前也只按精确 B 查图，并没有接入补桶。B15 等形状回退是本轮应修正的配置与集成问题，不能表述为原引擎不支持 padding。

## 阶段 A：目标和 draft 的独立补桶

- 保持 target、官方 draft、block=7、真实条件 q、T=1 指数竞争采样及拒绝采样语义。target 开关沿用 `spec_graph_pad`；新增 DSpark draft 独立开关。验证 query 数不齐仍走 `ragged_draft_len` fallback，不能用 batch padding 掩盖。
- draft backbone 可向上选择 1/2/4/8/16 桶；dummy 的 anchor/position 合法、context_len=0、block table 指向 DSpark sentinel；只返回真实前 B 行。为了 dummy 行不消耗真实行 RNG，实验路径将 backbone 图和 Markov 逐步采样拆开，采样只处理真实 B 行，仍使用原来的 `softmax(logits/T)/Exp(1)`；头部图按真实 B 有界缓存，不乘以上下文桶数。此拆分可能改变运行成本，三种 DSpark 模式统一使用它，上一轮未拆分结果仅作为历史对照，不能把差异全部归因于 padding。
- target verify 的 dummy slot=-1，目标 KV kernel 不写；DSpark 目标特征 KV 的 dummy 只落私有 sentinel，完成后清零。要实测 target logits、draft 原始 q、真实行次序、prefix/拒绝后缀、dummy 不覆盖物理 KV；图随机数对真实 B 的消耗不能受 padding 行数影响。
- CPU 与单卡 GPU 契约覆盖 B3/11/15，比较 eager、精确形状和补桶；对固定 seed 记录真实 token、真实行 RNG 状态及分布。图预热结束后重置 seed；运行期若仍抓新图，记录形状、次数和成本，不能宣称轨迹完全无扰动。只预热有限的上下文桶和桶形状，并记录图池/显存，捕图失败立即停止并保存证据。

## 性能与预算

针对 B8 和 B16，复用上一轮公开样本 manifest：1024 输入、256 输出、代码/数学/一般各 3 次，固定 seed，单张已确认空闲的 3090、TP1、BF16、相同 `gpu_memory_utilization=0.70`。DSpark 三个交错模式：S0 target/draft padding 都关；S1 target 开、draft 关；S2 target/draft 都开。另跑无投机和原 Qwen3-0.6B AR draft（已有 target+draft padding 开启）作为公平基线。先预热，逐 run 清 prefix 哈希；记录实际 active B、满批轮吞吐、完整请求墙钟吞吐、decode、TTFT/TPOT、接受前缀、精确/补桶命中、每类 fallback、图捕获和峰值显存。

上一轮固定 padding 关闭结果单独作为参考，特别注明 sampler 拆分和预热差异，不拿它当与本轮严格配对的控制。阶段 A 契约失败停止性能测试；容量/图内存不足保留失败和原始日志，不降低真实 B 冒充通过。每个 GPU 进程有独立 timeout；本轮包括失败的累计上限 2700 GPU 秒，只用现场复查空闲且 UUID 匹配的 GPU7，不接触其他任务。

## 阶段 B：仅在通过阶段 A 后决定

若 A 通过且本轮预算允许，在 B8/16 的一般文本上以小样本固定扫描候选前缀 k=1/2/4/7：始终运行官方完整 7 位非因果 draft block，再只截短送往 target 的候选及对应真实 q，不改变噪声块语义。先检查完整请求收益是否稳定；无稳定收益则停止，不扩展矩阵，也不直接实现 confidence 自适应调度。
