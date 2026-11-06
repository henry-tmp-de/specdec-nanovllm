# 单卡层间 prefill 实验实现

## 操作路径

默认 `enable_layer_prefill=False`，原 prefill/decode 和 scheduler 路径保持现状。显式开启只支持 Qwen3 BF16、TP1、prefix cache 关闭、无 draft/speculation、无 mixed overlap、无 target quantization；不支持的组合在配置阶段拒绝。`layer_prefill_chunk_tokens=128`、`layer_prefill_max_inflight=2`、`layer_prefill_mode=pipeline` 为启用后的默认参数；`serial` 使用相同分块和同一对角 DAG，但提交到单条 worker stream，不是按块跑完全层的 chunk-major，也不是旧 scheduler 的外层分块。`layer_prefill_nvtx=False`，NSYS 诊断时显式开启逐 task/layer 标签。

一个 scheduler prefill step 内，每条序列已分配的 token span 按块切分，使用各自绝对 start/end、block table 和 slot mapping。每个 task 的 context 设置 `cu_q=[0,chunk_len]`、`cu_k=[0,absolute_end]`，读取该序列完整可见前缀；不从全局 scheduler 水位推测每层已经完成的 KV。每个 DecoderLayer 原样消费/返回 `(hidden,residual)`，只在末层后运行最终 norm。输出按原 flat token 顺序拼接，恢复原 context 后一次调用 LM head；内部块不采样、不提前推进 scheduler 水位。

任务按最多两个块一组，以 layer 对角顺序提交；一个 task 固定一条 stream。每层等待自身前一层 event，同一序列的后一块还等待前一块在同层的 event。所有 event 在等待之前提交，主 stream 对输入和上下文元数据设置 source event，worker 输出的末 event 被主 stream 等待。输入、metadata、模型/KV 和输出跨流使用 `record_stream`；完成计数只在既有 sampler 的 `.tolist()` 返回后增加，不做逐层 CPU 同步。异常恢复原 context，向主 stream 排入 worker fence 并阻止失败 executor 复用。

初始化 warmup 在 KV 池分配前走原完整前向并记 `warmup_no_kv`；已分配 KV 后块表不齐是错误，不静默回退。单块输入记录 task 数与单流原因，不把两个已创建 stream 当作 GPU 已发生重叠。

## 验证边界

CPU 覆盖任务拓扑、same-sequence 同层依赖、不同序列隔离、257/513 和非零 cached start、块表/slot、hidden/residual、原序拼接、异常 context、event 顺序、开关及启动 fallback；fake CUDA 不证明设备执行或性能。

独立 Sol Medium 和主协调审核通过后，主协调在同一 1800 秒账本内安排 300 秒内 smoke；GPU 比对 A 原完整前向、B 同分块串行、C 同分块流水的 logits/KV 与固定后续轨迹，另记录实际 task/stream/fallback 和缓存配置。之后是否做不带 profiler 的吞吐 A/B/C 和 NSYS 由主协调按正确性和剩余预算决定；GPU overlap 与完整请求吞吐分开报告，不预设收益。
