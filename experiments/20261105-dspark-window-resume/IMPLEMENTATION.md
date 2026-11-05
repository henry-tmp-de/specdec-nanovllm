# 实现与证据边界

`dspark_window_tokens` 默认 0，`dspark_window_storage` 默认 `full_cache`；W>0 的 `full_cache` 仍分配与 target 块等量的 DSpark 缓存，只截断读取；`ring` 每活跃序列拥有 W/块大小个独立块。W>0 时 `DSparkProposer._inputs()` 仍返回 `(anchors, lengths, tables, bucket, temperatures)`：`lengths` 是绝对 anchor C；`tables` 是按历史顺序排列的逐 token 物理槽，补位 -1。W0 则保持原来的块表输入。可见历史是 `[max(0,C-W),C)`，noise block 使用绝对位置 `C..C+6`。target KV 不截断。

`enable_prefix_cache=False` 在 `BlockManager.can_allocate()` 中禁用查找和复用，并禁用哈希登记；W>0 必须选择该设置，其他 arm 也可显式选择。默认值为 True。池不够时 ring 不回退到不存在的全缓存。owner 绑定序列和分配代次，检查块表和连续已确认区间；preemption、完成和引擎 step 异常经 deallocate 清空，重新准入从头构建。当前不支持 TP>1、mixed overlap 和取消 API。

ring 的 scratch 行数为 `max(max_num_seqs*(spec_k+1),8)`，独立于 prefill token 预算，预算先扣实际环池、scratch 和 sentinel。target 验证图、补桶图、ragged eager 验证、普通 decode 图/回退均将对应 target feature 写入 scratch；采样结果产生后只提交继续运行序列的 anchor + 接受前缀，拒绝后缀不进入环。结束序列跳过提交并由 scheduler 释放；EOS/输出限制先裁剪返回值，后处理收到的就是实际可交付前缀。full_cache 可直接写相同绝对位置，但读取只取已确认窗口。

真实 prefill 在 Qwen 每个选层先按目标行 `index_select`，再作 residual 相加及拼接；target 的整个 prefill 仍完整执行。初始化 warmup 早于池分配、没有 owner，沿用全 feature 路径；KV 预算仍基于该 warmup 峰值，不能声称所有 prefill 峰值已消除。

计量接口：`runner.dspark_window_stats` 中的 `target_cache_bytes`、`dspark_cache_bytes`（含 sentinel）、`scratch_rows`、`scratch_bytes` 是 tensor 大小；`prefill_target_rows`、`prefill_feature_rows` 是累计行数，`max_feature_tensor_bytes`、`max_projection_tensor_bytes` 是相应单个张量的最大字节数，不是 allocator 峰值。`commit_host_enqueue_s` 及 proposer 的 `input_prepare_wall_s` 是主机侧墙钟，不是 GPU 时间。`context_bucket_histogram`、draft/verify 图命中、补桶与回退计数用于检查执行路径；`graph_mem` 的值是捕获时 allocated 增量，不等同整个进程 graph reserved 或峰值。完整显存应另行以 allocator/设备观测记录。

本地 CPU 测试用小模型、模拟特征和 CPU 图替身，没有加载正式权重、没有运行 CUDA 图/模型。它们不证明真实 GPU 数值、吞吐或容量；图精确/补桶/回退的实际 replay 和提交仍是 GPU 门禁。
