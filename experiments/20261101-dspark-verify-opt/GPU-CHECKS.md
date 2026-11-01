# GPU padding 契约冒烟

阶段结果：**通过约定阈值；尚未批准或执行正式消融**。测试代码 `2244a5a`，单张 RTX 3090 GPU7、TP1、BF16、真实 Qwen3-4B 与官方 DSpark，32-token 上下文，B3/11/15、k7。测试为取得精确形状参考临时额外捕获 B3/11/15；正式图桶计划仍是 1/2/4/8/16。

| 测试 | B3 | B11 | B15 |
|---|---:|---:|---:|
| target 精确图/补桶图实际命中 | 1 / 1 | 1 / 1 | 1 / 1 |
| draft 精确图/补桶图实际命中 | 1 / 1 | 1 / 1 | 1 / 1 |
| padded 与 eager target logits 最大 KL | 0 | 0.003681 | 0.003722 |
| 相同前驱 padded 与 eager draft q 最大 KL | 0 | 0.002479 | 0 |
| padded 与 eager target KV 相对 RMS | 0 | 0.004041 | 0.003734 |
| padded 与 eager DSpark context KV 相对 RMS | 0 | 0.006171 | 0.006000 |
| exact / padded CUDA RNG 状态一致 | 是 | 是 | 是 |
| target / draft guard 未改、sentinel 清零 | 是 | 是 | 是 |

B11 中相同 seed 的 exact/padded token 不完全相同，同时 RNG 状态一致、同前驱 q 的 KL 为 0.002479；不要把此结果描述为逐位等价。当前结果与不同 batch 形状导致的 BF16 数值差异相符，但未对底层 kernel 做归因。实采样分布检查用真实 target p 和 draft q 的 8192 次拒绝采样，top16+其他分桶 TV=0.014647；这不是整个输出序列分布的穷举证明。

候选长度不齐时确实命中一次 `ragged_draft_len` eager fallback，返回 116 个真实 query 行；不以 batch padding 冒充 ragged 支持。三组测试分别实际进行了 14 次真实 B head 图重放，预热以外 draft backbone/head 新捕图数均为 0。未发现真实 KV guard 被覆盖。

GPU allocator snapshot 的各图池 reserved：target decode 22 MiB，target verify 68 MiB，DSpark backbone 444 MiB，head 42 MiB；这些是本次短上下文冒烟，不能当作 1K 正式峰值。进程峰值 allocated 15.17 GiB、reserved 16.49 GiB。单次耗时 20.09 GPU 秒，已计入本轮 2700 秒上限。

原始 [JSON](evidence/gpu-contract.json)、[日志](evidence/gpu-contract.log)、[预算台账](evidence/gpu-budget.jsonl) 和 [退出状态](evidence/gpu-final-state.json)。结束时 GPU7 为 15 MiB、0% 利用率、无计算进程，端口 24719 无监听、本批无残留进程。
