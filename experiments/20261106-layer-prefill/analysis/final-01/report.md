# 层间 prefill：单次诊断与吞吐

只测 B4/4K/output128，每臂一条计时，无重复范围或置信区间。
正确性记录仅含逐位置指标，不含完整 logits/KV；本工具验证覆盖和门槛，GPU guard ledger 单独核验。
B 为相同对角DAG单流，C 为相同DAG双流；A/B 差含分块、发射、事件与元数据。
smoke有效：True；logit检查 102，KV检查 2592。

| 模式 | 有效 | 墙钟秒 | 输出token/s | target KV块 | 失败/缺测 |
|---|---|---:|---:|---:|---|
| full | True | 4.86018 | 105.346 | 207 | — |
| serial | True | 7.08076 | 72.3086 | 207 | — |
| pipeline | True | 7.21222 | 70.9907 | 207 | — |

单次吞吐比：{"serial_over_full": 0.6863915108505805, "pipeline_over_full": 0.673881166890311, "pipeline_over_serial": 0.9817737490011106}。

同预算不代表同 KV 容量；没有 profile 不能把原因或 GPU 跨流重叠写成已测结论。
