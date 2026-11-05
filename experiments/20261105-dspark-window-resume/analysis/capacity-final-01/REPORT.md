# DSpark 实验报告

状态：Interim（阶段性报告；正式完整矩阵尚未确认）。
本报告由 raw 逐请求/逐轮和守护台账离线生成；诊断和 capacity 单列。缺测不是 0，失败不会被过滤。
这是当前 nano-vLLM 分支内的模型与缓存策略对照，不是 vLLM、SGLang 或其他推理引擎之间的横向测试。DSpark 滑窗实验关闭 target prefix lookup，在线 mixed 流量不受支持；结果不外推到启用前缀复用或在线混批。
本报告不将历史 prefill、量化、kernel 或混批数字归入本轮 DSpark 结果；matched 对照只说明其列出的模型、负载、批量和条件，不能单凭这一报告推导普遍加速或任务质量。AR k=6、DSpark k=7；AR 块对齐窗口与 DSpark 最近 W 个历史 token 语义不同。W0 为原路径对照，不能将 W0 结果归因为新滑窗工程收益。

完整请求吞吐为真实交付 tokens / 入队至全部完成墙钟；decode span 是首个 decode 起点至末个 decode 终点，包含轮间 host 间隙。满 B 前缀只含最初连续满批轮。TTFT、TPOT 和交付批次间隔为每次运行内的 P50/P95/P99；批间隔不等于 SSE，也不插值成逐 token 间隔。

计划单元 2，观察到 1 次主/容量运行，完整 1，缺测 0，失败或无效 1。
若未提供完整计划，覆盖范围未知；少于三次的单元只展示观察值，不计算置信区间或作为简历对照。

4K 输入的填充比例尚无已核验记录；该补齐面板是合成吞吐负载，不是自然4K任务或任务质量测量。

## 各预算计划覆盖

| 计划 | phase | GPU预算 | 完整/计划 | 缺测 | 部分或失败 | 完整正式矩阵 |
|---|---|---:|---|---:|---:|---|
| experiments/20261105-dspark-window-resume/evidence/final-all-01/data/capacity4a085-plan.json | capacity | 0.850 | 1/2 | 0 | 1 | False |

## 单元与吞吐

| phase | 预算 | ctx | B | panel | arm | W | 状态 | repeat | 完整请求 tokens/s：全部值、中位、范围 | decode span tokens/s：全部值、中位、范围 | 满B前缀 tokens/s：全部值、中位、范围 |
|---|---:|---:|---:|---|---|---:|---|---|---|---|---|
| capacity | 0.850 | 16384 | 4 | code | dspark_full | 0 | failed_attempt_present | [] | 未采集 | 未采集 | 未采集 |
| capacity | 0.850 | 16384 | 4 | code | dspark_ring | 512 | complete | [0] | [50.728]；中位 50.728；范围 50.728–50.728 | [225.641]；中位 225.641；范围 225.641–225.641 | [236.613]；中位 236.613；范围 236.613–236.613 |

## 条件与负载边界

target/draft 身份以路径、配置哈希和权重文件名/大小记录；不表示本轮重新哈希权重内容。填充比例逐请求列出，实际任务截断数按 manifest 记录。

invalid 或未验证的原始条件仍列出，明确标注状态；它们不进入上面的已核验填充范围。

| phase/ctx/B/panel/arm/W | 审计状态 | k / 输出 / T | GPU / 源码提交 | target / draft 路径 | 图策略 / prefix / 显存预算 | 逐次 Wiki 比例 / 任务截断数 |
|---|---|---|---|---|---|---|
| capacity/16384/4/code/dspark_ring/512 | complete | 7 / 256 / 1.000 | GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91 / 4a4a64b29192e57b3e27eb24d6a8a3069dfbd18f | /home/ziru/nano-vllm/models/Qwen3-4B / /home/ziru/nano-vllm/dspark-integration-20261101/models/dspark-4b | {"buckets":[1,2,3,4],"draft_padding":false,"name":"exact_all","target_decode_padding":false,"target_verify_padding":false} / False / 0.850 | [{"repeat":0,"task_truncated":0,"valid":true,"wiki_fractions":[0.9908447265625,0.99139404296875,0.99322509765625,0.99114990234375]}] |

## 逐次延迟与内存

单位：延迟秒；缓存和 allocator 字节为采集值，具名图池 reserved 可能共享；模型权重文件大小不能证明权重内容一致。详情和全量图/窗口统计见 JSON。

| phase/预算/ctx/B/panel/arm/repeat | TTFT P50/P95/P99 | TPOT P50/P95/P99 | 真实交付批间隔 P50/P95/P99 | accepted/proposed / tau | KV / scratch bytes | peak allocated / reserved bytes | graph allocated GiB / graph pool reserved bytes / 预热全进程 allocΔ/resΔ bytes | graph 命中/回退 / 预热秒 | feature目标行/选中行 | host prepare / commit enqueue 秒 |
|---|---|---|---|---|---|---|---|---|---|---|
| capacity/0.85/16384/4/code/dspark_ring/0 | 7.832240/15.664742/15.664742 (n=4) | 0.033085/0.062947/0.062947 (n=4) | 0.093002/0.094824/7.926333 (n=187) | 833/1269 / 5.455 | {"dspark.cache":41963520,"dspark.scratch":655360,"runner.kv_cache":11286872064} | 21115646976 / 21483225088 | {"decode_graphs_gb":0.008,"verify_graphs_gb":0.0227} / {"dspark_backbone":188743680,"dspark_head":25165824,"target_decode":23068672,"target_verify":25165824} / 27693056/239075328 | {"bucket_histogram":{},"exact":0,"fallbacks":{},"padded":0} / {"active_batch_histogram":{"1":4,"2":2,"3":1,"4":44},"context_bucket_histogram":{"512":51},"decode_graph_bucket_histogram":{},"decode_graph_exact_hits":0,"decode_graph_fallbacks":{},"decode_graph_hits":0,"decode_graph_padded_hits":0,"graph_captures":0,"graph_exact_hits":51,"graph_fallbacks":{},"graph_hits":51,"graph_padded_hits":0,"graph_runtime_capture_s":0.0,"graph_runtime_captures":0,"graph_runtime_shapes":[],"head_graph_captures":0,"head_graph_hits":357,"head_runtime_batches":[],"head_runtime_capture_s":0.0,"head_runtime_captures":0,"input_prepare_calls":51,"input_prepare_wall_s":0.025971679016947746,"verify_exact_hits":44,"verify_fallbacks":{"ragged_draft_len":7},"verify_hits":44,"verify_padded_hits":0} / 0.167732 | 65536 / 2048 | 0.025972 / 0.008066 |

## 匹配对照

仅相同输入、代码提交、GPU、target 元数据和运行条件且至少三次配对时计算比值；输出哈希差异本身不是 T=1 正确性判断。

| 预算/ctx/B/panel/W | 候选 / 对照 | 状态 | 候选/对照吞吐比：全部值、中位、范围 | 每次输出哈希不同的请求数 |
|---|---|---|---|---|

## 容量对照（独立口径）

| ctx/B/panel/W/显存预算 | full vs ring | 相同 target KV 块数 |
|---|---|---|
| 16384/4/code/512/0.85 | not_comparable: full-cache cell is failed_attempt_present | 未采集 |

## 失败、缺测和诊断

- experiments\20261105-dspark-window-resume\evidence\final-all-01\results\capacity4a085-01-dspark_full-b4-c16384.jsonl:1：capacity/dspark_full，阶段 graph_prewarm/未采集，OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total capacity of 23.57 GiB of which 773.69 MiB is free. Including non-PyTorch memory, this process has 22.79 GiB memory in use. Of the allocated memory 22.38 GiB is allocated by PyTorch, with 37.88 MiB allocated in private pools (e.g., CUDA Graphs), and 33.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)。
- 台账 formal4a085-49-ar_window-b16-c4096: failed，outcome=child_failed_or_lingering，elapsed=36.558 秒。
- 台账 capacity4a085-01-dspark_full-b4-c16384: failed，outcome=child_failed_or_lingering，elapsed=19.661 秒。
- 台账 smoke-b3-ring-padded-02: failed，outcome=child_failed_or_lingering，elapsed=2.605 秒。
- 台账 padding-b11-01: failed，outcome=child_failed_or_lingering，elapsed=23.413 秒。
- 台账 warmup-rows-b8full-085-01: failed，outcome=child_failed_or_lingering，elapsed=30.473 秒。
- 台账 formal4a085-40-ar_full-b8-c4096: failed，outcome=child_failed_or_lingering，elapsed=28.671 秒。
- 台账 pilot085-01-dspark_full-b8-w0: failed，outcome=child_failed_or_lingering，elapsed=24.512 秒。
- 台账 formal4a085-46-dspark_ring-b16-c4096: failed，outcome=child_failed_or_lingering，elapsed=39.631 秒。
- 台账 formal4a085-41-dspark_full-b16-c1024: failed，outcome=child_failed_or_lingering，elapsed=27.591 秒。
- 台账 formal4a085-47-nospec-b16-c4096: failed，outcome=child_failed_or_lingering，elapsed=32.384 秒。
- 台账 formal4a085-50-dspark_full-b16-c4096: failed，outcome=child_failed_or_lingering，elapsed=30.523 秒。
- 台账 formal4a085-48-ar_full-b16-c4096: failed，outcome=child_failed_or_lingering，elapsed=36.472 秒。
- 台账 prewarm-desc-b8full-085-01: failed，outcome=child_failed_or_lingering，elapsed=26.303 秒。
- 台账 pilot-08-dspark_window_full-b8-w512: failed，outcome=child_failed_or_lingering，elapsed=32.491 秒。
- 台账 smoke-b3-ring-padded-01: failed，outcome=child_failed_or_lingering，elapsed=2.499 秒。

| 诊断来源 | 类型/B/策略 | 记录的 KL / KV RMS / 门槛 | 合同结果 |
|---|---|---|---|
| 未输入 | 未采集 | 未采集 | 未单独判定 |

曲线数据位于 JSON 的 `curves`：每个缺测/失败/不足三次的点为 null，`segments` 在缺口或来源变化处断开。
诊断 probe 原始记录及未匹配失败均保留在 JSON 的 `diagnostic_events` / `failures`；容量失败不等于 target KV 本身装不下。
