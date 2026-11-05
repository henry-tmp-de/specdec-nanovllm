# 普通 target decode 精确图桶

新增 `decode_graph_bs`，默认空 tuple，普通 target decode 使用原有桶和选择规则。显式指定时要求正整数、无重复、不超过 `max_num_seqs`，并包含 `max_num_seqs`；显式配置的最大 batch 当前不超过 512，符合现有 `run_model` 的图执行范围。捕获前排序，固定缓冲宽度覆盖最大指定桶。

`decode_graph_bs=tuple(range(1,B+1))` 且 `max_num_seqs=B` 时，普通 target decode 的每个实际 batch 都选择相同 B 的图。若显式桶稀疏，仍使用原规则选最小的 `>=B` 桶，会产生补桶。投机 draft/verify 的桶仍由独立的 `spec_graph_bs` 控制；评测可对所有 arm 同时设置两个完整精确桶列表并关闭对应补桶开关。捕获图数量、预热和图池开销需实际记录。

`runner.decode_graph_hits`、`decode_graph_exact_hits`、`decode_graph_padded_hits`、`decode_graph_bucket_histogram` 和 `decode_graph_fallbacks` 用于区分实际调用路径；`graph_mem.decode_graphs_gb` 沿用 allocated 增量口径，不代表全部 GPU reserved 峰值。

选择该方案不改变既有补桶判据：本轮 B11 target verify 的 padded KL 为 0.0201313533，超过事先设定的 0.02；同输入 eager 与 exact 的 KL、target KV RMS、scratch RMS 均为 0。原记录 `evidence/results/padding-b11-01.jsonl` 为失败记录，不能称补桶数值问题已修复或放宽其阈值。B3 padded 合同结果不代表 B11 或其他 batch 通过。

CPU 图替身执行真实 `capture_cudagraph()` 和 `run_model()` 的 Python 捕获/选择/缓冲路径，检查 B11/B15 逐批精确选择、动态 scratch 内容、活跃环不变和随机状态不消耗。它不捕获或运行真实 CUDA Graph；GPU 验收由主协调单独安排。

CPU 命令为 `D:\Anaconda\python.exe -B experiments\20261105-dspark-window-resume\cpu_decode_graph_checks.py`。6 个子进程均退出 0：四个新增测试文件同进程 51 项通过，旧 DSpark contract 7 项、pad 10 项通过，旧图补桶、mixed 调度和抢占复现脚本通过。命令、退出码及完整输出见 `decode-graph-cpu/results.json`；CPU 图替身不构成 GPU 图命中或数值证据。
