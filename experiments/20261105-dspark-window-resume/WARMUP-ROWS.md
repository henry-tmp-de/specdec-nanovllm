# 初始化热身行数覆盖

主协调记录的 0.85、B8、4K DSpark full 试验中，大形状优先后图预热完成，随后真实 warmup 请求的 target prefill 在 MLP activation 申请 608 MiB 时 OOM，现场报告剩余 471.69 MiB、private pools 2.77 GiB。该试验和先前 OOM 均保留；本次没有重跑 GPU，也不能断言本改动消除 OOM。

旧 `warmup_model()` 以 `max_num_batched_tokens // max_model_len` 向下取整生成等长序列。在 32768 token 预算、最大长度 4360、最多 8 序列时只输入 7×4360=30520 行，比可达预算少 2248 行。修正后，目标行数为 `min(max_num_batched_tokens, max_num_seqs * max_model_len)`；先放满长序列，再以一条非空短序列补尾，因此该例为 7×4360+2248=32768 行。超出序列数上限的预算不会构造超量序列；不修改 KV 预算公式、模型、dtype、输入或 attention backend。

初始化在模型前向前打印 `[warmup]` JSON，并保存在 `runner.warmup_profile`：`max_num_batched_tokens`、`max_model_len`、`max_num_seqs`、`target_rows`、`scheduled_rows` 和 `sequence_lengths`。这些是构造的输入元数据，不是成功完成前向或实际 GPU 峰值的证明。缓存尚未绑定时仍不建立 prefix 或窗口 owner；DSpark target 全量 feature 收集保持原样。

更完整的热身可能测得更大的峰值，并通过**原有**预算公式减少后续分配的 KV 块数；此影响应记录，不能解读成调整了预算或保证能准入同一批请求。

CPU 检查以模型与设备桩运行真实 `warmup_model → run → prepare_prefill → target`，核对真实输入行、每条长度、position、`cu_seqlens_q` 和 profile 与实际输入一致，并确认 W0/窗口缓存前无 owner：

- `D:\Anaconda\python.exe -c "import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s', 'tests/test_warmup_rows.py']))"`：修前 6 failed / 12 passed，3 种模式都为 30520 而非 32768，31/10 反例为 30 而非 31；`warmup-rows-cpu/before.json` 保存当时命令、退出码与原始输出字节（base64）。修后 21 passed，见 `after.log`。
- `D:\Anaconda\python.exe -c "import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s', 'tests/test_warmup_rows.py', 'tests/test_dspark_feature_rows.py', 'tests/test_dspark_window.py', 'tests/test_dspark_window_config.py', 'tests/test_dspark_window_runner.py']))"`：72 passed，见 `joint.log`。
- `D:\Anaconda\python.exe -c "import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s', 'tests/test_dspark_prewarm_order.py', 'tests/test_dspark_pad.py', 'tests/test_dspark_contract.py']))"`：20 passed，见 `dspark.log`。
- `D:\Anaconda\python.exe -c "import numpy, runpy; runpy.run_path('tests/test_graph_pad.py', run_name='__main__')"`：退出 0，见 `graph-pad.log`。这些均为 CPU 元数据或小模型/假图，不是 GPU 成功。

审核后由主协调在相同 0.85 预算执行一次 B8 full 试验；如果仍失败，保留失败并停止，不能据此宣称内存问题已解决或继续盲目改参数。
