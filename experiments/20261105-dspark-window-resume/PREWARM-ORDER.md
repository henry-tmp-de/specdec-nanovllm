# DSpark 图预热顺序试验

既有 0.85 预算、B8、4K、DSpark full 首个 pilot 在 backbone 预热中 OOM：日志记录当时申请 342 MiB、设备剩余 263.69 MiB、PyTorch private pools 5.04 GiB，原始文件为 `evidence/logs/pilot085-01-dspark_full-b8-w0.log`。这些是该次失败的记录；当前没有证据证明图捕获顺序是根因，也没有新的显存或吞吐结果。

本次仅将 `DSparkProposer.prewarm()` 的 backbone 捕获排序为 `(B*context, B, context)` 降序，head real B 降序；形状集合、已捕图跳过、池与输出缓冲、未命中回退和采样逻辑不变。如果调用时已有图池，排序无法撤销之前的分配。正式 harness 的 `run_once()` 在每次请求前重新设定种子；改变捕获顺序可能影响预热 RNG 消耗，不能仅凭 CPU 假图声称 CUDA 随机流逐位一致。

这一方向与 [vLLM v0.22 `capture_model` 的大形状优先说明](https://docs.vllm.ai/en/v0.22.0/api/vllm/v1/worker/gpu_model_runner/#vllm.v1.worker.gpu_model_runner.GPUModelRunner.capture_model) 一致，但两套实现、工作负载和池分配未证明相同。[PyTorch 共享图池文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#sharing-memory-across-captures) 的非并发及输出依赖约束仍适用；本项目当前限定 TP1、无 mixed overlap，backbone/head 输出缓冲在各自捕获外分配。真实 GPU 验证仍由主协调执行。

CPU 检查在 `prewarm-order-cpu/`：

- `D:\Anaconda\python.exe -c "import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s', 'tests/test_dspark_prewarm_order.py', 'tests/test_dspark_pad.py', 'tests/test_dspark_contract.py']))"`：20 passed；实际 `prewarm`/捕获方法在 CPU 图替身下按大形状先后被调用，覆盖重复/已有图、同一共享池、输出独立、真实 B 的条件采样、随机状态和未命中回退，也检查本次 1..8 × 4096/4352 的预热集合。
- `D:\Anaconda\python.exe -c "import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s', 'tests/test_dspark_feature_rows.py', 'tests/test_dspark_window.py', 'tests/test_dspark_window_config.py', 'tests/test_dspark_window_runner.py']))"`：51 passed，见 `window-tests.log`。
- `D:\Anaconda\python.exe -c "import numpy, runpy; runpy.run_path('tests/test_graph_pad.py', run_name='__main__')"`：退出 0，见 `graph-pad.log`。
- 测试开发中第一次直接调用 `_split_propose`，未进入生产 `propose_batch` 使用的 `inference_mode`，触发 inference tensor 原地更新异常；测试已按生产模式修正。失败退出码和原始输出字节以 base64 保存在 `initial-test-failure.json`，并未改引擎来掩盖该失败。

上述只验证 CPU 逻辑与假图调用；不能声称真实 CUDA capture、显存节省、OOM 消除或数值无差异。后续仅在审查通过后由主协调按原 0.85 预算对 B8 full 预热及完整请求执行一次验证，失败则保留并停止，另行讨论。
