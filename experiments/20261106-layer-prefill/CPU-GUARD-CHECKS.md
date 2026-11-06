# CPU 守护合同结果

本机 `D:\Anaconda\python.exe`，不使用 GPU 或 SSH：

```text
pytest -q experiments/20261106-layer-prefill/tests_layer_prefill_guard.py experiments/20261105-dspark-window-resume/tests_gpu_control.py
...........                                                              [100%]
11 passed in 0.18s
```

新测试核对 1800 秒总预算、300 秒 smoke 预算、45 秒清理预留，以及总预算或 smoke 预算不足时在 GPU 查询前停止；原守护测试覆盖 UUID、PID/start ticks、pidfd、TID→Tgid 和 ledger 计费。`gpu_control.py --help` 在本机也可运行，不查询设备。

最初把新测试命名为 `tests_gpu_control.py` 时，与旧目录同名文件在一次 pytest 联跑中发生模块名冲突、未执行测试；已将新测试改名为 `tests_layer_prefill_guard.py`，上面的结果为修正后联跑。此记录不代表 GPU 环境、实际运行或性能已验收。
