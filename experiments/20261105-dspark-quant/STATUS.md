# 状态

- 已从 `155cd51c33f33c43dde1e5353c523be95f3c97db` 建立隔离工作树和 `feat/dspark-quant-20261105`。
- 默认关闭的 DSpark 专用 W8A16 量化路径、严格候选检查、CPU 分块量化与运行时 buffer 字节清单已实现。原 AR draft 量化 guard 保留。
- CPU 使用现有 `D:/Anaconda/python.exe`，先 import numpy；执行 `D:/Anaconda/python.exe -B experiments/20261105-dspark-quant/cpu_checks.py`，联合 DSpark 量化/滑窗合同 90 passed，旧 bench/guard 13 passed。记录见 `cpu-evidence/`。真实量化函数参与测试，kernel 和 Graph 在 CPU 合同测试中为模拟，没有 GPU 数值或性能结果。
- 曾有一次自编 CPU 测试用 `patch.dict(sys.modules)` 全量恢复，导致后续 PyTorch 重复注册失败；已改为只恢复明确的模块 stub，重跑上述联合回归通过。这是测试隔离问题，不是模型/GPU 数值证据。
- 尚未运行 GPU。主协调和独立 Sol Medium 审核后，由主协调决定进入 GPU 门禁。
