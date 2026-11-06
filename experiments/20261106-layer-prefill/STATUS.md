# 状态

- worktree `D:/学习/specdec-layer-prefill`，branch `feat/layer-pipelined-prefill-20261106`，起点 `d051665281a3fa591a6f1f829be013bbe9de1f25`。
- 已实现默认关闭的 target-only 分层 prefill、同 DAG 单流对照、CPU planner、事件与 context 生命周期；benchmark/guard 为独立任务维护。
- `D:/Anaconda/python.exe -B experiments/20261106-layer-prefill/cpu_checks.py`：新合同 36 passed，旧 DSpark/量化 96 passed；已有 prefill 分块、轮转、块分配和 graph padding 回归均 exit 0。命令和日志见 `cpu-evidence/`；其中 fake CUDA 和 paged KV 均为 CPU 模拟。
- 未由实现者运行 GPU；准确性和吞吐须经独立审查与主协调 GPU 验收。
