# 状态

- worktree `D:/学习/specdec-layer-prefill`，branch `feat/layer-pipelined-prefill-20261106`，起点 `d051665281a3fa591a6f1f829be013bbe9de1f25`。
- 已实现默认关闭的 target-only 分层 prefill、同 DAG 单流对照、CPU planner、事件与 context 生命周期；benchmark/guard 为独立任务维护。
- `D:/Anaconda/python.exe -B experiments/20261106-layer-prefill/cpu_checks.py`：新合同 36 passed，旧 DSpark/量化 96 passed；已有 prefill 分块、轮转、块分配和 graph padding 回归均 exit 0。命令和日志见 `cpu-evidence/`；其中 fake CUDA 和 paged KV 均为 CPU 模拟。
- 主协调已完成实际GPU检查与三臂单次性能，Sol Medium独立审raw/报告通过：混长B2固定16步数值门槛通过，同块serial/pipeline记录误差0；功能确实执行128tasks/4608层，无fallback。
- 本次B4/4K/输出128的单点full/serial/pipeline分别105.346/72.309/70.991 tok/s，流水比full慢32.61%。只各一次primary和一次预热，无NSYS/追加扫描；不作稳定或生产结论。
- 默认关闭、保留实验分支、不合main。实际执行b98fc8f；首19cf参数解析prelaunch失败0GPU记录保留，修复后没有重复primary。
- 4stage总计107.142822秒，136个自有进程身份退出，GPU0已释放，无后续GPU任务。入口FINAL_REPORT.md/USAGE.md。
