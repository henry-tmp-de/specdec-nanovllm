# 状态

- 阶段一：`65f4b82` 修复普通 decode 位置映射；`aee5aa5` 修复混批抢占。完整命令、基线失败和修复后原始记录见 `PHASE1.md`。
- 阶段二：窗口配置、full_cache/ring、owner 生命周期、选层 feature 行、verify/decode scratch 确认提交和图形状接入已实现，待主协调及独立审核；CPU 数值与元数据检查见 `cpu-evidence/results.json` 及各日志。尚未运行正式权重、任何 GPU 模型或 CUDA Graph。
- 本机 D 盘 Anaconda Python 直接 `import torch` 发生 OpenMP Error #15；在单个测试进程先导入 NumPy 再导入 CPU PyTorch 可运行，未更改共享环境，也未使用 `KMP_DUPLICATE_LIB_OK`。测试命令由 `cpu_checks.py` 逐项记录。某次误用 pytest 运行独立脚本 `test_graph_pad.py` 返回 “no tests ran”/退出码 5，不算通过；之后按脚本入口执行并保留退出码和完整输出。
- 最后一次 `cpu_checks.py` 的 16 个进程均退出 0：新四文件联跑 33 项通过，单独运行新窗口/runner/feature/config pytest 分别 7/12/3/11 项通过，既有 DSpark contract/pad 分别 7/10 项通过，其余为可直接执行的既有回归和两项阶段一复现。小模型 float32 在 W 覆盖完整历史时 hidden/q 最大绝对差均为 0；W=4/8/12 的十轮 full/ring 对照中，hidden 最大绝对差分别为 4.77e-7/3.58e-7/2.98e-7，固定前驱条件 q 的最大绝对差为 2.03e-6/2.86e-6/1.91e-6。该比较不涉及正式权重。
- 测试期间发现缺少 `transformers` 的 CPU 本机环境；只在涉及真实 scheduler/runner 代码的测试中对外部导入作显式桩，随后重新执行通过。不能把这些桩解释为模型或推理测试。
- 主协调在 `937e10e` 上把新四个测试文件放进同一 pytest 进程时得到 20 passed / 13 failed，原因是模块级测试桩互相覆盖；本机另一加载顺序复现为 30 passed / 3 failed，原始输出 `cpu-evidence/joint-before.log`。修正为限定桩的导入/测试作用域后，两种文件顺序各为 33 passed，输出在 `joint-after.log` 和 `joint-alternate-order.log`；之前独立进程结果本身未变，但不证明联跑。
- 无训练、量化、远端操作、GPU 执行或性能/容量结论。GPU smoke、固定历史真实权重对照及 pilot 由主协调检查资源、审查入口后独立执行；失败及预算均记录。
