# 固定预设速度试验准备

依用户新的速度导向要求，额外提供固定预设 `q_o`（10 层）、`down`（5 层）和 `q_o_down`（15 层），分别只替换五个 decoder block 内指定的线性层。原有 `none/ffn/attention/all` 默认和含义不变，target、共享 embedding/LM head、fc、Markov、norm、confidence、KV 均不动。加载官方 BF16 checkpoint 后一次性量化所选层，后续正常预热/捕图；不使用条件质量诊断的双权重 bank 或重放切换前的图。

`bench_selected.py` 复用上轮 native fixed-batch 计时与已审的实际权重/buffer 清单，不修改旧实验脚本。入口仅允许 `none/q_o/down/q_o_down`、B1/B8、1K 输入、128 输出、W512、0.85 初始化预算，固定 code/math/general × 3 repeats、T=1、Graph-on 和 seed。`--manifest` 要求旧 1K workload manifest，而非质量诊断的 teacher-forced manifest。实际选层、scale 字节、Graph 统计、加载时间以及原始记录均沿用 native harness 的口径。

本阶段仅 CPU 准备：`D:/Anaconda/python.exe -B experiments/20261105-dspark-quant-quality/cpu_checks_selected.py`，质量/数据/新入口合同 33 passed，现有 DSpark 合同含新增预设 96 passed；命令与日志在 `cpu-evidence-selected/`，保留了质量阶段原有 CPU 记录。是否使用同一质量批次剩余 GPU 预算、运行哪些候选，均由主协调结合质量诊断结果决定。各候选相对 `none` 的端到端收益要在相同条件和新源码下重新测量；此前算子 micro 的正收益不代表完整请求加速。旧 `audit_quant.py` 对 preset 有旧列表，不能直接混合审计这些新增组，需新审计入口或逐组检查。
