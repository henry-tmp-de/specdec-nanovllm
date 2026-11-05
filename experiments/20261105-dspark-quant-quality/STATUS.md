# 状态

- 隔离工作树 `D:/学习/specdec-dspark-quant-quality`，分支 `feat/dspark-quant-quality-20261105`，起点 `49265569d2cb21015b4025148466ce21ff134fd0`。
- 已实现固定 teacher-forced 条件诊断、15 组离线切换与 1200/300 秒 guard；数据和汇总由独立任务负责。`q_o_down` 针对旧 micro 部分有速度潜力的投影组合，尚无该组合端到端速度结论。
- 本机无 transformers；主协调已只读核远端 Transformers 5.15.1 的 Qwen3 forward/cache 签名与 accelerate，实际加载和数值由本轮 GPU smoke 验证。
- CPU 执行 `D:/Anaconda/python.exe -B experiments/20261105-dspark-quant-quality/cpu_checks.py`：质量、数据与数学合同 21 passed，已有 DSpark 量化和窗口合同 90 passed；结果和命令见 `cpu-evidence/`。CPU fake forward/量化测试不能证明真实 GPU 行为。
- 尚无本轮 GPU 运行和质量结果；独立 Sol Medium 审核待完成。
