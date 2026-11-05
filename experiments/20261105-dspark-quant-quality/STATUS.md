# 状态

- 隔离工作树 `D:/学习/specdec-dspark-quant-quality`，分支 `feat/dspark-quant-quality-20261105`，起点 `49265569d2cb21015b4025148466ce21ff134fd0`。
- 已完成固定teacher-forced条件诊断与native固定权重速度验证；15个质量组及q_o/down/q_o_down三个部署候选均保留，默认none。
- 本机无 transformers；主协调已只读核远端 Transformers 5.15.1 的 Qwen3 forward/cache 签名与 accelerate，实际加载和数值由本轮 GPU smoke 验证。
- CPU 执行 `D:/Anaconda/python.exe -B experiments/20261105-dspark-quant-quality/cpu_checks.py`：质量、数据与数学合同 21 passed，已有 DSpark 量化和窗口合同 90 passed；结果和命令见 `cpu-evidence/`。CPU fake forward/量化测试不能证明真实 GPU 行为。
- 质量smoke及full均GPU通过：full为96case/24trajectory、15组每组672位置，1440评分/96首末恢复检查，独立复算480个汇总组一致。
- native共72次测量、324请求、41472输出，24组完整、18组54对。Q/O+down在B1代码+7.14%，B8通用-12.65%，不能称普遍加速；完整表见FINAL_REPORT.md。
- 主协调和Sol Medium已审实际代码与全部raw/报告。10阶段同一预算累计626.458494秒，274个自有进程身份及三个controller退出，GPU0释放。没有后续GPU任务；保持默认none，不合main。
