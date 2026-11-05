# 已审报告绘图

`plot_results.py` 读取 `report_results.py` 输出的 JSON，以本机 Matplotlib 输出三组 PNG/PDF：完整请求吞吐、DSpark 相对无投机的吞吐、请求 TTFT/TPOT P50。它只使用报告中已有的中位数、范围、配对比值和曲线分段，不从 raw 重算指标；AR 不另外计算相对比值。

以本轮正式报告为例，在仓库根目录运行：

```powershell
& 'D:\Anaconda\python.exe' -B 'experiments/20261105-dspark-window-resume/plot_results.py' `
  --report 'experiments/20261105-dspark-window-resume/analysis/formal-final-01/REPORT.json' `
  --out-dir 'experiments/20261105-dspark-window-resume/analysis/formal-final-01/plots-new' `
  --engine-commit 4a4a64b29192e57b3e27eb24d6a8a3069dfbd18f `
  --gpu-uuid GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91 `
  --budget 0.85 --seed 20261105
```

脚本检查所选来源、GPU、预算、seed 和模型身份，且不会覆盖同名输出。输出目录还包含 `plot-status.md` 和 `plot-status.json`，逐单元列出缺测、失败及指标是否绘出；未完成的点不填零，曲线不跨缺口。图上的误差线是三次重复的 min–max，**不是置信区间**；延迟先按每次运行取请求 P50，再显示三次的中位数及范围。

需要查看未闭合台账的阶段性快照时，显式加 `--allow-interim`，图会标注内部预览。图上的批次状态只指输入报告包含的账目和覆盖范围，独立容量实验及其最终总账须另行核对。中文字体默认使用本机已有字体，也可用 `--font` 指定字体文件。
