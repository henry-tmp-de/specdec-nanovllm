# 绘图覆盖与缺失

图只呈现已审、匹配指定条件且完整3次的单元。
状态：正式批量结束 · 覆盖不全；完整报告与全阶段GPU账目以终态复核为准。

相对无投机图只显示报告已导出的 DSpark full/ring；1×是参考定义，不补算AR比值。

| 上下文 | panel | arm | B1 | B2 | B4 | B8 | B16 |
|---|---|---|---|---|---|---|---|
| 1024 | code | 无投机 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | code | AR 全上下文（k=6） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | code | AR 窗口 512（块对齐） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | code | DSpark 全上下文（k=7） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 1024 | code | DSpark 环形窗口 512 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | math | 无投机 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | math | AR 全上下文（k=6） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | math | AR 窗口 512（块对齐） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | math | DSpark 全上下文（k=7） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 1024 | math | DSpark 环形窗口 512 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | general | 无投机 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | general | AR 全上下文（k=6） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | general | AR 窗口 512（块对齐） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 1024 | general | DSpark 全上下文（k=7） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 1024 | general | DSpark 环形窗口 512 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 |
| 4096 | code | 无投机 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | code | AR 全上下文（k=6） | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present | failed_attempt_present |
| 4096 | code | AR 窗口 512（块对齐） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | code | DSpark 全上下文（k=7） | 已审/3次 | 已审/3次 | 已审/3次 | missing | failed_attempt_present |
| 4096 | code | DSpark 环形窗口 512 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | math | 无投机 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | math | AR 全上下文（k=6） | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present | failed_attempt_present |
| 4096 | math | AR 窗口 512（块对齐） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | math | DSpark 全上下文（k=7） | 已审/3次 | 已审/3次 | 已审/3次 | missing | failed_attempt_present |
| 4096 | math | DSpark 环形窗口 512 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | general | 无投机 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | general | AR 全上下文（k=6） | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present | failed_attempt_present |
| 4096 | general | AR 窗口 512（块对齐） | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |
| 4096 | general | DSpark 全上下文（k=7） | 已审/3次 | 已审/3次 | 已审/3次 | missing | failed_attempt_present |
| 4096 | general | DSpark 环形窗口 512 | 已审/3次 | 已审/3次 | 已审/3次 | 已审/3次 | failed_attempt_present |

细分各metric是否绘出见 plot-status.json。缺测/失败不填零，曲线只沿报告给出的连续 segments 绘制。
