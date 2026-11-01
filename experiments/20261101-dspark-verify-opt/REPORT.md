# DSpark 图补桶消融：阶段 A

## 结论

**target 验证图补桶在这组 B8 请求上有收益；本实现的 DSpark draft 补桶没有显示额外收益。** B8 一般文本完整请求吞吐由拆分但不补桶的 356.2 提到仅 target 补桶的 399.1 token/s；再打开 draft 补桶仍为 399.1。B16 一般文本分别为 438.5、459.9、450.5 token/s。B8/B16 一般文本的最佳 DSpark 实测依然低于无投机 445.4/666.5，不能据此合并或宣称大批量普遍加速。

上一轮将 `spec_graph_pad=False`，所以 target verify 和旧 AR draft 都没有使用已有 P16 补桶，DSpark draft 也未接入补桶。这是**配置与集成遗漏**，不是引擎缺少原有 padding 能力。开启已有 padding 后，旧 AR 基线明显变快，必须按本轮实测重新比较。

## 口径

单张 RTX 3090、TP1、BF16，target Qwen3-4B；原 AR draft 为 Qwen3-0.6B k6，DSpark 为官方训练权重 block7、7 个候选；不量化、不滑窗、不混流。B8/16，输入 1024、输出 256、T=1，代码/数学/一般文本各 3 次，使用上一轮同一 [公开任务与 WikiText 填充 manifest](../20261101-dspark-integration/evidence/workload-manifest.json)、seed 和统一显存预算 0.70。实际达到指定 B，无抢占。该 panel 大部分是 WikiText 上下文填充，结果不等于真实 Agent 流量或任务质量。

完整请求吞吐=实际交付 token / 请求入队到全部完成墙钟，包含 prefill、所有 decode 和尾批；下面均为三次中位数，单位 token/s。S0：拆分 backbone/head，target 与 draft 都不补桶；S1：只打开已有 target verify 补桶；S2：target 与 DSpark draft 都补桶。拆分模式的 Markov/Exp 采样仅处理真实 B，dummy 不消费其 RNG；完整官方 7 位置 backbone 和真实条件 q 不变。原单图控制保留旧 `propose` 路径且 padding 关闭。无投机和旧 AR 控制也启用现有 padding 设置。

| B / panel | 无投机 | AR + pad | 原单图 | S0 拆分 | S1 target-pad | S2 双 pad |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 代码 | 447.1 | 406.9 | 619.6 | 620.1 | 634.1 | 632.6 |
| 8 / 数学 | 446.0 | 398.4 | 574.7 | 574.0 | 611.6 | 610.7 |
| 8 / 一般 | 445.4 | 279.9 | 358.8 | 356.2 | 399.1 | 399.1 |
| 16 / 代码 | 669.1 | 513.1 | 690.3 | 684.1 | 693.5 | 692.4 |
| 16 / 数学 | 667.3 | 491.7 | 632.5 | 628.9 | 609.4 | 609.2 |
| 16 / 一般 | 666.5 | 372.0 | 441.6 | 438.5 | 459.9 | 450.5 |

S1/S0 的配对比中位数：B8 代码/数学/一般为 1.022/1.066/1.121；B16 为 1.012/0.969/1.049。S2/S1 分别为 B8 的 0.999/1.000/1.000，B16 的 0.998/1.000/0.979。三次范围及配对比保存在 [原始数据审计](evidence/stage-a-audit.json)。

**不要把上述全部差异归因于 padding 本身。** 所有 36 个 panel×mode 组内的三次输出哈希稳定；但 B8 数学、一般的 S1 与 S0 输出哈希不同，B16 多数组间也不同。相同 seed 不保证不同 batch 形状的 BF16 logits 逐位一致。B8 代码三模式及原单图输出哈希相同，此处 S1 的 632.4–634.3 高于 S0 的 584.3–620.7；S0 和原单图首轮分别为 584.3 和 574.2，后两轮约 620，虽然运行期捕图均为 0，首轮额外墙钟成本的原因尚未定位，不能剔除。

## 补桶命中、尾批和内存

以每组第二次为例，S1 的 target 验证补桶次数：B8 代码/数学/一般为 3/8/18，B16 为 3/1/27；S2 在此之外的 draft 补桶次数分别为 B8 7/17/25、B16 11/13/36。S2 的 draft 形状未命中 fallback 为 0，但 verify 的 `ragged_draft_len` 仍分别为 B8 9/10/13、B16 11/20/20 次。batch 补桶不解决候选数不齐。

AR+pad 的 B8 代码/数学/一般 target padded verify 为 8/17/35 次，draft padded forward（含 catch-up）为 67/133/214 次；B16 为 target 11/16/45、draft 104/139/290 次。上一轮关闭 padding 的 AR 中位为 B8 254/249/140、B16 394/325/206；本轮为 407/398/280、513/492/372。B8 历史组显存预算为 0.85 而本轮为 0.70，且两轮预热方式不同；这些历史数字只说明原对照选择有明显影响，不能作为只改变 padding 的严格因果消融。

满批轮与完整请求也不同：B8 一般的 S1 在 active B8 轮约 585.7 tok/s，无投机约 555.0，但完整请求为 399.1/445.4；尾批和候选不齐必须计入。B16 一般 S1 的 TTFT 中位约 1.883 秒、TPOT 22.56 毫秒，无投机约 1.839 秒、16.88 毫秒。B16 一般 S1 中位完整墙钟 8.907 秒，其中 prefill 1.884 秒、decode 7.020 秒；draft/verify CUDA stream 事件跨度约 2.383/3.529 秒，可能包含等待，不能直接相加或作为 kernel 层归因。

原单图与 split 均按相同有限 backbone 形状预热：B8 五个、B16 六个；split 另预热真实 B 的 head 图。计时前均重置 seed，108 个有效 run 的运行期新捕图数为 0。两者内存分配方式不同：B16 原单图六个图池总 reserved 约 4594 MiB，split 的共享 backbone/head 图池约 1646+44 MiB；正式 run 峰值 reserved 分别约 21.46/18.81 GiB。B8 对应约 2306 MiB 与 844+24 MiB，进程峰值约 18.84/17.49 GiB。图池与 split 实现一起变化，不能把显存差只归因于 padding。

## 正确性、失败与边界

CPU 契约 17/17、旧 P16 测试通过；真实 GPU B3/11/15 的 target/draft 精确与 padded 图均实际命中，RNG 状态一致，guard/sentinel 不受 dummy 写入，ragged 安全回退。B11 相同前驱 draft q 的最大 KL 为 0.002479、target 最大 KL 0.003722，token 不逐位相同；真实 p/q 的 8192 样本分桶 TV=0.014647。这是有限 BF16 数值和分布检查，不是全功能或逐位等价证明。详见 [GPU 冒烟记录](GPU-CHECKS.md)。

第一轮 B8 原单图预热因实验 helper 未处于 `inference_mode` 报错，未产出正式结果；修复 helper 并重新 CPU 测试后，在相同配置重试成功。原始失败文件 `*-attempt1.*` 保留，失败时间计入预算，没有放宽数值阈值或降低负载。

[审计脚本](audit_stage_a.py) 对照 manifest 复算 36 个 panel×mode 单元 / 108 次有效 run 的输入、输出数、实际 active B、逐序列 remaining/accepted、开关、图捕获和预热形状，结果通过、无运行期捕图警告。原始 JSONL 和日志在 [evidence](evidence/)，包含每轮成本和尾批；所有统计见 [stage-a-audit.json](evidence/stage-a-audit.json)。

本轮含冒烟、失败和重试累计 **825.47 GPU 秒 / 2700 秒**；结束 GPU7 为 15 MiB、0% 利用率、无本批进程，端口 24719 无监听，见 [预算台账](evidence/gpu-budget-stage-a.jsonl) 和 [退出证明](evidence/gpu-stage-a-final-state.json)。阶段 B 固定 k 扫描尚未启动，等待主协调决定；未 push 或 merge。
