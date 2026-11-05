# DSpark 滑窗恢复计划

在 `feat/dspark-window-resume-20261105` 上恢复 `20261102-dspark-window/PLAN.md` 的窗口设计。当前授权实现和 CPU 检查；GPU 选择、资源和执行由主协调现场确认。本目录中基准与 GPU 脚本由实验代理独立准备和审核。

1. 修复目标普通 decode 预留尾块与 mixed 抢占两处缺陷，各自提交，保留 CPU 原始记录。
2. 保留 `dspark_window_tokens=0` 原路径。W>0 对官方 block7 DSpark 引入等语义的 `full_cache` 读截断和独立 `ring` 存储，target 仍保留完整历史；W 是不含 anchor 的历史 token 数，绝对位置用于 RoPE。保留 Markov 条件 q，不引入 sink、训练、动态 confidence 或量化。
3. W>0 明确要求块大小整倍数、TP1、不开 mixed overlap、不开 AR 滑窗/sink，并显式关闭 target prefix lookup。对照臂均使用 `enable_prefix_cache=False`；默认仍开启旧 prefix 行为。
4. ring 在 target verify/普通 decode 中先写固定地址 scratch，再按已确认输出提交；prefix 命中、owner 错误、有效区间缺口或池不足时拒绝继续。prefill 从每个选层只收集最终尾窗与当前 chunk 的交集。完成、抢占、失败后清理 owner；本引擎未提供独立取消接口。
5. CPU 对照覆盖原路径、W 覆盖历史、full/ring 同窗、回绕、接收 0/部分/7、尾部/EOS、fallback、多个 owner、分块、抢占重入、dummy 和错误状态。CPU 的图替身只验证静态元数据，不替代真实图。
6. 主协调审查固定提交后，另行确认空闲设备执行小型 GPU smoke（建议累计不超过 30 GPU 分钟），包括真实 target feature 和同窗 full/ring 的固定历史数值对照、拒绝写保护、图精确/补桶/回退、普通 decode 和多次回绕；通过后再决定小 pilot。测试失败、OOM、graph fallback 和已花时间均保留，CPU 成功不视作 GPU 验收。
7. 同卡同预算和相同输入/采样/图、prefix 策略下比较 W0/full_cache/ring；先少量样本，记录真实图命中、尾批、host 和 device 成本、真实池和 scratch、feature 张量、预热、完整请求和 decode 吞吐。没有对照证据前不写简历速度或显存收益。后续量化仅作单变量阶段，需另行指示。
