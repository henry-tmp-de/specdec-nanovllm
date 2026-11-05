# DSpark 滑窗与 CUDA Graph：本轮协调记录

日期：2026-11-05。用户当前授权：继续完成现有推理引擎的 DSpark 开发，采用滑动窗口注意力及 CUDA Graph，重新测算可用于简历的相关数据。

## 范围和分工

- 起点：`feat/dspark-window` / `88f6861`；在同一 worktree 上建立本轮隔离开发分支，保留 main、旧分支和历史实验。
- Terra Max：实现、针对性回归、实验工具。主协调：方向、实际 diff 与 CPU 证据审核、全部 GPU 操作、原始统计复算。独立 review 使用 GPT-6.1 Sol Medium（用户 2026-11-05 最新全局指令；Astra 停止后续审核）。
- 当前仅 DSpark block7、Qwen3-4B BF16、TP1、单卡。target 保持完整上下文，不训练，不加入动态 confidence、PD、多卡或新量化变量。
- 历史 DSpark 已测试，滑窗仅有设计。旧结果不视为本轮结果，旧 deadline、启动锁和批准记录不复用。

## 本轮工作

1. 独立修复已复现的 target decode 跨页写槽和同一步 prefill 被抢占问题，保留旧失败和新 CPU 原始输出。
2. DSpark 专属窗口开关，窗口读取控制与物理独立环池；先暂存验证特征，仅提交有效前缀；完整 target、绝对位置、真实 Markov q 不变。
3. 接通 eager、精确图、补桶和安全回退，明确 prefix 与生命周期边界；功能不支持时明确报错。
4. 代码/CPU 经协调者和独立审核后，现场选择一张完全空闲 3090，先 smoke，再小 pilot，之后按实际时间和正确性推进固定批量比较。
5. 比较普通 graph-on、原 AR full/window、DSpark full/window；同输入、采样、输出上限、缓存政策和显存预算。不能把候选 k、窗口语义、填充输入或图配置差异隐去。

## 资源与计量

- 本轮累计单卡上限 14,400 GPU 秒，包含加载、图预热、失败、重试和清理。首次 smoke 子上限 1,800 秒。按当前任务授权执行，不重新询问已有范围。
- 现场 2026-11-05T06:58:13Z 查询 GPU0/1/2 均为 15 MiB、0%、无 compute；此为历史快照，不是预约或持续可用证明。启动前重查 UUID/compute/显存/利用率，不动其他任务。
- 守护任务仅识别、结束本批自有进程；共享资源冲突、数值错误、OOM 或超预算时保留记录并停止受影响阶段，审核后决定是否有必要的最小修复和复测。
- 正式性能无 profiler；请求完整墙钟、真实最终交付 token、prefill/decode 分开。接受候选、含 bonus 的每轮推进、最终截断交付分别记账。
- 实际 graph 命中/回退、运行中捕图、active B、尾批、缓存策略、所有 KV/scratch/feature/graph 内存项必须可核对。

## 简历边界

旧 0.6B 的 1.39x、82.3→132.5、4.73→0.55 GiB、量化/PPL不能迁移给 DSpark。独立 kernel 与原 target-only 混批结果保留原实验条件；不声称本轮同时集成了尚不支持的 DSpark 双流混批。量化如未在本轮执行，最终资料明确沿用为旧 AR 独立成果或暂不写。

## 当前状态

引擎及实验工具已通过主协调/Sol Medium 代码与 CPU 审核，正在运行单卡真实权重数值门禁，尚未开始性能 pilot。最新状态以下方追加记录和远端真实 ledger/LAUNCH 为准。模型和大缓存不进入 Git。

### 阶段一已审核

- `65f4b8274e4e60b184cafb5464d157eecf74aa6b`：decode 按实际绝对位置选择 target 物理块。
- `aee5aa58f210187e3fd92d52fefed892b0e6e7cd`：混批抢占排除本步已排的 prefill 请求，普通非混批策略保持。
- 主协调亲读引擎 diff，独立运行 `cpu_decode_slot_repro.py` 和 `cpu_mixed_preemption_repro.py` 均退出 0。前者接受 0/2 后实际槽为 250/252；后者有其他 decode 受害者、无其他受害者及非混批控制均符合预期。
- Astra Ultra 用 `git show` 加载固定基线与新提交，独立重现旧失败并验证新路径，结论：无阻断，CPU 元数据门禁通过。此结论不替代模型/GPU 数值验收，也不是速度证据。
- 主协调首次调用混批复现时误拼脚本名，得到文件不存在；改正后成功，未修改实现或放宽测试。

### 审核模型变更

用户明确要求今后全部 review 改为 GPT-6.1 Sol Medium，并写入全局记忆。已停止正在进行的 Astra 审核；其此前完成的阶段一审核保留为历史记录，当前阶段二与 GPU 工具门禁交由新的 Sol Medium reviewer 接手。实现仍由 Terra Max 完成，GPU 仍仅由主协调操作。

### 2026-11-05 15:31 北京：本轮真实 GPU 状态

- 工作分支 `feat/dspark-window-resume-20261105`。核心 `937e10e`；测试桩隔离/归因修订 `f9af31a`。主协调和 Sol 各自独立新四文件联跑 33 passed；最初联跑确实失败，原因是模块级测试桩污染，已修并保留失败日志，非模型数值失败。
- 正式工具最终 `d2d48d2` 已获 Sol 审核；默认不注入 CUDA Events/额外 q 前向/窗口探针；`--diagnostic-cost` 单列排除正式吞吐。计划工具只生成计划，不能声称已执行。下一阶段仅 14 个校准 job：B2/8，4K，W0 加 W512/1024/2048 的 full_cache/ring，各 code 单次；尚未启动。
- 远端新隔离根 `/home/ziru/nano-vllm/dspark-window-resume-20261105`；SSH `lab3090`；Python `/home/ziru/nano-vllm/venv/bin/python`（torch2.10.0+cu128、transformers5.15.1）。权重沿用 `/home/ziru/nano-vllm/models/Qwen3-4B`、`Qwen3-0.6B` 和 `/home/ziru/nano-vllm/dspark-integration-20261101/models/dspark-4b`，未训练/下载/升级依赖。
- 现场选物理 GPU0 / `GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91`，绑定完整 UUID；不再使用后来被别人占用的 GPU2。端口仅本批环境 `NANOVLLM_PORT=24805`。所有阶段串行、同一 `results/gpu-budget.jsonl`、14,400秒总限/1,800秒smoke限，不重置。
- source 冻结通过 Git bundle/独立 clone，绝不运行中替换：首 e64b9fa、第二 00614a4（两次均在模型前因 UUID 返回格式误拒，原件保留）；真实使用 `code-e0d5f21` / `e0d5f21f86b411e7a262c490701255686dd016bf`。PyTorch `_CUuuid.bytes` 实测为16个0..255整数的list，严格规范化已修并经 Sol 审核。
- `smoke-b3-ring-padded-03` 成功：96=3×32 交付；decode93=68接受候选+25bonus；draft命中10（pad5），target verify命中6（pad4/exact2）+ragged回退4；运行期捕图0；首轮相同前驱q KL/最大绝对差0。主协调和Sol独立核raw；结束GPU15MiB/0%/无compute。诊断有额外前向，绝不能用它的TPS作简历。
- `fixed-replay-eager-01` 成功：三模式各532受控交付，固定接受0/3/7与普通decode，10个相同确认位置（252…775，跨两次完整256环绕），同窗 full_cache/ring q KL、全部可见draftKV RMS和抽样targetKV RMS均0；W0与W覆盖历史的2个位置也均0。强制采样诊断不是性能或无损采样分布证明。
- 当前运行 `fixed-replay-graph-01`，2026-11-05T07:31:16Z 启动，guard PID928510/start_ticks905861983/UID1006。入口/完整参数在远端 `fixed-replay-graph-01-LAUNCH.json`；日志 `logs/fixed-replay-graph-01{,-guard}.log`、raw `results/fixed-replay-graph-01.jsonl`。只查进度，不能重复启动。此前已完成阶段累计101.65秒左右；精确累计以 ledger 各end为准，当前运行仍在计费。
- 之后需先审核graph raw，再串行运行已审 `gpu_padding_contract.py` B3/11/15，各阶段新日志/raw/启动记录，同一ledger；失败停该阶段保存证据，主协调决定必要修复，不放宽数值门限。数值门限事前为KL<0.02、KV相对RMS<0.05，logit abs只观察。
- 4K workload 构造原180秒和40秒CPU超时，faulthandler定位逐token反复调用tokenizer.__len__，缓存词表大小 `88ca2c8` 后4.67秒成功。正式manifest为远端 `data/workload-4k.json`，使用不重复的连续 WikiText 片段加任务后缀，填充占比95.9%至99.6%，明确为合成固定长度吞吐负载，不是自然4K任务或质量评测。1K沿旧manifest。
- 本地已拉 `evidence/results`、`evidence/logs`（目前仅首smoke/预检，eager新数据待拉），root diagnostic审计 `evidence/smoke03-root-audit.json`。大模型/权重不入Git。

### 2026-11-05 16:20 北京：精确图与性能校准进展（覆盖上节运行态）

- 固定历史 eager/graph 均完成；旧补桶 `padding-b3-01` 通过，`padding-b11-01` 仅 target verify KL `0.0201313533 > 0.02` 失败，B15旧补桶未运行。所有失败原件保留，未放宽门限。该差异与 padded 路径相关，底层原因未独立证明。
- `60bb01b` 新增可选 `decode_graph_bs`，空默认旧行为；实验配置 decode/spec 均为 `1..B`，关闭两项补桶。主协调 51 项 CPU 联跑、Sol 独立 config/runner 41 项通过。工具 `490d43f` 经 Sol 审核后执行新 GPU。
- `code-490d43f` 的 B11/B15 ring exact 真实 T1 smoke 输出352/480；同query/scratch初态逐字节恢复后，target verify和普通decode的logits KL/target KV RMS/scratch RMS均0。exact多owner合同B11/B15均通过，0/3/7、ragged、owner scratch/旧KV、guard/sentinel/prefix检查通过；主协调和Sol各自复算。GPU均退出，数值门禁通过；诊断TPS不用作性能。
- `PILOT-LAUNCH-01.json` / PID953142 已结束。source490、0.70、4K、B2/8：前7个B2 job有效，首B8 full_cache W512因target池107块/27392槽只达到first decode B6而失败，余6未运行。原记录 `results/pilot-01...08*.jsonl`，计划 `data/pilot-plan-exact.json`，`PILOT-STATUS.json=stopped_on_failure`。不是B8性能结果，不能降B冒充。
- `a13a147` 经Sol独立17项CPU审核允许显式0.85、CLI默认0.70；plan/job/cell及raw/audit预算贯通。新独立计划 `data/pilot085-plan.json` 同14case，校准顺序B8 W0先、其余B8窗、再B2，未拼旧0.70。
- `PILOT085-LAUNCH-01.json` / PID966698 已结束。source `code-a13a147`，首 `pilot085-01-dspark_full-b8-w0` 在图预热 SDPA OOM（申请342MiB、free263.69MiB、private pools5.04GiB）；无正式run，余13 job未执行。`PILOT085-STATUS.json=stopped_on_failure`。不再自动提高预算、降低输入/B或改精度。
- 此刻没有本轮GPU任务；GPU0最后核15MiB/0%/无compute。账 `results/gpu-budget.jsonl` 已闭合，累计约517.56秒（准确数由完整end加总）。总限14,400秒；smoke限1,800秒，失败均记账。
- 当前代码工作：Terra `dspark_window_resume` 在同一分支最小试 DSpark prewarm 最大shape先捕、head B降序，保留相同图集合、backend、模型和预算，验证共享池占用是否下降；未预设节省或认定根因。Sol `dspark_sol_review` 审固定diff+CPU；通过后仅同0.85 B8 W0完整job验证，成功才同新source重做整14job。若仍失败再按证据决定，不盲扫参数。
- 报告工具由 Terra `dspark_resume_bench` 准备，`b0ab460/b8311d2` 主体与合法填充汇总经Sol审，`35ab66d`标明未完成矩阵为Interim；各来源分开报告，最终正文只用最终source自己的正式计划/raw，历史失败作附录不抹去。不能用旧failure直接污染新commit同名cell，也不能跨预算/源码做正收益配对。正在用真实旧pilot raw核schema，未形成简历结论。
- 本地 `evidence/results`、`evidence/logs`、`evidence/data` 已包含以上所有已结束raw、日志、两版pilot计划、4K/16K manifest。报告/曲线未最终发布。正式150cell/450次测量与16K容量均尚未启动。

### 2026-11-05 17:17 北京：当前正式任务在运行（优先读取）

- `d9e72c2` 最大shape优先预捕、`61a5263` 实际预热顺序记录、`4a4a64b29192e57b3e27eb24d6a8a3069dfbd18f` warmup覆盖完整行数，均经Sol Medium实际diff和CPU审核。相同0.85/B8/full/4K仍在真实prefill申请608MiB OOM，故停止该case盲调；其失败 `results/warmup-rows-b8full-085-01.jsonl` 原样保留，不宣称内存问题全解决。
- 最终运行源码冻结为远端 `code-4a4a64b`，主模型/草稿/precision/graph策略/输入不再改变。旧源码副本和失败不覆盖，local main未合并。
- source4a的0.85 pilot 14case终态：13成功、1已知B8 W0失败，成功62请求/15872实际输出。新计划 `data/pilot4a085-plan.json`，`PILOT4A085-STATUS.json=complete_with_known_failure`；不拼任何旧source或0.70数据。
- 按预写规则：ring W512/1024/2048在B2/B8各自最大实测TPS归一化后取几何均值；512=.98611057379、1024=.97299435631、2048=.96539262433，全部在best 3%内选最小W512。`WINDOW-SELECTION.json`和本地evidence副本可复算，不用缺失B8 W0作分母，不把pilot单次当正式加速。
- 追加最小新source GPU复验已通过：`source4a-fixed-graph-01`（跨context预捕逆序/实际升context、两回绕、W0/full/ring、0/3/7、ordinary），`source4a-b11-exact-01`（352交付，q/verify/ordinary same-preimage KL/KV/scratch RMS均0）。Sol亲读原始记录独立复算，数值与pilot门禁放行。全部旧/新诊断累计到当时1060.302144秒，smoke362.388721秒；此数不是当前最终账。
- **当前正式父任务**：`FORMAL-LAUNCH-01.json`，PID **1006447** / start_ticks **906411087** / UID1006，2026-11-05T09:02:47Z启动。`FORMAL-STATUS.json`、`logs/formal4a085-suite-01.log`，独立runner已脱离SSH。只查进度，禁止重复启动。实际控制脚本从核验身份的本进程argv原样保存为远端 `operator-formal-01.py`，旁证 `FORMAL-OPERATOR-SNAPSHOT.json`。
- 正式计划 `data/formal4a085-plan.json`：5arm（nospec/AR full/AR window/DS full/DS ring），B1/2/4/8/16 × ctx1024/4096 × code/math/general × 3 timing repeats；50模型进程组、150条件cell/450run是目标，不是成功声明。所有arm同source4a/.85/W512（适用窗口arm）/BF16/T1/ignoreEOS/prefix关闭/exact decode+spec 1..B，max_model_len=ctx+256+8，prefill token预算=B*ctx。实际输入高度Wiki补齐，非任务质量或真实在线服务。
- 已知B8 DS full/4K与pilot同配置初始化失败，正式标 `blocked_known_pilot` 不复跑，不填TPS，也不称9次各自实测OOM。其正式3panel×3repeat未测。其余事先定义仅OutOfMemoryError或明确fixed batch capacity error、guard全退出/释放无foreign时才继续下一原计划job；数值/未知/外来占用/清理失败即停整阶段，不缩shape/precision或改budget。
- 最新只读17组成功/153run（2026-11-05T09:15:35Z），正在group18 nospec/B2/4K。该进度会变化，以远端status及raw为准。GPU **仅物理0 UUID GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91**；端口24805，仅本进程环境，无sudo或全局设置修改。总账仍同一个results/gpu-budget.jsonl，总14400秒；当前阶段实际还在计费。
- `evidence/partial-formal-01`是早期11组/99run字段核验快照，snapshot明确主阶段仍在跑、其closed-ledger人为排除active阶段。Terra CPU逐条99valid，不作为最终总账或对外正式报告；最终必须拉完整闭合ledger/全raw。
- 报告脚本 `report_results.py` 及15项fixture经Sol审；之后小修合法填充汇总/Interim状态经定向审。只读适配已成功区分nospec/AR/DS full W0的N/A scratch与ring实际scratch。按来源/预算分别报告，不能用旧failure污染新commit同名cell，也不跨不同模型连接曲线。正式benchmark结束后先拉raw/manifest/全账、root复算、Sol独立证据审核，才写简历数字。
- **容量阶段还没开始**：已有CPU生成16K manifest，planned DS full/ring B4/ctx16384/output256、同0.85、prefill预算4096、W512，无量化/训练。需正式阶段终态、root/独立审核及现场空闲检查后单独执行，不与正式并行，不自动加PD/ZOS/MoE/量化/多卡。
- 全局review偏好已按用户新要求记入 `C:/Users/85935/.codex/memories/extensions/ad_hoc/notes/20261105-review-sol-medium.md`：今后全部review使用GPT-6.1 Sol Medium，禁止Astra。实现仍Terra Max，GPU/root负责。
