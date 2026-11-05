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
