# DSpark 近期窗口：阶段①设计，等待审核

状态：**仅设计，不开始实现或 GPU 测试**。工作区 `feat/dspark-window`，基础提交 `cefb5a2ab2f57ad2f709fd38493ac544d6b78989`。用户已将 PD 分离或 ZOS 调度作为当前主线；本文件只保留 DSpark 窗口的后续可审阅方案，不申请实验矩阵，也不预设窗口能提高吞吐。

## 目标与边界

- A：以现有全量 DSpark KV 存储为对照，真实读取最近的历史窗口，检查截断对 q、接受与成本的影响。此模式**不节省 KV 存储**。
- B：在相同窗口语义下，使用每序列独立的块级环形 DSpark KV 池，以实际分配量和可容纳请求数检查容量价值；target KV 仍是完整上下文。
- 默认关窗走现有代码；不训练，不引入新的 head、置信度算法、动态 k、sink、PD 或调度策略。官方 block-7、7 候选、Markov 条件 q 和 target 验证语义不变。
- 此处 `W` 是 draft 读取的**最近 W 个历史位置上限**；anchor 不计入 W，仍位于完整的七位置 noise block。设 `C = len(seq) - 1` 为 anchor 的绝对位置，则可见历史恰为 `[max(0, C-W), C)`，noise block 使用绝对位置 `C, ..., C+6`。`W=0` 表示原全上下文实现，`W>=C` 可见历史与全量相同。
- 历史 K/V 来自 target 的完整历史计算所产生的各选层 feature，经 `fc`/`hidden_norm` 和各 draft 层 K/V 投影；不会套用自回归 draft 的局部窗口重算假设。

## 现有路径与需要解决的问题

1. `nanovllm/spec_decode/dspark_model.py` 中的 `bind_cache()` 分配与 target 块数相同的物理槽，`write_context()` 用 target `slot_mapping` 写投影结果；`forward_hidden()` 再按 target 块表读取。
2. `ModelRunner._target_forward_hidden()` 在 target prefill、普通 decode、验证及其图中采集 feature 并投影写入。验证输入覆盖 anchor 和全部候选；被拒候选目前依靠 target 绝对位置与确认长度不可见，而没有事务性写入。
3. 若把第 2 点的写入直接映射到大小为 W 的环，尚未确认的后缀会覆盖旧窗口仍需要的位置；仅将有效水位夹回去也无法找回旧值。下面采用独立暂存，再提交确认前缀。
4. `Qwen3Model.forward()` 现为每个选层构造完整 token 的 feature，再 `torch.cat`。仅在投影前裁剪可节省后续投影，却不会消除完整 feature 列表和拼接的峰值；需在选层取 feature 时就按目标行提取，实际峰值须实测。
5. 现有 AR 的 `spec_draft_window=W` 使用块对齐起点，可见长度随位置变化；不能把相同 W 参数当成 AR 和 DSpark 逐 token 同窗。之后对比必须记录实际历史起点、终点和长度；若要求完全相同的历史集合，应另行设计对齐并审核。

## 建议的配置及读取接口

新增 DSpark 专属的 `dspark_window_tokens`（默认 0）与 `dspark_window_storage`（建议 `full_cache` 或 `ring`），不复用 AR 的 `spec_draft_window`：两者窗口语义不同。第一版 W>0 要求 W 为 target 块长 S 的整数倍；拒绝同时开启 AR 的窗口、sink 或其他不兼容选项。W=0 直接走旧实现，不进入任何新窗口处理路径。

两个 W>0 模式共用同一个窗口几何和 `forward_hidden` 的输入契约：

- 输入分开携带 `anchor_absolute_position=C` 与 `visible_len=min(C,W)`，不能用较短可见长度替换绝对 RoPE 位置。
- 为每个 batch 行生成一个固定桶宽的 `context_slots` 和有效 mask。有效项按历史绝对位置递增排列；其余项指向已初始化的 sentinel 并遮蔽。桶宽按可见长度上取 S 的倍数，而非按原始 C 扩张。graph 以 `(batch bucket, context bucket)` 为形状，重放前更新绝对位置、slots、长度及 mask，dummy 行长度为 0，不影响真实行 RNG。
- A 模式 `context_slots` 从原 target 块表换算，例如 `target_blocks[p//S]*S+p%S`；写入和存储仍沿用全量原始 DSpark cache。这样可把仅截断读取的影响与改变物理存储隔离。
- B 模式按下一节从独立环池产生同样的 `context_slots`；在相同 features 和 mask 下，A/B 的 K/V 读取顺序和逻辑输入相同。
- 按上式显式分离绝对历史位置、其在读取桶中的相对行、以及物理槽；不能把窗口起点当作 RoPE 的 0，也不能让 sentinel、旧 owner 或拒绝后缀变成有效 key。

## 独立环池和提交协议

设 `S = kvcache_block_size`、`M=W/S`，每个活跃序列分配独立的 M 个物理块 `ring[0:M]`，无开头常驻块。对于绝对位置 `p`：

```text
physical_block(p) = ring[(p // S) % M]
physical_slot(p)  = physical_block(p) * S + (p % S)
```

读当前 `[max(0,C-W),C)` 的逐位置槽位，保证读写都使用该公式。即使 W 是整块数，窗口若跨越 M+1 个逻辑块，最老逻辑块的尾部与最新逻辑块的头部也会共享同一物理块的不同 offset；因此不能以“物理块只属于一个绝对块”的单一标签认定整块有效。以序列 owner/分配代次和连续确认有效区间追踪有效性，读取前检查所需区间被覆盖；测试可额外逐槽保存绝对位置标签验证。块被释放或重分配后，新 owner 的有效区间清空，不能凭旧字节凑巧相同而复用。

**验证事务：**

1. 原 target 继续执行完整验证；DSpark 投影结果先写独立、可由 CUDA graph 使用的固定地址 scratch，不写活跃环。每个真实验证输入行有独立 scratch 槽；补桶行可以有自己独立的无效 scratch 行，但绝不能写入活跃环、推进水位或参与提交。
2. 旧的 anchor 绝对位置为 C；若前缀接受 `a` 个候选，则 target 此轮有效 feature 对应 `[C, C+a+1)`：anchor 加 `a` 个接受的候选。按绝对位置递增提交这 `a+1` 行，再将下一轮的有效历史上界推进到 `C+a+1`；拒绝候选和之后的行不提交。新增的 bonus 本轮还没有自己的 target feature，不能提交。
3. 即使 `a=0`，也要提交 anchor 这 1 行；全接受 `a=7` 则提交 8 行。提交过程中只覆盖新窗口已淘汰的最老位置。普通非投机 decode 也需在 target forward 成功后提交当前 anchor 一行，不能只处理 verify。
4. graph 和 eager 使用相同的真实行到 scratch 行的对应关系；现有 verify padding 是真实 query 排前、dummy 排后，但需在测试中强制核对，ragged 回退按 `cu_seqlens_q` 切分。只有对应 forward 成功、采样结果已确定、该行属于当前 sequence owner 时才可提交。
5. 在 EOS、达到输出上限或取消后，处理最终交付边界：继续运行的序列只提交合法确认行；已终止序列可直接释放而不提交。`Scheduler.postprocess_spec()` 当前在检查 EOS 前追加并登记，具体接入点需保证不会读取、登记或交付越过最终边界的环状态。错误或异常时不推进有效区间。

2026-11-02 的纯整数 CPU 模拟对 S=4、W=4/8/12/16、顺序/逆序块表、80 个位置、不同确认长度共检查了 4480 种提交情形：逐项读回的绝对位置与应有窗口一致。该模拟也演示 C=37、W=16、7 候选全部预写但接受 0 个时，会覆盖仍需保留的 7 个旧位置。它只验证地址算术，**不是**引擎集成、数值或 GPU 正确性证据。

## Prefill、前缀和生命周期

- 仅在所有 target forward 的输入行都可关联 sequence 和绝对位置后才能建立映射。prefill 已有完整输入 token，且目标模型仍处理完整历史；对每条序列只抽取其最终目标尾窗 `[max(0, input_len-W), input_len)` 与本 prefill chunk 的交集。分块 prefill 的较早 chunk 如不与尾窗相交，不创建或投影 DSpark features；后续选层提取的每行仍来自完整 target 前缀。已投影区间必须按实际完成的 chunk 连续推进，整个目标尾窗有效后才能提议。
- 对 prefill 调整 Qwen3 feature 收集为“每个选层先按本批目标行索引，再相加 residual 和拼接”，不改变 target 本身的完整前向；小于 W、跨块、非整齐 batch 和零选中行均需覆盖。记录选中 feature 行数、原始 target token 行数、feature 及投影临时量峰值、prefill 全部墙钟；若因 graph/实现限制仍构造全量 feature，必须原样计费，不宣称省了该部分开销。
- 独立环池不继承 target prefix block 的 DSpark KV；target prefix 命中时，跳过的 feature 无法由 target KV 唯一还原。首版建议在所有比较 arm 上显式关闭 target prefix **查找/复用**，包括无投机、AR full/window、DSpark full/window，并记录命中/跳过数恒为 0；若窗口模式遇到意外命中或有效区间缺口则 fail closed，不能读取陈旧槽位。默认 W=0 保留原引擎 prefix 行为；评测额外显式设置相同 prefix 策略。未来若要支持 prefix 命中，须另行设计带成本的安全重建或可证明同源的共享，不默认为免费。
- allocate 时完整分配每序列环，失败则不能沿用当前 AR 分支“退回 full”提示：B 模式并没有完整 DSpark 池，必须拒绝准入、等待可用环，或明确 fail closed。正常结束、EOS、抢占/驱逐、取消和异常路径均按 owner 释放；重建/重入时有效区间清空并重新 prefill。当前项目是否暴露取消 API 需先核对；未实现的接口不能声称已经验证。
- 本集成目前只支持 TP=1 且禁止 mixed overlap；scratch 可由本轮独占、串行复用。不能把这一约束无声扩展成并发 graph 多流安全。

## 分配与计量

在 BF16、DSpark 5 层、8 个 KV 头、每头 128 维条件下，DSpark 每 token 的物理 KV 为 `2*5*8*128*2 = 20,480 B = 20 KiB`；36 层 target 为 `144 KiB/token`。S=256 时分别为 5 MiB 和 36 MiB/块。不要套用旧 AR 28 层、112 KiB/token 的节省数字。

```text
full_draft_KV = target_pool_blocks * S * 20 KiB + draft_sentinel
ring_draft_KV = max_num_seqs * M * S * 20 KiB + draft_sentinel
scratch_KV    = max_live_or_graph_rows * 20 KiB (+ padding/metadata)
```

上述只是主要 buffer 的理论值；需实际记录 tensor 形状、dtype、每种池大小、已分配和峰值、scratch、feature buffer、模型权重与 CUDA graph 池。环池若按 `max_num_seqs` 全量预留，只有在该池小于旧 full draft 池时才有实际 draft-KV 节省。统一显存预算下，节约的 draft 空间可能被现有分配器转成更多 target 块，故**总保留显存未必下降**；容量对比应分别报告同 target 池的实际 draft 分配，以及固定总预算下可运行的最大 active B/长度和其 OOM/抢占情况。

一个仅用于核对数量级的无共享例子：每请求 4096 个 target 历史槽、W=512，full DSpark KV 为 80 MiB，环为 10 MiB；target KV 为 576 MiB。减去的 70 MiB 不包括 scratch、graph、feature 及 allocator 差异，也不是实测容量收益。

## 计划修改文件清单（当前均未修改）

| 文件 | 预计职责 |
|---|---|
| `nanovllm/config.py` | 声明 DSpark W / 存储模式和统一 prefix 控制；校验不兼容配置。 |
| `nanovllm/spec_decode/dspark_model.py` | 原 W=0 分支保留；新读取输入、scratch 投影及有效槽 gather。 |
| `nanovllm/spec_decode/dspark_proposer.py` | 将绝对位置、窗口可见长度、物理 slots 和 mask 装入 eager / graph 静态输入，记录 shape、命中与回退。 |
| `nanovllm/spec_decode/dspark_window.py`（新增） | 可独立 CPU 测试的窗口几何、槽映射、提交范围和有效性判定。 |
| `nanovllm/models/qwen3.py` | 可选 feature 行选择，仅影响 feature 收集，不截断 target 前向。 |
| `nanovllm/engine/model_runner.py` | 显存分池、prefill feature 行、target graph scratch、结果确认后提交、普通 decode；W=0 原路径不改变。 |
| `nanovllm/engine/sequence.py`、`nanovllm/engine/block_manager.py`、`nanovllm/engine/scheduler.py` | 环池 owner/有效区间、准入与回收、统一关闭 prefix，并连接 EOS/抢占生命周期；尽量复用现有私有池，但不能复用 AR 的自回归重算语义。 |
| `tests/test_dspark_window.py`（新增），必要时扩展 `tests/test_dspark_contract.py` / `tests/test_dspark_pad.py` | CPU 数学、数值及 scheduler 生命周期契约；避免只自证同一实现。 |
| `experiments/20261102-dspark-window/` | 日后另建 smoke/pilot 驱动和原始记录；必须获批准后才运行。 |

上述是设计中的预计范围，不构成对上述实现、资源或实验的批准。

## 后续验收门禁（本次均未执行）

**阶段②，若以后获实现批准：CPU 契约。**

- W=0 走原路径，行为及旧测试不退化；`W>=C` 对全量模式的隐藏状态、固定真实前驱条件下的 q 对照，CPU float32 建议 `rtol=1e-5, atol=1e-6`，另报告最大偏差。严格核验 RoPE 绝对位置、noise block 和 Markov 前驱。
- 以独立朴素 oracle 实现列表切片和真实历史构造；A/B 对相同 feature 的物理位置、读取内容、mask 和 logits 对照。覆盖 C 在块前/块上/块后，W-1/W/W+1、多个完整环绕、序列间不同位置、错误 owner、重复槽和 stale 内容。
- 明确构造接受 0、部分、7、非投机一步、末尾 ragged 候选：确认提交的 feature 行和下次窗口与“只处理真实确认前缀”的独立重建一致，拒绝后缀不得推进水位。覆盖 prefill 多 chunk、无选中行、前缀被禁用与强制意外命中、两序列复用、抢占和重新准入、EOS/上限、存在的取消接口、padding guard/sentinel、资源不足。
- 使用小词表枚举或采样对照实际条件 q 的拒绝采样与 target 联合分布；不得用条件 logits 对比代替分布测试，也不把有限样本当严格证明。
- 风险前置：现有 target 普通 decode 跨块回退写槽错误已通过下节 CPU 元数据契约复现；该问题须单独报告和处理，不能以 target 未修改掩盖它。任何最小修复均等待主协调明确批准。

**阶段③，另获单卡批准后：真实 GPU smoke。** 前提是主协调按机器现场进程/UUID 分配空闲卡，本代理当前没有启动授权；之前提出的本轮上限为单卡累计 30 GPU 分钟，包含加载、图预热、失败、重试和清理，且并非预授权。检查 eager/精确图/补桶图、graph 实际命中、dummy/guard、跨块和多轮接受/拒绝，逐行记录 KL、最大偏差、相同前驱 q 和样本分布；参考既有 GPU 检查的 KL 量级设置固定阈值，避免因随机路径差异强求全部输出逐 token 相同。CPU 通过不能替代此步骤。

**阶段④，smoke 审核通过并另获批准后：小 pilot。** 先同卡同资源比较 DSpark full / A / B，以 W=full、2048、1024、512、输入约 4K、B=2/8 和少量固定样本筛选；绝不自动铺大矩阵。记录实际 prompt/output token、active B 分布、prefix 策略、采样和 EOS、图捕获/回退、预热时间/预算、target+draft+verify+commit/sync 成本、完整请求吞吐、steady decode 和所有内存项；保留负结果。需作项目正式比较时，再引入无投机、AR full、AR window、DSpark full、DSpark window 五 arm，同一 padding/缓存策略和相应实际窗口计量，并另议 B16 容量与 1K 参考。上一轮 WikiText 补齐 panel 主要是固定长度吞吐，不代表真实任务质量。

## 本阶段补充：target 跨块回退写槽 CPU 复现

在主协调明确批准下，执行 [cpu_decode_slot_repro.py](cpu_decode_slot_repro.py)，记录见 [cpu-decode-slot-repro.json](cpu-decode-slot-repro.json)。脚本使用真实 `Scheduler`、`BlockManager`、`Sequence`、`ModelRunner.run()` 和 `ModelRunner.prepare_decode()`；tensor、模型、DSpark 候选生成与采样仅用 CPU 元数据桩替代，没有运行模型或 GPU。

真实调度序列从 249 token prompt prefill 采样出第 1 个 token，长度为 250；`spec_k=7` 的 verify 调度按需将 target 块表扩为 `[0,1]`；按真实后处理提交已接受候选和 target correction；此时剩 1 个输出名额，真实 `ModelRunner.run()` 将 7 个候选截为 0 并回退普通 decode。`postprocess_spec()` 没有释放多余尾块。

| 情况 | decode 绝对位置 | 应写 slot | 实际写 slot | 不变量 |
|---|---:|---:|---:|---|
| 无投机预留控制 | 249 | 249 | 249 | 通过 |
| verify 接受 0 个候选 | 250 | 250 | 506 | 失败 |
| verify 接受 2 个候选 | 252 | 252 | 508 | 失败 |

测试末尾的 `actual_slot == expected_slot` 不变量断言因此以退出码 1 失败，记录先于断言保存。复现命令为 `D:\Anaconda\python.exe -B experiments\20261102-dspark-window\cpu_decode_slot_repro.py`。这是**可达状态下的目标写槽元数据错误**；此阶段不声称 GPU 数值、输出或吞吐影响，也不修改引擎。脚本在问题修复后应该通过，届时须重新审核测试与源代码。

## 当前建议

完成本设计后暂停。当前主线是 PD 分离或 ZOS 服务闭环；DSpark 独立环池在长上下文、多活跃请求下可能提供容量价值，但净收益要扣除 scratch、feature、图和调度成本，现阶段没有速度或容量实测结论。是否重启、实施顺序及任何 GPU 资源，均等待主协调后续明确指令。
