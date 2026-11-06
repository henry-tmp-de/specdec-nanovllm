# 可选分层prefill流水：协调记录

## 用户目标与分工

用户提出：chunk A（0..127）进入第二层时，chunk B（128..255）开始第一层，像流水线一样交错推进；已明确优先整体吞吐，并批准独立分支、加入原引擎且可开关。

新工作区D:/学习/specdec-layer-prefill；分支feat/layer-pipelined-prefill-20261106；起点d051665281a3fa591a6f1f829be013bbe9de1f25。Terra Max负责实现和工具，GPT-6.1 Sol Medium独立审核，主协调负责方向、亲审、实际GPU及原始证据。旧main和所有已结束实验不动。

本批不是单独toy demo：正常LLM配置必须消费新开关，默认off保持旧路径，不创建多余streams/events或改变旧prefill/decode行为。第一版只支持Qwen3 target-only BF16、单GPU/TP1、关闭prefix/mixed/spec/draft/targetquant；不支持组合提前明确拒绝。DSpark、旧量化及PD分支不作为本批联合成果。

## 本批实现契约

仍按token分块，改变的是块之间的逐层执行顺序；不是让32个逻辑层拥有32份GPU资源。实际Qwen3-4B为36层。

- 每ChunkTask保存同一scheduled prefill span内的绝对token起止、原flat行偏移、h/residual、paged slot/context和layer进度。
- 同chunk的前一层，以及同序列前chunk的同一层（保守完整layer边界）都是依赖，使用已record的CUDA events等待；不同请求不错误互锁。单host线程按拓扑发射，两条streams，起步maxinflight2，不每层CPU同步。
- 每task attention均须读正确paged历史KV，cu_q=[0,chunklen]、cu_k=[0,absolute_end]，不能沿用首块block_tables=None而让后块只看自身。
- 保持原fused residual传递；只在末层完成后归并原flat顺序、恢复外Context，再执行原LM head和采样。一个外step内完成全部内部task后，原scheduler才推进全局水位和注册缓存。
- 元数据每task一次准备，不每层GPU.tolist/H2D；跨stream输入、h/resid、输出和元数据的依赖与生命周期显式维护。warmup尚无KV时允许清楚标明的原路径fallback，但真实测试必须证明新task/layer路径实际运行。
- off/full、同样chunks串行、两stream流水三种主要对照。旧外部分块路径只有真实调度记录证明启用时才列入，不能只看参数。

## 研究对照与结论边界

- NanoFlow：https://arxiv.org/abs/2408.12757 ，https://github.com/efeslab/Nanoflow 。侧重nano-batches和算子级异构资源重叠，与本吞吐目标最接近；本首版不移植其硬件资源划分或优化器。
- Layered Prefill：https://arxiv.org/abs/2510.08055 。以MoE层组调度减少专家权重重载，不等同于本用户的token块跨层wavefront，不搬用其数字。
- Bullet：https://arxiv.org/abs/2504.19516 。涉及层级prefill与decode及资源分配；本批不默认引入SM masking。
- FlowPrefill：https://arxiv.org/abs/2602.16603 。算子边界抢占与事件调度，主要解决SLO和HoL，本任务优先原始吞吐。

在同一3090上，层任务共享Tensor Core、SM和带宽；小GEMM、重复读权重、额外event/launch和多流争用可能使流水变慢。CPU NVTX重叠、两个stream存在或理论DAG并行都不等于实际kernel重叠，更不等于吞吐提升。

## 验收与资源范围

先CPU planner/配置/因果/上下文恢复/输出契约，主协调+Sol代码门禁，再真实单卡小smoke：同形状serial与流水的logits/KV/后续decode对拍，另与full检查合理浮点差异。通过后逐步测prefill输入token吞吐及完整请求输出吞吐，不混分母；加载/编译/预热单列且计GPU预算，至少保留原full基线和全部负结果。

原计划最多单卡1800秒，已被用户最新“少测、不多次重复”的指令替代：**当前总预算600秒，smoke最多300秒**，包含加载、编译、失败和清理，所有阶段同一账。只用现场UUID复核、无compute且低显存/利用率空闲的GPU；不操作别人的作业、不改驱动/时钟/电源/全局权限，不下载模型或重装依赖。

当前实际范围仅必要数值smoke（混长B2的257/513，三臂及固定历史16步）和一个性能工作点：B4、4K输入/128输出、chunk128、inflight2，A整段full、B同DAG单流serial、C同DAG双流pipeline各一次primary计时。每臂一次不计时预热用于编译/stream初始化，仍计GPU账，不能当重复成绩。无output1、B1、512扫描或NSYS采集，不跑完整矩阵，不因出现负收益自动重跑。

本批已经完成并释放GPU，无后续任务。功能与小样本数值通过、单次吞吐无益，保留可选实验分支和负结果，不合main或继续扫参数直到变快。最终入口FINAL_REPORT.md/USAGE.md。

## 核心交付与CPU门禁

核心6a5c61b已完成Config/逐task元数据/同层前缀事件/h+residual/Context恢复/原LM head及采样后完成计数的集成。主协调亲读完整diff与executor，并独立CPU36项通过（3.22秒）；Terra交新36、legacy96及4个旧脚本exit0日志，Sol Medium独立全diff/日志审阅并新36项通过，codegate无阻断。

CPU/fake CUDA不代表设备通过。明确serial使用相同对角DAG但一条worker stream，不是旧chunk-major。KV尚未分配的初始化warmup回原forward并计warmup_no_kv，已有KV缺表必须报错；真实测试须观察到tasks/layers/completed与nonfallback路径，不能以参数存在代替实际生效。

## 实际运行与最终验收

原19cf启动在guard pre-parser阶段因`--stage`被识别为timeout缩写而失败，没有CUDA/ledger/GPU运行；原receipt/log/lock未改，PRELAUNCH-FAILURE-01完整保留。b98fc8f最小修与完整argv/child隔离CPU回归12项通过，并经Sol独立审核后重新冻结；仅实际GPU尝试命名smoke02，没有重复primary。

smoke02实际混长257/513两请求及相同固定历史16步，三臂完成；102 logits和2592 K/V门槛检查全pass，同块serial/pipeline最大KL/RMS均0，full比分块最大KL.00047045、KV RMS.0465982。actual8task/288层/fallback0/completed8，有两不同workerstream IDs；不把它当NSYS实际kernel重叠证据。

三个primary按用户范围各1warmup/1timed，实际12req/1536out，full/serial/pipeline TPS105.345974/72.308582/70.990668，墙钟4.860176/7.080764/7.212216；prefill轮墙钟2.595089/4.814641/4.945918。三臂KV207块，输入/source/budget/exact decode127匹配，on完成128tasks/4608层无fallback；输出hash本次相同但不认证质量。主协调从raw独立复算，Sol重建计量/summary/CSV/MD一致，gate通过；保留-32.61%和-1.82%负例。

同一账4stage累计107.142821968秒、136自有identity无存活，3controller退出，现场GPU015MiB0%无compute。prelaunch GPU成本0不计primary重复；完整证据压缩包505312B，服务器与本地SHA一致。code源b98、核心6a，审计e6e29e5；报告最终小产物962849c。没有追加GPU、NSYS、调参或主分支合并。
