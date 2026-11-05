# DSpark W8A16 分层量化：最终结果

已完成实现与单卡实测。结论：可以对DSpark草稿进行选择性W8A16量化，但本轮融合算子实现的端到端收益小、依赖batch和输入；默认仍保留BF16，量化作为显式可选路径，不合入main。

## 量化了什么

在官方DSpark block-7的五层decoder中分别试验以下配置。按输出通道对称PTQ，权重INT8、scale和激活BF16、矩阵累加FP32；前向复用融合反量化Triton算子，不把完整BF16权重解压写回显存。

| preset | 选中线性层 | 数量 | 选中权重存储节省（含scale） |
|---|---|---:|---:|
| ffn | 各层gate_proj、up_proj、down_proj | 15 | 356.04 MiB |
| attention | 各层q_proj、k_proj、v_proj、o_proj | 20 | 124.92 MiB |
| all | 上述两组 | 35 | 480.96 MiB |

target Qwen3-4B、共享embedding/LM head、DSpark的fc、Markov head、norm、confidence和KV保持BF16；未训练模型，未测试W4/GPTQ/AWQ。配置入口为`dspark_quant_preset="none"/"ffn"/"attention"/"all"`，默认none。Checkpoint严格加载完成后、warmup和图捕获前替换模块；Attention的k/v在历史feature写入与proposal两条路径均使用量化模块。拒绝采样使用量化草稿实际产生的q。

## 测试条件

- 单张RTX 3090（24GB），官方DSpark block7，target Qwen3-4B BF16。
- draft独立ring窗口512，target完整上下文，TP1，prefix关闭、mixed overlap关闭。
- 输入1024 token，固定输出128 token，temperature=1、ignore EOS；代码、数学、通用文本三个固定长度输入面板；B=1/8/16，每个条件三个同seed计时重复。
- target decode、draft及verify使用精确batch CUDA Graph；运行期新捕获与padding为0，verify仍有ragged长度回退，所有108次测量共927次，不声称全部验证前向均走图。
- 同初始化显存预算系数0.85，记录实际KV池差异；基线是同一新源码下的BF16 DSpark（none），不是普通解码或旧0.6B草稿。
- 完整请求吞吐包含prefill、decode、调度、采样和尾批；加载、加载期PTQ与启动建图单列。不是HTTP在线压测，也不是任务准确率评测。

执行源码为`2ac595ee1565e1218bfabad8c827c32784a130b5`。本轮完整108次测量、900条请求、115,200个实际交付token；12个模型进程组各完成三面板三重复，全部计量有效。共有27组量化/none比较，每组三个输入及配置匹配的重复，共81对。输入源沿用上一轮固定长度面板，含Wiki填充，不推广为自然长任务质量结论。

## 端到端吞吐变化

以下为每次重复先计算量化/none吞吐比，再取三个比值中位数并转成百分比。正值更快，负值更慢；三次计时范围在详细表中，不是置信区间。

| B | 量化方案 | 代码 | 数学 | 通用文本 |
|---:|---|---:|---:|---:|
| 1 | FFN | +1.78% | +1.70% | -0.88% |
| 1 | Attention | +0.80% | +0.86% | +0.97% |
| 1 | 全部 | +2.24% | +2.17% | -0.30% |
| 8 | FFN | -0.72% | -1.85% | -5.86% |
| 8 | Attention | +0.50% | +2.82% | -5.33% |
| 8 | 全部 | -0.12% | -1.03% | -4.92% |
| 16 | FFN | -3.08% | +0.40% | -3.83% |
| 16 | Attention | -5.55% | -2.10% | -7.93% |
| 16 | 全部 | -4.05% | -0.87% | -5.18% |

绝对吞吐、逐次值、TTFT、TPOT、接受比例、实际每轮交付及图计数见[完整计量表](analysis/final-01/formal-report.md)和[逐run CSV](analysis/final-01/formal-runs.csv)。B1代码/数学约2.2%的收益仅支持这些具体条件；B16只有FFN数学微增0.4%，其余条件均下降。本轮不据此推荐全局启用量化。

## 为什么省权重没有带来同幅度加速

已测事实：五种layer0代表形状、B=1/8/16合计15个条件的数学和Graph对拍均通过。最大同量化权重参照relative RMS为0.001679，Graph/eager输出完全一致。B1 Graph的代表算子约快1.33–2.54倍，但B16五类代表算子均慢于BF16；[micro表](analysis/final-01/micro-report.md)独立列出eager与Graph，输入是合成BF16张量。

代码事实：DSpark一次处理七个候选位置，proposal GEMM行数为7B，即7/56/112；当前kernel在M不超过64时采用split-K=4，之后为1，最初配置针对原AR草稿的其他形状。此次没有调kernel或扫配置，不能认定某一开关是唯一慢因。

已测事实：量化改变了部分候选与接受比例。例如B16通用文本，none的接受候选比例/实际每轮交付约0.34430/3.33662，全部量化约0.32298/3.18495。完整请求还包含target验证及prefill等成本，算子速度不能直接迁移为请求速度。本轮没有新采NSYS/NCU或正式的draft/verify GPU分解，不编造各项占比。

## 存储、加载与容量

480.96 MiB指35个被量化线性层的存储减少，不是整个进程显存下降比例。实际全部35个候选层的存储为1,009,254,400字节（none）与504,934,400字节（all，含scale）；未量化的DSpark参数另计，共享target权重不重复计数。旧工具的unique_parameter_bytes不含新INT8 buffers，不能直接拿它作总权重对比。

| B | none target KV块 | FFN | Attention | 全部 |
|---:|---:|---:|---:|---:|
| 1 | 308 | 291 | 309 | 290 |
| 8 | 286 | 268 | 286 | 268 |
| 16 | 260 | 242 | 260 | 241 |

同预算不等于同KV容量。当前是先加载BF16、再在CPU按行分块PTQ并替换设备权重，仍存在初始化/分配器开销。B1加载期PTQ约FFN61.57秒、Attention24.34秒、all85.92秒；该时间已包含在load_initial_graph_s中，不能相加，也未计入生成吞吐。实际reserved/allocated也分别保留；未新增分配器归因实验，不把KV差异归结为已证实的单一原因。

## 质量与简历边界

- kernel对拍通过、实际q接入拒采，不等于完成任务质量或PPL评估；本轮未测DSpark PPL。
- 随机生成输出ID变化已记录，不能仅凭变化断言质量下降或用文本看似正常证明等价。
- 原Qwen3-0.6B量化的6.4%–7.1%及PPL+0.455%不属于DSpark，不迁移到本轮。当前输出128，与上一轮输出256的DSpark加速数字也不直接相乘。
- 可以描述选择性PTQ、Graph集成与分层消融；本轮收益不足以表述为普遍的大batch量化加速。

## 验收与资源收尾

Terra Max实现，GPT-6.1 Sol Medium独立审核代码及最终原始证据，主协调亲审diff、运行GPU并独立复算结果。13个阶段均正常结束，累计1,129.839秒（约18.83单卡分钟，含加载、量化、预热和清理；本轮没有失败阶段），未用满3,600秒预算。410个已记录自有PID/starttick身份及三个controller均无存活；现场GPU0为15MiB、0%、无compute，未动其他作业。没有后续GPU任务。

- [资源释放记录](analysis/final-01/RESOURCE-FINAL.json)
- 原始记录：`evidence/final-01/results`，日志及启动/门禁/负结果随同归档保留，不随Git上传完整token IDs。
- 本机归档：`D:/tools/codex-runs/dspark-quant-20261105/final-evidence-01.tar.gz`；服务器同实验根下同名文件，约1.53MB，不含模型权重或凭据。
- 分支：`feat/dspark-quant-20261105`；保持默认none，不合入main。
