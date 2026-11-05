# DSpark选层量化：条件质量与真实速度

本轮完成了从量化质量筛选到固定权重CUDA Graph速度验证的闭环。结论是：**可以按层找出局部加速，但尚未找到适合全局启用的方案。** Q/O+down在B1代码面板提升7.14%，在B8通用文本下降12.65%；默认保持BF16（none），三个新预设保留为显式实验选项，不合入main。

## DFlash/DSpark能不能测PPL

不能直接用不带target特征的独立语言模型PPL脚本。DFlash官方设计将target历史特征注入draft各层KV；本项目实际接入的是DSpark，另外有Markov条件修正。[DFlash设计说明](https://z-lab.ai/projects/dflash/)、[DeepSpec固定版本特征与mask实现](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py)。

不过，给定合法因果prefix特征、anchor及真实前驱，可以计算草稿对后续token的条件交叉熵。本轮定义：

`L = sum(-log q(label_j | prefix_features, anchor, true_previous_tokens)) / scored_tokens`

以`exp(L_quant - L_BF16)`比较同一协议下的变化。这里的`exp(L)`只是所选位置上的条件分数，依赖target、窗口、锚点及block协议；本批参考是target生成样本，不是标准WikiText语料的独立LM PPL。量化损失小通常看分数升幅小；负差值不等于泛化质量改善。

## 质量筛选怎么测

固定旧普通target Graph解码生成的24条轨迹（代码/数学/通用文本各8，T=1、seed20261105），每条取4个offset，得到96个锚点、每方案672个评分位置。15组涵盖Q/K/V/O、gate/up/down、组合及原粗粒度方案；原prompt含Wiki填充，96个锚点不是96篇独立文章，也不是人类gold。

target为同权重Qwen3-4B BF16的HF SDPA参考：先仅前向完整prefix产生特征，选定5层的最后512行交给draft；未来anchor和labels不进入这一步。再用完整target cache续算anchor与7个labels，取前7行预测对应labels。draft每方案重建自己的窗口KV，backbone仅见anchor+6mask，Markov逐位置用真实前驱。HF与native后端不宣称逐位一致。

质量诊断用eager，BF16原模块与INT8 bank同时驻留，仅用于切换比较；不拿其时间或显存作部署收益。96次BF16首末实际重算的NLL/KL差均为0，所有评分有限且归一化通过。

| 后续速度候选 | 量化模块数 | 条件分数相对变化 | KL(q_BF16‖q_quant)均值 |
|---|---:|---:|---:|
| Q/O | 10 | -0.222% | 0.00046535 |
| down | 5 | +0.124% | 0.00042223 |
| Q/O+down | 15 | -0.146% | 0.00052563 |

各方案整体分布扰动较小，但局部面板×位置有约1.4%–1.6%的分数升幅，每格仅32个位置。质量筛选不是质量无损认证。`sum(min(p_target,q))`也只是相同teacher-forced历史下的单步分布重叠，不是实际接受前缀比例或每轮交付tau。

[全部15方案的条件质量表](analysis/quality-full-01/report.md)、[质量CSV](analysis/quality-full-01/summary.csv)。

## 固定部署后是否真的更快

根据此前算子实测，B8时Q/O/down的图模式有收益，gate投影明显变慢，因此选择上述三个组合。生产可选预设新增`q_o`、`down`、`q_o_down`；权重按输出通道对称W8A16，激活和scale为BF16，target/共享embedding与LM head/FC/Markov/norm/KV保持BF16。内核算法未改。

native速度测试只保留一套固定模型权重，在加载和量化完成后重新捕获Graph，不使用质量bank。单张RTX3090，W512、TP1、prefix/mixed关闭，1K输入/128输出、T=1、ignore EOS、0.85初始化预算，B1/B8、三个面板各3次同seed计时。共72次测量、324请求、41472个真实输出，24组完整，18组共54对同输入及配置比较。

下表每次重复先计算量化/同版本BF16 DSpark吞吐比，再取中位数，正值更快。完整请求计时包含prefill、调度、草稿、验证、采样和尾批；加载/PTQ/启动建图排除，单列记录。

| B | 量化方案 | 代码 | 数学 | 通用文本 |
|---:|---|---:|---:|---:|
| 1 | Q/O | +0.09% | +0.11% | +0.31% |
| 1 | down | +6.00% | -0.07% | -2.53% |
| 1 | Q/O+down | +7.14% | +0.56% | +1.11% |
| 8 | Q/O | +0.91% | +0.14% | +1.50% |
| 8 | down | +0.47% | -0.46% | -1.23% |
| 8 | Q/O+down | +2.05% | +1.02% | -12.65% |

B1代码的Q/O+down：160.817→172.207 tok/s，逐次配对比范围1.069602–1.072153；B8通用文本：308.596→269.548 tok/s，范围0.871864–0.874491。都是三次同seed计时范围，不是置信区间或三个独立采样种子。B1每面板只有一条输入，不能推广为代码任务整体加速。

Q/O六个条件的小幅正收益尚不足以作为稳定优势；Q/O+down的强正例与强负例必须同时保留，不全局启用，不通过单一最好条件给简历写“整体加速7%”。[完整native计量表](analysis/native-final-01/formal-report.md)、[逐run CSV](analysis/native-final-01/formal-runs.csv)、[配对CSV](analysis/native-final-01/formal-pairs.csv)。

## 为什么质量接近、速度仍可能下降

1. **改变的只是部分draft投影。** target验证、共享LM head、Markov修正及其他计算仍在。本实现W8A16减少权重读取，矩阵计算仍使用BF16，不会自动得到INT8矩阵乘的算力收益。此前代表算子实测已表明部分shape变慢；不能把某一个算子的加速比迁移到整个请求。
2. **真实交付量也会改变。** B8通用文本中，Q/O+down的实际tau从3.564912降到3.175；序列解码轮次合计285→320，全局decode步骤51→66。即使部分投影更快，更多验证/尾批步骤也会拖慢请求。固定参考指标与真实采样历史不同，不能由这组路径反推固定幅度的任务质量损失。
3. **PPL类指标不衡量执行成本。** 固定参考下gate的条件分数略低，仍不值得在已有负收益shape上盲目启用。反过来，分布漂移小的Q/O+down也未保证通用文本速度。因此应联合看算子实测、分布/接受表现和真实墙钟。

本轮没有额外采集NSYS/NCU或完整GPU阶段breakdown，不给target/draft耗时占比，不将split-K阈值认定为唯一慢因。加载期CPU PTQ虽然增加冷启动时间，但已从生成吞吐中排除，不能拿它解释表中的运行期退化。

## 存储、容量与其他边界

三组被选权重含scale的存储节省分别约99.94、118.73、218.66 MiB，不是整个进程显存下降比例。各预设同0.85预算，但实际target KV块数B1为308/310/289/311，B8为286/287/266/288（顺序none/QO/down/QO+down），不称同容量；实际测试均无抢占，不能直接用池差异解释速度。

native实际有339次ragged verify回退，padding及运行期新capture为0，不称所有验证都走Graph。输出ID差异已记录；未测任务准确率或标准独立PPL。原0.6B量化结果不能移作DSpark数据；本次输出128的结果也不能直接乘到上一轮输出256的DSpark加速倍数。

下一步若继续追求大batch收益，优先依据profile核查验证前向、ragged回退及融合算子的具体shape，而不是继续扩大条件PPL筛选范围。这是基于本轮数据的方向判断，未在本批新增执行。

## 版本、审核与资源

质量执行源码a4e5a9c，native执行源码d878e86，报告/审计工具另有已审提交。Terra Max实现，GPT-6.1 Sol Medium独立审代码和全部真实结果；主协调亲审、实际操作GPU，并从raw独立复算质量/吞吐/字节/形状/配对及资源账。

共10个GPU阶段均ok，累计626.458494秒（约10.44单卡分钟，含加载、量化、预热和清理），未超过同一1200秒预算。274个已记录自有PID/starttick身份和三个controller全部退出；GPU0现场15MiB、0%、无compute。没有后续GPU任务，未动其他作业。

- [协调独立复算](analysis/COORDINATOR-AUDIT.json)、[资源释放记录](analysis/RESOURCE-FINAL.json)
- 原始记录与token清单保留在`evidence/final-01`及`data/`，不随Git上传完整token IDs。
- 完整归档：本机`D:/tools/codex-runs/dspark-quant-quality-20261105/final-evidence-01.tar.gz`，服务器本批根下同名，约1.22MB，不含模型权重或凭据。
- 分支`feat/dspark-quant-quality-20261105`，默认none，不合入main。
