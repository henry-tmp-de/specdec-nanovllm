# 本轮如何测试与解释结果

本轮是 nano-vLLM 内部的单卡固定批量推理基准，用于比较解码方案、窗口与图执行的实际成本。没有运行真正的 vLLM/SGLang 服务横向对比，也不是 HTTP 请求率压测或生产流量回放。

## 共同设置

- 单张 RTX 3090，物理 GPU0，UUID `GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91`；TP=1。启动前检查空闲，运行中检查本批进程身份，不使用其他人的任务。
- Target：Qwen3-4B BF16。AR draft：Qwen3-0.6B BF16。DSpark：官方 `deepseek-ai/dspark_qwen3_4b_block7`，5层、vanilla Markov rank256、7个候选；confidence scheduler未启用。权重、配置和文件元数据写在原始记录中；本轮不训练、不量化、不更新checkpoint。
- temperature=1、固定seed、ignore_eos=True、每请求输出上限256。重复运行重置seed，属于相同轨迹设置下的计时重复，不将三次重复当作任务质量评估或置信区间。
- 正式矩阵：B=1/2/4/8/16，输入1024/4096 token，code/math/general三个面板，各3次。每个arm/B/ctx新建一个模型进程，预热后复用模型测九次，面板顺序随repeat轮转；arm顺序随B/ctx组轮转。
- 每次共入队B条请求，完成后不补新请求。不同接受长度会使尾部active B下降，因此同时报告完整请求与固定满B阶段，不把尾部排除后称整体吞吐。
- 所有主对照同一冻结源码 `4a4a64b29192e57b3e27eb24d6a8a3069dfbd18f`、同物理卡、同输入IDs和采样设置；prefix lookup与注册关闭，不启用mixed overlap。此处没有证明DSpark滑窗与原双流混批或prefix复用同时兼容。

## 五个对照

| arm | 草稿与上下文 | 候选设置 |
|---|---|---|
| nospec | 普通target解码 | 不投机 |
| ar_full | 原0.6B AR draft，完整历史 | k=6 |
| ar_window | 原AR块对齐滑窗，开启已有对齐/重对齐 | k=6、窗口上限512 |
| dspark_full | DSpark完整历史feature KV | block7、7候选 |
| dspark_ring | DSpark独立环形KV，最近512个历史位置 | block7、7候选 |

AR窗口是块对齐的历史范围；DSpark窗口精确读取 `[max(0,C-W),C)`，anchor位置C不计入W，RoPE仍使用绝对位置。两者并非逐token完全相同的可见历史。DSpark的KV来自完整target计算产生的中间特征，不能理解为整个模型只看最近W个原文token。

校准另有 `dspark_window_full`：同样截断读取和prefill feature选行，但物理存储仍为全缓存。它与同W的ring对比，才用于观察物理布局、确认提交与预算差异；W0→窗口还同时改变读取、选层行选择和投影，不能全部归因于环池。

## 图与显存预算

普通target decode、draft和target verify显式覆盖1..B的精确图，关闭补桶；ragged候选的verify保留eager回退并记录。原B11 padded数值检查失败保留，没有通过调阈值将它改为成功。

Backbone按 `(B*context,B,context)` 降序预捕共享池，head按实际B降序。记录请求覆盖集合与实际捕获顺序、图命中/回退、运行期捕获、图池内存和预热时间。预热或CPU检查成功不等于真实图数值正确，本轮另有真实权重的固定历史与多owner诊断。

主实验显式 `gpu_memory_utilization=0.85`，`max_num_batched_tokens=B*ctx`，`max_model_len=ctx+256+8`。0.70试验与不同源码的失败均另行保留，不拼接成新版本重复。

该显存比例用于初始化KV预算，不能解释为整个进程显存一定小于该比例。现有预算依据模型加载后的used、初始化warmup峰值与当前allocated量计算；后续捕图的私有池和真实临时张量仍可能使配置OOM。本轮因此保留实际OOM，未以理论KV槽数代替实测可运行并发。

初始化warmup现在覆盖 `min(max_num_batched_tokens,max_num_seqs*max_model_len)` 行，尾部使用合法短序列。原先floor计算漏行的问题已修，日志中的 `warmup_profile` 只是排入行数，不单独证明前向完成或峰值。

## 输入与计时

任务后缀来自本地DeepSpec的HumanEval/GSM8K/Alpaca文件；用WikiText连续片段补齐长度，关闭thinking，不重复同一填充片段来伪造自然长任务。原始manifest保存完整token IDs、指纹、填充量和截断标记。4K面板约95.9%–99.6%是填充；它衡量固定长度系统成本，不是长程任务能力或生产代表性证明。

模型加载、初始化期间的JIT、初始化捕图、显式预捕、16-token请求热身单列，不计入每次请求的吞吐墙钟；这些时间计入本轮GPU总账。实际请求中若发生新编译，其成本仍留在墙钟里；无运行期捕图不等于已证明没有新编译。每次正式run使用新请求、重置RNG与峰值计数，保留真实尾批和fallback。

正式primary不注入额外q前向、CUDA Event或逐轮窗口探针。交付hook、CPU轮次记账和引擎原生计数保留，测量本身不是零开销。强制接受、额外数值对拍或 `--diagnostic-cost` 结果明确作为诊断，不纳入主速度比较。

## 指标定义

| 指标 | 本轮口径 |
|---|---|
| 完整请求输出吞吐 | 所有最终交付output token数 / 入队开始至全部请求完成并同步的墙钟；包括prefill、调度、draft、verify、采样、KV维护与尾批 |
| TTFT | 每请求首次实际交付时刻减入队时刻；没有包含HTTP或网络传输 |
| TPOT | 每请求 `(末次交付-首次交付)/(实际输出token数-1)`，然后统计请求分布 |
| delivery batch gap | 相邻两次实际交付事件的间隔；一次投机交付可能包含多个token，不是SSE chunk或插值生成的单token ITL |
| decode span吞吐 | 全部decode交付token / 首decode轮开始至末decode轮结束，包含轮间CPU间隙 |
| step-sum decode吞吐 | decode轮内计时之和的局部口径，另列，不能冒充完整decode墙钟 |
| 满B前缀吞吐 | 首个连续active B等于请求B的decode阶段，按完整span计算，单列尾批影响 |
| 接受比例 | 实际送验候选中的raw接受前缀数 / 送验候选数；不是每个位置条件接受概率，未必包括draft已生成但未送验的末尾候选 |
| 平均推进τ | 实际decode交付token / decode序列轮数；包括实际交付的bonus，排除prefill首token |

AR尾部可能验证了超出剩余输出名额的候选；分别保留raw accepted、verified output与最终截断delivered。不能把已验证但没交付的token放入吞吐分子。

TTFT/TPOT等先保留每次原始值与样本量，再给3次的中位数和范围。小固定批的P99是有限样本描述，不等同于在线服务尾延迟SLO。

## 内存与失败

分别报告target KV、draft KV、ring、scratch、唯一权重、feature/projection张量、allocator allocated/reserved及各图池。共享target embedding/LM head只计一次。allocator总量包含命名张量，不能相加重复记账。allocated、reserved、单张量最大值、进程峰值和活跃KV不是同一指标。

`target_pool_blocks*256`表示物理预分配槽数，不能直接当作某长度下可服务请求数。并发结论要求实际达到所称B、完整上下文和输出、无抢占；未达到或OOM均作为失败记录。

已知source4a/B8/4K的DSpark全缓存prefill OOM在正式计划中标为blocked_known_pilot，引用原失败，不重跑也不虚构九次各自实测失败。其余仅明确OOM/固定批容量不足且安全退出才继续下一预登记job；数值、未知或外部占用则停止。

正式配置是否全部成功，与整个计划是否已到终态分开报告。容量失败或未测点不填零，不跨缺失连接曲线，不从失败基线算加速比；仅同源码/卡/输入/精度/预算/图策略的完整三次对照用于简历。

旧0.6B量化PPL、旧混批/在线压测和独立attention kernel结果均有自己的证据口径，不能迁移为本轮DSpark结果。
