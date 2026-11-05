# DSpark 滑窗与 CUDA Graph：最终结论

本轮开发、正式基准和容量诊断均已结束，GPU0已释放。实现保留在 `feat/dspark-window-resume-20261105`，没有自动合并到main。

**实际推荐方案是 `dspark_ring`：草稿窗口512，target保留完整历史。`dspark_full`仅作实验对照，相关OOM不阻止交付已通过的滑窗方案；本轮不再继续优化全历史草稿路径。** 使用配置见 [USAGE-SWA.md](USAGE-SWA.md)。

实测实现版本：`4a4a64b29192e57b3e27eb24d6a8a3069dfbd18f`。后续绘图和本文提交不改变远端已冻结的推理实现。

## 完成的实现

- 在已有官方DSpark接入上实现精确历史窗口、独立环形KV池和owner生命周期；target始终保留完整上下文。
- 验证前向先写scratch，仅提交确认前缀，避免拒绝后缀覆盖仍需读取的历史；覆盖prefill、普通decode、ragged和结束清理。
- 配置普通decode/draft/verify的1..B精确CUDA Graph，避免本轮B11补桶数值超门限；保留旧补桶失败，不把精确图通过称作补桶已修复。
- 按大shape优先预捕共享池，记录实际捕获顺序；修正初始化warmup的floor欠覆盖问题。
- 修复投机预留尾块后的decode写槽错误，以及mixed调度中本步prefill被抢占释放的问题。两项修复独立留有旧失败与CPU回归。

滑窗当前要求TP1、关闭prefix复用和mixed overlap。原混批、量化与独立kernel成果仍属于各自实现和实验，没有移作新DSpark结果。

## 正式实验与计量

RTX 3090单卡，Qwen3-4B target，BF16、temperature=1、输出256、ignore EOS；输入1K/4K，B=1/2/4/8/16，code/math/general三个固定长度面板，各3次。

五个对照是普通Graph解码、0.6B AR全历史、AR窗口512、DSpark全历史、DSpark环形窗口512。AR k=6，DSpark block7；AR窗口按块对齐，DSpark读取最近精确512个历史位置。统一初始化显存预算系数0.85、prefix关闭，实际图与fallback成本保留。

这是引擎内固定批量基准，完整请求吞吐包含prefill、decode、调度、采样和尾批；不是HTTP在线压测或vLLM/SGLang引擎横测。4K实际成功请求有96.34%–99.54%的Wiki填充，不代表自然长上下文任务质量。方法和公式见 [METHODS.md](METHODS.md)。

正式计划50组模型进程/150条件单元，终态为：

- 42组成功，形成126个完整三重复单元、378次测量。
- 7组资源失败；1组引用同源码同配置的已知失败，没有重跑或虚构九次OOM。
- 成功正式测量共1,782请求、456,192个实际输出token。失败/未测点不填零。

## DSpark窗口方案相对普通Graph解码

下表为三次同条件配对吞吐比的中位数。大于1表示更快。

| 输入 | B | 代码 | 数学 | 通用文本 |
|---|---:|---:|---:|---:|
| 1K | 1 | 2.939x | 3.385x | 2.340x |
| 1K | 2 | 2.412x | 1.976x | 1.536x |
| 1K | 4 | 2.011x | 1.612x | 1.278x |
| 1K | 8 | 1.575x | 1.562x | 0.973x |
| 1K | 16 | 1.144x | 1.134x | 0.746x |
| 4K | 1 | 2.075x | 2.286x | 1.582x |
| 4K | 2 | 1.805x | 1.670x | 1.394x |
| 4K | 4 | 1.452x | 1.335x | 1.137x |
| 4K | 8 | 1.275x | 1.207x | 0.988x |
| 4K | 16 | 容量未通过 | 容量未通过 | 容量未通过 |

在1K/B16，窗口DSpark相对本轮0.6B AR窗口512分别为代码1.274x、数学1.338x、通用1.183x；但相对普通解码，通用文本下降25.4%。因此可以写指定代码/数学负载的大batch收益，不能写大batch普遍加速，也不能把窗口512的AR对照称为所有AR参数下的最优方案。

在4K/B1、2、4的九个条件中，窗口DSpark相对全历史DSpark的配对完整请求吞吐提升4.4%–38.8%。1K也有窗口负收益，例如B4通用文本约下降8.3%；全部DSpark收益不等于窗口这一项的贡献。

## 16K容量诊断

同source、0.85预算、B4、每请求16,384输入/256输出，prefill token预算4096，code单次；不做最大并发搜索，也不作为三重复速度比。

| 项目 | DSpark全历史 | DSpark窗口512/环池 |
|---|---:|---:|
| 执行结果 | 图预热OOM，申请1.02GiB失败 | 四请求完整完成 |
| 正式容量诊断输出 | 未产生有效run | 1,024 token，首轮decode B4，无抢占 |
| target KV分配块 | 263 | 299 |
| target KV池槽数 | 67,328 | 76,544（+13.7%） |
| DSpark KV张量字节 | 1,378,897,920 | 41,963,520 |
| 验证scratch字节 | 全历史路径不分配 | 655,360 |

全历史侧KV字节由已完成分配的263块记录和BF16配置精确复算：`(263*256+1)*20,480`，不是完整成功run的具名张量/进程峰值实测。环池KV为40.02MiB，连同0.625MiB scratch为40.64MiB；这部分缓存相对1.284GiB下降96.91%。不能写总GPU显存下降96.9%，槽数增长也不是最大并发增长。

窗口同时改变历史读取、prefill feature/投影、图context形状和物理存储；四请求成功是组合配置结果，不能只归因于环池。

## 数值、原始记录与收尾

真实权重诊断覆盖相同窗口full/ring、W覆盖全历史、跨两次环绕、0/3/7接受、普通decode、精确图、ragged、多owner、scratch确认提交、拒绝写保护及guard。新源码的固定历史和B11精确图补验通过。有限诊断不等于所有模型/负载的逐位或质量证明；T1跨算法输出hash不同独立列出。

历史0.70准入不足、0.85预热/真实prefill OOM、B11补桶KL超门限以及UUID兼容预检失败均保留。没有通过降低B/输入、换精度或跨源码拼接把失败改成成功。

本轮89个GPU阶段全部闭合，累计4,172.785秒，约69.55 GPU分钟，包含加载、预热、失败和清理，不是kernel忙时。两主runner退出，记录的2,927个自有进程身份无存活；GPU0最终15MiB、0%、无compute进程。没有后续GPU任务。

- [正式逐次结果与所有对照](analysis/formal-final-02/REPORT.md)；[机器可读统计](analysis/DERIVED-SUMMARY.json)；[378次指标CSV](analysis/RUN-METRICS.csv)
- [容量逐次记录](analysis/capacity-final-01/REPORT.md)
- [吞吐曲线](analysis/formal-final-01/plots-4a085-v3/throughput.png)；[相对普通解码](analysis/formal-final-01/plots-4a085-v3/relative-to-nospec.png)；[TTFT/TPOT](analysis/formal-final-01/plots-4a085-v3/latency-ttft-tpot.png)
- 最终原始记录：`evidence/final-all-01/{results,logs,data}`，最终释放/账目 `evidence/final-all-01/FINAL-RELEASE.json`。
- 完整中间JSON和原始token IDs/日志/输入manifest留在本地及服务器，不随Git提交；公开的CSV和派生摘要附原始source/line、指纹及压缩包校验值，复现入口脚本同目录可用。
- [可用于简历的条件与文本](RESUME-DATA.md)。自动详细报告的Interim表示成功coverage不足，不表示本轮仍在运行。
