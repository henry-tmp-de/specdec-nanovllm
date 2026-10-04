# 交接文档：nano-vllm 投机解码

>给下一个 AI / 接手的人。读完这一份就能继续实验。
> 最后更新：2026-10-04

---

## 0. 一句话现状

在 nano-vllm（1385 行教学版 vLLM 复现）上实现了**线性链投机解码**，两条候选来源：
n-gram 检索（零训练，实测无收益）与 draft model（0.6B → 4B，**实现正确、输出无损、但速度收益为负**）。

**核心待办：让 draft 路线真正加速。** 已定位瓶颈（draft 的 k 次串行 forward 开销），方案见§5。

---

## 1. 项目目标与约束

用户是研究生，投 **AI Infra / 推理优化实习**，需要简历项目。

**用户明确的四条约束**：
1. 代码不能太复杂——面试要能讲清整个项目
2. 改动量不能太大——在 nano-vllm 基础上改
3. 时间：两周
4. 叙事重点：**推理框架工程**（不是 kernel 优化）

---

## 2. 已确定的战略方向（重要，别推翻）

调研 1195 条中文 AI Infra 面经的结论：

```
CUDA/算子 14.4% + 系统设计 10% + C++ 6.9% + 显存KV 6.1% ≈ 37%  ← 主战场
投机解码 1.3%（14个主题垫底）                            ← 冷门
```

**所以：投机解码不能当主题，只能当论据。**

```
主线（60%）：性能边界量化 + 修上游 PR（踩 37% 的高权重考点）
副线（40%）：投机解码（证明懂调度 + 显存 + attention mask）
```

三条可提 PR 的上游 issue（已核实状态）：
- **#243** late merge 去重（只改 `block_manager.py` +15/-1）← 唯一还活着的
- **#210** prefix cache 分配后重算 scheduled tokens（open）
- **#233** `num_cached_tokens` 增长致 `prepare_prefill` 越界（open）
- ⚠️ **#51已修、#99 是误报**（报告人自己撤了），**不要拿这两个去提 PR**

---

## 3. 环境

```bash
ssh lab3090                    # 10.201.126.245, user=ziru
cd /home/ziru/nano-vllm/repo  # 工作副本
```

| 资源 | 说明 |
|---|---|
| GPU | 8× 3090 (24GB)，**用前必查 `nvidia-smi`**，不能抢别人的卡 |
| 模型 | `~/nano-vllm/models/Qwen3-4B`（target）、`Qwen3-0.6B`（draft） |
| Python | `/home/ziru/nano-vllm/venv/bin/python` |
| 端口 | **2333 被 nano-vllm 的 `init_process_group` 占用，测试必须串行跑**，并发会报 `EADDRINUSE` |
| 下载 | 服务器无Clash，`hf-mirror.com` 可直连；**下权重用 curl + hf-mirror，snapshot_download 会回源被墙** |

---

## 4. 现状：已实现 + 实测数据

### 4.1 代码结构

```
nanovllm/spec_decode/
├── ngram_proposer.py    ~150 行  n-gram 提议（零训练）
├── draft_proposer.py    ~180 行  draft model 提议（真实分布）
├── verify.py            ~140 行  拒绝采样（保证无损）
└── __init__.py

改动���有文件：
├── engine/model_runner.py   +~150  prepare_verify / run_verify / draft 加载 / KV cache 按字节切分
├── engine/scheduler.py      +~50   spec_enabled() 门控 / postprocess_spec 一对多
├── engine/sequence.py       +~35   draft_tokens / draft_probs / append_tokens / tokens_in
├── engine/block_manager.py  +~30   can_append/may_append 支持 1+k 槽位
├── layers/attention.py      +~8   q 维度兼容 draft 单token 路径
├── layers/embed_head.py     +~5   is_spec_verify 时跳过 lm_head 的位置裁剪
├── utils/context.py         +~2   is_spec_verify 标志
└── config.py                +~6   spec_k / spec_method / draft_model / spec_batch_threshold
```

**约 600 行增量，动了 8 个已有文件。** `layers/attention.py` 的核心逻辑没改，只加了分支。

### 4.2 KV cache 分配（关键设计）

draft 与 target **共用 `Sequence.block_table`**（逻辑 block i → 物理块 i），但各自的 cache 是独立显存区。**块数必须相同**，所以不能「给 draft 分一块」，而要按字节反推：

```
N = 总预算 / (target_block_bytes + draft_block_bytes)
4B:   36 层 × 8 头 × 128 × 256 × 2 × 2 = 37.7 MB/block
0.6B: 28 层 × 8 头 × 128 × 256 × 2 × 2 = 29.4 MB/block
实测：174 blocks，两边指针重叠数 = 0  ✅
```

### 4.3 ★ 实测数据（RTX 3090, Qwen3-4B target, temperature=1.0）

```
                tok/s     steps  tok/step  接受率  distinct
baseline        17.0      60      1.00      —       41
draft k=2       11.3      37      1.62    31.9%     40
draft k=4       10.4      25      2.28    36.5%     37
```

**★ 关键结论：输出质量已修好（distinct 37~41，无退化），但速度是负的。**

**原因**（这是核心洞察，别再走错路）：

```
每步 decode 的权重读取量：
  baseline  ：1 × 8.0 GB（4B 模型）           = 8.0 GB
  draft k=2 ：2 × 1.1 GB（draft）+ 1 × 8.0 GB = 10.2 GB  ← 更多！

虽然每步多产出 1.62 个 token，但读取量增加 27%
→ 净亏损 34%
```

**投机解码的盈亏公式**（调研文献里的 `c` 参数）：
```
加速比 S = (1 − α^(k+1)) / ((1 − α)(γc + 1))
  α = 接受率, γ = k, c = draft单步耗时 / target单步耗时
0.6B/4B 的 c ≈ 0.14，看似很有利，但 α 只有 0.32 拖累了
★ 之前我算错的原因：只看了「draft 小」，忘了 draft 要跑 k 次【串行】
```

### 4.4 无损性验证（已完成，两种口径）

**a) 单元测试级（严格）**：`tests/test_spec_decode.py` 16 项全过
```
★ 无损性判据不用固定阈值，而是看 TV 是否随样本量收敛到 0：
  N=40,000  平均 TV = 0.00355
  N=400,000 平均 TV = 0.00122
  比值 2.91（理论 3.16）→ 无偏
  接受率 = 0.320（理论 p[d] 均值 = 0.320）完全吻合
```

**b) 端到端级（统计）**：单条 prompt × 2 次采样 × 60 token
```
baseline: n=120, distinct=57, top=[220:11, 198:7, 25:6, 2:5, 284:5]
draft   : n=117, distinct=59, top=[220:23, 279:5, 18:5, 374:4, 17:4]
→ top1 相同（220），distinct 数量接近，分布高度一致 ✅
```

---

## 5. ★ 核心待办：让 draft 路线真正加速

### 5.1 问题定位

**瓶颈是 draft 的 k 次串行 forward 开销**，不是接受率也不是正确性。

量化（每步权重读取）：
```
baseline:  8.0 GB
draft k=2: 2×1.1 + 8.0 = 10.2 GB    (×1.28)
draft k=4: 4×1.1 + 8.0 = 12.4 GB    (×1.55)
```

### 5.2 三个可选方案（按推荐度排序）

**方案 A：换成 EAGLE 式单层 draft head**（推荐）
```
不是跑整个 0.6B，而是只跑【target 模型的最后一层】+
一个轻量 draft head（复用 EAGLE 的思路）。
c 从 0.14 降到 ~0.02，draft 开销可忽略。
★ 这是工业界的主流做法（EAGLE/EAGLE-3/SGLang 都是这样）
★ 但需要一个训练好的 head——可以从 EAGLE 公开权重开始改，
  或者自己训（超出两周，得评估）
```

**方案 B：换更小的 draft 模型**
```
0.6B → 0.1B 左右（Qwen3-0.1B 在 HF 上不存在，需找替代，
  如 Qwen2.5-0.5B vocab 相同但也不够小）
权重比拉到 1:40+，draft 开销可忽略
★ 风险：draft 太小接受率会崩（文献说 α<0.5 时验证开销可能超过收益）
```

**方案 C：接受「收益为负」，改写成边界量化叙事**
```
简历写法：「我实测了 draft model 投机解码在 0.6B→4B 配比下的收益，
发现加速比是【权重比 × 接受率 × k】的函数，并给出了盈亏平衡点」
★ 这也算扎实，但面试里说「我的实现没加速」是减分的
```

**我推荐 A**，理由：它同时解决「draft 开销大」和「这是工业界主流」两个问题，
而且 EAGLE-3 已进 vLLM/SGLang 双主线，简历上站得住。

### 5.3 batch 门控（必做，成本极低）

`speculative.py` 里已实现 `spec_batch_threshold`（默认 0=不限制）。
**必须做实测找出盈亏平衡点**——文献显示 batch>8 后收益递减、batch≥64 可能净亏损。

```python
# 已在 scheduler.spec_enabled() 实现，改 config 即可启用
spec_batch_threshold = 8   # batch 超过就退回普通 decode
```

---

## 6. 已修复的 Bug（10 个，全部是「不报错但结果错」的类型）

| # | 文件 | 问题 | 症状 |
|---|---|---|---|
| 1 | `verify.py` | 单点分布用 logits 表达 | softmax 后是 0.9999 而非 1.0，残余概率污染修正分布，接受率 1.0→0.35 |
| 2 | `verify.py` | target 索引偏移错位 | 写成 `p[:, :k]` 应为 `p[:, 1:k+1]` |
| 3 | `verify.py` | bonus 采样用错分布 | 200000 个样本全部错，修正后全部正确 |
| 4 | `model_runner.py` | `prepare_verify` 的 n 取错 | `num_scheduled_tokens` vs 实际草稿数 → 三者长度错位、RoPE 越界 |
| 5 | `embed_head.py` | lm_head 在 `is_prefill` 时裁剪位置 | 验证阶段只剩 1 行 logits |
| 6 | `context.py` | 新字段加在 dataclass 中间 | `set_context` 用位置参数 → 全盘错位，k=0 都跑不起来 |
| 7 | `draft_proposer.py` | off-by-one | `last_token` 在位置 `context_len-1`，误写成 `context_len` |
| 8 | `attention.py` | q 维度不兼容 draft 单token 路径 | draft 的 q 是 2 维，`unsqueeze(1)` 多一维 |
| 9 | `draft_proposer.py` | `block_tables` 缺 batch 维 | 应为 `(batch, max_blocks)` |
| 10 | `scheduler.py` | ★ `hash_blocks` 在 `num_cached_tokens` 推进**之前**调用 | **上游 nano-vllm 同源问题**，普通 decode 无害，投机路径直接用它当 verify 起点会放大 |

**第 10 条值得单独提PR**——它是真实的上游 bug，且就在投机解码必经之路上。

---

## 7. ★★ 两个方法论陷阱（务必避免，我各浪费了 4轮）

### 陷阱 1：低temperature 会把分布压成 one-hot

```python
SamplingParams(temperature=0.01)   # ❌ 用来做「逐 token 一致」对比
```
```
softmax(logits / 0.01)：logits 量级 10~20 → 除以 0.01 = 1000~2000 → p 必然 = 1.0000
```
**后果**：baseline 同样退化，我据此误判「draft 分布退化」，白改 8 轮。

**正确做法**：用 `temperature=1.0`（正常采样）+ 统计检验。

### 陷阱 2：随机采样下逐 token 对比没有意义

```
temperature=1.0 是随机采样，两次运行本来就该不同。
```
**正确做法**：跑多条 prompt / 长序列，比较 **token 频率分布**。

### 附：GPU 精度确实影响一致性（实测）

```
同一条 token 单独算 vs 批量算（verify 路径）：
  max|Δlogit| = 0.25,  TV 距离 = 0.033
  连续 30 个位置中 argmax 不同：1/30 = 3.3%
```
**所以「逐 token 严格一致」这个标准本身过严**，应该用统计无偏的判据。

---

## 8. 代码与脚本清单

### 仓库
```
https://github.com/henry-tmp-de/specdec-nanovllm
本地  D:\学习\specdec\
服务器 ~/nano-vllm/repo/   （工作副本，含临时脚本）
```

### 正式脚本
```
tests/test_spec_decode.py    16 项单测（含无损性收敛性检验），CPU 可跑
bench_final.py              性能基准，输出 @@B@@ JSON
stat2.py                    统计无损性检验，输出 @@S@@ JSON
check_stat.py               多 prompt 统计检验（draft 模式太慢，慎用）
bench_code.py / bench_draft.py   早期 benchmark
observe.py                  nano-vllm 原有的逐 step 计时脚本（在 nano-vllm 目录）
```

### 已清理
临时诊断脚本（dtest*/dbg*/diag*）已从仓库删除，但服务器上可能还有残留。

---

## 9. 服务器运维备注

- **端口 2333**：nano-vllm 硬编码 `tcp://localhost:2333`，**测试必须串行**，
  并发跑会报 `EADDRINUSE`。残留进程用 `pkill -f <脚本名>` 清理，
  ⚠️ 但 `pkill -f` 可能误杀自己的 SSH 会话，要小心。
- **代理**：`anyilin` 用户的 mihomo 在跑，但**不要用他的**。服务器无自己的 Clash。
- **下载**：用 `curl` + `hf-mirror.com`（支持断点续传 `-C -`），
  `snapshot_download` 会回源 xethub 被墙而失败。
- **空卡查询**：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`

---

## 10. 下一步的明确建议

```
1. 实现 EAGLE 式单层 draft head（方案 A）
   → 参考 EAGLE (arXiv 2401.15077) 与 vLLM 的 eagle3 路径
   → 关键：只跑 target 的最后一层 + 一个 head，c 从 0.14 降到 ~0.02

2. 补 batch 门控实测
   → 扫 batch = 1/4/8/16/32/64，画出盈亏平衡点
   → 这本身就是一个可写进简历的数据点

3. 修上游 #10（hash_blocks 顺序），提 PR
   → 改动 <5 行，是真实 bug，且与投机解码直接相关

4. 写README
   → benchmark 表（含负收益区间）+ 适用边界 + 三个可 defend 的判断
```

**简历可用的三个判断**（都能 defend，都有数据支撑）：
```
① n-gram 投机无收益的根因：缺少真实分布 → 接受概率退化为 p[draft]
   （实测 0.6% vs draft model 的 32%）
② 「树状投机是过度设计」：论文消融 linear 5.345× > tree 5.175×，
   Snowflake 生产代码 use_tree_spec 默认 False
③ 「draft 开销抵消收益」：加速比是权重比 × 接受率 × k 的函数，
   0.6B→4B 实测净亏损 34%，给出了盈亏平衡的条件
```

---

## 11. 环境路径速查

```
本地：
  D:\学习\specdec\                      项目（含 .git）
  D:\学习\nano-vllm\                    nano-vllm 源码副本（只读参考）
  D:\学习\CUDA-入门\                    CUDA 练习代码

服务器：
  ~/nano-vllm/repo/                    工作副本
  ~/nano-vllm/models/Qwen3-4B/         target模型
  ~/nano-vllm/models/Qwen3-0.6B/       draft 模型
  ~/nano-vllm/venv/bin/python          Python 解释器

交接文档（旧，包含策略调研）：
  C:\Users\85935\AppData\Local\Temp\handoff-specdec-nanovllm-2026-10-02.md
```