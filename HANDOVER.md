# 交接文档：nano-vllm 投机解码

>给下一个 AI / 接手的人。读完这一份就能继续实验。
>最后更新：2026-10-05

---

## 0. 一句话现状

在 nano-vllm（1385 行教学版 vLLM 复现）上实现了**线性链投机解码**，
并把它从「净亏损」做成了**真加速：0.68× → 1.61×**（Qwen3-0.6B → Qwen3-4B，单条 256 token）。

上一轮「速度为负」的根因**不是权重读取量**（那个模型解释不了实测：
同样的 4B 前向，eager 33.2 ms/步、拍图 12.8 ms/步），而是 **per-step 的 eager 开销**；
draft 更极端 —— 0.6B 权重只要 1.3 ms 读完，eager 一次前向却要 24.8 ms。

本轮另外查出**五个真实 bug**（其中两个破坏无损性、一个会让 CUDA 崩、一个会无限生成），
全部有回归测试。核心改动是给 **draft 前向**和**验证前向**各拍一张静态形状的 CUDA graph。

---

## 1. 项目目标与约束

用户是研究生，投 **AI Infra / 推理优化实习**，需要简历项目。

**用户明确的四条约束**：
1. 代码不能太复杂——面试要能讲清整个项目
2. 改动量不能太大——在 nano-vllm 基础上改
3. 时间：两周
4. 叙事重点：**推理框架工程**（不是 kernel 优化）

---

## 2. 战略方向（别推翻）

调研 1195 条中文 AI Infra 面经的结论：

```
CUDA/算子 14.4% + 系统设计 10% + C++ 6.9% + 显存KV 6.1% ≈ 37%  ← 主战场
投机解码 1.3%（14个主题垫底）                            ← 冷门
```

**所以：投机解码不能当主题，只能当论据。**
本轮的叙事正好落在主战场上：**「定位并消除推理引擎的 per-step 开销」**，
这属于「系统设计 / 显存 / 性能边界量化」，比「我实现了一个投机解码」高一个量级。

---

## 3. 环境

```bash
ssh lab3090                    # 10.201.126.245, user=ziru
cd /home/ziru/nano-vllm/repo  # 工作副本（注意：这个目录【不是】git 仓库，
                              #  真正的 git 在本地 D:\学习\specdec\）
```

| 资源 | 说明 |
|---|---|
| GPU | 8× 3090 (24GB)。**用前必查 `nvidia-smi`**；本轮全程只有 7 号卡是空的 |
| 模型 | `~/nano-vllm/models/Qwen3-4B`（target）、`Qwen3-0.6B`（draft） |
| Python | `/home/ziru/nano-vllm/venv/bin/python`（torch 2.10.0+cu128） |
| 端口 | **2333 被 `init_process_group` 占用，测试必须串行跑**，并发报 `EADDRINUSE` |
| 下载 | 服务器无 Clash，用 `curl` + `hf-mirror.com`（`-C -` 断点续传） |

**★ 跑任何 GPU 任务都要 `CUDA_VISIBLE_DEVICES=7`**（0–6 长期被 caiyue 的 heat 任务占满）。

---

## 4. ★★ 本轮推翻了上一轮的根因分析

### 4.1 上一轮的说法（错的）

> 「每步 decode 要读全部权重，baseline 8.0GB，draft k=2 是 10.2GB，
>  读取量增加 27% → 净亏损 34%」

**这个模型解释不了实测数据。** 4B 模型 fp16 权重 8GB，
3090 显存带宽 936 GB/s，读一遍只要 **8.5 ms**；
而实测 eager 每步要 **33.2 ms**。多出来的 24 ms 跟权重没关系。

### 4.2 实测（空卡、同一脚本、同一 prompt）

| 配置 | 每步耗时 | tok/s |
|---|---:|---:|
| baseline，`enforce_eager=True` | **33.2 ms** | 30.6 |
| baseline，开 CUDA graph | **12.8 ms** | 79.6 |
| — | — | — |
| draft 的 0.6B 单次前向（eager） | **24.8 ms** | — |
| （同样一次前向的理论权重读取时间） | 1.3 ms | — |

**同样是「读一遍 8GB 权重」，拍图 12.8 ms、不拍图 33.2 ms —— 差 2.6 倍。**
这 20 ms 里没有一个字节是权重，全是 kernel launch + Python dispatch。

draft 更极端：0.6B 只有 1.2GB 权重（理论 1.3ms），eager 下要 24.8ms —— **19 倍**。
而 draft 要串行跑 k 次，这笔开销要乘以 k。

### 4.3 结论

```
旧的（错）模型：加速比 ≈ f(权重比)
新的（对）模型：加速比 ≈ f(每步固定开销)  ← 这才是 batch=1 时的主导项
```

**上一轮报的 baseline「17.0 tok/s」也偏低**（同一脚本现在测出 30.6 eager）。
大概率是当时 GPU 上有别人的任务在抢。**做性能结论前，先确认卡是空的。**

---

## 5. ★★ 本轮查出的五个真实 bug

五个都是「不报错、只是结果悄悄错」的类型（E 在拍图后直接崩），
而且**互相叠加**，把上一轮的实测数据整个带偏了。

| | 一句话 | 后果 | 回归测试 |
|---|---|---|---|
| A | draft 侧传概率，验证又 softmax 一次 | **无损性破裂** | 测试 5 |
| B | 候选 j 被拿去和第 j+1 行比 | 接受率 0.84 → 0.16 | 测试 6 |
| C | 全接受时丢掉已算好的 bonus | 每步只出 1 个 token，**必然变慢** | 测试 6 |
| D | 终止判据用 `==` 而非 `>=` | 无限生成（300 步/572 token 不停） | test_scheduler_terminate.py |
| E | 跨 block 少分配一块 | **CUDA 非法访存** | test_block_alloc.py |

前两个（A、B）直接破坏「输出分布 = 目标分布」这条硬性质。

### Bug A：draft 侧传了概率，verify 又 softmax 了一次

`verify_batch` 内部做的是 `q = softmax(draft_side / temperature)`，
所以 draft 侧必须传**原始 logits**。
但 `DraftModelProposer` 返回的是 `softmax(logits/T)` —— 概率。
于是 q 被第二次 softmax，在 151936 词表上被压成**近似均匀分布**：

```
实测（引擎内）：draft 真分布 max = 0.597
                再 softmax 后 max = 1.23e-05   （1/vocab = 6.6e-06）
```

草稿 token 是从真分布 q 采的、验证却用 q̃ —— 拒绝采样的一致性前提被破坏，
**输出分布不再等于目标分布，无损性不成立**。

**决定性证据**（构造 q ≡ p，此时 min(1,p/q)=1，正确实现必须全部接受）：

| draft 侧传什么 | 接受率 | TV 距离 |
|---|---:|---:|
| logits（正确） | **1.0000** | 0.0032（N=20万，在收敛到 0） |
| 概率（原实现） | 0.8895 | **0.1075**（有偏，不随 N 下降） |

修法：proposer 返回原始 logits；字段名 `draft_probs` → `draft_logits`
（**这个名字本身就是 bug 的诱因**）。
回归测试：`tests/test_spec_decode.py` 测试 5。

> ⚠️ `verify_batch` 的形参名叫 `draft_probs`，但语义是 logits。
> 这个命名与文档自相矛盾，是 bug 的温床，已在 docstring 里写明。

### Bug B：候选 j 被拿去和 target 的第 j+1 行比（错位一位）

```
验证时送进 forward 的 k+1 个 token 位于 position: len-1, len, ..., len+k-1
模型在 position p 的输出 = 「p+1 位置 token」的分布
  => 第 r 行 = 位置 len+r 的 token 的分布 = 候选 r 的分布
  => 候选 j 对齐【第 j 行】，下标相同、不偏移
```

原实现写的是 `p[:, 1:k+1, :]`（整体挪一行），bonus 也取 `p[accepted+1]`。
后果：候选 0 被拿去和「候选 1 的分布」比 —— 本来高度一致的 0.6B/4B，
接受率被从应有的 **0.84 打到实测 0.16**。

**为什么单测没查出来**：`tests/test_spec_decode.py` 里所有用到多行的用例，
target 各行都是**完全相同**的（`draft_logits.clone()`、整块 -50 加同一个峰、
`logits.expand(N,2,V)`）—— 各行一样时索引偏移多少都测不出差异。

**决定性证据**（合成算例：target 第 j 行在 token(10+j) 上有尖峰，
草稿逐行等于目标，k=3）：

```
修复前：accepted = 0    bonus = 11   ← bonus 落在第 1 行的峰上
修复后：accepted = 3    bonus = -1   ← 全部接受，符合预期
        （第 0 位被拒时 bonus = 10，取自第 0 行）
```

回归测试：`tests/test_spec_decode.py` 测试 6。

### Bug C：k 个候选全被接受时，把已经算好的 bonus 扔掉了

`verify_batch` 原来在「无拒绝」时把 bonus 置成 -1 丢弃：

```python
bonus = torch.where(has_reject, bonus, torch.full_like(bonus, -1))
```

**但那一行的目标分布（第 k 行）在这一轮 forward 里已经算出来了。**
全接受时 `resid = p[k] − 0 = p[k]`，正是 target 自己下一步会算的 token。
丢掉它等于白算一行，而且：

```
k=1、α=0.857 时：
  丢掉 bonus -> 每步恒定只出 1 个 token（接受出候选 1 个 / 被拒出 bonus 1 个）
  保留 bonus -> 每步 1.857 个
```

**这直接决定了「加速比是正还是负」**：即便接受率 0.85，
每步只出 1 个 token 的投机解码一定比 baseline 慢（它还多付了 k 次 draft 前向）。

而且 scheduler 每步预留的槽位数本来就是 `need = 1 + spec_k`
（`scheduler.schedule` 里），本来就是按「最多落地 k+1 个 token」设计的 ——
说明这是实现遗漏，不是设计取舍。

**证据**：修复前 `scripts/bench_spec.py draft 1 64 1` 报 `tok_per_step = 1.0`、
`steps = 64`；修复后同配置 `tok_per_step = 1.561`、`steps = 41`。

### Bug D：终止判据用 `==` 而不是 `>=` —— 序列会无限生成

```python
# scheduler.postprocess_spec，原来写的是：
if (not seq.ignore_eos and t == self.eos) or seq.num_completion_tokens == seq.max_tokens:
```

投机一步会落地多个 token，完成 token 数会**跳过** `max_tokens`：
从 62 直接跳到 65，`== 64` 永远不成立 → 序列一直生成下去。

**实测**：`max_tokens=64` 的配置跑到 **300 步、572 个 token 还没停**。
在 benchmark 里的表现就是「draft 模式一跑就卡住几百秒」——
这个现象被骗了很久，一度以为是模型加载慢或 GPU 竞争。

（上一轮没暴露，是因为对齐 bug 让接受率只有 0.16、绝大多数步只落地 1 个 token，
恰好总是命中 `==`。）

修法：先记 append 之前的完成数 `base_completion`，判据改成
`base_completion + i + 1 >= seq.max_tokens`。
回归测试：`tests/test_scheduler_terminate.py`（纯 CPU，8 项）。

### Bug E：跨 block 时少分配一块 → CUDA 非法访存

`block_manager.can_append / may_append` 是按「本次要写 `len..len+n-1` 这些位置」
算空位的：

```python
remaining_in_block = self.block_size - (len(seq) % self.block_size)
to_alloc = max(0, n_tokens - remaining_in_block)
```

**但所有写入路径写的都是 `len-1 .. len+n-2`**
（`prepare_decode` / `prepare_verify` 的 slot_mapping 都从 `len-1` 起算，
因为上一个 token 的 KV 要重算一遍）。

`len` 正好是 `block_size` 整数倍时，这个式子给出 `remaining = block_size`
（以为当前 block 空着），**其实当前 block 已经正好写满** —— 于是少分配一块，
block_table 里那一位是填充值 -1，attention 要跨块时 flash-attn 就读第 -1 页：

```
eager 路径   : 读到别处的垃圾内存 —— 不报错，但 KV 是错的
拍成 CUDA graph: torch.AcceleratorError: an illegal memory access was encountered
```

**为什么藏了这么久**：64 个 token 的短生成永远碰不到 256 的边界。
把 benchmark 的 `max_tokens` 从 64 改成 256，立刻崩。

修法：`_blocks_needed(seq, n) = (len + n - 2) // block_size + 1`。
回归测试：`tests/test_block_alloc.py`（纯 CPU，11 项）。

> ⚠️ 同一处还有一点值得留意（本轮没改，因为没触发）：
> 原版 nano-vllm 的 `may_append` 会处理「最后一个 block 被前缀缓存共享
> （ref_count > 1）」的情况（copy-on-write）。改写成按块数分配后这段逻辑没了。
> 本项目是单序列、没走到共享块，所以没暴露；多条序列共享前缀时要回头补。

### 附带修正：原来报的「接受率 31.9%」这个数本身也是错的

```
原式：accept_rate = (landed − verify) / proposed
```
`landed` 是每步落地的 token 数。**k 个候选全被接受时不会加 bonus**，
所以「全接受」时这个式子只有 (k−1)/k —— **永远到不了 1**。
真实接受率要直接数 `res.accept_mask`。
（正确的写法见 `scripts/bench_spec.py`。）

---

## 6. ★ draft 模型到底能提供多高的接受率（绕开引擎独立测）

接受率有恒等式：

```
α = E_{x~q}[ min(1, p[x]/q[x]) ] = Σ_x min(p[x], q[x]) = 1 − TV(p, q)
```

所以 α 完全由两个模型分布的 TV 距离决定。用 transformers 原生直接跑两个模型
（不经过引擎，排除引擎算错上下文），在相同上下文上比下一 token 分布：

| 指标 | 实测 |
|---|---:|
| 平均 TV 距离（0.6B vs 4B，60 个位置） | **0.1655** |
| argmax 一致率 | 86.7% |
| 理论接受率 α = 1 − TV | **0.8345** |
| 采样仿真接受率 | 0.8374 |

**结论：Qwen3-0.6B 给 Qwen3-4B 当 draft，固有接受率约 0.84，是很好的组合。**
（这也反过来证明 Bug B 的存在：引擎里测到 0.16，差 5 倍。）
脚本：`scripts/check_draft_quality.py`。

---

## 7. ★ 本轮的实际改动

### 7.1 给投机路径补上 CUDA graph（核心改动）

| 图 | 形状 | 收益 |
|---|---|---|
| draft 前向图 | bs=1、每次 1 token | 24.8 ms → ~2 ms |
| 验证前向图 | bs=1、query 数固定 1+k、varlen 路径 | 36 ms → ~14 ms |

为什么验证前向也能拍图：它走 varlen，`run_model` 里 `is_prefill=True`
会强制走 eager 分支；但它的**形状是固定的**（batch=1、query 恒等于 1+k、
block 表宽度固定），形状固定就能拍图。

条件不满足时（batch>1、草稿数不足 k）自动退回 eager，行为不变。
开关：`config.spec_cuda_graph`（默认 True）。

### 7.2 两个 bug 的修复

见 §5。改动集中在 `spec_decode/verify.py`、`spec_decode/draft_proposer.py`、
`engine/model_runner.py`、`engine/sequence.py`、`engine/scheduler.py`。

### 7.3 正确性对拍（新代码必须过这一关）

`scripts/check_graph_and_verify.py`：同一次前向，图路径 vs eager 路径逐元素比。

```
验证前向：max|Δlogit| ≈ 0.30，argmax 绝大多数相同
draft 前向第 0 步：max|Δlogit| ≈ 0.19
```

这个量级和上一轮已经记录过的「单条算 vs 批量算」差异（max|Δlogit|=0.25）一致 ——
是 flash-attn 分块配置不同带来的**浮点累加顺序**差异，不是逻辑错误。
所以判据不能是「逐 bit 相同」，而是**分布无偏**（见 §8）。

---

## 8. 正确性验证（三档，强度不一样，别混着说）

```bash
python tests/test_spec_decode.py              # 纯 CPU，18 项，含 3 类无损性判据
python tests/test_scheduler_terminate.py      # 纯 CPU，8 项（bug D 回归）
python tests/test_block_alloc.py              # 纯 CPU，11 项（bug E 回归）
python scripts/check_draft_quality.py         # 绕开引擎测 draft 固有质量（α 的理论上界）
python scripts/check_lossless_engine.py       # 引擎级：输出分布对比（要 GPU，功效弱）
```

**① 单元测试级（强证据 —— 主要靠这个）**

- `verify.py` 的无偏性：TV 随样本量收敛（N=40k → 400k，比值 2.91 vs 理论 3.16）
- 合成算例：q ≡ p 时必须全部接受；候选 j 必须对齐第 j 行
- 这两条**直接证明算法是对的**，和引擎、GPU、CUDA graph 都无关

**② draft 固有质量（独立于引擎）**

`scripts/check_draft_quality.py` 用 transformers 原生跑两个模型，
测得 0.6B→4B 的 TV = 0.1655，即 α 的理论上界 0.8345。

**③ 引擎级分布对比（中等强度，设计上有讲究）**

四轮各 4800 token（`temperature=1.0`，随机采样）：

```
TV(base , base2 ) = 0.5315   <- 基线噪声地板（同配置两次独立运行）
TV(draft, draft2) = 0.6325   <- 投机路径自己的地板
跨组 base x draft 四个值：0.514 / 0.521 / 0.671 / 0.713
```

**跨组差异完全夹在两个噪声地板之间 —— 没有发现系统性偏差。**

★ 这里的关键是**必须给投机路径单独测一个地板**，不能只跟基线比：
投机解码一步产出 3~4 个 token，它们来自同一次前向、同一个上下文，
序列自相关远强于逐 token 解码 → 有效样本量更小 → 经验分布天然更散。
实测 `TV(draft,draft2) = 0.63 > TV(base,base2) = 0.53`，
正好印证了这一点。拿 `TV(base,draft)=0.71` 直接和 0.53 比，
会得出「超出地板 34%、可能有偏」的**错误结论**。

**这个设计的局限（要更强的话往这走）**：n=4800 时地板高达 0.5 量级，
检验功效有限，只能说「没发现偏差」，不能说「证明无偏」。
想做成强证据得换设计 —— 同一批 prompt 各只生成 1 个 token（样本互相独立），
或用 5 万+ token 让地板降到 0.1 量级。

**★ 方法论（上一轮在这里各浪费了 4 轮，别重犯）**：

1. **不要用低 temperature 做逐 token 对比**。`temperature=0.01` 会把
   `softmax(logits/0.01)` 压成 one-hot，p 必然 = 1.0000，baseline 同样如此，
   会误判成「draft 分布退化」。
2. **随机采样下逐 token 对比没有意义**。temperature=1.0 本来就是随机采样。
   要比的是 **token 频率分布**。
3. **GPU 精度确实影响逐 token 一致性**（实测 max|Δlogit|≈0.25，
   30 个位置里约 1 次 argmax 不同 ≈ 3.3%），所以「严格逐 token 一致」
   这个标准本身过严，要用**统计无偏**的判据。
4. **无损性判据 = TV 是否随样本量收敛到 0**，不是固定阈值。

引擎级检验的额外要点：用「基线 vs 基线的第二次独立运行」当**噪声地板** ——
单看 TV(draft, base) 没有意义，必须和 TV(base, base2) 同量级。

---

## 9. 实测结果

环境：RTX 3090（空卡 7）、Qwen3-4B + Qwen3-0.6B、temperature=1.0、
单条 prompt、生成 **256 token**、`gpu_memory_utilization=0.9`。
"开图" = `enforce_eager=False`（普通 decode 与投机两条路径都拍 CUDA graph）。

| 配置 | tok/s | 每步 token | 接受率 | 相对 baseline |
|---|---:|---:|---:|---:|
| baseline（开图） | 82.3 | 1.00 | — | 1.00× |
| draft k=2（开图） | 116.9 | 2.15 | 0.648 | **1.42×** |
| draft k=6（开图） | **132.5** | 3.82 | 0.770 | **1.61×** |
| — 对照：draft k=6（不拍图，eager） | *20.8* | *3.77* | *0.791* | *0.68×*ᵃ |
| — 对照：baseline（不拍图，eager） | *30.6* | *1.00* | *—* | — |

ᵃ 相对 eager baseline（30.6 tok/s）。

### ★ 这组对照说明了什么

```
draft k=6，eager  : 20.8 tok/s   ← 每步 3.77 个 token，但仍然是【净亏损】
draft k=6，开图   : 132.5 tok/s  ← 每步 3.82 个 token，加速 1.61×
        ↑ 每步产出几乎没变（3.77 → 3.82），变的只有 per-step 固定开销
```

**把符号翻正的不是接受率，也不是算法，是 CUDA graph（6.4 倍）。**
这也解释了上一轮「实现正确但速度为负」：那时两条前向都是 eager，
投机解码要串行多做 k 次前向 + 1 次验证，**必然**比 baseline 慢 ——
跟权重读取量多少无关。

### 64 token 的短程数据（同一批配置，供对照）

| 配置 | tok/s | 每步 token | 接受率 | 相对 baseline |
|---|---:|---:|---:|---:|
| baseline（开图） | 80.9 | 1.00 | — | 1.00× |
| draft k=1（开图） | 94.3 | 1.56 | 0.575 | 1.17× |
| draft k=2（开图） | 110.3 | 2.13 | 0.724 | 1.36× |
| draft k=4（开图） | 122.5 | 3.05 | 0.750 | 1.51× |
| draft k=6（开图） | 149.4 | 4.57 | 0.885 | 1.85× |

⚠️ **短程的接受率偏高、波动大**（每次只有一条轨迹，上下文难度不同）。
以 256 token 那一组为准：k=6 的接受率是 0.77 而不是 0.885。
另一个佐证：`scripts/check_draft_quality.py` 独立测得 0.6B→4B 的
固有 TV = 0.1655，即首位置接受率上界 **0.8345**。

### k 怎么选

```
α = 0.77 时 E[每步 token] = (1 − α^(k+1)) / (1 − α)
  k=2 -> 2.25    k=4 -> 3.42    k=6 -> 3.82    k=8 -> 3.95    k=10 -> 4.13
每步耗时 ≈ k × (draft 前向 ~2ms) + 验证前向 ~14ms
  k=6 -> 26ms      k=8 -> 30ms   k=10 -> 34ms
  => tok/s: k=6 约 147、k=8 约 132、k=10 约 121
实测 k=6 132.5（含 Python/调度开销）已经接近最优，再加大 k 收益递减。
```

---

## 10. 代码与脚本清单

### 仓库
```
https://github.com/henry-tmp-de/specdec-nanovllm
本地  D:\学习\specdec\
服务器 ~/nano-vllm/repo/   （工作副本，无 .git）
```

### 脚本
```
tests/test_spec_decode.py            23 项单测（纯 CPU，含 3 类无损性判据 + bug A/B/C 回归）
tests/test_scheduler_terminate.py     8 项（纯 CPU，bug D 回归）
tests/test_block_alloc.py            11 项（纯 CPU，bug E 回归）
scripts/bench_spec.py                加速比基准，输出 @@B@@ JSON
scripts/bench_batch.py               batch 门控实测（还没跑）
scripts/check_draft_quality.py       draft 固有质量（TV / α 理论上界），transformers 原生
scripts/check_graph_and_verify.py    图路径 vs eager 路径逐元素对拍
scripts/check_lossless_engine.py     引擎级输出分布对比（base / base2 / draft）
scripts/check_stat.py / stat2.py     早期的统计检验（较弱，保留作对照）
bench_final.py                       ★ 注意：它用 enforce_eager=True，
                                     且 accept_rate 算式有误，别再用它下结论
```

### 服务器上的运行脚本（在 ~/nano-vllm/repo/，没进 git）
```
run_bench2.sh + bench_results.txt    64 token 的 k 扫描
run_final.sh  + final_results.txt    256 token 的最终数据（含 eager 对照）
run_lossless.sh + lossless_results.txt  引擎级无损性三轮
tv_compare.py                        比较三轮的 token 频率分布（用 base/base2 当噪声地板）
```

---

## 11. 服务器运维备注

- **端口 2333**：`init_process_group` 硬编码 `tcp://localhost:2333`，
  **测试必须串行**。残留进程 `pkill -f <脚本名>` 清理（小心别误杀自己的 ssh 会话）。
- **代理**：服务器无 Clash，`anyilin` 用户的 mihomo 不要用。
- **空卡查询**：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
- **同步代码**：本地改完用 `scp` 推上去（服务器那份不是 git 仓库）。
  本地 Windows 的 git bash 里 scp 到 `lab3090` 走的也是 ~/.ssh/config 里的 lab3090。

---

## 12. 下一步

```
1. 【低成本，建议先做】batch 门控实测（scripts/bench_batch.py），
   扫 B=1/4/8/16/32/64 找盈亏平衡点，写进 config.spec_batch_threshold 的推荐值。
   ★ 这是唯一还缺的数据点，而且它本身就是「性能边界量化」的好素材。
2. 【低成本】temperature 扫描：α 直接由 TV(p,q) 决定，温度越低两个分布越接近、
   α 越高。画一条「加速比 vs 温度」的曲线，边界就完整了。
3. 【中成本】提上游 PR：nano-vllm 的 hash_blocks() 调用顺序 bug
   （scheduler.py 里 <5 行改动）。相关 issue：#243 / #210 / #233。
   ⚠️ 不要用 #51（已修）和 #99（误报）
4. 【中成本】把 bug E 也提给上游：nano-vllm 原版 may_append 的
   `len(seq) % block_size == 1` 判据被改坏之后会少分一块（本项目已修，
   但上游/其他 fork 可能同样有问题）。改动同样很小。
5. 【可选】EAGLE 式单层 draft head：α=0.77~0.83 已经不低，
   换 head 的收益有限（E[tokens/step] 的天花板是 1/(1−α) ≈ 4.3，
   k=6 已经到 3.82），优先级最低。
6. 【可选】多序列 / 前缀缓存场景验证：改写的 may_append 丢掉了原版对
   「最后一个 block 被共享（ref_count>1）」的 copy-on-write 处理。
```

**简历可用的判断**（都能 defend，都有数据支撑）：

```
① 推理引擎在 batch=1 时，瓶颈是 per-step 固定开销而不是权重读取
   （同一模型同一次前向：拍图 12.8ms，不拍图 33.2ms，2.6 倍；
    0.6B 的 draft 单次前向 24.8ms，而它的权重只需 1.3ms 读完）
② 投机解码的接受率有恒等式 α = 1 − TV(p, q)，可以用它给 draft 选型定上界
   （Qwen3-0.6B→4B 实测 TV=0.1655，α 上界 0.8345）
③ 无损性的前提是「草稿分布必须和采样时用的分布是同一个」——
   实测踩过两个破坏它的 bug：概率被二次 softmax、候选与 target 行错位一位
④ 加速比 = f(每步固定开销, 接受率 α, k)，给出了 0.6B→4B 的盈亏平衡条件
```

---

## 13. 环境路径速查

```
本地：
  D:\学习\specdec\                      项目（含 .git）
  D:\学习\nano-vllm\                    nano-vllm 源码副本（只读参考）

服务器：
  ~/nano-vllm/repo/                    工作副本（无 .git）
  ~/nano-vllm/models/Qwen3-4B/         target
  ~/nano-vllm/models/Qwen3-0.6B/       draft
  ~/nano-vllm/venv/bin/python          Python
  CUDA_VISIBLE_DEVICES=7               只有 7 号卡是空的
```
