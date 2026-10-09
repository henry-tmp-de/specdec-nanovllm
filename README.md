# Speculative Decoding on nano-vllm

在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（1385 行教学版 vLLM 复现，
纯 PyTorch + Triton + FlashAttention）上实现**线性链投机解码**，
并回答一个具体问题：**投机解码在什么条件下真正加速？**

RTX 3090 · Qwen3-4B (target) + Qwen3-0.6B (draft) · temperature=1.0 · `enforce_eager=False`

## 文档索引

| 文档 | 看什么 |
|---|---|
| **`docs/优化总览.md`** | **所有优化 + 实测效果 + 证据**（5 个 bug、三版 kernel 阶梯、方法论复盘、已知边界） |
| **`docs/面试考点分析.md`** | **172 道真实面经的考点分布**，逐桶列原题，判断该补什么 |
| `docs/面试考点-逐题分桶.md` | 上面那份的原始数据（每桶全部原题） |
| `HANDOVER.md` | 交接：环境、坑、下一步 |
| `nanovllm/kernels/README.md` | kernel 那条线的完整记录（含踩坑） |

---

## 结果

单条 prompt、生成 **256 token**、temperature=1.0：

| 配置 | tok/s | 每步 token | 接受率 | 相对 baseline |
|---|---:|---:|---:|---:|
| baseline（普通 decode + CUDA graph） | 82.3 | 1.00 | — | 1.00× |
| draft k=2 | 116.9 | 2.15 | 0.65 | **1.42×** |
| draft k=6 | **132.5** | 3.82 | 0.77 | **1.61×** |
| *（对照）draft k=6，不拍图（eager）* | *20.8* | *3.77* | *0.79* | *0.68×*† |
| *（对照）baseline，不拍图（eager）* | *30.6* | *1.00* | *—* | — |

† 相对 eager baseline（30.6 tok/s）。

### 这组对照才是重点

```
draft k=6，eager : 20.8 tok/s    ← 每步 3.77 个 token，仍然是【净亏损】
draft k=6，拍图  : 132.5 tok/s   ← 每步 3.82 个 token，加速 1.61×
   ↑ 每步产出几乎没变（3.77 → 3.82），变的只有 per-step 固定开销
```

**把符号翻正的不是接受率，也不是算法，是 CUDA graph（6.4 倍）。**
上一轮「实现正确但速度为负」，根因就在这里：两条前向都是 eager 时，
投机解码要串行多跑 k 次 draft 前向再加一次验证，**必然**比 baseline 慢 ——
跟权重读取量多少无关。

---

## 一、上一轮的结论是错的：瓶颈不是权重读取量

上一轮的报告说：

> 每步 decode 读全部权重，baseline 8.0 GB，draft k=2 要 10.2 GB，
> 读取量涨 27% → 净亏损 34%

**这个模型解释不了实测。** 4B 模型 fp16 权重 8 GB，3090 显存带宽 936 GB/s，
读一遍只要 **8.5 ms**；而实测每步要 **33 ms**。多出来的 24 ms 跟权重没关系。

同一个模型、同一次前向，只差一个开关：

| | 每步耗时 | tok/s |
|---|---:|---:|
| `enforce_eager=True` | 33.2 ms | 30.6 |
| 开 CUDA graph | **12.8 ms** | **79.6** |

**差 2.6 倍，而这 20 ms 里没有一个字节是权重读取，全是 kernel launch + Python dispatch。**

draft 侧更极端：0.6B 只有 1.2 GB 权重（理论 1.3 ms 读完），
eager 下实测一次前向 **24.8 ms** —— **19 倍**。draft 要串行跑 k 次，这笔开销要乘以 k。

```
旧模型（错）：加速比 ≈ f(权重比)
新模型（对）：加速比 ≈ f(每步固定开销)     ← batch=1 时的主导项
```

**结论：batch=1 的推理引擎，先把 per-step 固定开销压掉，再谈投机解码。**
本项目的做法：给 draft 前向和验证前向**各拍一张静态形状的 CUDA graph**。

---

## 二、四个真实 bug（都是「不报错、只是结果悄悄错」）

投机解码有一条硬性质：**输出分布必须严格等于目标模型的分布**。
这四个 bug 里有两个直接破坏它。

### Bug A：draft 侧传了概率，验证时又被 softmax 一次 → 无损性破裂

`verify_batch` 内部做的是 `q = softmax(draft_side / temperature)`，
所以 draft 侧必须传**原始 logits**。而 proposer 返回的是已经 softmax 过的概率。

结果是在 151936 词表上被二次 softmax，压成**近似均匀分布**：

```
实测：draft 真分布 max = 0.597
      再 softmax 后 max = 1.23e-05     （1/vocab = 6.6e-06）
```

草稿 token 是从真分布 q 采样的，验证却用 q̃ —— 拒绝采样的一致性前提被破坏。

**决定性证据**（构造 q ≡ p，此时 `min(1,p/q)=1`，正确实现必须全部接受）：

| draft 侧传什么 | 接受率 | TV 距离 |
|---|---:|---:|
| logits（正确） | **1.0000** | 0.0032（N=20 万，在收敛到 0） |
| 概率（原实现） | 0.8895 | **0.1075**（有偏，不随 N 下降） |

### Bug B：候选 j 被拿去和 target 第 j+1 行比（错位一位）

验证时送进 forward 的 k+1 个 token 在位置 `len-1 … len+k-1`，
模型第 r 行的输出是「位置 len+r 的 token」的分布 —— 所以**候选 j 对齐第 j 行**。
原实现写的是 `p[:, 1:k+1]`，整体挪了一行。

后果：候选 0 被拿去和「候选 1 的分布」比。本来高度一致的 0.6B/4B，
接受率被从应有的 **0.84 打到实测 0.16**。

> 现有单测查不出来：所有多行用例的 target 各行都是**完全相同**的
> （`clone()` / 整块同值 / `expand`），索引偏移多少都没差异。
> 补的合成算例（各行峰不同）第一次跑就暴露了。

### Bug C：全接受时把已经算好的 bonus 扔了

`verify_batch` 原来在无拒绝时把 bonus 置 -1。但那一行的目标分布
**在这一轮 forward 里已经算出来了**（`resid = p[k] − 0 = p[k]`），
正是 target 自己下一步会算的 token。

丢掉它，k=1、α=0.857 时每步恒定只出 1 个 token —— **必然比 baseline 慢**。
而 scheduler 每步预留的槽位本来就是 `need = 1 + spec_k`，
本来就是按「最多落地 k+1 个 token」设计的。

### Bug D：终止判据用 `==` 而不是 `>=` → 序列无限生成

投机一步落地多个 token，完成数会**跳过** `max_tokens`（62 → 65，`== 64` 永不成立），
序列就一直生成下去。实测 `max_tokens=64` 跑到 **300 步 / 572 token 还没停**，
在 benchmark 里表现为「draft 模式一卡就是几百秒」。

### Bug E：跨 block 时少分配一块 → CUDA 非法访存

所有写入路径（`prepare_decode` / `prepare_verify`）写的都是位置
`len-1 … len+n-2`（上一个 token 的 KV 要重算一遍），
但块数公式按 `len … len+n-1` 算空位：`len` 正好是 `block_size` 整数倍时，
它以为当前 block 空着，**其实已经写满** → 少分一块 → 那一位是填充值 `-1`
→ `cache_seqlens` 要跨块时 flash-attn 去读第 -1 页：

```
eager        : 读到别处的垃圾内存，不报错但 KV 是错的
拍成 CUDA graph: torch.AcceleratorError: an illegal memory access was encountered
```

64 token 的短生成永远碰不到 256 的边界，把 `max_tokens` 改成 256 立刻崩。

---

## 三、无损性怎么验

证据强度分三档，**别混着说**：

```bash
python tests/test_spec_decode.py              # 纯 CPU，18 项 —— 主证据
python tests/test_scheduler_terminate.py      # 纯 CPU，8 项（Bug D 回归）
python tests/test_block_alloc.py              # 纯 CPU，11 项（Bug E 回归）
python scripts/check_draft_quality.py         # 绕开引擎测 draft 固有质量
python scripts/check_lossless_engine.py       # 引擎级（弱，见下）
```

**① 单元测试级（强证据，主要靠这个）**

judge 不用固定阈值，而是看 TV 距离是否随样本量收敛到 0：

```
N = 40,000   平均 TV = 0.00355
N = 400,000  平均 TV = 0.00122
比值 2.91（理论 3.16）→ 无偏
```

外加两个合成算例直接钉死对齐关系：`q ≡ p` 时必须全部接受（实测接受率 1.0000）；
候选 `j` 必须对齐 target 第 `j` 行。这两条和引擎、GPU、CUDA graph 都无关。

**② draft 固有质量（独立于引擎）**

用 transformers 原生跑两个模型，测得 TV = 0.1655 → α 理论上是 0.8345。

**③ 引擎级分布对比**

四轮各 4800 token（temperature=1.0、随机采样）：

```
TV(base , base2 ) = 0.5315   基线噪声地板
TV(draft, draft2) = 0.6325   投机路径自己的地板
跨组 base x draft：0.514 / 0.521 / 0.671 / 0.713
```

**跨组差异完全夹在两个地板之间 —— 没有发现系统性偏差。**

★ 关键是**必须给投机路径单独测一个地板**：投机一步产出 3~4 个 token
（同一次前向、同一个上下文），序列自相关远强于逐 token 解码，
有效样本量更小、经验分布天然更散 —— 实测 `TV(draft,draft2)=0.63 > TV(base,base2)=0.53`
正好印证。拿 `TV(base,draft)=0.71` 直接跟 0.53 比，会得出「超出 34%、可能有偏」的**错误结论**。

局限：n=4800 时地板有 0.5 量级，功效有限，只能说「没发现偏差」。
要做成强证据得换设计（同一批 prompt 各只生成 1 个 token，样本独立）。

**三个方法论陷阱（都真踩过）：**

1. **别用低 temperature 做逐 token 对比**。`temperature=0.01` 会把
   `softmax(logits/0.01)` 压成 one-hot，p 恒为 1.0000，baseline 同样如此 ——
   会误判成「draft 分布退化」。
2. **随机采样下逐 token 对比没有意义**，要比 token 频率分布。
3. **GPU 精度确实影响逐 token 一致性**（实测 `max|Δlogit| = 0.25`，
   30 个位置里约 1 次 argmax 不同 ≈ 3.3%），所以「严格逐 token 一致」
   这个标准本身过严。同理，CUDA graph 路径与 eager 路径
   `max|Δlogit| ≈ 0.30` —— 那是 flash-attn 分块配置不同导致的浮点累加顺序差异，
   不是逻辑错误；判据要落在「分布无偏」上。

---

## 四、draft 模型能提供多高的接受率

接受率有恒等式：

```
α = E_{x~q}[ min(1, p[x]/q[x]) ] = Σ_x min(p[x], q[x]) = 1 − TV(p, q)
```

所以 α 完全由两个模型分布的 TV 距离决定。用 transformers 原生直接跑两个模型
（不经过引擎，排除引擎算错上下文），在相同上下文上比下一 token 分布：

| 指标 | 实测 |
|---|---:|
| 平均 TV 距离（0.6B vs 4B，60 个位置） | 0.1655 |
| argmax 一致率 | 86.7% |
| 理论接受率 α = 1 − TV | **0.8345** |

**Qwen3-0.6B 给 Qwen3-4B 当 draft，固有接受率约 0.84，是很好的组合。**
（这也反证了 Bug B：引擎里只测到 0.16。）

---

## 五、适用边界

**成立的条件：**
- **batch=1 或很小**。投机解码的收益来自「用一次 target 前向换多个 token」，
  batch 越大 target 前向的算力越吃紧，而验证要付 B·(k+1) 的算力。
  `config.spec_batch_threshold` 可设门控，超过就退回普通 decode。
- **有接受率足够高的 draft**。α 直接决定每步产出 token 数
  （E[tokens/step] = (1−α^(k+1))/(1−α)）；α < 0.5 时收益很快被 draft 开销吃掉。
- **draft 与 target 词表相同**，否则无法比较同一 token 的概率。

**不成立 / 需要额外工作：**
- **draft 的 per-step 开销没有降下来**。对照行里 `draft k=6 不拍图` 只有 20.8 tok/s
  —— 即使五个 bug 全修好，eager 下投机解码**仍然是净亏损**。
  把符号翻正的是 CUDA graph，不是接受率。
- **batch 很大时**。见上。
- **draft 与 target 分布差距大时**。α 是 TV 距离决定的，选 draft 之前先算 TV。
- **多序列共享前缀（前缀缓存命中）时**暂未验证：改写的 `may_append` 没有保留
  原版对「最后一个 block 被共享（ref_count>1）」的 copy-on-write 处理。

---

## 六、实现要点

**draft 与 target 共享 block_table，KV cache 按字节反推块数**

两者共用 `Sequence.block_table`（逻辑块 i → 物理块 i），但各自的 cache 独立。
块数必须相同，所以不能「给 draft 分一块」，要按字节算：
`N = 总预算 / (target_block_bytes + draft_block_bytes)`。

**草稿 token 绝不进 `token_ids`**

`hash_blocks()` 从 `token_ids` 算前缀缓存哈希。草稿混进去，被拒后 `token_ids` 变了
而旧哈希还在 → 下次命中错误缓存 → **静默输出错误 token**。
所以草稿单独存 `seq.draft_tokens`，验证通过才写进 `token_ids`。

**两张静态形状的 CUDA graph**

| 图 | 形状 | 收益 |
|---|---|---|
| draft 前向 | bs=1、每次 1 token | 24.8 ms → ~2 ms |
| 验证前向 | bs=1、query 恒 1+k、varlen | 36 ms → ~14 ms |

验证前向走 varlen，`run_model` 里 `is_prefill=True` 会强制走 eager 分支，
但它的**形状是固定的**，所以照样能拍图。形状不匹配时自动退回 eager。

**图路径里采样结果留在 GPU 上**

draft 的 k 步自回归中间不做 `.item()`，直接把 token 写回静态输入缓冲区，
只在最后 stack 成 list 时同步一次 —— 否则 k 步就是 k 次 pipeline stall。

**单点分布不能用 logits 表达**

n-gram 路线的 q 是单点分布，`logits=[30,-30,...]` 过 softmax 是 **0.9999 而非 1.0**，
残余概率会污染修正分布 `max(0, p−q)`（实测让接受概率从 1.0 掉到 0.35）。

---

## 七、改动范围

```
新增  nanovllm/spec_decode/         ~500 行  三个模块，可脱离引擎单测
改动  engine/model_runner.py        +~240    prepare_verify / run_verify / draft 加载
                                             / 两张静态 CUDA graph / KV cache 按字节切分
      engine/scheduler.py           +~55     spec_enabled 门控 / postprocess_spec 一对多
      engine/sequence.py            +~40     draft_tokens / draft_logits / append_tokens
      engine/block_manager.py       +~30     can_append/may_append 支持 1+k 槽位
      layers/attention.py           +~8      q 维度兼容 draft 单 token 路径
      layers/embed_head.py          +~5      is_spec_verify 时跳过位置裁剪
      utils/context.py              +~2      is_spec_verify 标志
      config.py                     +~14     spec_k / spec_method / draft_model
                                             / spec_batch_threshold / spec_cuda_graph
```

全部用 `spec_k` 开关控制，默认关闭时与原版逐 token 一致。

---

## 运行

```bash
python tests/test_spec_decode.py              # 23 项单测，纯 CPU
python tests/test_scheduler_terminate.py      # 终止判据回归，纯 CPU
python scripts/bench_spec.py draft 6 64 1     # 加速比基准（k=6，64 token，开图）
python scripts/bench_spec.py base  0 64 1     # 对照
python scripts/check_draft_quality.py         # draft 固有质量（需 GPU）
```

```python
llm = LLM(model, spec_k=6, spec_method="draft", draft_model=draft_path)
```

---

*基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（MIT License）开发。*
