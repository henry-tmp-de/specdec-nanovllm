# Speculative Decoding on nano-vllm

在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（1385 行教学版 vLLM 复现，纯 PyTorch + Triton + FlashAttention）上实现**线性链投机解码**，并量化其收益边界。

**研究目标**：投机解码在什么条件下真正加速？加速比由哪些因素决定？

---

## 实测结果

RTX 3090 · Qwen3-4B (target) + Qwen3-0.6B (draft) · temperature=1.0

| 配置 | tok/s | 每步产出 token | 接受率 | 输出 distinct |
|---|---:|---:|---:|---:|
| baseline（无投机） | **17.0** | 1.00 | — | 41 |
| draft k=2 | 11.3 | 1.62 | 31.9% | 40 |
| draft k=4 | 10.4 | 2.28 | 36.5% | 37 |

**在 0.6B→4B 这个配比下，投机解码是净亏损。** 这不是实现问题——下节给出量化分析。

### 为什么亏损

每步 decode 必须把**全部模型权重**从显存读一遍：

```
baseline   : 1 × 8.0 GB（target）                    =  8.0 GB
draft k=2  : 2 × 1.1 GB（draft 跑两次）+ 1 × 8.0 GB  = 10.2 GB   ×1.28
draft k=4  : 4 × 1.1 GB（draft 跑四次）+ 1 × 8.0 GB  = 12.4 GB   ×1.55
```

**虽然每步多产出 1.62~2.28 个 token，但权重读取量增加得更多。**

通用形式：
```
S = (1 − α^(k+1)) / ((1 − α)(γ·c + 1))
  α = 接受率, γ = k, c = draft单步耗时 / target单步耗时
```

关键在于 **draft 要串行跑 k 次**，不能只看"draft 模型小"。

---

## 三个可验证的判断

### ① n-gram 投机解码为什么无效

n-gram 从历史文本检索候选，**它不是模型，没有真实分布**。把它的分布建模为单点分布后：

```
接受概率 = min(1, p_target / q_draft) = min(1, p_target / 1) = p_target
```

而 `p_target` 在 151936 词表上通常只有 0.001~0.06 → **接受率必然趋近于 0**。

实测对比（相同 target，同一批 prompt）：

| 提议来源 | 接受率 |
|---|---:|
| n-gram（无真实分布） | **0.6%** |
| draft model（有真实分布） | **31.9%** |

两者只差"草稿分布是不是真的"，接受率差**50倍**。

### ② 树状投机是过度设计

SuffixDecoding（arXiv 2411.04975, NeurIPS 2025）论文自己的消融：

| 基准 | linear | tree |
|---|---:|---:|
| AgenticSQL | **5.345×** | 5.175× |
| SWE-Bench | **2.452×** | ~2.15× |
| Spec-Bench | 1.659× | 1.661× |

而且 Snowflake 生产代码 `speculate(..., use_tree_spec=False)` 默认就是线性链。

**所以本实现只做线性链**，省掉了自定义 mask + 换 SDPA + 失去 FlashAttention 后端的一整套代价。

### ③ 验证阶段不需要树状掩码

验证阶段要一次 forward 算 `1+k` 个位置，它们的关系恰好是**标准因果顺序**（候选 j 能看到 prompt + 候选 0..j-1 + 自己）。因此可以直接复用 nano-vllm 已有的 chunked prefill 路径：

```
vLLM 的 _prepare_inputs 根本不问这是 prefill 还是 verify
一条 k+1 线性链就是一个 ragged varlen prefill chunk
唯一区别：position 从 num_computed_tokens 起步而非 0
```

**结果：`layers/attention.py` 的核心逻辑一行未改。**

---

## 无损性验证

投机解码不是"近似加速"，而是数学上保证输出分布与原模型一致。两种口径都做了验证：

**单元测试级**（16 项全过）——判据不用固定阈值，而看 TV 距离是否随样本量收敛到 0：
```
N = 40,000   平均 TV = 0.00355
N = 400,000  平均 TV = 0.00122
比值 2.91（理论 3.16）→ 无偏
接受率 = 0.320（理论 p[d] 均值 = 0.320）完全吻合
```

**端到端级**——token 频率分布对比：
```
baseline: n=120, distinct=57, top=[220:11, 198:7, 25:6, ...]
draft   : n=117, distinct=59, top=[220:23, 279:5, 18:5, ...]
```

> ⚠️ 注意：GPU 精度确实影响逐 token 一致性。实测同一条 token 单独算 vs 批量算，`max|Δlogit| = 0.25`，30 个位置中约 1 次（3.3%）argmax 不同。所以**「逐 token 严格一致」这个标准本身过严**，应该用统计判据。

---

## 实现要点

### KV cache：draft 与 target 共享 block_table

两者共用 `Sequence.block_table`（逻辑 block i → 物理块 i），但各自的 cache 是独立显存区。**块数必须相同**，所以不能简单分配，要按字节反推：

```python
N = 总预算 / (target_block_bytes + draft_block_bytes)
```

### 草稿 token 绝不进 token_ids

`block_manager.hash_blocks()` 从 `token_ids` 算前缀缓存哈希。若草稿混进去，被拒后 `token_ids` 变了而旧哈希还在 → 下次相同前缀命中错误缓存 → **静默输出错误 token**。

所以草稿单独存 `seq.draft_tokens`，验证通过后才写进 `token_ids`。

### 拒绝采样的 q_pad 技巧

给 q 末尾补一行零，让「全部接受」和「第 j 位拒绝」合并成同一表达式：
```
accepted == k 时，(p_k − q_k)+ 中的 q_k 变 0 → resid 退化成 p_k（bonus 分布）
```

### 单点分布不能用 logits 表达

`logits=[30,-30,...]` 过 softmax 后是 **0.9999 而非 1.0**，残余概率会污染修正分布 `max(0, p−q)`。
实测让接受率从 1.0 掉到 0.35。

---

## 改动范围

```
新增  nanovllm/spec_decode/         ~470 行   三个模块，可脱离引擎单测
改动  engine/model_runner.py        +~150    prepare_verify / run_verify / draft 加载
      engine/scheduler.py           +~50    batch 门控 / 一对多 postprocess
      engine/sequence.py            +~35    draft_tokens / append_tokens
      engine/block_manager.py       +~30    支持 1+k 槽位
      layers/attention.py           +~8     q 维度兼容
      layers/embed_head.py          +~5     跳过位置裁剪
      utils/context.py              +~2     is_spec_verify 标志
      config.py                     +~6     spec_k / spec_method / draft_model
```

约 600 行增量，8 个已有文件。全部用 `spec_k` 开关控制，默认关闭时与原版逐 token 一致。

**附带修复了一个上游 bug**：`hash_blocks()` 在 `num_cached_tokens` 推进**之前**被调用，登记进前缀缓存的哈希对应旧的完成量（上游 issue 待提 PR）。

---

## 运行

```bash
python tests/test_spec_decode.py     # 16 项单测，纯 CPU
python bench_final.py base 0 60      # 性能基准
python stat2.py draft               # 统计无损性检验
```

```python
llm = LLM(model, spec_k=4, spec_method="draft", draft_model=draft_path)
```

---

## 下一步

当前 0.6B→4B 配比为净亏损。三个方向：

- **EAGLE 式单层 draft head**：只跑 target 最后一层 + 轻量 head，`c` 从 0.14 降到 ~0.02
- **换更小的 draft**：权重比拉到 1:40+，但需权衡接受率下降
- **batch 门控实测**：扫 batch = 1~64 找出盈亏平衡点

---

*基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（MIT License）开发。*