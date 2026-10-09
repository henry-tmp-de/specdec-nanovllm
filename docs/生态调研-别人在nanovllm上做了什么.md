# 生态调研：别人在 nano-vllm 上做了什么

> 调研时间：2026-10-07。来源：GitHub API 全量检索 `nano-vllm in:name`（343 个结果）+ 逐个核实，
> 以及 vLLM / SGLang 官方文档的特性对照。
> **目的：避免重复造轮子，以及知道"哪些方向已经饱和"**。
>
> ⚠️ 本文件是**访谈式汇总**，不是逐条 200 状态码核验过的引用。引用具体项目前请自行打开确认。

---

## 一、上游项目现状

| | |
|---|---|
| 仓库 | `GeeeekExplorer/nano-vllm` |
| Star / Fork | **15,726 ★ / 2,663 fork** |
| 创建 / 最后 push | 2025-06-09 / **2026-04-26**（之后基本停更） |
| PR 总数 | 约 205，**merged 仅 20 个**；open issues 99 |

**上游真正合并的只有：chunked prefill（#204/#218）+ 一堆 bug 修复 + 零散微优化。**
**量化 / LoRA / MoE / 采样扩展 / server —— 一个都没合并。**

> ⚠️ 2026-04 之后的新 PR 里有一批明显是 AI agent 批量生成的（同一作者同时提 3 组几乎重复的 PR），
> 评估时需打折。

**结论：这是个教学仓，上游不打算做成生产引擎。所有功能扩展都在 fork 里。**

---

## 二、社区重复做得最多的方向（= 已经饱和）

| 方向 | 独立实现数 | 代表 |
|---|---|---|
| **量化**（FP8 权重 / FP8 KV / INT8 KV） | **≥6** | `cuber726579/Nano-VLLM-Quant`(FP8 权重+FP8 KV)、`pzsacc/nano-vllm-lite`(FP8 KV)、`naalo2/nano-vllm-kv-compression`(int8 KV)、`dzhengAP` PR#184(int8 KV) |
| **投机解码** | **≥5** | `limei1221/nano-vllm-speculative-decoding`、`tianyuantong/nano-vllm-speculative-decoding`、`banfeb/nano-VLLM-MS2`(n-gram)、`ovshake`、PR#147/#266 |
| **MoE** | **≥6** | `ZeroKernel798/nano-vllm-moe`、`banfeb/nano-VLLM-MS2`、`Hang-get` PR#251、`2419322417/GLM4_MOE` |
| **非 CUDA 后端移植** | **≥8** | 昇腾 `linzm1007/nano-vllm-ascend`(143★) + 3 个小仓、Rust `ssvgopal/nano-vllm-rs`(38★)、C++ `GaoXiangYa/nano-vllm.cpp`、Java `raydac/nano-vllm-java`(5★)、Metal、XPU、Jetson、CPU |
| **Kernel 重写** | **≥5** | `Wenyueh/MinivLLM`(1,060★,自带 paged+flash attention)、`52Hz-kiko/mini-vllm-qwen3-serving`(105★, **Triton Split-KV GQA PagedAttention**)、`cosmoliu2002/nano-vllm-triton`(14★) |
| **多模态** | ≥4 | `Sakana-31/nano-vllm-Multimodal`、`86MaxCao/nano-vllm`(Qwen3-VL) |
| **OpenAI 兼容 server** | ≥4 | Sakana-31、`Pla-Yer/Start-from-Nano-VLLM`、linzm1007 |

---

## 三、★ 和我们直接相关的几个（务必先读，避免撞车）

### 3.1 `limei1221/nano-vllm-speculative-decoding`（15★）—— 和我们的投机解码高度重合

- **draft model 路线**，约 560 行 / 10 文件：Propose → verify → rollback → reconcile 全链路
- 双 KV cache（draft 独立分页池）、单次 verifier、**verify-mode CUDA graph**（按 `B·(K+1)` 形状捕获）
- 实测（RTX 4090，Qwen3-8B + Qwen3-0.6B，K=5）：

| batch | 加速比 |
|---|---:|
| 1 | **1.85×** |
| 8 | 1.30× |
| 64 | **0.59×** |
| 256 | **0.57×** |

**→ 他们的 batch 数据正好补上我们一直没测的那一块（我们是 batch=1 的 1.61×）。
面试如果被问「并发下投机解码还成立吗」，这是现成的参照。**

### 3.2 `52Hz-kiko/mini-vllm-qwen3-serving`（105★）—— 名字不像 fork，但源码是 nanovllm 风格

- **Triton Split-KV GQA PagedAttention Decode Kernel**（和我们 v3 同一条路）
- 基准（RTX 4070 Laptop、Batch=1、ctx=4096、Q/KV Heads=64/8、FP16）：

```
旧 Decode Kernel        2.431 ms
Split-KV PagedAttention   110.9 µs
加速比                   21.93×
```

⚠️ **那个 21.93× 是相对他们自己的旧 kernel（很朴素），不是 flash-attn。**
按各自卡的带宽归一化，他们的 kernel 效率是 151 GB/s（4070 Laptop 峰值约 256 GB/s，约 59%），
**比我们的 77% 低**。绝对值 110.9µs vs 我们的 25.9µs 也受硬件影响，不可直接比。

### 3.3 `pzsacc/nano-vllm-lite`（33★）—— 做小算子融合的那条路

- CUDA **Add+RMSNorm 融合 kernel**（带宽 3-4× vs eager）、in-place RoPE、**FP8 KV cache**（靠 Triton 重写 paged attention decode 实现）、decode-first chunked prefill
- 实测（RTX 5090）：CUDA Graph 让 8 并发 decode TPOT 从 36.30 → 3.90 ms（9.3×）；prefix cache 命中率 90% 时 TTFT −67%；FP8 KV 让 4096 上下文容量 61 → 123 seqs（2×）

### 3.4 知乎「白牛」（tpoisonooo，231 赞）—— **一条负面结论，值得记住**

> 他试过**手写 Triton kernel 去超越 `torch.compile` 的 cutlass rms_norm，失败了**。
> 原因是 Qwen3 这里 hidden(C) 小、token_num(N) 大，简单 Triton kernel 在该 shape 上表现不佳。

**→ 这解释了为什么「小算子融合」那条路不是随便写写就能赢的。**
（我们在 attention 这个**大**算子上走通了，因为那是访存瓶颈、且形状不同。）

---

## 四、★ 社区**很少**碰的方向（= 还有空间）

| 方向 | 现状 |
|---|---|
| **LoRA** | **未找到任何实现** |
| **top-p / top-k 采样** | 只有 1 个 PR（#260），上游 sampler 只有 temperature |
| **前缀缓存调度 / 淘汰** | 只有 2 个小仓：`KawHimmy`(3★，cache-aware prefill scheduler)、`AEM001`(2★) |
| **结构化输出** | 1 个仓（`wtr0504`，1★） |
| **抢占 swap（换出到 CPU）** | 1 个仓：`chengy-sysu/nano-vllm-swap`(2★) |
| **PD 分离** | 有仓声称做了（`Sebastian-dong/nano-vllm-updatev1`），**但 README 只有一行、无实现说明、无 benchmark** —— 可信度低 |
| **RadixTree 前缀缓存** | 1 个仓：`RealJosephus/radix-turn-aware-nano-vllm`(12★)，且自述"只提供 KVCache 最小实现" |

⚠️ **`chengy-sysu/nano-vllm-swap` 的一条实测值得看**：A100-40GB + Qwen3-0.6B，
swap 配置 509 tok/s vs 纯 recompute 560 tok/s → **只有 0.91×**（小模型 prefill 重算便宜，
PCIe 拷贝固定开销反而亏）。**作者预期 7B+ 大模型、长序列才划算。**
→ 如果要做 swap，这直接说明「要选对场景才有收益」。

---

## 五、vLLM / SGLang 有而 nano-vllm 没有的（按官方文档核实）

### 5.1 调度 / batching
- **Multi-step scheduling**（一次调度跑多步 decode，摊薄 CPU 调度开销）
- **Overlap / zero-overhead scheduler**（SGLang 代表性：调度器提前一拍，CPU 调度与 GPU 计算重叠）
- **Cache-aware 调度（LPM 最长前缀匹配优先）** ← SGLang
- **优先级调度**
- **Dual Batch Overlap (DBO)**

### 5.2 KV cache
- KV 量化（FP8/INT8/NVFP4）
- **KV 换出到 CPU / 分层卸载**（vLLM Offloading Connector；SGLang HiCache 三级：HBM/CPU/外部存储）
- PD 分离 + KV 传输连接器（NIXL / Mooncake / Moriio）
- **Session/context caching**（SGLang：session_id 软保护，长多轮 Agent 命中率提升）
- 混合/稀疏 KV 管理（vLLM hybrid_kv_cache_manager / hisparse）

### 5.3 注意力 / 算子
- FlashInfer 后端（nano 只有 flash-attn）
- **CUDA Graph 的 PIECEWISE / FULL_AND_PIECEWISE 分档**（vLLM `-O1`/`-O2`）
- 自定义 all-reduce（绕 NCCL 的小消息低延迟通信）
- Fused MoE、量化 GEMM（Marlin / W4A16）

### 5.4 并行
- PP / DP / CP / EP（nano 只有 TP）

### 5.5 采样
- top-k / top-p / min-p、repetition penalty、logprobs、best_of/n、beam search、
  **结构化输出（XGrammar / Outlines）**、确定性推理（batch-invariant kernels）

### 5.6 服务
- OpenAI 兼容 server、SSE streaming、async engine、metrics/Prometheus、多 LoRA 热挂载、sleep mode

### 5.7 ★ 三条对 nano-vllm 的**具体**源码级观察（已核实）

1. **chunked prefill 是"部分实现"**：`nanovllm/engine/scheduler.py:51` 注释写明
   `only allow chunked prefill for the first seq` —— 只有队首请求可被切块。
2. **前缀缓存没有淘汰策略，也没有 CoW**：`block_manager.py` 用 `free_block_ids = deque`
   （FIFO）+ xxhash 链；块只在 `ref_count==0` 时释放，无 LRU/LFU/SLRU，无共享块写时复制。
3. **block_size 是 256**（`config.py:17`，且 `:39` 断言必须是 256 的倍数），
   而 vLLM 默认 16、SGLang 可到 1 —— 这是"映射表开销 vs 内部碎片"的经典权衡素材。

---

## 六、一句话总结

**这个生态里已经没有「没人做过」的便宜可捡了** —— 15.7k star、2663 个 fork。
量化 / 投机 / MoE / 移植 / kernel 都有多个独立实现。

所以差异化只能来自：**做得更深**（同样的方向走到别人没走的深度）、
**做得更严谨**（有对照实验、有被证伪的假设、有边界量化），
**或者选那些"有人碰但做得浅"的方向**（见第四节）。
