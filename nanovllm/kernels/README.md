# Paged decode attention：从 roofline 到三个版本的实测

一句话：**decode 阶段的 attention 是纯访存瓶颈，所以优化只有两条路 ——
少搬字节、把带宽用满。我做了三步，每一步的动因都来自前一步的实测数据。**

环境和口径：RTX 3090、Qwen3-4B 的 attention 配置（32 q头 / 8 kv头 / head_dim 128 /
block_size 256 / bf16）、单条序列。
跑：`CUDA_VISIBLE_DEVICES=7 python scripts/bench_paged_decode.py`

---

## 一、先算清楚它到底是不是访存瓶颈

context = 1024 时：

```
要搬的字节（KV 各读一遍）= 1024 × 2 × 8头 × 128 × 2B   =  4.19 MB
要算的浮点（QKᵀ 与 PV）    = 4 × 32头 × 128 × 1024      = 16.78 MFLOP
算术密度 = 16.78e6 / 4.19e6                            =  4.0 FLOP/Byte
```

3090 的 roofline 拐点 = 35.6 TFLOP/s ÷ 936 GB/s ≈ **38 FLOP/Byte**。

**算术密度 4.0 只有拐点的 1/10 → 铁定的访存瓶颈。**
这意味着：算力有大把富余，你写的 kernel 快不快，只取决于它搬字节的效率。

⚠️ **参照系要选对**：936 GB/s 是规格值。实测（纯 copy 流式）：

```
纯 copy  32MB -> 815 GB/s
纯 copy 128MB -> 838 GB/s
纯 copy 512MB -> 842 GB/s     ← 这台卡实际的天花板
```

**下面所有百分比都用 842 GB/s 算**，用 936 会低估自己的效率。

---

## 二、三个版本，三次实测，两个不同的瓶颈

| ctx=4096 | 时间 | 程序数 | 搬的字节 | 实测带宽 | 占实测天花板 |
|---|---:|---:|---:|---:|---:|
| v1 每 (序列, q头) 一个 program | 192.4 µs | 32 | **67.1 MB**（4× 冗余） | 348 GB/s | 41% |
| v2 按 kv头 分组，共用 K/V | 112.2 µs | **8** | 16.8 MB | **149 GB/s** | **17.7%** |
| v3 split-K（把 context 切开并行） | **66.7 µs** | 64 | 16.8 MB | 251 GB/s | **29.9%** |
| （参照）flash-attn | 42.0 µs | — | 16.8 MB | 400 GB/s | 47% |

（v1 的带宽按它**实际**搬的 4× 字节算，才可比。）

> ⚠️ **这张表是墙钟时间**。对 v1/v2 它约等于 kernel 时间（启动开销只有 2µs），
> **但对 v3 是错的** —— v3 要启动两个 kernel，46µs 的 Python 启动开销把
> 31µs 的 kernel 盖成了 76µs。**真实情况见第三节**：v3 的 kernel 其实比
> flash-attn 快 1.31 倍。这个坑我踩了整整一轮才爬出来，详见 §三。

### v1 → v2：先把重复读消掉

GQA 是 32 q头 : 8 kv头 = 4:1。v1 每个 q头 各读一遍同一份 K/V ——
**同一个 kv头 的数据在显存里读了 4 次**。

证据（这条最漂亮）：v1 和 flash-attn 的**实测带宽几乎一样**（348 vs 400 GB/s），
但 v1 搬了 4 倍的字节、慢 4.6 倍。
→ **v1 不是「带宽用不满」，是「搬了不该搬的字节」。**

v2 让一个 program 负责一个 kv头、把同组 4 个 q头 一起算，K/V 只读一次。

### v2 → v3：把带宽用满（这一步的动因完全来自 v2 的数据）

v2 搬的字节少了 4 倍，却只快了 1.7 倍 —— 因为它的**带宽利用率掉到了 17.7%**。

原因：v2 只启动 `num_seqs × num_kv_heads` = **8 个 program**，而 3090 有 **82 个 SM**。
decode 每个序列只有 1 个 query，能并行的只有 head 维度，8 个 CTA 连一波都填不满。

v3 把 **context 方向也切开**（split-K / flash-decoding）：每个 (序列, kv头)
派 NUM_SPLITS 个 program 各算一段的局部 (m, l, acc)，再用第二个 kernel 按
online-softmax 的规则归约。并行度 = 8 × splits。

split 数扫描（ctx=4096）：

```
splits  程序数   时间      带宽      占天花板
     1      8   112.5us  149 GB/s   15.9%    ← 等于 v2
     2     16    73.7us  228 GB/s   24.3%
     4     32    68.6us  245 GB/s   26.1%
     8     64    68.0us  247 GB/s   26.4%    ← 饱和
    16    128    69.3us  242 GB/s   25.9%
    32    256    68.9us  244 GB/s   26.0%
```

**splits=2 就吃掉了大部分收益，8 以后完全平掉。** 这说明并行度只是从
「严重不足」补到「够用」，再往上加没有用 —— 剩下的差距不再是并行度问题。

---

## 三、★ 结论反转：v3 的 kernel 比 flash-attn 快 1.3 倍，我一开始量错了

### 3.1 先看最后的数据

用 `torch.profiler` 读**纯 kernel GPU 时间**，再用 CUDA Graph 把启动开销消掉：

| 变体 | 墙钟(us) | **纯 kernel(us)** | CPU 开销 | **+CUDA Graph** | Graph 后带宽 | 占实测天花板 |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 193.4 | 191.4 | 2.1 | 194.4 | 86.3 GB/s | 10.3% |
| v2 | 113.3 | 111.2 | 2.1 | 111.8 | 150.1 GB/s | 17.8% |
| **v3** | 76.1 | **31.4** | **45.9** | **32.0** | **523.8 GB/s** | **62.2%** |
| flash-attn | 42.9 | 41.2 | 1.7 | 40.9 | 410.2 GB/s | 48.7% |

**v3 的 kernel 是 31.4µs，flash-attn 是 41.2µs —— 快 1.31 倍。**
按带宽利用率算：**62.2% vs 48.7%**。

### 3.2 我之前错在哪

前面用「50 次连发 + CUDA 事件包住整段」测墙钟，得到 v3 = 76µs、只到天花板 29%，
于是开始逐个排查假设 —— **四个全被证伪**：

| 假设 | 怎么验 | 结果 |
|---|---|---|
| 并行度不足 | splits 从 1 扫到 32 | ❌ splits=2 就饱和 |
| 没有软件流水线 | `num_stages` 1 vs 3 | ❌ 一模一样 |
| warp 太少 | `num_warps` 4 vs 8 | ❌ 一模一样 |
| KV 访存模式太碎 | 转成 head-major 连续区间 | ❌ 反而慢 5% |
| block 粒度 | `block_n` 64/128/256 | ❌ 时间不变 |

**「改什么都不动」本身就是最强的信号 —— 说明被测的根本不是 kernel。**
real 原因：v3 要启动**两个** kernel（partial + combine），Triton 每次启动的
Python 侧开销 ~20µs，两次合计 **45.9µs**，把 31.4µs 的 kernel 整个盖住了。

CUDA Graph 一捕获，墙钟 76µs → **32µs**，和纯 kernel 时间吻合。

### 3.3 这段的真正价值：和一个更大的结论同构

**这个项目的主命题在两层都出现了：**

```
投机解码层：per-step 固定开销让 eager 33.2ms 变成 graph 12.8ms（2.6 倍）
kernel 层  ：启动开销 45.9us 盖住了 31.4us 的 kernel，看起来慢了 2.4 倍
```

**「小尺寸下，固定开销支配一切」** —— 这不是两个巧合，是同一个规律。
而且两次的解法也一样：**CUDA Graph**。

### 3.4 还能更好的地方（诚实记录）

- v2/v3 走 `tl.dot`，M 被补到 16（GQA 比只有 4），12/16 行白算 —— 但它吃的是
  寄存器和 shared memory，会影响能塞多少个 CTA；换成手写 CUDA 有优化空间。
- v3 需要两个 kernel，启动开销天然比 flash-attn 高一倍。**如果能合并成一个
  （用 atomic/semaphore 做跨 CTA 归约），连 CUDA Graph 都不需要。**
- `tl.trans(k)` 很可能多走了一趟 shared memory，Triton 层面看不到。

### ⚠️ 一次差点被骗过去的数据

布局实验第一次跑出来「两种布局一样快」，但输出差异是 **0.323608**（应该 ~0）——
自检抓到是我在 v3 分支里把 stride 写死了，没用到按 layout 算出来的值。
**没有这个对拍，我就会得出「布局无关」这个完全错误的结论。**
修完之后才拿到真实结果：head-major 反而慢 5%。

---

## 四、两个被数据抓出来的 bug（面试可以讲这个）

### 1. `context_lens.max().item()` 把流水线打断了

v3 第一版在 wrapper 里用 `.item()` 取最大 context 来算 chunk 大小 ——
那是一次 **device→host 同步**，每次调用都要 CPU 往返。

症状：**split 数扫描完全平的**（1→32 个 split，173→183µs）。
因为时间全被同步吃掉了，kernel 本身的优化根本显不出来。
改用 `block_table.shape[1] * block_size` 算静态上界，修掉后 splits=1 从 173µs → 112µs。

**教训**：在热路径上做任何 `.item()` / `.cpu()` / `.numpy()` 都会毁掉性能测量，
而且症状很有迷惑性（看起来像「优化无效」）。

### 2. chunk 算错，导致每个 split 都跑整个 context

```python
# ✗ 错：把「每份的位置数」又乘了一遍 block_n
chunk = max(block_n, triton.cdiv(max_ctx, splits) * block_n)
#   4096/32 = 128，再 ×128 = 16384 -> 每个 split 都去跑整个 4096 的 context

# ✓ 对：先按 splits 平分，再向上取整到 BLOCK_N 的倍数
chunk = max(block_n, triton.cdiv(triton.cdiv(max_ctx, splits), block_n) * block_n)
```

症状同样是「split 数扫描完全平的」，但这次是真的每个 split 在做重复劳动。
修掉后 ctx=4096 从 176µs → 67µs。

**教训**：这个 bug 只从「扫描曲线是平的」这一个信号看出来的 ——
如果只跑一个配置，永远发现不了。

### 附：一次判据过严导致的误判

初版正确性判据是「对 fp32 参照的相对误差 < 5e-3」，结果 v2/v3 在 ctx=7 判 FAIL。
查下来**不是 kernel 错**：v2/v3 走 `tl.dot`（bf16 tensor core），
bf16 的天然精度下限就在 1e-2 绝对误差量级，拿 5e-3 去卡本身就不合理。

改成两个参照后反而更清楚：

```
vs fp32 参照：v3 的误差 0.0005 ~ 0.002（bf16 的正常水平）
vs flash-attn：v3 吻合到 2e-3 以内（同为 bf16，同口径）
               —— 反而 fp32 的 v1 因为「算得更准」，离 bf16 的 flash-attn 更远
```

**教训**：判据的精度不该超过被测量的分辨率。

---

## 五、这段能回答哪些面试题

| 面试问法 | 用这里的什么回答 |
|---|---|
| 你写过什么算子？工作流程？ | paged decode attention：分页寻址 → online softmax 流式累加 → tensor core + split-K 两阶段归约 |
| 它的瓶颈是什么？效率如何？ | 算术密度 4.0 vs 拐点 38 → 访存瓶颈；**实测 523.8 GB/s = 实测天花板 842 的 62.2%，比 flash-attn 的 48.7% 高，kernel 快 1.31×** |
| 怎么优化的？ | 两步打两个不同的瓶颈：v1→v2 少搬字节（GQA 4× 冗余），v2→v3 提并行度（8→64 program） |
| 怎么知道该优化哪个？ | 看数据：v2 搬得少了但带宽掉到 17.9% → 说明瓶颈换成了并行度 |
| **怎么测一个 kernel 的真实性能？** | **墙钟 ≠ kernel 时间。Triton 一次启动的 Python 开销 ~20µs，两个 kernel 就 46µs，能把 31µs 的 kernel 盖成 76µs。要用 profiler 读 cuda_time，再和墙钟对照** |
| FlashAttention 的原理？ | online softmax 的 (m, l, acc) 递推；本 kernel 就是 decode 版的这个套路 |
| PagedAttention 的原理？ | block_table 把位置映射到物理块；这里用 `pos // block_size` + `pos % block_size` |
| 为什么 decode 和 prefill 的优化方向不同？ | prefill 是计算密集、decode 是访存密集，本 kernel 的 roofline 就是证据 |
| CUDA Graph 为什么有用？ | 启动开销在**小尺寸**下支配一切。这个项目在两层都撞到了同一个规律：投机解码 33.2ms→12.8ms，kernel 76µs→32µs |

---

## 六、文件

```
nanovllm/kernels/paged_decode_attn.py   kernel（v1/v2/v3 + fp32 参照实现）
scripts/bench_paged_decode.py           正确性对拍 + 性能 + roofline + 带宽天花板
scripts/ncu_paged_decode.py             ncu 驱动（本机被禁，见下）
scripts/run_ncu.sh
```

⚠️ **本机的 ncu 用不了**：`ERR_NVGPUCTRPERM - The user does not have permission to
access NVIDIA GPU Performance Counters`。所以拿不到 `dram__bytes_read` 这类硬件计数器，
roofline 的字节数是**理论推算 + 时间反推**（同一个 kernel 的「搬字节数 ÷ 时间」），
不是硬件读数。这一点在面试里要讲清楚，别说成「我用 ncu 量的」。
