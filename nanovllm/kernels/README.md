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
| 你用 CUDA 重写过 Triton kernel 吗？赢了多少？ | 赢了 1.09×（28.03 vs 30.66 µs）。**但更重要的是先量出了天花板**：写了个只读探针（访存逐字节相同、mma 全删）跑 23.08 µs，说明整个计算侧只值 2.79 µs —— 于是「M=16 白算 75%」这个假设被自己的数据证伪了（真去写了个 M=4 的 FFMA 版，反而慢 1.84×）。 |
| 那你这次优化到底靠什么拿到的？ | 靠 Triton 表达不了的两件事：**手写 shared memory 布局 + 流水线**。`cp.async` 三级流水一步就值 1.43×；XOR swizzle 消掉 ldmatrix 的 8 路 bank conflict。不是靠「少算」。 |
| 怎么知道一个 kernel 还能不能更快？ | 写一个**只保留访存、删掉全部计算**的孪生 kernel。它和真实 kernel 的差距就是「计算侧的开销」，也是优化的上界。这一步比任何 profiling 工具都直接（本机 ncu 还被禁了）。 |
| 遇到过最阴的 bug？ | 两个：(1) ldmatrix 行索引漏加 `warp*PW`，导致「ctx≤7 全对、ctx≥256 全错」；(2) q_sm 跨线程写后少一个 `__syncthreads()`，症状像精度问题（差 1e-2）其实是 race。都是**只跑一个配置就永远发现不了**的那种。 |

---

## 六、CUDA 版：把 Triton 表达不了的东西手写出来

**结论先说：赢了，但只赢 1.09×（28.03 µs vs 30.66 µs）。
而这次最重要的产出其实是那条「只读地板」，它把一个很诱人的假设直接证伪了。**

文件：`paged_decode_attn_cuda.cu` / `.py`（Ampere sm_86 深度实现）

### 6.1 用上了哪些 Ampere 特性

| 特性 | 怎么用的 |
|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | QKᵀ 和 PV 全走张量核 |
| `ldmatrix.sync.aligned.m8n8.x2.shared.b16` | K 用**非转置**、V 用**转置**（fragment 映射在 .cu 里有完整推导） |
| 手写 XOR swizzle | `physical_chunk = c ^ (row & 7)`，消 ldmatrix 的 8 路 bank conflict |
| `cp.async.cg` 多级流水 | 双/三级缓冲，把下一块 K/V 压在 mma 下面 |
| `__threadfence()` + `atomicAdd` | split 之间的归约由**最后一个到达的 CTA** 顺手做掉，省掉第二个 kernel |

（sm_86 不支持 TMA / 2-SM cluster / TMEM，都没碰。）

### 6.2 最终数字（ctx=4096、batch=1、纯 kernel 时间、取 3 次中位数）

| | 纯 kernel 时间 | 带宽 | 占实测天花板 842 |
|---|---:|---:|---:|
| **Triton v3**（partial+combine 两个 kernel） | 30.66 µs | — | — |
| flash-attn（对照） | 40.80 µs | 411 GB/s | 48.8% |
| **CUDA 只读探针**（访存逐字节相同、mma 全删） | **23.08 µs** | **727 GB/s** | **86.4%** |
| CUDA 主 kernel，st1（不用 cp.async） | 36.91 µs | 455 GB/s | 54.0% |
| CUDA 主 kernel，st2（双缓冲） | 27.63 µs | 607 GB/s | 72.1% |
| **CUDA 主 kernel，st3（三级流水）** | **25.87 µs** | **648 GB/s** | **77.0%** |
| CUDA 完整（st3，combine 另起 kernel） | 28.27 µs | — | — |
| **CUDA 完整（st3，融合归约）** | **28.03 µs** | — | **1.094× vs Triton** |

**每一步优化值多少（这就是那张阶梯表）：**

```
不用 cp.async  ->  三级流水 cp.async     36.91 -> 25.87 us   1.43x   ← 最大的一步
双缓冲        ->  三级流水                27.63 -> 25.87 us   1.07x
combine 另起  ->  threadfence 融合归约    28.27 -> 28.03 us   1.009x  ← 几乎白干
```

按 context 长度铺开（CUDA 都取当日最优配置）：

| ctx | Triton v3 | flash-attn | CUDA | vs v3 |
|---:|---:|---:|---:|---:|
| 256 | 9.14 µs | 12.82 | **7.39** | **1.24×** |
| 512 | 9.31 | 13.33 | **7.79** | **1.20×** |
| 1024 | 9.99 | 14.63 | 10.00 | 1.00× |
| 2048 | 20.94 | 27.73 | **17.82** | 1.17× |
| 4096 | 30.66 | 40.81 | **28.01** | 1.09× |

（ctx=512 那格是 `st1` 更快——小尺寸下 `cp.async` 的三级流水还没热起来就结束了，
所以按 ctx 各自取最优配置。其余各格都是 `s8/n64/w4/st3 + 融合归约`。）

**赢在哪**：全程都赢，ctx 越大越稳（数据量大、固定开销摊薄）。
**输在哪**：ctx=1024 基本打平 —— 那时总共只有 4.19 MB，两边的 kernel 都还没跑热，
十来个 µs 里全是启动/收尾，谁也没有优势。这也是这个算子的性质：**小尺寸下固定开销支配一切**
（和 §3.3 那个结论同构）。

### 6.3 ★ 核心假设被证伪：M=16 的白算不是瓶颈

出这道题时的假设是：

> Triton 的 `tl.dot` 要求 M ≥ 16，而 GQA 比只有 4，所以 75% 的 MMA 行是白算的。
> CUDA 可以正好算 4 行 —— 这是 Triton 表达不了、CUDA 能表达的地方。

**我把「正好算 4 行」的版本写出来了**（`paged_decode_ffma_kernel`，完全不用张量核，
lane 各管 4 个 dim + 5 步 butterfly 归约），结果：

| | 纯 kernel 时间 |
|---|---:|
| mma 版（M 补到 16，75% 白算） | **25.87 µs** |
| FFMA 版（正好 M=4，零 padding） | **47.73 µs** |

**正好 4 行的版本慢 1.84 倍。** 寄存器 96~128、几乎无 spill（ptxas -v 核过），
所以这不是「实现太糙」导致的，是结构性的：

> **张量核的价值不在算力，在于它把 head_dim 方向的归约做掉了。**
> FFMA 版要自己用 shuffle 做归约 —— 每个 (位置, q头) 要 5 步 butterfly，
> 一个 warp 每 8 个位置就是 160 条 shuffle。而归约走张量核是零指令。

还有一条更硬的证据 —— **只读探针**：

> 我把访存部分（同样的分页寻址、同样的 `cp.async` 流水、同样的 grid）原样保留，
> **把 mma 和 softmax 全部删掉**，只留拷贝。它跑 **23.08 µs**。
> 也就是说：**整个张量核 + softmax + 同步加起来只值 2.79 µs（10.8%）。**

M=16 白算的是**张量核的吞吐**，而张量核的吞吐本来就有大量富余（roofline 拐点 38，
这个算子算术密度只有 4）。把 4 倍的算力浪费丢掉，最多也只能碰那 2.79 µs 里的一小块 ——
而为了丢掉它，你反而要引入更多指令（shuffle）。**这笔账是亏的。**

所以：**这个假设在「访存瓶颈」这个前提成立时是站不住的。** 真正决定快慢的是
「搬字节的效率」，而 CUDA 相对 Triton 的优势也不在 M 方向，在**能精确控制 shared memory
布局和流水线**（13.9 个百分点的带宽就是这么来的）。

### 6.4 踩的坑（都是真踩过的）

**1. `ldmatrix` 的行索引忘了加 warp 偏移。**

K/V tile 是整个 CTA 共用的，每个 warp 只该读自己那 PW 个位置的行。
我写成了 `row = nt*8 + (r&7)`，等于**所有 warp 都去读第 0..16 行** —— 只有 warp 0 是对的。

症状极具迷惑性：`ctx=1` 和 `ctx=7` **全对**（那两档下只有 warp 0 有有效位置，
其余 warp 的 p 全被 mask 成 0，读错也看不出来），`ctx≥256` 直接崩。
**光看小 case 会以为全对。** 修：`row = warp*PW + nt*8 + (r&7)`。

**2. 一个真 race：q_sm 是别的线程写的，ldmatrix 直接就读了。**

只在 Q 写入之后、`aq` 的 ldmatrix 之前漏了一个 `__syncthreads()`。
症状是「大部分 ctx 都对、个别 ctx 差 1e-2」，**看起来完全像精度问题**，
其实是读到没写完的 shared memory。加一句 `__syncthreads()` 后 0.0294 → 0.00098。

**3. combine kernel 慢到 8.0 µs —— 串行依赖链，不是带宽问题。**

独立出来的归约 kernel 第一版：

```cpp
for (int s = 0; s < splits; ++s) M = fmaxf(M, pm[(base + s) * QPK + h]);   // ✗
```

`splits` 是运行期变量，编译器不展开 → 一条 load、一条 max、再一条 load 全串起来，
16 个 split 就是 16 次显存往返 ≈ 16 × 600ns ≈ 9.6 µs。**小 kernel 上「延迟」比「带宽」重要得多。**

改法两条：(1) 把 SPLITS 做成模板参数让 `#pragma unroll` 全展开；
(2) 每个 lane 用 `float4` 读 4 个连续 dim，一个 warp 一次读满 512B 连续，
避免「每线程跨 2KB 步长读 16 个标量」的 sector 放大。**8.0 µs → 3.3 µs。**

**4. sm_86 每个 block 只能用 99KB shared memory —— 流水线深度是被它卡住的。**

`(block_n=64, 3 级)` 的 K/V 缓冲 = 96 KB，再加跨 warp 归约要的 8 KB 就超了。
解法：归约缓冲直接**叠在已经用完的 K stage 上**（循环结束后 K 的 stage 不再用），省 8 KB。
`block_n=128` 因此只能跑单缓冲（cp.async 就上不了），实测也更慢。

**5. 融合归约几乎没有收益（1.009×）。**

用 `__threadfence()` + `atomicAdd` 让最后一个 CTA 顺手做归约，省掉第二个 kernel 的启动 ——
理论上应该省 2~3 µs，实测只省 0.24 µs。因为**那个 CTA 是最后完成的一批**，
归约的活落在关键路径尾巴上，和单独起一个 kernel 的费用基本抵消。

### 6.5 没做成的：warp specialization

按计划实现了生产者/消费者分离的版本（4 个消费者 warp 做 mma、4 个生产者 warp 只发
`cp.async`，用命名 barrier `bar.arrive` / `bar.sync` 做「stage 就绪 / stage 空闲」握手，
见 `paged_decode_ws_kernel`）。

**它跑得起来、不挂死，但结果不对**：`ctx=1` 就能复现，误差 2.5 量级，
而且**每次跑错的地方还不一样**（竞态）。修了 4 轮（补 CTA 级 `__syncthreads` 防 k_sm 被复用、
`__threadfence_block()` 两侧加栅栏、尾巴那几轮 group 计数改成保守 `wait_all`）
都没解决，最后停手。

**为什么没有继续投入**：只读探针已经给出上界 —— 访存路径（用的还是现在这套
`__syncthreads` 结构）能跑 23.08 µs，而完整 kernel 是 25.87 µs。
**warp specialization 既不能让访存快过 23.08，也不会减少计算量**，天花板就是那 2.79 µs。
所以这条路的上限本来就只有 ~10%，是个低价值目标。

### 6.6 口径说明（别被数字骗了）

* 所有时间都是 `torch.profiler` 的 `self_device_time_total`，**不是墙钟**。
  Triton 每次启动的 Python 开销 ~20µs，用墙钟会把结论带偏（§三 就是这么踩过来的）。
* 每一个数字都是**跑 3 次取中位数**，三次之间抖动 < 0.1% —— 这台卡上很稳。
* 带宽分母用**实测天花板 842 GB/s**（纯 copy 流式），不是规格值 936。
* 本机 **ncu 被禁**（`ERR_NVGPUCTRPERM`），拿不到 `dram__bytes_read`，
  所有带宽都是「字节数 ÷ 时间」反推的。roofline 那节也一直是这个口径，不要写成「ncu 量的」。

---

## 七、文件

```
nanovllm/kernels/paged_decode_attn.py        Triton kernel（v1/v2/v3 + fp32 参照实现）
nanovllm/kernels/paged_decode_attn_cuda.cu   CUDA kernel（mma/ldmatrix/swizzle/cp.async
                                             + FFMA M=4 对照版 + 只读探针 + warp spec 尝试）
nanovllm/kernels/paged_decode_attn_cuda.py   CUDA 版 Python 入口（JIT 编译）
scripts/bench_paged_decode.py                Triton：正确性对拍 + 性能 + roofline + 带宽天花板
scripts/bench_cuda_paged_decode.py           三方对比：CUDA vs Triton v3 vs flash-attn
scripts/sweep_cuda_paged_decode.py           CUDA 配置扫描（block_n × warps × stages × splits）
scripts/probe_kv_access.py                   KV 访存模式探针
scripts/ncu_paged_decode.py                  ncu 驱动（本机被禁，见下）
scripts/run_ncu.sh
```

跑 CUDA 版：`CUDA_VISIBLE_DEVICES=<空卡> python scripts/bench_cuda_paged_decode.py`

⚠️ **本机的 ncu 用不了**：`ERR_NVGPUCTRPERM - The user does not have permission to
access NVIDIA GPU Performance Counters`。所以拿不到 `dram__bytes_read` 这类硬件计数器，
roofline 的字节数是**理论推算 + 时间反推**（同一个 kernel 的「搬字节数 ÷ 时间」），
不是硬件读数。这一点在面试里要讲清楚，别说成「我用 ncu 量的」。
