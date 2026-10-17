# P16：投机路径 CUDA Graph 的「未精确捕获 batch size」向上取整（padding）

> 分支 `p16-graph-pad`（本地，未 push）。日期 2026-10-17。
> 一句话：**把「B=3 拿不到 B=3 的图 ⇒ 整个退回 eager」变成「向上取整到 B=4 的图 + 空行补齐」**，
> 默认关（`spec_graph_pad=False`），关着时逐字节等于改动前。

---

## 1. 问题（已定位，不是重新找）

投机路径的两张 CUDA Graph 都是**精确匹配** batch 的：

| 图 | 索引 | 代码位置 |
|---|---|---|
| 验证前向图 | `verify_graphs[(B, k)]` | `nanovllm/engine/model_runner.py` `run_verify_forward()` |
| draft 前向图 | `draft_graphs[B]` | `nanovllm/spec_decode/draft_proposer.py` `_forward_group()` |

而 `config.spec_graph_bs = (1, 2, 4)` 只捕获这三个桶 ⇒ **B=3 的混合 step 整个退回 eager**。

实测（`scripts/p16_graphpad_probe.py`，3 条 decode + 1 条 16K prefill 的混合批）：

```
verify_graph_fallbacks = {"shape_not_captured(B=3,k=6)": 18~23}   ← 多次运行 18/20/21/22/23，其中一次正好 22
draft 前向退回 eager    = 108 ~ 138 次
```

★ 这不是 P15 引入的，是 `spec_graph_bs` 这条**既有配置边界**（见 `docs/P15重叠执行混合批-开发记录.md` §412）。
原版 decode 早就用同一套办法绕开了它（`run_model` 里的 `next(x for x in self.graph_bs if x >= bs)`），
投机路径当时没跟上。

---

## 2. 做法

### 2.1 桶选择：向上取整，正好命中就用自己

`nanovllm/spec_decode/graph_pad.py::pick_bucket(b, buckets)`：返回 **>= b 的最小桶**；
b 比所有桶都大时返回 `None`（**必须退回 eager**，不能硬塞 —— 真实行数超过图的行数就是写越界）。

```
buckets=(1,2,4):  B=1→1(pad 0)  B=2→2(pad 0)  B=3→4(pad 1)  B=4→4(pad 0)  B=5→None(退回 eager)
```

这正是 vLLM `_compute_bs_to_padded_graph_size` 的语义（**正好命中 → 映射到自己，零浪费；落在缝隙才上取整**）。

### 2.2 两个调用点

| 位置 | 改动 |
|---|---|
| `model_runner.run_verify_forward()` | 精确匹配失败且开关打开时，取同一 `k` 的最近桶，多出来的行用空行补齐；**只把前 `B*(k+1)` 行交出去** |
| `draft_proposer._forward_group()` | 精确匹配失败且开关打开时，取最近桶，多出来的行用空行补齐；返回 `gv["logits"][:B]` |

★ **返回值必须切到真实行**：`run_verify` 用**真实**的 `cu_seqlens_q` 切分 logits，并断言
`cu[-1] == logits.shape[0]` —— 整块返回会让这个断言直接炸。

★ **不适用 padding 的形状**（按任务书要求，继续退回 eager，一行没动）：
`ragged_draft_len`（每条请求的候选数不齐）—— 那是「每条请求要算几个」**真的变了**，
不是 batch 维度的问题，补零只会污染拒绝采样。

### 2.3 开关与默认

```python
# config.py
spec_graph_pad: bool = False    # 默认关 ⇒ 老行为逐字节不变
```

运行期也能翻（`mr.graph_pad` / `mr.spec_proposer.graph_pad`），同一份二进制里做消融。

---

## 3. 空行的安全处理（★ 空行不是「什么都不做」，它会被真的算一遍）

### 3.1 四个位置，四个不变量

| 位置 | 填什么 | 为什么 | 常量 |
|---|---|---|---|
| `slot_mapping` | **-1** | `store_kvcache_kernel` 见到 -1 直接 `return`，一个字节都不写 | `PAD_SLOT=-1` |
| `block_table` | **0**（块 0） | 块 0 永远落在预分配的 KV cache 张量里 ⇒ 读它合法；留 -1 会读第 -1 页 | `PAD_ROW_BLOCK=0` |
| `cu_seqlens_k` / `context_lens` | **0**（归零） | 空行读不到任何 key（与 vLLM 的 `seq_lens=0` 一致） | `PAD_CTX_LEN=0` |
| `input_ids` / `positions` | **0** | 合法值；算出来的 logits 丢弃 | `PAD_TOKEN=0` |

★ 硬约束：`cu_seqlens` 是**前缀和**，必须**单调不减**（flash-attn 的硬要求）。
空行是在真实总和之上继续累加（步长 0 ⇒ 等值台阶），天然单调。

### 3.2 与 vLLM 的四件套对照（coordinator 提供的清单）

| vLLM | 本实现 | 说明 |
|---|---|---|
| `slot_mapping` 尾部 -1 | ✅ 一致 | |
| `block_table` 尾部 0 | ✅ 一致（值相同） | **但语义不同**，见 §3.4 |
| `seq_lens` 尾部 0 | ✅ 一致 | 验证侧走 `cu_seqlens_k` 台阶；draft 侧 `context_lens=0` |
| `query_start_loc` 尾部 = 最后一个累计值（q 长度 0） | ⚠️ **默认不采用** | 见 §3.3 |

### 3.3 「空行长度归零」是**实测**过的，不是照抄

任务书原来的注释写着「`cu_seqlens=[0,0]` 会让 varlen 内核退化成长度 0」，所以
「尽量归零，会崩就给小的合法值，并如实记录」。**我们实测了三种形状**（`scripts/p16_pad_variant_probe.py`，
RTX 3090 / flash-attn 2.8.3.post1 / CTX=4096 / B=3→4 / k=6）：

| 变体 | q 长度 | k 长度 | 崩不崩 | vs eager max\|Δ\| | argmax 一致 | 三变体互相对拍 |
|---|---|---:|---|---:|---:|---|
| A（原实现） | nq | 1 | 不崩 | 1.1719 | 100% | — |
| **B（现默认）** | nq | **0** | 不崩 | 1.1719 | 100% | **A ≡ B 逐位相同** |
| C（vLLM 式） | **0** | **0** | 不崩 | 1.1719 | 100% | **A ≡ C 逐位相同** |

draft 侧同样实测：`context_lens=0` 与 `=1` **逐位相同**（max\|Δ\| = 0.0，且确认走的是图路径 `padded=6`）。

**结论：归零安全，采用 B**（k=0、q 保持 nq）。理由：
- k=0 时空行读不到任何 key，块 0 的内容根本用不上 —— 这一项与 vLLM 对齐；
- q 保持 nq ⇒ `cu_seqlens_q` 是一条干净的等差数列、`cu_seqlens_q[-1] == q.shape[0]`，
  保住「query 表总长 = q 行数」这条常见不变量；C 组虽也可行，但它把这个不变量打破，
  省下的那点算力（空行本来就被 k=0 跳过了）不值得换这条耦合。
- 需要 vLLM 那种做法时，把 `graph_pad.PAD_Q_LEN` 设成 0 即可（已实测可行）。

★ 为什么三种形状对真实行**零影响**：varlen 是 block-diagonal 的 ——
每条序列只在自己的 query/key 段里做注意力，空行与真实行互不干扰。

### 3.4 ★ 没有为 padding 行新增保留 null block（如实记录取舍）

vLLM 的 block 0 是**保留的 NULL_BLOCK_ID**；**我们的块分配器没有这个保留块**
（`BlockManager.__init__` 里 `free_block_ids = deque(range(num_blocks))`，块 0 是普通可分配块）。

**选择：不新增保留块，空行的 `block_table` 指向普通块 0。** 理由：

1. **安全性不依赖「保留」**：`self.kv_cache = torch.empty(2, layers, num_blocks, block_size, ...)`
   是**一次性预分配**的整块张量，块号 0 永远落在合法地址上。就算块 0 被 free 再分给别人，
   它的**显存仍然存在** —— 读它不会非法访存，只会读到无意义的值，而那些值我们直接丢弃。
2. **而且现在根本不读**：`PAD_CTX_LEN=0` ⇒ 空行的 key 长度为 0 ⇒ 内核不读任何 K/V。
   `block_table=0` 是**纯防御性**的（万一某个后端仍然去取指针）。
3. **新增保留块要动全局账**：`num_blocks` 少一块会改变 KV 容量、进而改变所有既有基线数字
   （§2.5/§2.7 的推荐配置、池子块数 174→253 那类结论），属于「为了更干净而污染所有对照」。

⇒ 结论：**值上对齐 vLLM（0），语义上不引入保留块**，并把这个差异写在这里。

### 3.5 代码位置

```
nanovllm/spec_decode/graph_pad.py        新增（纯函数 + 缓冲区写入，不碰 CUDA，可 CPU 单测）
  ├─ pick_bucket()            桶选择
  ├─ pad_cu_seqlens()         空行的 cu_seqlens（device 上算，无 D2H 同步）
  ├─ draft_pad_rows()         draft 图空行的逐行元数据
  └─ fill_verify_static()     验证图静态缓冲区写入（真实行 + 空行）
```

---

## 4. 实测数字

环境：RTX 3090（GPU 0，全空闲）· Qwen3-4B + Qwen3-0.6B · `enforce_eager=False` ·
`spec_graph_bs=(1,2,4)` · `k=6` · `CTX=16384` · `CHUNK=1024` · `N_DEC=3`。

### 4.1 退回次数：`shape_not_captured(B=3,k=6)` → 0

| 口径 | pad 关（默认） | pad 开 |
|---|---:|---:|
| `shape_not_captured(B=3,k=6)` | **18 ~ 23**（多次运行；其中一次正好 **22**） | **0** |
| draft 前向退回 eager | 108 ~ 138 | **0** |
| 验证图走 padding 进图次数 | 0 | 16 ~ 24 |
| draft 图走 padding 进图次数 | 0 | 96 ~ 144 |

（该工作负载是随机采样的，接受率不同 ⇒ 混合 step 数不同 ⇒ 退回次数在 18~23 间波动。
任务书里的「22 次」落在区间内，来源是 `docs/P15重叠执行混合批-开发记录.md`。）

### 4.2 逐前向对拍（钩在真实调用点上，`scripts/p16_graphpad_probe.py`）

- `off` = `graph_pad=False`：B 未捕获 ⇒ **eager**；B 已捕获 ⇒ **精确匹配的图**
- `on` = `graph_pad=True`：B 未捕获 ⇒ **向上取整 + 空行**；B 已捕获 ⇒ 同一段精确匹配代码
- `eager` = 临时清空图集合后的**纯 eager 基线**
- ★ draft 三次调用之间**复位 CUDA 随机数状态**（候选会被写进 draft KV 供下一步读，
  不复位的话第 2..k 步输入天然不同）。自洽性检查（同路径连调两次）**6 次全为 0.0** ⇒ 无噪声地板。

| 路径 | B | off ≡ on（逐位） | on vs eager max\|Δ\| | argmax | **off vs eager（图 vs eager 地板）** |
|---|---:|---|---:|---:|---:|
| verify | 2（精确桶） | **1/1 逐位** | 0.7188 | 92.86% | 0.7188 |
| verify | **3（补 1 行）** | 0/2（off 是 eager，on 是图） | **0.5625** | **100%** | **0.0000**（证明 off 确实是纯 eager） |
| verify | 4（精确桶） | **3/3 逐位** | 0.8125 | 100% | 0.8125 |
| draft | 2（精确桶） | **1/1 逐位** | 0.1633 | 100% | 0.1633 |
| draft | **3（补 1 行）** | 0/2 | **0.1875** | **100%** | **0.0000** |
| draft | 4（精确桶） | **3/3 逐位** | 0.2188 | 100% | 0.2188 |

**怎么读这张表**（三条结论，强度递减）：

1. **精确桶上 `off ≡ on` 逐位相同**（verify 1+3=4 次、draft 1+3=4 次全部逐位）
   ⇒ **padding 的代码完全没有扰动原有的精确匹配路径**。这是最强的证据。
2. **B=3 的 `off` 就是纯 eager**（`off vs eager = 0.0000`，逐位）⇒ 表里 `on vs eager`
   这一列确实在比「补进图的 B=3」与「真 eager」。
3. **补进图的 B=3 与 eager 的差（0.5625 / 0.1875），落在既有「图 vs eager」地板（0.72~0.81 / 0.16~0.22）之内，
   而且 argmax 一致率 100%** ⇒ 这点差异是 flash-attn 分块累加顺序的固有差异，
   **不是 padding 引入的新误差**（HANDOVER §7.3 早就记过：图 vs eager 的 max\|Δlogit\| ≈ 0.30，
   30 个位置里约 1 次 argmax 不同）。

### 4.3 ★ 单次验证前向耗时（这项优化的真实收益）

同一次测量里同时报 **eager** 与 **补进图**，并用 `verify_graph_hits` / `verify_graph_padded`
确认图**真的 replay 了**（否则会退化成"两边都是 eager"的假对照 —— 第一版就踩了这个坑，见 §5）。

**轻档（批里全是短上下文，ctx ≈ 120~130）**：

| B | 用的桶 | eager | 补进图后 | 加速 | 图命中 | 其中 padding |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1（精确） | 40.53 ms | **13.73 ms** | 2.95× | 23/23 | 0 |
| 2 | 2（精确） | 40.46 ms | **14.83 ms** | 2.73× | 23/23 | 0 |
| **3** | **4（补 1 行）** | **40.86 ms** | **15.55 ms** | **2.63×** | 23/23 | **23** |
| 4 | 4（精确） | 40.55 ms | **15.51 ms** | 2.61× | 23/23 | 0 |

（20 次平均，CTX=16384。CTX=2048 的表几乎一样：13.7 / 15.0 / 15.8 / 15.5 ms ——
eager 那 40 ms 与上下文长度无关，再次印证「eager 的代价是 per-step 固定开销」。）

★ **补 1 行的 B=3（15.55 ms）与精确命中的 B=4（15.51 ms）只差 0.04 ms（0.3%）**
⇒ **padding 的代价可以忽略**，而 B=3 从 40.86 ms 降到 15.55 ms（**省 25.3 ms/次**）。

**★ 重档（批里混进一条 ~16K 的长序列）—— 一个必须一起写的边界**：

| B | 桶 | 是否补 | ctx_lens | eager | 图 | 加速 |
|---:|---:|---|---|---:|---:|---:|
| 2 | 2 | **精确命中（padded 0）** | [128, 16392] | 49.07 ms | 46.88 ms | **1.05×** |
| **3** | **4** | **补 1 行** | [132, 127, 16391] | **50.29 ms** | **47.51 ms** | **1.06×** |

⇒ **当批里有一条长上下文序列时，验证前向变成 GPU 工作主导，图能省的固定开销从 25 ms 缩到 ~2 ms。**
★★ **但这不是 padding 造成的**：`B=2` 那一行是**精确命中、`padded=0`**，同样掉到 1.05×。
而且把空行换成别的形状再量一遍，三种形状**完全一样**：

```
B=3 重档：q=nq,k=0 → 47.51 ms | q=nq,k=1 → 47.67 ms | q=0,k=0 → 47.52 ms
```

⇒ **空行的长度对耗时没有任何影响**（也和它对真实行 logits 无影响一致，§3.3）。
重档的性能特征是**既有的验证图**在长上下文下的固有性质，P16 只是把它继承过来 ——
**即便如此，补进图仍然严格优于 eager（47.51 < 50.29）**，方向没变，只是幅度小了。

★ 按 4.1 的 18~23 次退回算：轻档下这条路径每次生成省 **0.5 ~ 0.6 s** 的纯验证开销；
重档下省得少（每次 ~2.8 ms × 次数）。

### 4.4 桶配置的旁证（coordinator 的建议：把 `spec_graph_bs` 扩到 `(1,2,3,4)`）

| `spec_graph_bs` | pad 关 | pad 开 | 说明 |
|---|---:|---:|---|
| `(1,2,4)`（默认） | 退回 18~23 | **0** | 本项优化的场景 |
| `(1,2,3,4)` | **0** | 0 | **零浪费命中**（B=3 直接捕） |
| `(2,4)`（压力档） | 退回 B=1 1 次 + B=3 18 次，draft 退回 1140 次 | **0**（B=1/2/3 全靠 padding） | 验证 padding 在「大段缺口」下也不崩 |

⇒ **两条路都能把退回归零**：扩桶是「零浪费命中」，padding 是「不改配置也能兜住」。
**默认仍是 `(1,2,4)`** —— 改默认会让 `spec_graph_pad=False` 时的行为也变（B=3 从 eager 变成命中 B=3 的图），
与验收标准「默认关 ⇒ 逐字节等于改动前」冲突。要扩桶请显式设 `spec_graph_bs=(1,2,3,4)`。

### 4.5 默认关 = 老行为（逐字节）

三条独立证据：

1. **全量 CPU 测试**：`p16` 树与 `main` 干净树（`git worktree add --detach main`）
   跑**同一批** `tests/test_*.py`，逐文件 `diff` stdout：
   **14/16 完全一致**；2 个（`test_engine_step` / `test_overlap_mixed`）只差**墙钟数值与对象内存地址**
   （非确定性输出），断言结果全同。
2. **A/B 对拍**（`ab_draft_graph.py`，把 `main` 版与新版的 `draft_proposer.py` 装进同一个进程，
   同一批假图、同一串调用）：3 次调用的 logits **逐位相同**、模型看到的 context **完全相同**、
   静态缓冲区（含未用列）**完全相同**、统计计数相同。
3. **全量 CPU 测试通过**：**514/514**（基线 455 + 新增 `tests/test_graph_pad.py` 59 项），0 失败。

---

## 5. 边界与没做的（★ 必读）

### 5.1 没做 / 测不出

| 项 | 状态 | 说明 |
|---|---|---|
| **引擎级无损性统计检验** | **没做** | `scripts/check_lossless_engine.py` 那套（4 轮 × 4800 token 的 TV 对比 + 噪声地板）本轮没跑 —— 它要几十轮生成，与本次「形状/元数据」的验证目标不同。本次的无损证据是**逐前向逐位/量级对比**（§4.2）+ 既有测试套件。 |
| **端到端 token 序列是否一致** | **测了但不成立** | pad 关/开两档的 `delivered` 序列不同。**但这不是可比对照**：两档的步数不同（eager 那档的 step 更慢 ⇒ 混合批的时序不同，`mixed_steps` 1 vs 0）、采样是 temperature=1.0 随机采样。**不能拿它当"有偏"的证据，也不能当"无偏"的证据。** |
| **`spec_graph_bs` 默认值** | **没改** | 与验收标准冲突，只给实测建议（§4.4）。 |
| **保留 null block** | **没加** | 理由见 §3.4。 |
| **`ragged_draft_len`** | **按任务书要求，继续退回 eager** | 那是「每条请求算几个」变了，不是 batch 维度。 |
| **draft 图的耗时** | **没单独测** | 只测了验证图（§4.3）。draft 每轮要 replay k=6 次，量级应类似，但没实测。 |
| **`ncu` kernel 级归因** | **测不了** | 该机 `RmProfilingAdminOnly: 1`、无 sudo，ncu 被禁（HANDOVER 已记）。 |
| **B > 4 的 padding** | **没测** | `max_num_seqs=4`，桶最大 4，没有更大的桶可补。 |
| **重档为什么掉到 1.05×** | **只测了现象，没定位机理** | 已知「批里有一条 16K 序列 ⇒ 验证前向变成 GPU 工作主导」，且**精确命中（padded=0）的 B=2 同样如此** ⇒ 与 padding 无关。但**为什么图在轻档省 25 ms、重档只省 2 ms，没有隔离实验**（候选解释：host 侧 dispatch 与 GPU 工作重叠；图捕获时 `max_seqlen_k` 被烘成 `max_model_len`）。**这是本项留给下一轮的开口。** |
| **TP > 1** | **没测** | 单卡场景。 |

### 5.2 踩过的坑（写下来免得下次再踩）

1. **第一版计时测出了「图跟 eager 一样快（都 39.8 ms）」—— 假对照。**
   原因：`run_verify_forward` 的形状判定读的是 `self._verify_extra`，而计时循环里没按每个 B
   重设它，于是每次都被判成 `static_shape_mismatch`、**两边都走了 eager**。
   ⇒ **做「图 vs eager」的耗时对照，必须同时报 `verify_graph_hits` / `fallbacks`**，
   否则你根本不知道图有没有被用上。
2. **第一版 draft 对拍全是大数字（max\|Δ\| = 22~27）—— 也是假的。**
   原因：`propose_batch` 每步采样候选、候选又写进 draft KV 供下一步读；
   三次对照调用不复位随机数状态 ⇒ 第 2..k 步的**输入 token 天然不同**。
   ⇒ 修法：复位 `torch.cuda.get_rng_state()`，并**只比第 0 个候选位置**；
   再加一条「同路径连调两次」的自洽性检查当噪声地板（实测 0.0）。
3. **`tar --exclude='models'` 会连 `nanovllm/models/` 一起排掉**，同步到服务器直接
   `ModuleNotFoundError: No module named 'nanovllm.models'`。要用 `--exclude='./models'`。
4. **本机没有 rsync**（Git Bash 里也没有），同步用 `tar -czf - . | ssh host 'tar -xzf - -C DIR'`。

---

## 6. 复现命令

```bash
# ---- CPU（本地，无需 GPU）----
python tests/test_graph_pad.py                       # P16 新增，59 项
for f in tests/test_*.py; do python "$f"; done       # 全量 514 项

# ---- 同步到服务器（本机无 rsync）----
cd D:/学习/specdec-p16
tar -czf - --exclude='./.git' --exclude='__pycache__' --exclude='./models' \
    --exclude='./logs' --exclude='./out' --exclude='./data' . \
  | ssh lab3090 'tar -xzf - -C /home/ziru/nano-vllm/pad-work'

# ---- GPU 探针（★ 串行跑：端口 2333 被 init_process_group 占用）----
ssh lab3090
cd /home/ziru/nano-vllm/pad-work
# ① 退回次数 + 逐前向对拍 + 耗时表
CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2620 GBS=1,2,4 CTX=16384 N_DEC=3 \
  /home/ziru/nano-vllm/venv/bin/python -u scripts/p16_graphpad_probe.py
# ② 空行「零长度」安全性（三种形状互相对拍）
CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2606 CTX=4096 \
  /home/ziru/nano-vllm/venv/bin/python -u scripts/p16_pad_variant_probe.py
```

---

## 7. 改动的文件

| 文件 | 性质 |
|---|---|
| `nanovllm/spec_decode/graph_pad.py` | **新增**（桶选择 + 空行元数据 + 缓冲区写入，纯 CPU 可测） |
| `nanovllm/config.py` | 新增 `spec_graph_pad: bool = False`（默认关） |
| `nanovllm/engine/model_runner.py` | `run_verify_forward` 加 padding 分支；新增 `graph_pad` / `verify_graph_padded`；`bind_cudagraph` 传开关 |
| `nanovllm/spec_decode/draft_proposer.py` | `_forward_group` 加 padding 分支；`bind_cudagraph(graph_pad=)`；新增 `n_padded` |
| `tests/test_graph_pad.py` | **新增** CPU 测试（59 项） |
| `scripts/p16_graphpad_probe.py` | **新增** 探针（退回次数 / 逐前向对拍 / 耗时表） |
| `scripts/p16_pad_variant_probe.py` | **新增** 探针（空行零长度安全性） |
| `docs/P16图形状padding-开发记录.md` | 本文档 |
| `docs/优化过程复盘.md` | 补录 `[25]` |

★ **旧代码路径一行都没删**：所有新行为都在 `if pad:` / `if self.graph_pad` 分支里，
`pad == 0`（默认关，或开着但正好命中）时走的是**原文不动**的那几行。
