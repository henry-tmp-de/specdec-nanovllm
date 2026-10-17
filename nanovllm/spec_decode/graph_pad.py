"""P16：投机路径两张 CUDA graph 的「未精确捕获 batch size」向上取整（padding）。

问题
----
投机路径的两张图都是**精确匹配** batch 的：`verify_graphs[(B, k)]`（验证前向）
与 `draft_graphs[B]`（draft 前向），而 `config.spec_graph_bs = (1, 2, 4)` 只捕获
这三个桶。于是 B=3 的混合 step（3 条 decode 请求凑不出 B=3 的图）整个退回 eager
—— 台账里实测 22 次，单次验证前向 eager 36 ms vs 图 14 ms。

参考实现（原版 decode 早就这么干）
----------------------------------
`ModelRunner.run_model` 里普通 decode 的图路径：

    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]   # 3 条用 4 的图
    graph_vars["slot_mapping"].fill_(-1)                             # 整行先填 -1
    graph_vars["slot_mapping"][:bs] = context.slot_mapping           # 只填前 bs 行

本模块把同一套做法搬到投机路径的两张图上：**向上取整到最近的桶 + 多出来的行用
空行（padding 行）补齐**。桶选择、空行元数据、缓冲区写入都收在这里，
不碰 CUDA（只用 CPU 张量就能跑），可在 CPU 上单测（`tests/test_graph_pad.py`）。

★ 空行不是「什么都不做」——图会把 pad 行**真真切切算一遍**。
  所以每一行的元数据都必须落在安全的位置上，规则就是下面四个常量。
"""

from typing import Dict, Iterable, Optional

import torch

# =====================================================================
# 空行（padding 行）的元数据常量 —— 每一条对应一个必须成立的安全不变量
# =====================================================================
PAD_SLOT = -1
#   slot_mapping = -1 ⇒ `store_kvcache_kernel` 见到就 `return`，一个字节都不写。
#   ★ 绝不能给真实槽号：pad 行背后没有请求，写进去就是覆盖别人的 KV（静默污染）。

PAD_ROW_BLOCK = 0
#   空行的 block_table 指向【块 0】。块号从 0 开始发、KV cache 张量按 num_blocks
#   一次性分配，所以块 0 永远是一块**已分配的合法显存**，读它安全。
#   ★ 若留 -1，flash-attn 会去读第 -1 页 → 非法访存（不是算错，是崩）。
#   ★ 别和 draft_proposer 里的 `_PAD_BLOCK = -1` 搞混：那个是**真实行**块表里
#     「没用到的那几列」的填充值（那几列永远不被读），语义不同。

PAD_CTX_LEN = 0
#   空行的可见长度（key 侧）—— **归零**，与 vLLM 的 `seq_lens=0` 一致：
#   空行读不到任何 key，它的 attention 输出与后续 logits 都没有意义（我们丢弃）。
#   ★ 归零**实测安全**（scripts/p16_pad_variant_probe.py，RTX 3090 / flash-attn 2.8.3）：
#     k=0 与 k=1 两种空行得到的真实行 logits **逐位相同**（max|Δ| = 0.0），
#     也没有崩。原因：varlen 是 block-diagonal 的，空行与真实行互不影响。
#   ★ capture 注释里那句「cu_seqlens=[0,0] 会让内核退化成长度 0」说的是
#     **整张表全 0**（连真实行也是 0）那种情况；这里是「真实行照常、只有尾部
#     若干段长度为 0」，属于标准 padding，内核按 0 长度段直接跳过。
#   ★ 前提是 cu_seqlens 仍然**单调不减**（pad_cu_seqlens 保证）。

PAD_Q_LEN = None
#   空行占多少个 query 位置。None = 与真实行同宽（nq = 1+k）。
#   ★ 为什么不用 vLLM 的 `query_start_loc` 台阶（= 0）：
#     实测 q=0 与 q=nq **也逐位相同**（p16_pad_variant_probe.py 的 C 组），
#     但 q=0 会让 `cu_seqlens_q[-1] != q.shape[0]`，把「query 表总长 = q 行数」
#     这条常见不变量打破；而 q=nq 时 cu_seqlens_q 是一条干净的等差数列，
#     与图捕获时的形状逐字一致。省下的那点 attention 算力（空行本来就被
#     k=0 跳过了）不值得换这条耦合，所以默认取 None。
#     需要 vLLM 那种做法时把它设成 0 即可。

PAD_TOKEN = 0
#   空行的 input_ids / positions：填 0（合法值 —— 词表内 token / 位置 0）。
#   算出来的 logits 直接丢弃，调用方只取真实行。


# =====================================================================
#  1. 桶选择
# =====================================================================
def pick_bucket(b: int, buckets: Iterable[int]) -> Optional[int]:
    """向上取整到最近的桶：返回 >= b 的**最小**桶；b 比所有桶都大时返回 None。

    ★ 返回 None 表示「现有图覆盖不到这个 batch」——调用方必须**退回 eager**，
      不能硬塞：真实行数超过图的行数就是写越界。
    ★ b 恰好等于某个桶时返回该桶本身（pad = 0，等价于精确命中，老路径不变）。
    ★ 只挑桶，不关心桶是怎么捕获的 —— 纯整数运算，便于单测。
    """
    cands = [int(x) for x in buckets if int(x) >= int(b)]
    return min(cands) if cands else None


# =====================================================================
#  2. 空行的元数据（纯函数：输入输出都是张量，不读不写外部状态）
# =====================================================================
def pad_cu_seqlens(cu_q_real: torch.Tensor, cu_k_real: torch.Tensor,
                   nq: int, pad: int):
    """给验证图构造**补了 pad 行空行之后**的 cu_seqlens_q / cu_seqlens_k。

    参数
    ----
    cu_q_real, cu_k_real : (B+1,) 真实请求的 cu_seqlens（前缀和，首项为 0）
    nq                   : 每条请求的 query 数（= 1 + k）
    pad                  : 要补的空行数

    返回
    ----
    (cu_q, cu_k)，各 (B+pad+1,)，device / dtype 与输入一致。

    ★ 两条不变量（都是 flash-attn 的硬要求）：
      ① cu_seqlens 是**前缀和**，必须单调不减 —— 空行是在真实总和之上继续累加，
         天然单调。
      ② query 侧必须与图捕获时的形状自洽：默认每条空行占 nq 个 query（步长恒为
         nq）；若 PAD_Q_LEN 设成 0，则空行不占 query，cu_seqlens_q 末尾变成
         一段**等值**的台阶（长度 0）—— 仍然是单调不减的。
         这也正是 `ragged_draft_len` 不能 padding 的原因：那是**每条请求要算几个**
         真的变了，不是 batch 维度的问题。
    ★ 空行的 key 长度取 PAD_CTX_LEN（见该常量的注释）。
    ★ 全程在 device 上算（**没有** `.tolist()`）——本函数在每步的图路径上被调用，
      一次 D2H 同步就会把图省下来的收益吃掉一截。
    """
    if pad <= 0:
        return cu_q_real, cu_k_real
    dev = cu_q_real.device
    steps = torch.arange(1, pad + 1, dtype=cu_q_real.dtype, device=dev)
    q_step = int(nq) if PAD_Q_LEN is None else int(PAD_Q_LEN)
    cu_q = torch.cat([cu_q_real, cu_q_real[-1:] + steps * q_step])
    cu_k = torch.cat([cu_k_real, cu_k_real[-1:] + steps * PAD_CTX_LEN])
    return cu_q, cu_k


def draft_pad_rows(pad: int, device, dtype_i64=torch.int64,
                   dtype_i32=torch.int32) -> Dict[str, torch.Tensor]:
    """draft 图空行的逐行元数据，每个张量长度 = pad（pad <= 0 时返回空 dict）。

    返回的 key 与 draft 图的静态缓冲区**同名**，调用方可以两行写完：

        for key, val in draft_pad_rows(pad, dev).items():
            gv[key][B:] = val

    ★ 值全部来自上面四个常量 —— 也就是「不写 KV / 读块 0 / 可见长度 1 / token 0」。
    ★ block_tables 是二维的，不在这里返回，由调用方按行 fill_(PAD_ROW_BLOCK)。
    """
    if pad <= 0:
        return {}
    return dict(
        input_ids=torch.full((pad,), PAD_TOKEN, dtype=dtype_i64, device=device),
        positions=torch.full((pad,), PAD_TOKEN, dtype=dtype_i64, device=device),
        slot_mapping=torch.full((pad,), PAD_SLOT, dtype=dtype_i32, device=device),
        context_lens=torch.full((pad,), PAD_CTX_LEN, dtype=dtype_i32, device=device),
    )


# =====================================================================
#  3. 验证图：把「真实行 + 空行」写进静态缓冲区
# =====================================================================
def fill_verify_static(gv: Dict[str, torch.Tensor], B: int, nq: int, pad: int,
                       input_ids: torch.Tensor, positions: torch.Tensor,
                       slot_mapping: torch.Tensor, cu_q_real: torch.Tensor,
                       cu_k_real: torch.Tensor,
                       block_tables: Optional[torch.Tensor]) -> int:
    """验证图的静态缓冲区写入（P16 的 padding 分支），返回真实行数 Q。

    布局：[0, Q) 是 B 条真实请求（每条 nq = 1+k 个 query），[Q, Q+pad*nq) 是空行。

    ★ 真实行的写法与精确命中路径**逐字一致**（元数据原样拷进去）；
      空行的构造全部走 pad_cu_seqlens / 上面那几个常量。
    ★ 返回值 Q = B*nq 是**必须**的：调用方要 `gv["logits"][:Q]` 交出去 ——
      `run_verify` 用【真实】的 cu_seqlens_q 切分 logits，并断言
      `cu[-1] == logits.shape[0]`；整块返回会让这个断言直接炸。
    """
    Q = input_ids.numel()
    # 空行先整块铺成合法值，再把真实行覆盖上去
    gv["input_ids"].zero_()
    gv["input_ids"][:Q].copy_(input_ids)
    gv["positions"].zero_()
    gv["positions"][:Q].copy_(positions)
    gv["slot_mapping"].fill_(PAD_SLOT)
    gv["slot_mapping"][:Q].copy_(slot_mapping)
    cu_q, cu_k = pad_cu_seqlens(cu_q_real, cu_k_real, nq, pad)
    gv["cu_seqlens_q"].copy_(cu_q)
    gv["cu_seqlens_k"].copy_(cu_k)
    # 空行与真实行的未用列一律指向块 0（已分配，读它安全）；
    # 真实行自己的块表随即覆盖上去（bt 里不足宽度的列本来就是 -1）。
    gv["block_tables"].fill_(PAD_ROW_BLOCK)
    if block_tables is not None:
        gv["block_tables"][:B, :block_tables.size(1)].copy_(block_tables)
    return Q
