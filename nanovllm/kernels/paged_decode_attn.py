"""Paged decode attention —— 自己写的 Triton kernel。

为什么写这个
------------
nano-vllm 里 attention 全靠 flash-attn，整个项目只有一个手写 Triton kernel
（`layers/attention.py` 的 `store_kvcache_kernel`）。但 decode 阶段的 attention
恰恰是推理引擎最核心、最值得讲的一个算子：

* decode 时每个序列只有 **1 个 query**，KV 要从分页缓存里按 block_table 捞 ——
  访存模式跟 prefill 完全不同；
* 它是**访存瓶颈**而不是计算瓶颈：算术密度极低，
  绝大部分时间花在把 KV 从显存搬进来；
* 面试高频追问就是「你写的算子瓶颈是什么、效率如何、怎么优化」，
  而这些必须自己量过才答得出来。

数学
----
对每个序列 s、每个 query 头 h：

    o[h] = softmax(q[h] · Kᵀ · scale) · V

其中 K/V 不是连续存放的，而是分页的：
第 p 个位置的 KV 在 `block_table[s, p // block_size]` 号物理块的 `p % block_size` 行。

用 **online softmax**（FlashAttention 的核心技巧）流式累加，
不物化 (context_len × context_len) 的注意力矩阵：

    m_new = max(m_old, max_j(qk_j))
    l     = l * exp(m_old - m_new) + Σ_j exp(qk_j - m_new)
    acc   = acc * exp(m_old - m_new) + Σ_j exp(qk_j - m_new) * v_j

三个版本，递进对比（这就是要讲的故事）
--------------------------------------
v1  每 (序列, query头) 一个 program
    -> 正确，但 GQA 下同一个 kv 头被 4 个 query 头各读一遍，K/V 访存浪费 4 倍
v2  每 (序列, kv头) 一个 program，把同组的 Q_PER_KV 个 query 头一起算
    -> K/V 只读一次，访存量降到 1/4；代价是寄存器压力上升
v3  split-K（flash-decoding）：把 context 切给多个 program 并行，再归约
    -> batch=1 时并行度只有 num_kv_heads 个 program，喂不满 SM；
       见 `paged_decode_attn_split`（如果实现了）

用法
----
    from nanovllm.kernels.paged_decode_attn import paged_decode_attention
    o = paged_decode_attention(q, k_cache, v_cache, block_table, context_lens, scale)

形状约定（和 nano-vllm 现有 KV cache 布局一致）
    q           : (num_seqs, num_q_heads, head_dim)
    k_cache     : (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache     : (num_blocks, block_size, num_kv_heads, head_dim)
    block_table : (num_seqs, max_blocks)  int32，不足的位补 -1
    context_lens: (num_seqs,)             int32
    o           : (num_seqs, num_q_heads, head_dim)
"""

import torch
import triton
import triton.language as tl

# v3 的中间缓冲区（partial 的 m / l / acc）按形状缓存复用。
# 逐步 decode 时形状固定，没必要每步重新分配。
_buf_cache: dict = {}


# ======================================================================
# v1：每 (序列, query头) 一个 program
# ======================================================================
@triton.jit
def _paged_decode_v1(
    q_ptr, k_ptr, v_ptr, bt_ptr, cl_ptr, o_ptr,
    stride_qs, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_bs,
    stride_os, stride_oh,
    scale,
    NUM_Q_HEADS: tl.constexpr,      # 用来算 GQA 分组比
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,       # KV cache 的块大小（元素个数）
    BLOCK_N: tl.constexpr,          # 每次沿 context 方向处理多少个位置
):
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // (NUM_Q_HEADS // NUM_KV_HEADS)

    ctx_len = tl.load(cl_ptr + seq)
    offs_d = tl.arange(0, HEAD_DIM)

    # q 只有 1 个 token，直接整个载入寄存器
    q = tl.load(q_ptr + seq * stride_qs + qh * stride_qh + offs_d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # ★ 沿 context 方向分块流式处理，不物化注意力矩阵
    for blk in range(0, tl.cdiv(ctx_len, BLOCK_N)):
        pos = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = pos < ctx_len

        # ---- 分页寻址：位置 pos -> (物理块, 块内偏移) ----
        logical = pos // BLOCK_SIZE
        offset = pos % BLOCK_SIZE
        phys = tl.load(bt_ptr + seq * stride_bs + logical, mask=mask, other=0)

        k_ptrs = (k_ptr + phys[:, None] * stride_kb + offset[:, None] * stride_kt
                  + kvh * stride_kh + offs_d[None, :])
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # ---- online softmax ----
        qk = tl.sum(k * q[None, :], axis=1) * scale
        qk = tl.where(mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)

        v_ptrs = (v_ptr + phys[:, None] * stride_kb + offset[:, None] * stride_kt
                  + kvh * stride_kh + offs_d[None, :])
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    l_i = tl.where(l_i > 0, l_i, 1.0)          # context_len = 0 的兜底
    o = acc / l_i
    tl.store(o_ptr + seq * stride_os + qh * stride_oh + offs_d,
             o.to(o_ptr.dtype.element_ty))


# ======================================================================
# v2：每 (序列, kv头) 一个 program，同组的 query 头共用一次 K/V 载入
# ======================================================================
@triton.jit
def _paged_decode_v2(
    q_ptr, k_ptr, v_ptr, bt_ptr, cl_ptr, o_ptr,
    stride_qs, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_bs,
    stride_os, stride_oh,
    scale,
    Q_PER_KV: tl.constexpr,         # 每个 kv 头对应几个 query 头（GQA 比）
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,          # tl.dot 的 M 方向，必须 ≥16 且是 2 的幂
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)

    ctx_len = tl.load(cl_ptr + seq)
    offs_d = tl.arange(0, HEAD_DIM)
    # ★ tl.dot 要求 M ≥ 16，而 GQA 比通常只有 4 —— 把 M 方向补到 16，
    #   多出来的行喂成全 0（q=0 时 qk 恒为 0，softmax 不会出 NaN），
    #   最后只写回前 Q_PER_KV 行。跑不满的算力白扔，但这个 kernel 是访存瓶颈，
    #   算力本来就有大量富余。
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < Q_PER_KV

    # ---- 一次把同组的 Q_PER_KV 个 query 头全载进来（保持 bf16，喂给 tensor core）----
    q_ptrs = (q_ptr + seq * stride_qs + (kvh * Q_PER_KV + offs_m)[:, None] * stride_qh
              + offs_d[None, :])
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)      # (BLOCK_M, HEAD_DIM)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for blk in range(0, tl.cdiv(ctx_len, BLOCK_N)):
        pos = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = pos < ctx_len

        logical = pos // BLOCK_SIZE
        offset = pos % BLOCK_SIZE
        phys = tl.load(bt_ptr + seq * stride_bs + logical, mask=mask, other=0)

        k_ptrs = (k_ptr + phys[:, None] * stride_kb + offset[:, None] * stride_kt
                  + kvh * stride_kh + offs_d[None, :])
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)    # (BLOCK_N, HEAD_DIM) bf16

        # ★★ 关键区别：这一块 K 被 Q_PER_KV 个 query 头共用，只从显存读了一次。
        #    qk = Q · Kᵀ 走 tensor core
        qk = tl.dot(q, tl.trans(k)) * scale                   # (BLOCK_M, BLOCK_N)
        qk = tl.where(mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))           # (BLOCK_M,)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])                       # (BLOCK_M, BLOCK_N)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (v_ptr + phys[:, None] * stride_kb + offset[:, None] * stride_kt
                  + kvh * stride_kh + offs_d[None, :])
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)    # (BLOCK_N, HEAD_DIM)

        # acc += P · V，同样走 tensor core（P 转 bf16，和 flash-attn 的做法一致）
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_i = tl.where(l_i > 0, l_i, 1.0)
    o = acc / l_i[:, None]
    o_ptrs = (o_ptr + seq * stride_os + (kvh * Q_PER_KV + offs_m)[:, None] * stride_oh
              + offs_d[None, :])
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


# ======================================================================
# v3：split-K（flash-decoding）—— 把 context 切开并行，再归约
#
# 为什么需要它（实测驱动，不是照抄论文）
# ------------------------------------
# v2 已经把 GQA 的重复读消掉了（KV 只读一遍），但实测带宽利用率反而掉到 16%：
#     v1: 32 个 program（num_q_heads=32），跑满一部分 SM，带宽 37%
#     v2:  8 个 program（num_kv_heads=8），SM 大量空转，带宽 16%
# 根因是【并行度】：decode 每个序列只有 1 个 query，能并行的只有 head 维度。
# 3090 有 82 个 SM，8 个 CTA 连一波都填不满。
#
# 解法：把 context 方向也切开。每个 (序列, kv头) 派 NUM_SPLITS 个 program，
# 各算一段的局部 (m, l, acc)，再用第二个 kernel 按 online-softmax 的规则合并。
# 这样并行度 = num_seqs × num_kv_heads × NUM_SPLITS。
# ======================================================================
@triton.jit
def _paged_decode_split_partial(
    q_ptr, k_ptr, v_ptr, bt_ptr, cl_ptr,
    pm_ptr, pl_ptr, pacc_ptr,
    stride_qs, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_bs,
    stride_pm_s, stride_pm_h, stride_pm_k,      # pm: (seqs, kvh, splits, BLOCK_M)
    stride_pa_s, stride_pa_h, stride_pa_k,      # pacc: (seqs, kvh, splits, BLOCK_M, D)
    scale,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK: tl.constexpr,                        # 每个 split 负责多少个位置
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

    ctx_len = tl.load(cl_ptr + seq)
    start_pos = split * CHUNK
    end_pos = tl.minimum(start_pos + CHUNK, ctx_len)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < Q_PER_KV

    q_ptrs = (q_ptr + seq * stride_qs + (kvh * Q_PER_KV + offs_m)[:, None] * stride_qh
              + offs_d[None, :])
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 只处理本 split 负责的那一段 [start_pos, end_pos)
    for blk in range(0, tl.cdiv(tl.maximum(end_pos - start_pos, 0), BLOCK_N)):
        pos = start_pos + blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = pos < end_pos

        logical = pos // BLOCK_SIZE
        offset = pos % BLOCK_SIZE
        phys = tl.load(bt_ptr + seq * stride_bs + logical, mask=mask, other=0)

        k_ptrs = (k_ptr + phys[:, None] * stride_kb + offset[:, None] * stride_kt
                  + kvh * stride_kh + offs_d[None, :])
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (v_ptr + phys[:, None] * stride_kb + offset[:, None] * stride_kt
                  + kvh * stride_kh + offs_d[None, :])
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # 写局部结果（没分到活的 split 会写成 m=-inf, l=0，合并时权重为 0）
    base = seq * stride_pm_s + kvh * stride_pm_h + split * stride_pm_k + offs_m
    tl.store(pm_ptr + base, m_i, mask=mask_m)
    tl.store(pl_ptr + base, l_i, mask=mask_m)

    pa_ptrs = (pacc_ptr + seq * stride_pa_s + kvh * stride_pa_h + split * stride_pa_k
               + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    tl.store(pa_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _paged_decode_split_combine(
    pm_ptr, pl_ptr, pacc_ptr, o_ptr,
    stride_pm_s, stride_pm_h, stride_pm_k,
    stride_pa_s, stride_pa_h, stride_pa_k,
    stride_os, stride_oh,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, NUM_SPLITS)
    mask_m = offs_m < Q_PER_KV

    # 先把所有 split 的 m 读进来求全局 max
    pm_ptrs = (pm_ptr + seq * stride_pm_s + kvh * stride_pm_h
               + offs_k[None, :] * stride_pm_k + offs_m[:, None])
    m_all = tl.load(pm_ptrs, mask=mask_m[:, None], other=float("-inf"))   # (M, K)
    m_g = tl.max(m_all, axis=1)                                           # (M,)
    # ctx_len == 0 时全是 -inf，兜底避免 NaN
    m_g = tl.where(m_g == float("-inf"), 0.0, m_g)

    w = tl.exp(m_all - m_g[:, None])                                      # (M, K)

    pl_ptrs = (pl_ptr + seq * stride_pm_s + kvh * stride_pm_h
               + offs_k[None, :] * stride_pm_k + offs_m[:, None])
    l_all = tl.load(pl_ptrs, mask=mask_m[:, None], other=0.0)             # (M, K)
    l_g = tl.sum(l_all * w, axis=1)                                       # (M,)

    # acc 是 (M, K, D)，分 K 次累加，避免一次开一个三维大张量
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for k in tl.static_range(NUM_SPLITS):
        pa_ptrs = (pacc_ptr + seq * stride_pa_s + kvh * stride_pa_h + k * stride_pa_k
                   + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
        a_k = tl.load(pa_ptrs, mask=mask_m[:, None], other=0.0)
        wk = tl.sum(tl.where(offs_k[None, :] == k, w, 0.0), axis=1)       # (M,)
        acc += a_k * wk[:, None]

    l_g = tl.where(l_g > 0, l_g, 1.0)
    o = acc / l_g[:, None]

    o_ptrs = (o_ptr + seq * stride_os + (kvh * Q_PER_KV + offs_m)[:, None] * stride_oh
              + offs_d[None, :])
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


# ======================================================================
# Python 入口
# ======================================================================
def paged_decode_attention(q, k_cache, v_cache, block_table, context_lens, scale,
                           version=2, block_n=128, splits=8,
                           num_warps=None, num_stages=None, layout="interleaved"):
    """num_warps / num_stages 是给调参用的，None 表示交给 Triton 默认值。

    ★ 一开始没暴露这两个参数，结果是 v3 只有 ~3 warp/SM —— 线程太少，
      藏不住 DRAM 延迟，带宽上不去。这是「跑得慢」的第一大嫌疑。
    """
    """分页 decode attention。

    参数
    ----
    q            : (num_seqs, num_q_heads, head_dim)
    k_cache      : (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache      : 同 k_cache
    block_table  : (num_seqs, max_blocks) int32
    context_lens : (num_seqs,) int32
    scale        : float，通常是 head_dim ** -0.5
    version      : 1 或 2（见模块 docstring）
    block_n      : 每次沿 context 处理多少位置，调优用
    """
    num_seqs, num_q_heads, head_dim = q.shape
    if layout == "head_major":
        # (num_blocks, num_kv_heads, block_size, head_dim)：同一个头的所有位置连续
        num_blocks, num_kv_heads, block_size, _ = k_cache.shape
        stride_kb, stride_kh, stride_kt = (k_cache.stride(0), k_cache.stride(1),
                                           k_cache.stride(2))
    else:
        # (num_blocks, block_size, num_kv_heads, head_dim)：nano-vllm/flash-attn 的默认布局
        num_blocks, block_size, num_kv_heads, _ = k_cache.shape
        stride_kb, stride_kt, stride_kh = (k_cache.stride(0), k_cache.stride(1),
                                           k_cache.stride(2))
    assert num_q_heads % num_kv_heads == 0, \
        f"GQA 要求 num_q_heads 能整除 num_kv_heads：{num_q_heads} vs {num_kv_heads}"

    q = q.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.contiguous()
    o = torch.empty_like(q)

    common = dict(
        stride_qs=q.stride(0), stride_qh=q.stride(1),
        stride_kb=stride_kb, stride_kt=stride_kt, stride_kh=stride_kh,
        stride_bs=block_table.stride(0),
        stride_os=o.stride(0), stride_oh=o.stride(1),
        scale=scale, HEAD_DIM=head_dim, BLOCK_SIZE=block_size, BLOCK_N=block_n,
    )
    tune = {}
    if num_warps is not None:
        tune["num_warps"] = num_warps
    if num_stages is not None:
        tune["num_stages"] = num_stages

    if version == 1:
        _paged_decode_v1[(num_seqs, num_q_heads)](
            q, k_cache, v_cache, block_table, context_lens, o,
            NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads, **common, **tune)
    elif version == 2:
        q_per_kv = num_q_heads // num_kv_heads
        # tl.dot 的 M 必须 ≥16：GQA 比一般只有 4，补到 16
        block_m = max(16, triton.next_power_of_2(q_per_kv))
        _paged_decode_v2[(num_seqs, num_kv_heads)](
            q, k_cache, v_cache, block_table, context_lens, o,
            Q_PER_KV=q_per_kv, BLOCK_M=block_m, **common, **tune)
    elif version == 3:
        q_per_kv = num_q_heads // num_kv_heads
        block_m = max(16, triton.next_power_of_2(q_per_kv))
        # ★★ 绝对不要在这里用 context_lens.max().item()！
        #   那是一次 device→host 同步，会把流水线打断 —— 实测代价 ~60-140us，
        #   比 kernel 本身还贵，而且会让「split 数扫描」看起来毫无效果
        #   （因为时间全被同步吃掉了）。
        #   改用 block_table 的宽度算一个静态上界，不需要任何同步。
        max_ctx = block_table.shape[1] * block_size
        # 每个 split 负责 CHUNK 个位置，先按 splits 平分、再向上取整到 BLOCK_N 的倍数。
        # ★ 这里差点写错成 `cdiv(max_ctx, splits) * block_n`：
        #   那等于把「每份的位置数」又乘了一遍 block_n（4096/32=128 -> 16384），
        #   结果每个 split 都去跑整个 context，32 个 split 全在做重复劳动，
        #   实测表现是「split 数扫描完全平的」—— 从数据才看出是这里错了。
        chunk = max(block_n, triton.cdiv(triton.cdiv(max_ctx, splits), block_n) * block_n)

        # 缓冲区按形状缓存复用，省掉每步 3 次显存分配
        key = (num_seqs, num_kv_heads, splits, block_m, head_dim, q.device)
        buf = _buf_cache.get(key)
        if buf is None:
            pm = torch.empty(num_seqs, num_kv_heads, splits, block_m,
                             dtype=torch.float32, device=q.device)
            pl = torch.empty_like(pm)
            pacc = torch.empty(num_seqs, num_kv_heads, splits, block_m, head_dim,
                               dtype=torch.float32, device=q.device)
            buf = _buf_cache[key] = (pm, pl, pacc)
        pm, pl, pacc = buf

        _paged_decode_split_partial[(num_seqs, num_kv_heads, splits)](
            q, k_cache, v_cache, block_table, context_lens,
            pm, pl, pacc,
            stride_qs=q.stride(0), stride_qh=q.stride(1),
            # ★ 必须用上面按 layout 算出来的 stride_kb/kt/kh，
            #   不能写死 k_cache.stride(1)/(2) —— 写死的话换布局时
            #   传进去的就是错的 stride，结果会静默算错（实测输出差 0.32）。
            stride_kb=stride_kb, stride_kt=stride_kt, stride_kh=stride_kh,
            stride_bs=block_table.stride(0),
            stride_pm_s=pm.stride(0), stride_pm_h=pm.stride(1), stride_pm_k=pm.stride(2),
            stride_pa_s=pacc.stride(0), stride_pa_h=pacc.stride(1),
            stride_pa_k=pacc.stride(2),
            scale=scale, Q_PER_KV=q_per_kv, HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size, BLOCK_N=block_n, BLOCK_M=block_m, CHUNK=chunk, **tune)

        _paged_decode_split_combine[(num_seqs, num_kv_heads)](
            pm, pl, pacc, o,
            stride_pm_s=pm.stride(0), stride_pm_h=pm.stride(1), stride_pm_k=pm.stride(2),
            stride_pa_s=pacc.stride(0), stride_pa_h=pacc.stride(1),
            stride_pa_k=pacc.stride(2),
            stride_os=o.stride(0), stride_oh=o.stride(1),
            Q_PER_KV=q_per_kv, HEAD_DIM=head_dim, BLOCK_M=block_m, NUM_SPLITS=splits,
            **tune)
    else:
        raise ValueError(f"未知 version: {version}")
    return o


def paged_decode_attention_ref(q, k_cache, v_cache, block_table, context_lens, scale):
    """纯 torch 参照实现：把分页 KV 拼回连续序列，再做普通 attention。

    只用来对拍正确性，慢是应该的。
    """
    num_seqs, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    rep = num_q_heads // num_kv_heads
    outs = []
    for s in range(num_seqs):
        L = int(context_lens[s])
        if L == 0:
            outs.append(torch.zeros(num_q_heads, head_dim, device=q.device, dtype=q.dtype))
            continue
        nblk = (L + block_size - 1) // block_size
        phys = block_table[s, :nblk].long()
        k = k_cache[phys].reshape(-1, num_kv_heads, head_dim)[:L]      # (L, kvh, d)
        v = v_cache[phys].reshape(-1, num_kv_heads, head_dim)[:L]
        k = k.repeat_interleave(rep, dim=1).float()                    # (L, qh, d)
        v = v.repeat_interleave(rep, dim=1).float()
        logits = torch.einsum("hd,lhd->hl", q[s].float(), k) * scale   # (qh, L)
        p = torch.softmax(logits, dim=-1)
        outs.append(torch.einsum("hl,lhd->hd", p, v))
    return torch.stack(outs).to(q.dtype)
