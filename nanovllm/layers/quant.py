"""INT8 权重量化（W8A16：int8 存储 + bf16 计算）。

★ 核心约束：**反量化必须融合**，不能"先解压再算"。

    ✗ 读 int8 权重 -> 反量化成 bf16 写回显存 -> 调 cuBLAS 读 bf16
      每参数 = 读 1B + 写 2B + 读 2B = 5B，比原来的 2B 还多 —— 完全白做。
    ✓ 读 int8 权重 + scale -> 在【寄存器里】反量化 -> 直接进 MMA
      每参数显存流量 = 1B（+ 每 128 通道 2B 的 scale）≈ 1.016B。

本文件提供：
  · quantize_weight / dequantize_weight —— 纯 torch，量化与参照反量化（也在离线脚本里用）
  · int8_linear                         —— 融合反量化的 Triton GEMM（W8A16）
  · set_quant_mode / get_quant_mode     —— 让 LinearBase 在构造期决定用 int8 还是 bf16 权重

s8 张量核（mma.sync.m16n8k32.s8.s8.s32）在 sm_86 上确实有，但那是 **W8A8**
（要动态量化激活）。本轮按任务书做 **W8A16**：int8 只用于存储，乘加仍是 bf16 张量核。
"""
import re
import torch
import torch.nn as nn

import triton
import triton.language as tl

QMAX = 127

# ---------------------------------------------------------------------------
# HF（checkpoint）张量名 -> nano-vllm 引擎模块名
#   checkpoint 里 q/k/v 是分开的三块，引擎里合并成 qkv_proj；gate/up 同理。
# ---------------------------------------------------------------------------
_HF_TO_ENGINE = [
    (re.compile(r"^(.*)\.self_attn\.(?:q|k|v)_proj\.weight$"), r"\1.self_attn.qkv_proj"),
    (re.compile(r"^(.*)\.self_attn\.o_proj\.weight$"), r"\1.self_attn.o_proj"),
    (re.compile(r"^(.*)\.mlp\.(?:gate|up)_proj\.weight$"), r"\1.mlp.gate_up_proj"),
    (re.compile(r"^(.*)\.mlp\.down_proj\.weight$"), r"\1.mlp.down_proj"),
]


def hf_names_to_modules(hf_names):
    """把 quant_config.json 里的 HF 张量名翻译成引擎模块名（自动识别用）。

    合并模块必须【整套】在场：qkv_proj 要求 q/k/v 三个都在，gate_up 要求 gate/up 都在。
    缺一个就报错，避免静默只量化半边。
    """
    groups = {}
    for n in hf_names:
        for rx, rep in _HF_TO_ENGINE:
            if rx.match(n):
                groups.setdefault(rx.sub(rep, n), set()).add(n)
                break
        else:
            raise ValueError(f"无法识别的量化层名: {n}")
    out = set()
    for mod, src in groups.items():
        if mod.endswith("qkv_proj") and len(src) != 3:
            raise ValueError(f"{mod} 需要 q/k/v 三块都在场，实际只有 {sorted(src)}")
        if mod.endswith("gate_up_proj") and len(src) != 2:
            raise ValueError(f"{mod} 需要 gate/up 两块都在场，实际只有 {sorted(src)}")
        out.add(mod)
    return sorted(out)


# ---------------------------------------------------------------------------
# 把已加载的模型【就地】换成 int8 权重
# ---------------------------------------------------------------------------
def apply_int8_quant(model, module_names, granularity="per_channel", group_size=128,
                     verbose=False):
    """只量化 module_names 里点名的 LinearBase 子模块，其余保持 bf16。

    ★ 为什么"就地替换"而不是"构造期决定"：
      范围是【按名字】选的（比如「只量化 draft 的 FFN」），构造期模块还没挂到树上、
      拿不到最终名字。就地替换既简单又天然支持任意层筛选。
    ★ 反量化来源是【已加载的 bf16 权重】；与 offline 脚本 `quantize_int8.py`
      用同一个 `quantize_weight`，可逐位对拍。
    """
    from nanovllm.layers.linear import LinearBase
    want = set(module_names)
    done = []
    for name, mod in model.named_modules():
        if name not in want:
            continue
        if not isinstance(mod, LinearBase):
            raise TypeError(f"{name} 不是 LinearBase（{type(mod).__name__}），拒绝量化")
        w = mod.weight.data
        assert w.dtype == torch.bfloat16, f"{name} 权重不是 bf16（{w.dtype}），拒绝量化"
        q, s = quantize_weight(w, granularity, group_size)
        dev = w.device
        mod.weight = nn.Parameter(q.to(dev).contiguous(), requires_grad=False)
        mod.weight_scale = nn.Parameter(s.to(dev).contiguous(), requires_grad=False)
        mod.quant_granularity = granularity
        mod.quant_group_size = group_size
        bytes_before = w.numel() * 2
        bytes_after = q.numel() + s.numel() * 2
        # 误差在 GPU 上按【前 SAMPLE 行】估算，别整层展开成 fp32。
        # ★ 为什么要省这几百 MB：allocate_kv_cache 的预算是
        #   `0.9*total - used - peak + current` —— 加载期的一次性峰值会【永久】吃掉
        #   KV 池容量。整层 fp32 展开会把 peak 顶上去 ~0.3 GB（实测 174→163 块）。
        SAMPLE = min(w.shape[0], 256)
        wf = w[:SAMPLE].to(torch.float32)
        deq = dequantize_weight(q[:SAMPLE], s[:SAMPLE], granularity, group_size)
        diff = wf - deq
        rel = (diff.norm() / wf.norm()).item()
        del wf, deq, diff
        done.append({"name": name, "shape": list(w.shape),
                     "rel_err": rel,
                     "saturate_frac": (q.abs() == QMAX).float().mean().item(),
                     "scale_min": float(s.float().min()), "scale_max": float(s.float().max()),
                     "bytes_before": bytes_before, "bytes_after": bytes_after})
        if verbose:
            print(f"  [int8] {name:<46} {tuple(w.shape)} rel_err={rel:.5f}")
    missing = want - {d["name"] for d in done}
    if missing:
        raise KeyError(f"这些模块名在模型里找不到: {sorted(missing)}")
    # 量化过程中 freed 的 bf16 权重还在 caching allocator 手里；
    # 不还回去的话 `used`（= total-free）会偏大，KV 池跟着变小。
    torch.cuda.empty_cache()
    return done


# ---------------------------------------------------------------------------
# 量化 / 反量化（纯 torch）
# ---------------------------------------------------------------------------
def quantize_weight(w: torch.Tensor, granularity: str = "per_channel", group_size: int = 128):
    """对称量化，零点恒为 0。返回 (q_int8, scale_bf16)。

    w: [N, K]
      per_channel -> scale [N]
      per_group   -> scale [N, K // group_size]
    """
    assert w.dim() == 2, f"只处理 2D 权重，收到 {tuple(w.shape)}"
    N, K = w.shape
    wf = w.to(torch.float32)
    if granularity == "per_channel":
        amax = wf.abs().amax(dim=1, keepdim=True)                       # [N,1]
        scale = (amax / QMAX).clamp_min(1e-8)
        q = torch.round(wf / scale).clamp(-QMAX, QMAX).to(torch.int8)
    elif granularity == "per_group":
        assert group_size > 0 and K % group_size == 0
        amax = wf.view(N, K // group_size, group_size).abs().amax(dim=2, keepdim=True)
        scale = (amax / QMAX).clamp_min(1e-8)                           # [N, K/G, 1]
        q = torch.round(wf.view(N, K // group_size, group_size) / scale)
        q = q.clamp(-QMAX, QMAX).to(torch.int8).view(N, K)
        scale = scale.squeeze(-1).contiguous()                          # [N, K/G]
    else:
        raise ValueError(granularity)
    if granularity == "per_channel":
        scale = scale.squeeze(1).contiguous()                           # [N]
    return q.contiguous(), scale.to(torch.bfloat16)


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor,
                      granularity: str = "per_channel", group_size: int = 128):
    """参照反量化（fp32 精度），只用于正确性对拍，不上热路径。"""
    N, K = q.shape
    qf = q.to(torch.float32)
    if granularity == "per_channel":
        return qf * scale.to(torch.float32).unsqueeze(1)
    G = group_size
    return (qf.view(N, K // G, G) * scale.to(torch.float32).unsqueeze(-1)).view(N, K)


# ---------------------------------------------------------------------------
# 融合反量化的 GEMM（Triton）
# ---------------------------------------------------------------------------
@triton.jit
def _int8_dequant_gemm_kernel(
    X, W, S, BIAS, PART, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr,        # 0 = per-channel
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,           # 1 = 不分 K
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """★ 访存要点：W 是 [N, K] 行主序，**k 方向连续**。

    所以沿 N 方向切 tile 时，必须把 W 读成 [BLOCK_N, BLOCK_K]（末轴 k 连续，
    可向量化），再用 tl.trans 翻给 tl.dot；**不能**读成 [BLOCK_K, BLOCK_N]
    —— 那样末轴是 stride=K 的 n，访存完全散掉。
    这是第一版慢的真原因：qkv/gate_up 只跑到 530 GB/s（cuBLAS bf16 是 730）。
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # K 方向切分（split-K）
    k_tile = tl.cdiv(K, SPLIT_K * BLOCK_K) * BLOCK_K
    k_start = pid_k * k_tile
    k_end = tl.minimum(k_start + k_tile, K)
    offs_k = k_start + offs_k

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N
    for _ in range(0, tl.cdiv(k_end - k_start, BLOCK_K)):
        k_mask = offs_k < k_end
        x = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
        # ★ w 是 [BLOCK_N, BLOCK_K]，k 在【末轴】；掩码必须是 k_mask[None, :]，
        #   写成 k_mask[:, None] 在 BN==BK 时能编译但语义是错的（按 n 掩码，K 尾巴会读脏）。
        w = tl.load(w_ptrs, mask=n_mask & k_mask[None, :], other=0)
        wf = w.to(tl.bfloat16)
        if GROUP_SIZE > 0:
            g = offs_k // GROUP_SIZE
            sc = tl.load(S + offs_n[:, None] * (K // GROUP_SIZE) + g[None, :],
                         mask=n_mask, other=1.0)
            wf = wf * sc
        acc = tl.dot(x, tl.trans(wf), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    if GROUP_SIZE == 0:
        sc = tl.load(S + offs_n, mask=offs_n < N, other=1.0)
        acc = acc * sc[None, :]

    if SPLIT_K > 1:
        # 部分和写进 scratch，由 reduce kernel 汇总（decode 时 M 很小，reduce 很便宜）
        p_ptrs = PART + (pid_k * M * N) + offs_m[:, None] * N + offs_n[None, :]
        tl.store(p_ptrs, acc, mask=m_mask & (offs_n[None, :] < N))
        return

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask & (offs_n[None, :] < N))


_COMBINE_BLOCK_N = 1024
_SPLITK_MAX_M = 64      # M 超过它就退回 split-K=1（prefill 不需要 split-K，且省掉大 scratch）


@triton.jit
def _splitk_reduce_kernel(PART, BIAS, Y, M, N, SPLIT_K: tl.constexpr,
                          HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(PART + s * M * N + pid_m * N + offs_n, mask=n_mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=n_mask, other=0.0)
    tl.store(Y + pid_m * N + offs_n, acc.to(tl.bfloat16), mask=n_mask)


# 由 scripts/bench_int8_gemm.py 的 SWEEP 选出（draft FFN 与 target 形状共用同一组）：
#   BM16xBN64xBK64 w4 s4 split-K=4 -> draft gate_up 1.97x / down 1.42x（M=1）
_BLOCK = {"per_channel": dict(BM=16, BN=64, BK=64, warps=4, stages=4, split=4),
          "per_group": dict(BM=16, BN=64, BK=64, warps=4, stages=4, split=4)}


def int8_linear(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
                bias: torch.Tensor | None = None,
                granularity: str = "per_channel", group_size: int = 128,
                out_features: int | None = None, cfg_: dict | None = None):
    """y = x @ dequant(qweight).T + bias，反量化融合在 kernel 里。

    x: [..., K] bf16（行主序，最后两维连续即可）
    qweight: [N, K] int8
    scale:   [N]（per_channel）或 [N, K//group_size]（per_group）
    """
    assert qweight.dtype == torch.int8
    N, K = qweight.shape
    N = out_features or N
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    assert x2.shape[1] == K, f"K 不匹配: x {tuple(x2.shape)} vs w {tuple(qweight.shape)}"
    x2 = x2.contiguous()
    qw = qweight if qweight.is_contiguous() else qweight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    cfg = dict(_BLOCK[granularity])
    if cfg_:
        cfg.update(cfg_)
    # ★ split-K 只用于【decode】(M 小、并行度不够)；prefill 的 M 很大，
    #   N/M 方向本身就有足够并行度，split-K 不但没用，还会分配
    #   [SPLIT_K, M, N] 的 fp32 scratch —— warmup 的 prefill M=2048 时这就有 200 MB，
    #   把 allocate_kv_cache 的 peak 顶上去，**KV 池直接少十几块**（实测 174→163）。
    sk = int(cfg["split"]) if M <= _SPLITK_MAX_M else 1
    grid = (triton.cdiv(N, cfg["BN"]), triton.cdiv(M, cfg["BM"]), sk)
    part = None
    if sk > 1:
        part = torch.empty((sk, M, N), device=x.device, dtype=torch.float32)
    _int8_dequant_gemm_kernel[grid](
        x2, qw, scale, bias if bias is not None else x2, part if part is not None else x2, out,
        M, N, K,
        x2.stride(0), x2.stride(1),
        qw.stride(0), qw.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=(group_size if granularity == "per_group" else 0),
        HAS_BIAS=bias is not None,
        SPLIT_K=sk,
        BLOCK_M=cfg["BM"], BLOCK_N=cfg["BN"], BLOCK_K=cfg["BK"],
        num_warps=cfg["warps"], num_stages=cfg["stages"],
    )
    if sk > 1:
        _splitk_reduce_kernel[(M, triton.cdiv(N, _COMBINE_BLOCK_N))](
            part, bias if bias is not None else x2, out, M, N,
            SPLIT_K=sk, HAS_BIAS=bias is not None, BLOCK_N=_COMBINE_BLOCK_N,
            num_warps=4)
    return out.view(*x.shape[:-1], N)
