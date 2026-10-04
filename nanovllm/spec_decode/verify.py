"""
投机验证 —— 拒绝采样，保证【无损】。

无损的含义
----------
投机解码不是「近似加速」，而是【数学上保证输出分布与原模型逐 token 一致】。
标准做法（Leviathan et al. 2023）：

    对每个候选 token x_i，草稿分布 q_i、目标分布 p_i：
        以概率  min(1, p_i[x_i] / q_i[x_i])  接受它
        一旦某位被拒绝 → 它之后的所有候选全部作废
        并用修正分布  normalize(max(0, p_i − q_i))  采一个 bonus token 顶上

这样得到的输出分布，严格等于目标模型的分布。

实现要点
--------
用 q_pad 技巧把「全部接受」和「第 n 位拒绝」合并成一个表达式，
少一个分支就少一类 bug：

    给 q 末尾补一行零，
    则 (p_n − q_n)+ 在 n == k 时（全部接受）自动退化成 p_n — 0 = p_n，
    正好就是 bonus token 该用的分布。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class VerifyResult:
    """验证结果。"""
    accepted: torch.Tensor      # (batch,) 实际接受的候选个数（长度前缀）
    bonus: torch.Tensor         # (batch,) 被拒位置上重采的 token；全接受时为 -1
    n_proposed: torch.Tensor    # (batch,) 提议了几个候选
    accept_mask: torch.Tensor   # (batch, k) 逐位的接受掩码，便于统计每步接受率


@torch.no_grad()
def verify_batch(
    draft_logits: torch.Tensor,    # (batch, k, vocab)     草稿分布的 logits
    target_logits: torch.Tensor,   # (batch, k+1, vocab)   目标模型的 logits
    draft_tokens: torch.Tensor,    # (batch, k) 草稿采出的 token
    temperatures: Optional[torch.Tensor] = None,  # (batch,)
) -> VerifyResult:
    """验证一批序列的候选 token。

    参数
    ----
    draft_logits : (batch, k, vocab)
        草稿模型对 k 个候选位置的输出。
    target_logits : (batch, k+1, vocab)
        目标模型的输出。多一个位置是因为 target 要同时算出
        「验证 k 个候选」+「补一个 bonus」的分布。
    draft_tokens : (batch, k)
        草稿模型实际采出的候选 token。
    temperatures : (batch,) 或 None
        逐请求温度。None 视为全 1（贪心）。

    返回
    ----
    VerifyResult
    """
    batch, k, vocab = draft_logits.shape
    device = draft_logits.device

    # ---------- 1. 转成概率 ----------
    q = _softmax_with_temp(draft_logits, temperatures)      # (batch, k, vocab)
    p = _softmax_with_temp(target_logits, temperatures)     # (batch, k+1, vocab)

    # ---------- 2. 逐位计算接受概率 ----------
    # q_tok[i, j] = 草稿模型给第 i 条序列第 j 个候选的概率
    # draft_tokens 是 (batch, k)，q 是 (batch, k, vocab) → 需要补一维才能 gather
    q_tok = q.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)   # (batch, k)
    # p_tok[i, j] = 目标模型在同一位置的同一个 token 的概率
    p_tok = p[:, :k, :].gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)  # (batch, k)

    # accept_prob = min(1, p/q)，用 clamp 避免除零
    ratio = torch.where(
        q_tok > 0,
        torch.clamp(p_tok / q_tok.clamp_min(1e-10), max=1.0),
        torch.zeros_like(q_tok),
    )

    u = torch.rand(batch, k, device=device, dtype=ratio.dtype)
    accept_mask = u < ratio                                    # (batch, k)

    # ---------- 3. 找最长 True 前缀 ----------
    # 被拒之后的候选全部作废，所以接受长度 = 第一个 False 的下标
    is_rejected = ~accept_mask
    has_reject = is_rejected.any(dim=1)                        # (batch,)
    first_reject = is_rejected.float().argmax(dim=1)           # (batch,) 全 True 时是 0
    accepted = torch.where(has_reject, first_reject, torch.full_like(first_reject, k))
    accepted = accepted.long()                                 # (batch,)

    # ---------- 4. bonus token ----------
    # q_pad 技巧：给 q 末尾补一行零。
    # 当 accepted == k（全部接受）时，(p_k − q_k)+ 中的 q_k 变成 0，
    # 于是 resid 自动退化成 p_k —— 正好是 bonus token 该用的分布。
    # 这样「全部接受」和「第 n 位拒绝」合并成同一个表达式，少一个分支。
    q_pad = torch.cat([q, torch.zeros_like(q[:, :1, :])], dim=1)      # (batch, k+1, vocab)
    # p 侧同理补一行，避免 accepted == k 时越界（补的��会被 resid 抵消掉）
    p_pad = torch.cat([p, torch.zeros_like(p[:, :1, :])], dim=1)      # (batch, k+1, vocab)

    idx = accepted.unsqueeze(1).unsqueeze(2).expand(batch, 1, vocab)
    resid_p = p_pad.gather(1, idx).squeeze(1)                    # (batch, vocab)
    resid_q = q_pad.gather(1, idx).squeeze(1)                   # (batch, vocab)

    resid = torch.clamp(resid_p - resid_q, min=0.0)
    # 数值兜底：理论上 resid.sum() > 0（因为既然被拒，p≠q）
    resid_sum = resid.sum(dim=-1, keepdim=True)
    resid = torch.where(resid_sum > 0, resid, resid_p)
    resid = resid / resid.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    # 全接受的序列不需要 bonus（下一个位置就是 target 自己算的）
    bonus = _gumbel_argmax(resid)                               # (batch,)
    bonus = torch.where(has_reject, bonus, torch.full_like(bonus, -1))

    n_proposed = torch.full((batch,), k, dtype=torch.long, device=device)
    return VerifyResult(
        accepted=accepted,
        bonus=bonus,
        n_proposed=n_proposed,
        accept_mask=accept_mask,
    )


def _softmax_with_temp(logits: torch.Tensor, temperatures: Optional[torch.Tensor]) -> torch.Tensor:
    """按逐请求温度做 softmax。logits 是 (..., vocab)。"""
    x = logits.float()
    if temperatures is not None:
        shape = [1] * (x.dim() - 1) + [-1]
        x = x / temperatures.reshape(shape)
    return torch.softmax(x, dim=-1)


def _gumbel_argmax(probs: torch.Tensor) -> torch.Tensor:
    """按 probs 采样一个 token。

    用 Gumbel-max：argmax(log p + G)，等价于按 Categorical(p) 采样。
    写成 probs / Exp(1) 省掉两次 log，且与 nano-vllm 现有 Sampler 的技巧一致。
    """
    noise = torch.empty_like(probs).exponential_(1.0).clamp_min_(1e-10)
    return (probs / noise).argmax(dim=-1)