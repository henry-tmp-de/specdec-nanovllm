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
    draft_probs: torch.Tensor,     # (batch, k, vocab)   草稿分布的【概率】
    target_logits: torch.Tensor,   # (batch, k+1, vocab)  目标模型的 logits
    draft_tokens: torch.Tensor,    # (batch, k) 草稿采出的 token
    temperatures: Optional[torch.Tensor] = None,  # (batch,)
    draft_is_point_mass: bool = False,
) -> VerifyResult:
    """验证一批序列的候选 token。

    参数
    ----
    draft_probs : (batch, k, vocab)
        草稿分布的**概率**（不是 logits）。
        ★ 必须传概率而不是 logits —— 见下方 draft_is_point_mass 的说明。
    target_logits : (batch, k+1, vocab)
        目标模型的 logits。多一个位置是因为 target 要同时算出
        「验证 k 个候选」+「补一个 bonus」的分布。
    draft_tokens : (batch, k)
        草稿模型实际采出的候选 token。
    temperatures : (batch,) 或 None
        逐请求温度。None 视为全 1（贪心）。
    draft_is_point_mass : bool
        True 表示 draft_probs 是【单点分布】（n-gram 提议就是这种）。
        ★ 关键：单点分布无法用 logits 表达 ——
          logits=[30,-30,...] 过softmax 后得到的是 0.9999 而不是 1.0，
          那些「残余概率」会让修正分布 max(0, p−q) 算错，
          最终 bonus 分布被污染（实测：接受概率 1.0 的样本只被接受了 35%）。
          所以这种情况直接【跳过归一化】，q 就是给定的概率。

    返回
    ----
    VerifyResult
    """
    batch, k, vocab = draft_probs.shape
    device = draft_probs.device

    # ---------- 1. 转成概率 ----------
    if draft_is_point_mass:
        q = draft_probs                                # 已是概率，别再 softmax
    else:
        q = _softmax_with_temp(draft_probs, temperatures)
    p = _softmax_with_temp(target_logits, temperatures)     # (batch, k+1, vocab)

    # ---------- 2. 逐位计算接受概率 ----------
    # ★★ 索引对齐是这里最容易错的地方：
    #   draft_logits 的第 j 个候选，对应 target_logits 的第 j+1 个位置，
    #   因为 target 的第 0 个位置是「本该正常 decode 的那个 token」，
    #   第 1..k 个位置才是对候选 0..k-1 的验证。
    #   写成 p[:, :k, :] 会整体偏移一位 —— 不会报错，但接受判断全错。
    q_tok = q.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)          # (batch, k)
    # target 传k+1 个位置（末尾多一个用于 bonus）；传 k 个时按无偏移处理。
    if p.shape[1] >= k + 1:
        p_for_draft = p[:, 1:k + 1, :]       # ★ 偏移 1 位
    else:
        p_for_draft = p[:, :k, :]            # 退化情况：target 只给了 k 个
    p_tok = p_for_draft.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)  # (batch, k)

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
    # ★★ 这里必须用【p[:, 1:k+1]】(偏移后的验证位)，不是 p[:, :k]。
    #
    #   布局：target 的第 0 位是「本该正常 decode 的那个 token」，
    #         第 1..k 位才是候选 0..k-1 的真值分布。
    #   所以「第 j 个候选被拒」时，要用 p[j+1] 重采 ——
    #   写成 p[:, :k] 会取到【错一位】的分布，
    #   结果是 bonus 均匀落在词表上（实测），而不是集中在 target 真正偏好的 token。
    #
    # q_pad 技巧：给 q 末尾补一行零。当 accepted == k（全部接受）时，
    #   下一个 token 本该由 target 自己算，其分布是 p[k]，
    #   而 (p_k − q_k)+ 里的 q_k 变成 0 后 resid 正好退化成 p[k]
    #   —— 这样「全部接受」和「第 j 位拒绝」合并成同一个表达式，少一个分支。
    q_pad = torch.cat([q, torch.zeros_like(q[:, :1, :])], dim=1)      # (batch, k+1, vocab)
    # 末尾再补一个位置：accepted==k 时用它取 p[k]（下一步的分布）
    p_bonus_src = p                                    # (batch, k+1, vocab)
    p_bonus_pad = torch.cat([p_bonus_src, torch.zeros_like(p_bonus_src[:, :1, :])], dim=1)

    # 索引偏移：候选 j 被拒 -> 取 p[j+1]
    has_bonus_slot = p.shape[1] >= k + 1
    bonus_idx = (accepted + 1).clamp(max=p.shape[1] - 1) if has_bonus_slot else accepted
    q_idx = accepted.clamp(max=k)
    idx = bonus_idx.unsqueeze(1).unsqueeze(2).expand(batch, 1, vocab)
    resid_p = p_bonus_pad.gather(1, idx).squeeze(1)         # (batch, vocab)
    resid_q = q_pad.gather(1, q_idx.unsqueeze(1).unsqueeze(2).expand(batch, 1, vocab)).squeeze(1)

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