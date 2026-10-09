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
    bonus: torch.Tensor         # (batch,) 本轮额外产出的 token（见下，永远有效）
    n_proposed: torch.Tensor    # (batch,) 提议了几个候选
    accept_mask: torch.Tensor   # (batch, k) 逐位的接受掩码，便于统计每步接受率


@torch.no_grad()
def verify_batch(
    draft_probs: torch.Tensor,     # (batch, k, vocab)   草稿模型的【原始 logits】
    target_logits: torch.Tensor,   # (batch, k+1, vocab)  目标模型的 logits
    draft_tokens: torch.Tensor,    # (batch, k) 草稿采出的 token
    temperatures: Optional[torch.Tensor] = None,  # (batch,)
    draft_is_point_mass: bool = False,
) -> VerifyResult:
    """验证一批序列的候选 token。

    参数
    ----
    draft_probs : (batch, k, vocab)
        草稿模型的**原始 logits**（不是概率！）。
        ★ 参数名里的 "probs" 是历史遗留，容易误导，这里按【logits】用：
          内部会做 q = softmax(draft_probs / temperature)。
          传 softmax 过的概率进来的话，会被再 softmax 一次、压成近均匀分布
          （151936 词表上 q 的最大值只剩 1.2e-5），而草稿 token 是从真分布
          q 采的 —— 一致性前提被破坏，无损性不再成立。
          实测 q≡p 时：传 logits 接受率 1.0000 / TV 0.0032；
                       传概率   接受率 0.8895 / TV 0.1075（有偏，不随 N 收敛）。
          唯一例外是 draft_is_point_mass=True（n-gram 路线），见下。
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
    # ★★ 对齐关系（这里最容易错，错了不报错、只是接受率悄悄变低）：
    #
    #   验证时送进 forward 的 k+1 个 token 位于
    #       position:  len-1, len, len+1, ..., len+k-1
    #   其中 len-1 是最后一个已确认 token，len+j 是第 j 个候选。
    #   模型在 position p 的输出是「p+1 位置的 token」的分布，所以
    #       第 r 行  =  位置 len+r-1 的输出  =  位置 len+r 的 token 的分布
    #     => 候选 j  对齐【第 j 行】，下标相同、不偏移。
    #   （对照：q_j 来自 draft 在 position len+j-1 的输出，同一个 position，
    #     所以 q 的第 j 行和 p 的第 j 行本来就是同一个位置的分布。）
    #
    #   ✗ 曾经的错误写法：p[:, 1:k+1]（整体挪一行）。
    #     后果：候选 0 被拿去和第 1 行比 —— 0.6B→4B 这种本来就高度一致的
    #     组合，接受率从应有的 ~0.84 掉到实测 0.16，而且无损性也一并破裂
    #     （见 tests/test_spec_decode.py 测试 6 的合成算例：
    #       草稿逐行等于目标时，正确实现必须【全部接受】，错位实现接受 0 个）。
    q_tok = q.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)          # (batch, k)
    p_for_draft = p[:, :k, :]                # 候选 j <-> 第 j 行
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
    # ★★ 和上面同一套对齐：候选 j <-> 第 j 行，所以
    #   「第 accepted 个候选被拒」-> bonus 落在位置 len+accepted -> 取【第 accepted 行】。
    #   全部接受（accepted == k）-> bonus 落在位置 len+k -> 取第 k 行。
    #   两种情况的下标都是 accepted —— 这正是 q_pad 技巧要利用的巧合：
    #
    # q_pad：给 q 末尾补一行零（q 只有 k 行）。当 accepted == k 时，
    #   下一个 token 本该由 target 自己算，其分布是 p[k]；而 (p_k − q_k)+ 里的
    #   q_k 被补成 0 后，resid 正好退化成 p[k]
    #   —— 「全部接受」和「第 j 位拒绝」合并成同一个表达式，少一个分支。
    q_pad = torch.cat([q, torch.zeros_like(q[:, :1, :])], dim=1)      # (batch, k+1, vocab)
    # p 正常有 k+1 行（引擎路径）；退化成 k 行时（无 bonus 位）夹到 k-1，别越界。
    p_idx = accepted.clamp(max=p.shape[1] - 1).unsqueeze(1).unsqueeze(2).expand(batch, 1, vocab)
    q_idx = accepted.clamp(max=k).unsqueeze(1).unsqueeze(2).expand(batch, 1, vocab)
    resid_p = p.gather(1, p_idx).squeeze(1)                          # (batch, vocab)
    resid_q = q_pad.gather(1, q_idx).squeeze(1)                      # (batch, vocab)

    resid = torch.clamp(resid_p - resid_q, min=0.0)
    # 数值兜底：理论上 resid.sum() > 0（因为既然被拒，p≠q）
    resid_sum = resid.sum(dim=-1, keepdim=True)
    resid = torch.where(resid_sum > 0, resid, resid_p)
    resid = resid / resid.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    # ★★ bonus 【每一轮都要给】，包括 k 个候选全被接受的情况。
    #
    #   全接受时 accepted == k，而 q_pad 把 q 的第 k 行补成了 0，
    #   所以 resid 正好退化成 p[k] —— 也就是 target 自己下一步会算的那个 token。
    #   它【已经在这一轮 forward 里算出来了】（第 k 行），不取就是白算一行。
    #
    #   实测（k=1、α=0.857）：丢掉这个 token 时每步恒定只出 1 个 token
    #   （接受 -> 出候选 1 个；被拒 -> 出 bonus 1 个），
    #   投机解码反而比 baseline 慢；补上之后每步 1.857 个。
    #
    #   而且 scheduler 每步预留的槽位数本来就是 1+k
    #   （见 scheduler.schedule 里的 need = 1 + spec_k），
    #   本来就是按「最多落地 k+1 个 token」设计的。
    bonus = _gumbel_argmax(resid)                               # (batch,)

    n_proposed = torch.full((batch,), k, dtype=torch.long, device=device)
    return VerifyResult(
        accepted=accepted,
        bonus=bonus,
        n_proposed=n_proposed,
        accept_mask=accept_mask,
    )


def _softmax_with_temp(logits: torch.Tensor, temperatures: Optional[torch.Tensor]) -> torch.Tensor:
    """按【逐请求】温度做 softmax。logits 形状 (batch, ..., vocab)。

    ★★ 温度必须沿【请求维】广播。请求维是第 0 维、vocab 是最后一维，所以
       温度要 reshape 成 (B, 1, ..., 1)：
           (B, k, V) -> (B, 1, 1)
           (B, V)    -> (B, 1)
       旧写法 `shape = [1]*(ndim-1) + [-1]` 把 B 个温度塞进了【最后一维】也就是
       vocab 维：B=1 时因为长度 1 的维广播无差别而被掩盖（引擎现在就是逐序列
       调用，B=1，所以一直没暴露），B>1 时直接错 ——
         · V ≠ B：reshape/广播直接 RuntimeError；
         · V == B：每个词的概率被除以【别人的】温度，q 与 p 一起错位，
           min(1, p/q) 随之算错，无损性破裂。
       批量验证（P6/批量投机）一上来就是 B>1，所以这是必须先修的正确性问题。
    """
    x = logits.float()
    if temperatures is not None:
        if temperatures.dim() == 0:
            x = x / temperatures                    # 标量温度：无需广播
        else:
            shape = [temperatures.shape[0]] + [1] * max(0, x.dim() - 1)
            x = x / temperatures.reshape(shape)
    return torch.softmax(x, dim=-1)


def _gumbel_argmax(probs: torch.Tensor) -> torch.Tensor:
    """按 probs 采样一个 token。

    用 Gumbel-max：argmax(log p + G)，等价于按 Categorical(p) 采样。
    写成 probs / Exp(1) 省掉两次 log，且与 nano-vllm 现有 Sampler 的技巧一致。
    """
    noise = torch.empty_like(probs).exponential_(1.0).clamp_min_(1e-10)
    return (probs / noise).argmax(dim=-1)