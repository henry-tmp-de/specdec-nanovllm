"""Teacher-forced distribution scores and token-weighted NLL aggregation.

The scorer keeps tensors on their original device and performs no scalar read
or synchronization. It does not sample tokens or infer acceptance / tau.
"""
from __future__ import annotations

import math

POSITIONS = 7
POSITION_FIELDS = ("nll_quant", "nll_bf16", "nll_target", "kl_bf16_quant", "kl_target_quant",
                   "overlap_target_quant", "argmax_agree_bf16", "argmax_agree_target")
DIAGNOSTIC_FIELDS = ("logits_finite", "scores_finite", "target_norm_max_abs", "bf16_norm_max_abs",
                     "quant_norm_max_abs")


def score_distributions(target_logits, bf16_logits, quant_logits, labels):
    """Score seven matching histories; all three logits must be [7, vocab].

    target_logits are target p; bf16_logits and quant_logits are draft q on
    the identical teacher-forced predecessors. Temperature is fixed to 1.
    Each returned position field is a length-7 device tensor; diagnostic fields
    are scalar device tensors. Convert to host only after the case finishes.
    """
    import torch

    tensors = (target_logits, bf16_logits, quant_logits)
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all logits must be tensors")
    shape = target_logits.shape
    if (len(shape) != 2 or shape[0] != POSITIONS or shape[1] < 2 or
            any(x.shape != shape or x.device != target_logits.device or not x.is_floating_point() for x in tensors)):
        raise ValueError("logits must be same-device floating tensors of shape [7, vocab]")
    if isinstance(labels, (list, tuple)):
        if len(labels) != POSITIONS or any(type(x) is not int or not 0 <= x < shape[1] for x in labels):
            raise ValueError("labels must be seven valid token IDs")
        label = torch.tensor(labels, dtype=torch.int64, device=target_logits.device)
    elif isinstance(labels, torch.Tensor) and labels.dtype == torch.int64 and labels.shape == (POSITIONS,):
        # Tensor label range must be validated before GPU transfer by the caller.
        label = labels.to(device=target_logits.device)
    else:
        raise ValueError("labels must be a list or int64 tensor of length seven")

    finite_logits = torch.stack([torch.isfinite(x).all() for x in tensors]).all()
    log_target, log_bf16, log_quant = [torch.log_softmax(x.float(), dim=-1) for x in tensors]
    p_target, p_bf16, p_quant = [x.exp() for x in (log_target, log_bf16, log_quant)]
    gather = label[:, None]
    result = {
        "nll_quant": -log_quant.gather(-1, gather).squeeze(-1),
        "nll_bf16": -log_bf16.gather(-1, gather).squeeze(-1),
        "nll_target": -log_target.gather(-1, gather).squeeze(-1),
        "kl_bf16_quant": (p_bf16 * (log_bf16 - log_quant)).sum(dim=-1),
        "kl_target_quant": (p_target * (log_target - log_quant)).sum(dim=-1),
        "overlap_target_quant": torch.minimum(p_target, p_quant).sum(dim=-1),
        "argmax_agree_bf16": quant_logits.argmax(dim=-1) == bf16_logits.argmax(dim=-1),
        "argmax_agree_target": quant_logits.argmax(dim=-1) == target_logits.argmax(dim=-1),
        "logits_finite": finite_logits,
        "target_norm_max_abs": (p_target.sum(dim=-1) - 1).abs().max(),
        "bf16_norm_max_abs": (p_bf16.sum(dim=-1) - 1).abs().max(),
        "quant_norm_max_abs": (p_quant.sum(dim=-1) - 1).abs().max(),
    }
    result["scores_finite"] = torch.stack([torch.isfinite(result[name]).all() for name in
                                           POSITION_FIELDS[:6] + DIAGNOSTIC_FIELDS[2:]]).all()
    return result


def pooled_nll(losses):
    """Compute exp(total token NLL / total tokens), never average case exps."""
    values = list(losses)
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("losses must be a nonempty finite token-level sequence")
    total = math.fsum(values)
    mean = total / len(values)
    try:
        score = math.exp(mean)
    except OverflowError as exc:
        raise ValueError("pooled exp(mean NLL) overflows") from exc
    if not math.isfinite(total) or not math.isfinite(mean) or not math.isfinite(score):
        raise ValueError("nonfinite pooled NLL")
    return {"tokens": len(values), "total_nll": total, "mean_nll": mean, "exp_mean_nll": score}


def paired_nll(baseline_losses, candidate_losses):
    """Pool matching positions, then exp(candidate mean NLL - BF16 mean NLL)."""
    baseline, candidate = pooled_nll(baseline_losses), pooled_nll(candidate_losses)
    if baseline["tokens"] != candidate["tokens"]:
        raise ValueError("baseline and candidate counts differ")
    delta = candidate["mean_nll"] - baseline["mean_nll"]
    try:
        relative = math.exp(delta)
    except OverflowError as exc:
        raise ValueError("relative exp(delta NLL) overflows") from exc
    if not math.isfinite(relative):
        raise ValueError("nonfinite relative score")
    return {"baseline": baseline, "candidate": candidate, "delta_mean_nll": delta,
            "relative_exp_delta_nll": relative}
