"""CPU contracts for score direction, normalization and token pooling."""
import importlib.util
import math
from pathlib import Path

import numpy  # noqa: F401 -- initialize before the local CPU torch build
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_math", HERE / "quality_metrics.py")
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


def test_scores_direction_overlap_and_argmax():
    p, q0, q = [0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.25, 0.5, 0.25]
    logits = lambda probs: torch.tensor([[math.log(x) for x in probs]] * 7)
    result = metric.score_distributions(logits(p), logits(q0), logits(q), [0, 1, 2, 0, 1, 2, 0])
    assert result["logits_finite"].item() and result["scores_finite"].item()
    assert all(result[k].shape == (7,) for k in metric.POSITION_FIELDS)
    assert result["nll_quant"][0].item() == pytest.approx(-math.log(q[0]), abs=1e-6)
    assert result["nll_target"][1].item() == pytest.approx(-math.log(p[1]), abs=1e-6)
    assert result["kl_bf16_quant"][0].item() == pytest.approx(sum(a * math.log(a / b) for a, b in zip(q0, q)), abs=1e-6)
    assert result["kl_target_quant"][0].item() == pytest.approx(sum(a * math.log(a / b) for a, b in zip(p, q)), abs=1e-6)
    assert result["overlap_target_quant"][0].item() == pytest.approx(1 - sum(abs(a - b) for a, b in zip(p, q)) / 2, abs=1e-6)
    assert result["argmax_agree_bf16"].all().item()
    assert not result["argmax_agree_target"].any().item()
    assert max(result[k].item() for k in ("target_norm_max_abs", "bf16_norm_max_abs", "quant_norm_max_abs")) < 1e-6


def test_same_distribution_zero_kl_and_stable_large_logits():
    logits = torch.tensor([[10000.0, 0.0, -10000.0]] * 7)
    result = metric.score_distributions(logits, logits, logits, [2] * 7)
    assert result["scores_finite"].item()
    assert torch.equal(result["kl_bf16_quant"], torch.zeros(7))
    assert torch.equal(result["kl_target_quant"], torch.zeros(7))
    assert torch.equal(result["overlap_target_quant"], torch.ones(7))
    assert result["nll_quant"][0].item() == 20000.0


def test_nonfinite_and_bad_shapes_not_silent():
    logits = torch.zeros((7, 3))
    bad = logits.clone()
    bad[0, 0] = float("inf")
    result = metric.score_distributions(logits, logits, bad, [0] * 7)
    assert not result["logits_finite"].item() and not result["scores_finite"].item()
    with pytest.raises(ValueError):
        metric.score_distributions(logits, logits, logits[:6], [0] * 7)
    with pytest.raises(ValueError):
        metric.score_distributions(logits, logits, logits, [3] * 7)


def test_pool_is_token_weighted_before_exponentiation():
    result = metric.pooled_nll([0.0, 0.0, 0.0, 4.0])
    assert result["tokens"] == 4 and result["total_nll"] == 4.0
    assert result["exp_mean_nll"] == pytest.approx(math.e)
    assert result["exp_mean_nll"] != pytest.approx((1 + math.exp(4)) / 2)
    pair = metric.paired_nll([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
    assert pair["delta_mean_nll"] == pytest.approx(-0.5)
    assert pair["relative_exp_delta_nll"] == pytest.approx(math.exp(-0.5))
    with pytest.raises(ValueError):
        metric.paired_nll([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        metric.pooled_nll([float("nan")])
