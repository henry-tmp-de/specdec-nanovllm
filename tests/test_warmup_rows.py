"""CPU-only coverage of actual warmup -> run -> prepare_prefill input rows."""
from types import SimpleNamespace

import numpy  # load the local OpenMP runtime before CPU torch
import pytest
import torch

from test_dspark_window_runner import DSparkModel, ModelRunner, runner_module


CASES = [
    (32768, 4360, 8, [4360] * 7 + [2248]),  # observed B8 4K undercoverage
    (31, 10, 8, [10, 10, 10, 1]),             # a short last sequence
    (6, 10, 8, [6]),                           # budget smaller than model length
    (30, 10, 8, [10, 10, 10]),                # exact division
    (100, 10, 3, [10, 10, 10]),               # max_num_seqs is limiting
    (16384, 4096, 512, [4096] * 4),          # default configuration
]


def make_draft():
    cfg = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=4, attention_bias=False, intermediate_size=24,
                          num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                          markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                          mask_token_id=3, enable_confidence_head=True,
                          confidence_head_with_markov=True, rope_theta=10000)
    return DSparkModel(cfg, torch.randn(37, 16), torch.randn(37, 16))


@pytest.mark.parametrize("mode", ["no_spec", "dspark_w0", "dspark_window"])
@pytest.mark.parametrize("budget,max_len,max_seqs,expected", CASES)
def test_warmup_covers_reachable_rows(monkeypatch, mode, budget, max_len, max_seqs, expected):
    tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(*args, **kwargs)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner_module, "_mem_snap", lambda *args: None)

    observed = {}
    class Target:
        def __call__(self, ids, positions, layers=None):
            observed["target_rows"] = ids.numel()
            observed["positions"] = positions.tolist()
            observed["cu_seqlens_q"] = runner_module.get_context().cu_seqlens_q.tolist()
            observed["feature_rows"] = ids.numel() if layers is not None else None
            hidden = torch.zeros(ids.numel(), 1).expand(-1, 16)
            if layers is None:
                return hidden
            return hidden, torch.zeros(ids.numel(), 1).expand(-1, 32)

        def compute_logits(self, hidden):
            return hidden

    class Proposer:
        def reset_watermark(self, tokens):
            pass

    r = ModelRunner.__new__(ModelRunner)
    r.config = SimpleNamespace(max_num_batched_tokens=budget, max_model_len=max_len,
                               max_num_seqs=max_seqs,
                               dspark_window_tokens=8 if mode == "dspark_window" else 0)
    r.block_size = 4
    r.rank, r.enforce_eager, r.draft_model = 0, True, None
    r.dspark_model = None if mode == "no_spec" else make_draft()
    r.dspark_target_layers = (0, 1)
    r.spec_proposer = None if mode == "no_spec" else Proposer()
    r.model = Target()
    r.prepare_sample = lambda seqs: torch.ones(len(seqs))
    r.sampler = lambda logits, temperatures: torch.zeros(len(temperatures), dtype=torch.long)
    real_prepare = r.prepare_prefill
    def prepare(seqs, advance_draft=True):
        observed["seqs"] = [(len(s), s.num_scheduled_tokens, list(s.block_table), s.dspark_owner)
                            for s in seqs]
        ids, positions = real_prepare(seqs, advance_draft)
        observed["prepared_rows"] = ids.numel()
        return ids, positions
    r.prepare_prefill = prepare

    r.warmup_model()
    if budget in (32768, 31):
        lengths = [item[0] for item in observed["seqs"]]
        print(f"{mode} budget={budget} lengths={lengths} "
              f"target_rows={observed['target_rows']} expected_rows={sum(expected)}")
    assert observed["seqs"] == [(n, n, [], None) for n in expected]
    assert observed["prepared_rows"] == observed["target_rows"] == sum(expected)
    assert observed["positions"] == [p for n in expected for p in range(n)]
    assert observed["cu_seqlens_q"] == [sum(expected[:i]) for i in range(len(expected) + 1)]
    assert observed["feature_rows"] == (None if mode == "no_spec" else sum(expected))
    assert r.warmup_profile == {
        "max_num_batched_tokens": budget, "max_model_len": max_len, "max_num_seqs": max_seqs,
        "target_rows": sum(expected), "scheduled_rows": observed["target_rows"],
        "sequence_lengths": expected,
    }
    if mode != "no_spec":
        assert r.dspark_model.cache is None


@pytest.mark.parametrize("budget,max_len,max_seqs", [(0, 4, 1), (4, 0, 1), (4, 4, 0)])
def test_invalid_warmup_limits_fail_before_forward(monkeypatch, budget, max_len, max_seqs):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner_module, "_mem_snap", lambda *args: None)
    r = ModelRunner.__new__(ModelRunner)
    r.config = SimpleNamespace(max_num_batched_tokens=budget, max_model_len=max_len, max_num_seqs=max_seqs)
    r.run = lambda *args: pytest.fail("invalid limits must not reach model forward")
    with pytest.raises(ValueError, match="positive"):
        r.warmup_model()
    assert not hasattr(r, "warmup_profile")
