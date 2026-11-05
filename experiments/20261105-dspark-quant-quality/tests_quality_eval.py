"""CPU contracts for causal alignment, KV rebuild and quant-bank restoration.

Fake target/draft forwards are explicitly CPU simulations, not GPU evidence.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy  # noqa: F401 -- initialize before local CPU torch
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import quality_eval as qe
from quality_metrics import score_distributions


def synthetic_manifest():
    cases = []
    for panel_index, panel in enumerate(("code", "math", "general")):
        for rid in range(8):
            prompt = [100 + panel_index * 10 + rid] * qe.BASE_CONTEXT
            generated = [1000 + panel_index * 300 + rid * 300 + i for i in range(256)]
            for offset in qe.OFFSETS:
                prefix = prompt + generated[:offset]
                cases.append({"case_id": f"{panel}-{rid}-{offset}", "panel": panel, "request_id": rid,
                              "offset": offset, "context_tokens": len(prefix), "prefix_ids": prefix,
                              "anchor_id": generated[offset], "label_ids": generated[offset + 1:offset + 8],
                              "prompt_sha256": qe.token_hash(prompt), "prefix_sha256": qe.token_hash(prefix),
                              "output_sha256": qe.token_hash(generated), "input_metadata": {}})
    return {"schema": "dspark-quant-quality-manifest-v1", "seed": 20261105,
            "context_tokens": 1024, "window_tokens": 512, "block_size": 7,
            "offsets": list(qe.OFFSETS), "provenance": {"source_raw_sha256": "a" * 64}, "cases": cases}


def test_manifest_contract_and_cross_offset_alignment():
    data = synthetic_manifest()
    assert len(qe.manifest_cases(data)) == 96
    bad = copy.deepcopy(data)
    case = bad["cases"][1]
    case["prefix_ids"][1024] += 1
    case["prefix_sha256"] = qe.token_hash(case["prefix_ids"])
    with pytest.raises(ValueError, match="offsets"):
        qe.manifest_cases(bad)
    bad = copy.deepcopy(data)
    bad["cases"][0]["label_ids"] = bad["cases"][0]["label_ids"][:-1]
    with pytest.raises(ValueError, match="seven labels"):
        qe.manifest_cases(bad)


def test_causal_inputs_exclude_anchor_and_shift_markov_labels():
    case = {"prefix_ids": [1, 2, 3], "anchor_id": 4, "label_ids": [5, 6, 7, 8, 9, 10, 11]}
    prefix, target_suffix, previous = qe.causal_inputs(case)
    assert prefix == [1, 2, 3]
    assert target_suffix == [4, 5, 6, 7, 8, 9, 10, 11]
    assert previous == [4, 5, 6, 7, 8, 9, 10]


class FakeCache:
    def __init__(self, length):
        self.length = length
    def get_seq_length(self):
        return self.length


class FakeTargetTransformer:
    def __init__(self):
        self.calls = []
    def __call__(self, *, input_ids, attention_mask, position_ids, use_cache, output_hidden_states,
                 past_key_values=None):
        ids = input_ids.tolist()[0]
        self.calls.append({"ids": ids, "mask": attention_mask.tolist()[0],
                           "positions": position_ids.tolist()[0], "past": past_key_values,
                           "hidden": output_hidden_states})
        assert use_cache
        if past_key_values is None:
            length = len(ids)
            states = tuple(torch.full((1, length, 2), layer, dtype=torch.bfloat16) +
                           torch.arange(length, dtype=torch.bfloat16)[None, :, None] / 16
                           for layer in range(35))
            return SimpleNamespace(hidden_states=states, past_key_values=FakeCache(length))
        past_key_values.length += len(ids)
        hidden = torch.tensor(ids, dtype=torch.bfloat16).view(1, -1, 1).expand(1, -1, 2)
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values)


class FakeTarget:
    def __init__(self):
        self.model = FakeTargetTransformer()
        self.embedding = nn.Embedding(64, 2, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(2, 16, bias=False, dtype=torch.bfloat16)
    def get_input_embeddings(self):
        return self.embedding


def test_target_prefix_only_full_cache_positions_and_layer_index():
    target = FakeTarget()
    case = {"prefix_ids": [1, 2, 3, 4, 5], "anchor_id": 20,
            "label_ids": [21, 22, 23, 24, 25, 26, 27]}
    features, logits, fingerprint = qe.target_reference(target, case, window=2)
    assert features.shape == (2, 10) and logits.shape == (7, 16)
    assert features[0].tolist() == [2.1875, 2.1875, 10.1875, 10.1875, 18.25, 18.25,
                                    26.25, 26.25, 34.25, 34.25]
    prefix, suffix = target.model.calls
    assert prefix["ids"] == case["prefix_ids"] and prefix["positions"] == list(range(5))
    assert prefix["mask"] == [1] * 5 and prefix["hidden"] is True
    assert suffix["ids"] == [20, 21, 22, 23, 24, 25, 26, 27]
    assert suffix["positions"] == list(range(5, 13)) and suffix["mask"] == [1] * 13
    assert suffix["hidden"] is False and suffix["past"].get_seq_length() == 13
    teacher_inputs = torch.tensor([20, 21, 22, 23, 24, 25, 26], dtype=torch.bfloat16).view(7, 1).expand(7, 2)
    torch.testing.assert_close(logits, target.lm_head(teacher_inputs))
    changed = {**case, "anchor_id": 30, "label_ids": [31, 32, 33, 34, 35, 36, 37]}
    features2, _, fingerprint2 = qe.target_reference(target, changed, window=2)
    assert torch.equal(features, features2) and fingerprint == fingerprint2
    assert target.model.calls[2]["ids"] == case["prefix_ids"]


def load_dspark_model():
    spec = importlib.util.spec_from_file_location("quality_tiny_dspark_model", ROOT / "nanovllm/spec_decode/dspark_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_model():
    cfg = SimpleNamespace(hidden_size=16, intermediate_size=24, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, attention_bias=False, num_hidden_layers=5,
                          rms_norm_eps=1e-6, vocab_size=31, markov_rank=4, target_layer_ids=[0, 1],
                          block_size=7, mask_token_id=3, enable_confidence_head=True,
                          confidence_head_with_markov=True, rope_theta=10000)
    embedding = torch.randn(31, 16, dtype=torch.bfloat16)
    lm_head = torch.randn(31, 16, dtype=torch.bfloat16)
    before = torch.get_default_dtype()
    model = qe.construct_dspark(load_dspark_model().DSparkModel, cfg, embedding, lm_head)
    assert torch.get_default_dtype() == before
    assert model.fc.weight.dtype == torch.bfloat16 and model.inv_freq.dtype == torch.float32
    assert model.embedding_weight is embedding and model.lm_head_weight is lm_head
    return model


def names():
    return tuple(f"layers.{i}.{suffix}" for i in range(5) for suffix in qe.SUFFIXES["all"])


class FakeQuant(nn.Module):
    pass


def fake_apply(model, preset, fail_at=None):
    assert preset == "all"
    records = []
    for i, name in enumerate(names()):
        qe._put(model, name, FakeQuant())
        records.append({"name": name})
        if i == fail_at:
            raise RuntimeError("synthetic quantization failure")
    return records


def test_group_membership_bank_restoration_and_rope():
    model = tiny_model()
    original = {name: model.get_submodule(name) for name in names()}
    bank = qe.build_bank(model, lambda preset: names(), fake_apply, FakeQuant)
    assert {key: len(value) for key, value in bank.variants.items()} == {
        "none": 0, "q": 5, "k": 5, "v": 5, "o": 5, "gate": 5, "up": 5, "down": 5,
        "q_o": 10, "q_o_down": 15, "kv": 10, "gate_up": 10, "ffn": 15, "attention": 20, "all": 35}
    assert set(bank.variants["q_o_down"]) == {
        f"layers.{i}.{suffix}" for i in range(5)
        for suffix in ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")}
    protected = (model.fc, model.markov_head, model.confidence_head, model.embedding_weight,
                 model.lm_head_weight, model.inv_freq)
    for variant, selected in bank.variants.items():
        with bank.selected(variant):
            assert {name for name in names() if isinstance(model.get_submodule(name), FakeQuant)} == set(selected)
            assert (model.fc, model.markov_head, model.confidence_head, model.embedding_weight,
                    model.lm_head_weight, model.inv_freq) == protected
        assert all(model.get_submodule(name) is original[name] for name in names())
    with pytest.raises(RuntimeError):
        with bank.selected("q"):
            raise RuntimeError("synthetic case failure")
    assert all(model.get_submodule(name) is original[name] for name in names())


def test_partial_bank_creation_failure_restores_all_originals():
    model = tiny_model()
    original = {name: model.get_submodule(name) for name in names()}
    with pytest.raises(RuntimeError, match="synthetic"):
        qe.build_bank(model, lambda preset: names(), lambda m, p: fake_apply(m, p, fail_at=3), FakeQuant)
    assert all(model.get_submodule(name) is original[name] for name in names())


def test_real_quantizer_builds_and_restores_bank_on_cpu(monkeypatch):
    # Import the actual torch quantizer/wrapper with only Triton decorators
    # stubbed: this does not execute a Triton kernel or establish GPU behavior.
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(ROOT / "nanovllm")]
    with pytest.MonkeyPatch.context() as mp:
        for key, value in {"triton": triton, "triton.language": language, "nanovllm": package}.items():
            mp.setitem(sys.modules, key, value)
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            mp.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            return module
        quant = load("nanovllm.layers.quant", ROOT / "nanovllm/layers/quant.py")
        dsq = load("quality_actual_dspark_quant", ROOT / "nanovllm/spec_decode/dspark_quant.py")
        model = tiny_model()
        original = {name: model.get_submodule(name) for name in names()}
        bank = qe.build_bank(model, dsq.expected_names,
                             lambda m, p: dsq.apply_dspark_quant(m, p, chunk_rows=3), dsq.DSparkInt8Linear)
        assert all(model.get_submodule(name) is original[name] for name in names())
        with bank.selected("kv"):
            for name in bank.variants["kv"]:
                module = model.get_submodule(name)
                q, s = quant.quantize_weight(original[name].weight, "per_channel")
                assert torch.equal(module.qweight, q) and torch.equal(module.scale, s)
            assert model.inv_freq.dtype == torch.float32
        assert all(model.get_submodule(name) is original[name] for name in names())


def test_target_checkpoint_inventory_must_match_sample_source():
    current = {"config_sha256": "a" * 64, "weight_files": [{"name": "a.safetensors", "bytes": 123}]}
    source = {"source_target_config_sha256": "a" * 64,
              "source_target_provenance": {"config_sha256": "a" * 64, "weight_files": current["weight_files"]}}
    qe.check_target_matches_manifest(current, source)
    with pytest.raises(ValueError, match="configuration"):
        qe.check_target_matches_manifest({**current, "config_sha256": "b" * 64}, source)
    with pytest.raises(ValueError, match="inventory"):
        qe.check_target_matches_manifest({**current, "weight_files": []}, source)


class FakeDraft:
    def __init__(self):
        self.fc = nn.Linear(1, 1, dtype=torch.bfloat16)
        self.window_tokens, self.rebuilds, self.writes, self.priors = 2, 0, [], []
    def bind_cache(self, num_blocks, block_size):
        self.rebuilds += 1
        self.sentinel = num_blocks * block_size
    def write_context(self, features, positions, slots):
        self.writes.append((features.clone(), positions.tolist(), slots.tolist()))
    def base_logits(self, anchors, lens, slots, window):
        assert anchors.tolist() == [10] and lens.tolist() == [5]
        assert slots.tolist() == [[0, 1]] and window == 2
        return torch.ones(1, 7, 4, dtype=torch.bfloat16)
    def markov_head(self, previous):
        self.priors.append(previous.item())
        return torch.full((1, 4), previous.item() / 128, dtype=torch.bfloat16)


def test_every_draft_forward_rebuilds_kv_and_uses_true_predecessors():
    case = {"prefix_ids": [1, 2, 3, 4, 5], "anchor_id": 10,
            "label_ids": [11, 12, 13, 14, 15, 16, 17]}
    draft = FakeDraft()
    features = torch.ones(2, 4, dtype=torch.bfloat16)
    first = qe.draft_logits(draft, case, features, window=2, block_size=1)
    second = qe.draft_logits(draft, case, features, window=2, block_size=1)
    assert draft.rebuilds == 2 and len(draft.writes) == 2
    assert all(positions == [3, 4] and slots == [0, 1] for _, positions, slots in draft.writes)
    assert draft.priors == [10, 11, 12, 13, 14, 15, 16] * 2
    assert first.dtype == torch.bfloat16 and torch.equal(first, second)
    assert first[0, 0] == torch.tensor(1, dtype=torch.bfloat16) + torch.tensor(10 / 128, dtype=torch.bfloat16)


def test_metric_rows_gate_probability_validity_and_replay():
    torch.manual_seed(7)
    target = torch.randn(7, 13, dtype=torch.bfloat16)
    baseline = torch.randn(7, 13, dtype=torch.bfloat16)
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    result = score_distributions(target, baseline, baseline, labels)
    rows, diagnostics = qe.metric_rows(result)
    assert len(rows) == 7 and all(0 <= r["overlap_target_quant"] <= 1.000001 for r in rows)
    assert diagnostics["logits_finite"] and diagnostics["scores_finite"]
    assert qe.baseline_replay_check(score_distributions, target, baseline, baseline.clone(), labels)["pass"]
    perturbed = baseline.clone()
    perturbed[:, 0] += 2
    assert not qe.baseline_replay_check(score_distributions, target, baseline, perturbed, labels)["pass"]
    result["quant_norm_max_abs"] = torch.tensor(0.1)
    with pytest.raises(RuntimeError, match="normalization"):
        qe.metric_rows(result)
    result = score_distributions(target, baseline, baseline, labels)
    result["logits_finite"] = torch.tensor(False)
    with pytest.raises(RuntimeError, match="non-finite"):
        qe.metric_rows(result)


def test_guard_actual_budgets():
    spec = importlib.util.spec_from_file_location("quality_guard_test", HERE / "gpu_control.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    guard = wrapper.load_guard()
    assert guard.TOTAL_LIMIT_S == 1200 and guard.SMOKE_LIMIT_S == 300
