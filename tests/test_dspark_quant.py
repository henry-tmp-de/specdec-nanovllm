"""CPU-only contracts: real quantizer and explicitly simulated kernel dispatch."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy  # noqa: F401 -- initialize before the local CPU torch build
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    # Triton decorators are needed to import the actual torch quantizer; no
    # Triton code or GPU kernel executes in this CPU test suite.
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(ROOT / "nanovllm")]
    with pytest.MonkeyPatch.context() as mp:
        for name, value in {"triton": triton, "triton.language": language, "nanovllm": package}.items():
            mp.setitem(sys.modules, name, value)
        quant = load("nanovllm.layers.quant", ROOT / "nanovllm/layers/quant.py", mp)
        dsquant = load("isolated_dspark_quant", ROOT / "nanovllm/spec_decode/dspark_quant.py", mp)
        model = load("isolated_dspark_model_quant_test", ROOT / "nanovllm/spec_decode/dspark_model.py", mp)
        proposer = load("isolated_dspark_proposer_quant_test", ROOT / "nanovllm/spec_decode/dspark_proposer.py", mp)
        yield quant, dsquant, model, proposer


def tiny(model_mod):
    torch.manual_seed(1234)
    cfg = SimpleNamespace(hidden_size=16, intermediate_size=24, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, attention_bias=False,
                          num_hidden_layers=5, rms_norm_eps=1e-6, vocab_size=31,
                          markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                          mask_token_id=3, enable_confidence_head=True,
                          confidence_head_with_markov=True, rope_theta=10000)
    embedding = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    lm_head = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    model = model_mod.DSparkModel(cfg, embedding, lm_head).to(dtype=torch.bfloat16).eval()
    return model


def requested(preset):
    ffn = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    attn = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    suffixes = {"none": (), "ffn": ffn, "attention": attn, "all": ffn + attn}[preset]
    return {f"layers.{i}.{suffix}" for i in range(5) for suffix in suffixes}


@pytest.mark.parametrize("preset,count", [("none", 0), ("ffn", 15), ("attention", 20), ("all", 35)])
def test_precise_selection_bytes_and_protected_tensors(modules, preset, count):
    quant, dsq, model_mod, _ = modules
    model = tiny(model_mod)
    all_names = requested("all")
    original = {name: model.get_submodule(name) for name in all_names}
    weights = {name: m.weight.detach().clone() for name, m in original.items()}
    protected = {name: p for name, p in model.named_parameters()
                 if not (name.endswith(".weight") and name[:-7] in all_names)}
    shared = (model.embedding_weight, model.lm_head_weight)
    records = dsq.apply_dspark_quant(model, preset, chunk_rows=3)
    assert {r["name"] for r in records} == requested(preset)
    assert len(records) == count
    for name, old in original.items():
        current = model.get_submodule(name)
        if name in requested(preset):
            q, scale = quant.quantize_weight(weights[name], "per_channel")
            assert isinstance(current, dsq.DSparkInt8Linear)
            assert torch.equal(current.qweight, q) and torch.equal(current.scale, scale)
            assert current.qweight.dtype == torch.int8 and current.scale.dtype == torch.bfloat16
            assert list(dict(current.named_parameters())) == []
            assert set(dict(current.named_buffers())) == {"qweight", "scale"}
            record = next(x for x in records if x["name"] == name)
            assert record["shape"] == list(q.shape)
            assert record["bytes_before"] == weights[name].numel() * 2
            assert record["bytes_after"] == q.numel() + scale.numel() * 2
        else:
            assert current is old and torch.equal(current.weight, weights[name])
    assert all(dict(model.named_parameters())[name] is p for name, p in protected.items())
    assert (model.embedding_weight is shared[0] and model.lm_head_weight is shared[1])
    assert model.fc.weight.dtype == torch.bfloat16
    model.bind_cache(2, 4)
    assert model.cache.dtype == torch.bfloat16 and model.cache.device == model.fc.weight.device


@pytest.mark.parametrize("problem", ["bad_preset", "bad_chunk", "missing", "extra", "wrong_dtype",
                                      "wrong_shape", "alias", "shared", "bias", "repeat", "nan",
                                      "wrong_fc_dtype", "wrong_shared_dtype"])
def test_invalid_inventory_and_inputs_fail(modules, problem):
    _, dsq, model_mod, _ = modules
    model = tiny(model_mod)
    name = "layers.0.mlp.gate_proj"
    if problem == "bad_preset":
        with pytest.raises(ValueError, match="preset"):
            dsq.apply_dspark_quant(model, "bad")
        return
    if problem == "bad_chunk":
        with pytest.raises(ValueError, match="chunk_rows"):
            dsq.apply_dspark_quant(model, "ffn", chunk_rows=True)
        return
    if problem == "missing":
        model.layers[0].mlp.gate_proj = nn.Identity()
    elif problem == "extra":
        model.layers[0].mlp.extra = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    elif problem == "wrong_dtype":
        model.layers[0].mlp.gate_proj.float()
    elif problem == "wrong_shape":
        model.layers[0].mlp.gate_proj = nn.Linear(16, 25, bias=False, dtype=torch.bfloat16)
    elif problem == "alias":
        model.layers[0].mlp.up_proj.weight = model.layers[0].mlp.gate_proj.weight
    elif problem == "shared":
        model.__dict__["embedding_weight"] = model.layers[0].mlp.gate_proj.weight
    elif problem == "bias":
        model.layers[0].mlp.gate_proj = nn.Linear(16, 24, bias=True, dtype=torch.bfloat16)
    elif problem == "repeat":
        dsq.apply_dspark_quant(model, "ffn")
    elif problem == "nan":
        with torch.no_grad():
            model.get_submodule(name).weight[0, 0] = float("nan")
    elif problem == "wrong_fc_dtype":
        model.fc.float()
    elif problem == "wrong_shared_dtype":
        model.__dict__["embedding_weight"] = model.embedding_weight.float()
    with pytest.raises((ValueError, TypeError)):
        dsq.apply_dspark_quant(model, "all")


def test_simulated_kernel_routes_quantized_logits_into_proposer(modules, monkeypatch):
    _, dsq, model_mod, proposer_mod = modules
    calls = []
    def cpu_kernel(x, q, scale, **kwargs):
        calls.append((tuple(x.shape), tuple(q.shape), kwargs))
        # Same mathematical order as the channel-scale GPU kernel, not a GPU test.
        return ((x.float() @ q.float().T) * scale.float()).to(torch.bfloat16)
    monkeypatch.setattr(dsq, "int8_linear", cpu_kernel)
    model = tiny(model_mod)
    dsq.apply_dspark_quant(model, "all", chunk_rows=3)
    model.bind_cache(2, 4)
    model.write_context(torch.randn(3, 32, dtype=torch.bfloat16), torch.arange(3), torch.arange(3))
    assert len(calls) == 10  # five k + five v projections populate context KV
    config = SimpleNamespace(enforce_eager=True, spec_cuda_graph=False, spec_graph_bs=(1,),
                             dspark_split_graph=True, dspark_graph_pad=False, max_num_seqs=1,
                             dspark_window_storage="full_cache")
    proposer = proposer_mod.DSparkProposer(model, config)
    class Seq:
        last_token, temperature, block_table, token_ids = 4, 1.0, [0], [1, 2, 3, 4]
        def __len__(self):
            return len(self.token_ids)
    tokens, logits = proposer.propose_batch([Seq()])
    assert len(calls) == 45  # all 35 backbone projections were executed
    assert logits.shape == (1, 7, 31)
    a = torch.tensor([4])
    lengths = torch.tensor([3])
    tables = torch.tensor([[0]], dtype=torch.int32)
    base = model.base_logits(a, lengths, tables, 4)
    prev = a
    for i, token in enumerate(tokens[0]):
        expected = base[:, i] + model.markov_head(prev)
        torch.testing.assert_close(logits[:, i], expected, rtol=0, atol=0)
        prev = torch.tensor([token])
    with pytest.raises(ValueError, match="BF16"):
        model.layers[0].mlp.gate_proj(torch.zeros(1, 16, dtype=torch.float32))


def test_config_default_and_invalid_presets(tmp_path):
    fake_tf = types.ModuleType("transformers")
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_):
            return SimpleNamespace(max_position_embeddings=4096)
    fake_tf.AutoConfig = FakeAutoConfig
    fake_quant = types.ModuleType("nanovllm.layers.quant")
    fake_quant.hf_names_to_modules = lambda names: names
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(ROOT / "nanovllm")]
    with pytest.MonkeyPatch.context() as mp:
        for name, value in {"transformers": fake_tf, "nanovllm": package,
                            "nanovllm.layers.quant": fake_quant}.items():
            mp.setitem(sys.modules, name, value)
        mod = load("isolated_config_quant_test", ROOT / "nanovllm/config.py", mp)
        assert mod.Config(str(tmp_path)).dspark_quant_preset == "none"
        for bad in ("bad", "int8", None, True):
            with pytest.raises(ValueError, match="dspark_quant_preset"):
                mod.Config(str(tmp_path), dspark_quant_preset=bad)
        with pytest.raises(ValueError, match="spec_method"):
            mod.Config(str(tmp_path), dspark_quant_preset="ffn")
        with pytest.raises(ValueError, match="spec_k"):
            mod.Config(str(tmp_path), spec_method="dspark", spec_k=6, dspark_quant_preset="ffn")
        c = mod.Config(str(tmp_path), spec_method="dspark", spec_k=7, dspark_quant_preset="all")
        assert c.draft_quant_weights is None


def test_pilot_guard_uses_actual_smaller_limits(modules):
    exp = ROOT / "experiments/20261105-dspark-quant"
    with pytest.MonkeyPatch.context() as mp:
        guard_wrapper = load("isolated_quant_guard", exp / "gpu_control.py", mp)
        guard = guard_wrapper.load_guard()
        assert guard.TOTAL_LIMIT_S == 3600
        assert guard.SMOKE_LIMIT_S == 600


@pytest.mark.parametrize("preset", ["none", "ffn", "attention", "all"])
def test_live_inventory_includes_quant_buffers_and_excluded_parameters(modules, preset):
    _, dsq, model_mod, _ = modules
    model = tiny(model_mod)
    stats = dsq.apply_dspark_quant(model, preset, chunk_rows=3)
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=model.embedding_weight)),
                             lm_head=SimpleNamespace(weight=model.lm_head_weight))
    runner = SimpleNamespace(dspark_model=model, model=target, dspark_quant_stats=stats, dspark_quant_host_s=0.0,
                             config=SimpleNamespace(dspark_quant_preset=preset))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "nanovllm.spec_decode.dspark_quant", dsq)
        wrapper = load("isolated_quant_bench", ROOT / "experiments/20261105-dspark-quant/bench_formal.py", mp)
        inv = wrapper.runtime_inventory(runner, preset)
    assert inv["quantized_modules"] == len(requested(preset))
    assert len(inv["candidate_modules"]) == 35
    assert sum(c["total_bytes"] for c in inv["candidate_modules"]) == inv["candidate_total_bytes"]
    assert sum(c["scale_bytes"] > 0 for c in inv["candidate_modules"]) == len(requested(preset))
    assert {x["name"] for x in inv["noncandidate_parameters"]} >= {
        "fc.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
        "confidence_head.proj.weight", "norm.weight", "hidden_norm.weight"}
    assert all(x["aliases_target"] for x in inv["shared_target_weights"])


def test_bench_wrapper_forwards_preset_and_attaches_raw_inventory(modules, monkeypatch):
    exp = ROOT / "experiments/20261105-dspark-quant"
    with pytest.MonkeyPatch.context() as mp:
        wrapper = load("isolated_quant_bench_entry", exp / "bench_formal.py", mp)
        fake = SimpleNamespace(options_for=lambda *a, **kw: {"original": True},
                               run_once=lambda *a, **kw: {"wall_s": 1.0})
        fake.main = lambda: None
        mp.setattr(wrapper, "load_bench", lambda: fake)
        mp.setattr(wrapper, "runtime_inventory", lambda runner, preset: {"preset": preset})
        mp.setattr(sys, "argv", ["bench_formal.py", "--dspark-quant-preset", "attention", "--phase", "pilot"])
        wrapper.main()
        assert sys.argv == ["bench_formal.py", "--phase", "pilot"]
        options = fake.options_for("dspark_ring", 1, 1024, 128, 512, "target", "ar", "ds", 1024)
        assert options == {"original": True, "dspark_quant_preset": "attention"}
        with pytest.raises(ValueError, match="pilot"):
            fake.options_for("dspark_ring", 2, 1024, 128, 512, "target", "ar", "ds", 2048)
        result = fake.run_once(SimpleNamespace(model_runner=object()))
        assert result == {"wall_s": 1.0, "dspark_quant": {"preset": "attention"}}
