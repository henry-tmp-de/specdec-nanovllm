"""CPU contracts for real-batch sampling and padded DSpark metadata."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def load(name):
    path = Path(__file__).resolve().parents[1] / "nanovllm/spec_decode" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_pad_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load("dspark_model")
P = load("dspark_proposer")


def make_model():
    torch.manual_seed(22)
    c = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, attention_bias=False, intermediate_size=24,
                        num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                        markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                        mask_token_id=3, enable_confidence_head=True,
                        confidence_head_with_markov=True, rope_theta=10000)
    m = M.DSparkModel(c, torch.randn(37, 16), torch.randn(37, 16)).eval()
    m.bind_cache(32, 4)
    m.cache.zero_()
    return m


def setup_batch(m, batch):
    torch.manual_seed(813)
    anchors = torch.arange(4, batch + 4) % m.cfg.vocab_size
    lengths = torch.tensor([3 + i % 5 for i in range(batch)], dtype=torch.long)
    tables = torch.tensor([[2 * i, 2 * i + 1] for i in range(batch)], dtype=torch.int32)
    feats, positions, slots = [], [], []
    for i, n in enumerate(lengths.tolist()):
        feats.append(torch.randn(n, 2 * m.cfg.hidden_size))
        positions.extend(range(n))
        slots.extend(8 * i + j for j in range(n))
    m.write_context(torch.cat(feats), torch.tensor(positions), torch.tensor(slots))
    return anchors, lengths, tables


@pytest.mark.parametrize("batch,cap", [(3, 4), (11, 16), (15, 16)])
def test_dummy_metadata_real_logits_rng_and_cache_isolation(batch, cap):
    model = make_model()
    anchors, lengths, tables = setup_batch(model, batch)
    static = (torch.full((cap,), 99, dtype=torch.long),
              torch.full((cap,), 99, dtype=torch.long),
              torch.full((cap, 2), 99, dtype=torch.int32))
    P.DSparkProposer._fill_backbone(static, anchors, lengths, tables)
    padded_anchors, padded_lengths, padded_tables = static
    assert torch.equal(padded_anchors[:batch], anchors)
    assert torch.equal(padded_lengths[:batch], lengths)
    assert torch.equal(padded_tables[:batch], tables)
    assert torch.count_nonzero(padded_anchors[batch:]) == 0
    assert torch.count_nonzero(padded_lengths[batch:]) == 0
    assert torch.all(padded_tables[batch:] == -1)
    snapshot = model.cache.clone()
    real_base = model.base_logits(anchors, lengths, tables, 8)
    padded_base = model.base_logits(padded_anchors, padded_lengths, padded_tables, 8)[:batch]
    torch.testing.assert_close(real_base, padded_base, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model.cache, snapshot)  # draft backbone must not write context KV
    temperature = torch.linspace(0.65, 1.2, batch)
    def sample(base):
        previous = anchors
        tokens, logits = [], []
        for i in range(7):
            previous, corrected = model.sample_step(base[:, i], previous, temperature)
            tokens.append(previous)
            logits.append(corrected)
        return torch.stack(tokens, 1), torch.stack(logits, 1), torch.get_rng_state().clone()
    torch.manual_seed(911)
    direct_tokens, direct_q, direct_rng = sample(real_base)
    torch.manual_seed(911)
    padded_tokens, padded_q, padded_rng = sample(padded_base)
    assert torch.equal(direct_tokens, padded_tokens)
    assert torch.equal(direct_rng, padded_rng)
    torch.testing.assert_close(direct_q, padded_q, rtol=1e-5, atol=1e-5)


def test_split_eager_preserves_original_exp_race_and_rng():
    m = make_model()
    anchors, lengths, tables = setup_batch(m, 3)
    temp = torch.tensor([0.8, 1.0, 1.2])
    cfg = SimpleNamespace(enforce_eager=True, spec_cuda_graph=False, spec_graph_bs=(1, 2, 4),
                          max_num_seqs=4, dspark_split_graph=True, dspark_graph_pad=True)
    p = P.DSparkProposer(m, cfg)
    torch.manual_seed(4321)
    old_tokens, old_q = m.propose(anchors, lengths, tables, 8, temp)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(4321)
    new_tokens, new_q = p._split_propose(anchors, lengths, tables, 8, temp)
    assert torch.equal(old_tokens, new_tokens)
    assert torch.equal(old_rng, torch.get_rng_state())
    torch.testing.assert_close(old_q, new_q, rtol=0, atol=0)
    assert p.graph_fallbacks == {"graph_disabled": 1}


def test_unsafe_draft_padding_configuration_is_rejected():
    cfg = SimpleNamespace(dspark_split_graph=False, dspark_graph_pad=True)
    with pytest.raises(ValueError, match="real-batch split sampling"):
        P.DSparkProposer(make_model(), cfg)


def test_bucket_selection_and_static_shape_failure():
    assert [P._pick_bucket(x, [1, 2, 4, 8, 16]) for x in (3, 5, 6, 7, 9, 10, 11, 15, 17)] == [4, 8, 8, 8, 16, 16, 16, 16, None]
    with pytest.raises(ValueError, match="shape mismatch"):
        P.DSparkProposer._fill_backbone((torch.zeros(2), torch.zeros(2), torch.zeros(2, 1)),
                                        torch.zeros(3), torch.zeros(3), torch.zeros(3, 1))
    cfg = SimpleNamespace(enforce_eager=False, spec_cuda_graph=True, spec_graph_bs=(1, 2, 4),
                          max_num_seqs=4, dspark_split_graph=True, dspark_graph_pad=True)
    p = P.DSparkProposer(make_model(), cfg)
    with pytest.raises(ValueError, match="unsupported prewarm shape"):
        p.prewarm([(3, 8)], [])
    assert not p.prewarm_in_progress and not p.graphs


@pytest.mark.parametrize("batch,cap", [(3, 4), (11, 16), (15, 16)])
def test_split_graph_dispatch_samples_only_real_rows(batch, cap, monkeypatch):
    m = make_model()
    anchors, lengths, tables = setup_batch(m, batch)
    temps = torch.linspace(0.7, 1.0, batch)
    cfg = SimpleNamespace(enforce_eager=False, spec_cuda_graph=True, spec_graph_bs=(1, 2, 4, 8, 16),
                          max_num_seqs=16, dspark_split_graph=True, dspark_graph_pad=True)
    p = P.DSparkProposer(m, cfg)
    monkeypatch.setattr(p, "_graphs_enabled", lambda _: True)
    ga = torch.zeros(cap, dtype=torch.long)
    gl = torch.zeros(cap, dtype=torch.long)
    gt = torch.full((cap, 2), -1, dtype=torch.int32)
    gbase = torch.empty(cap, 7, m.cfg.vocab_size)
    hb = torch.empty(batch, m.cfg.vocab_size)
    hp = torch.empty(batch, dtype=torch.long)
    ht = torch.empty(batch)
    htok = torch.empty(batch, dtype=torch.long)
    hlogits = torch.empty(batch, m.cfg.vocab_size)
    class FakeGraph:
        def __init__(self, fn):
            self.fn = fn
        def replay(self):
            self.fn()
    def backbone():
        gbase.copy_(m.base_logits(ga, gl, gt, 8))
    def sample():
        t, q = m.sample_step(hb, hp, ht)
        htok.copy_(t)
        hlogits.copy_(q)
    p.graphs[(cap, 8)] = (FakeGraph(backbone), (ga, gl, gt), gbase)
    p.head_graphs[batch] = (FakeGraph(sample), (hb, hp, ht), (htok, hlogits))
    torch.manual_seed(316)
    expected_tokens, expected_q = m.propose(anchors, lengths, tables, 8, temps)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(316)
    actual_tokens, actual_q = p._split_propose(anchors, lengths, tables, 8, temps)
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(actual_q, expected_q, rtol=1e-5, atol=1e-5)
    assert p.graph_padded_hits == 1 and p.graph_exact_hits == 0 and p.head_graph_hits == 7
    assert p.graph_captures == 0 and p.head_graph_captures == 0
