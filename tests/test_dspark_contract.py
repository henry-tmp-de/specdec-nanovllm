"""CPU tests for DSpark batching, slot isolation and proposal distributions."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


def load_module(name):
    path = Path(__file__).resolve().parents[1] / "nanovllm" / "spec_decode" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load_module("dspark_model")
P = load_module("dspark_proposer")
V = load_module("verify")


@pytest.fixture
def tiny():
    torch.manual_seed(17)
    cfg = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=4, attention_bias=False, intermediate_size=24,
                          num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=31,
                          markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                          mask_token_id=3, enable_confidence_head=True,
                          confidence_head_with_markov=True, rope_theta=10000)
    embedding = torch.randn(cfg.vocab_size, cfg.hidden_size)
    lm_head = torch.randn(cfg.vocab_size, cfg.hidden_size)
    model = M.DSparkModel(cfg, embedding, lm_head).eval()
    model.bind_cache(num_blocks=4, block_size=4)
    # Two independent histories in physical blocks [0,1] and [2,3].
    features = torch.randn(8, 2 * cfg.hidden_size)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    slots = torch.tensor([0, 1, 2, 8, 9, 10, 11, 12])
    model.write_context(features, positions, slots)
    return model


def test_one_backbone_call_for_ragged_batch_matches_individual(tiny):
    anchors = torch.tensor([4, 5])
    lengths = torch.tensor([3, 5])
    tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    batch = tiny.forward_hidden(anchors, lengths, tables, context_bucket=8)
    alone = torch.cat([tiny.forward_hidden(anchors[i:i+1], lengths[i:i+1],
                                          tables[i:i+1], context_bucket=8)
                       for i in range(2)])
    torch.testing.assert_close(batch, alone, rtol=1e-5, atol=1e-6)
    assert batch.shape == (2, 7, 16)


def test_unconfirmed_suffix_and_graph_padding_are_invisible(tiny):
    anchors = torch.tensor([4])
    lengths = torch.tensor([3])
    tables = torch.tensor([[0, 1]], dtype=torch.int32)
    before = tiny.forward_hidden(anchors, lengths, tables, 8)
    # A speculative token may have been written immediately after the prefix.
    tiny.write_context(torch.randn(1, 32), torch.tensor([3]), torch.tensor([3]))
    after = tiny.forward_hidden(anchors, lengths, tables, 8)
    torch.testing.assert_close(before, after)
    # Graph padding must never overwrite a live cache slot.
    snapshot = tiny.cache[:, :, :tiny.sentinel].clone()
    tiny.write_context(torch.randn(2, 32), torch.tensor([0, 0]), torch.tensor([-1, -1]))
    torch.testing.assert_close(tiny.cache[:, :, :tiny.sentinel], snapshot)
    assert torch.count_nonzero(tiny.cache[:, :, tiny.sentinel]) == 0
    # Once accepted, that same feature becomes visible without copying caches.
    accepted = tiny.forward_hidden(anchors, torch.tensor([4]), tables, 8)
    assert not torch.allclose(before, accepted)


def test_markov_logits_condition_on_sampled_prefix(tiny):
    anchors = torch.tensor([4, 5])
    lengths = torch.tensor([3, 5])
    tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    temperatures = torch.tensor([0.7, 1.0])
    torch.manual_seed(111)
    tokens, logits = tiny.propose(anchors, lengths, tables, 8, temperatures)
    assert tokens.shape == (2, 7)
    assert logits.shape == (2, 7, 31)
    base = F.linear(tiny.forward_hidden(anchors, lengths, tables, 8), tiny.lm_head_weight)
    previous = anchors
    for i in range(7):
        torch.testing.assert_close(logits[:, i], base[:, i] + tiny.markov_head(previous))
        previous = tokens[:, i]
    assert tiny.confidence(tiny.forward_hidden(anchors, lengths, tables, 8)[:, 0], anchors).shape == (2,)


def test_proposer_batches_requests_and_uses_anchor_exclusive_context(tiny, monkeypatch):
    cfg = SimpleNamespace(enforce_eager=True, spec_cuda_graph=False, spec_graph_bs=(1, 2, 4))
    proposer = P.DSparkProposer(tiny, cfg)
    seqs = [SimpleNamespace(last_token=4, temperature=0.7, block_table=[0], token_ids=[0] * 4),
            SimpleNamespace(last_token=5, temperature=1.0, block_table=[2, 3], token_ids=[0] * 6)]
    # SimpleNamespace does not implement __len__, so give the fake the engine's contract.
    class Seq:
        def __init__(self, values):
            self.__dict__.update(values.__dict__)
        def __len__(self):
            return len(self.token_ids)
    seqs = [Seq(s) for s in seqs]
    original = tiny.forward_hidden
    observed = []
    def spy(anchors, lens, tables, bucket):
        observed.append((anchors.tolist(), lens.tolist(), bucket))
        return original(anchors, lens, tables, bucket)
    monkeypatch.setattr(tiny, "forward_hidden", spy)
    tokens, logits = proposer.propose_batch(seqs)
    assert observed == [([4, 5], [3, 5], 8)]
    assert len(tokens) == 2 and all(len(row) == 7 for row in tokens)
    assert logits.shape == (2, 7, 31)
    assert proposer.active_batch_histogram == {2: 1}
    assert proposer.graph_fallbacks == {"graph_disabled": 1}


def test_released_contract_requires_seven_candidates():
    cfg = SimpleNamespace(block_size=7, num_hidden_layers=5, markov_rank=256,
                          markov_head_type="vanilla", mask_token_id=151669,
                          enable_confidence_head=True, confidence_head_with_markov=True,
                          target_layer_ids=[1, 9, 17, 25, 33], hidden_size=2560,
                          vocab_size=151936, num_attention_heads=32,
                          num_key_value_heads=8, head_dim=128, attention_bias=False,
                          hidden_act="silu")
    M.validate_config(cfg, cfg, 7)
    with pytest.raises(ValueError, match="seven candidates"):
        M.validate_config(cfg, cfg, 6)


def test_conditional_markov_q_with_rejection_matches_target_joint_distribution():
    """A sampled q_i must be the same conditional q_i passed to verify."""
    torch.manual_seed(125)
    n = 40000
    p0 = torch.tensor([0.2, 0.5, 0.3])
    pnext = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.25, 0.5, 0.25]])
    q0_logits = torch.tensor([0.7, -0.4, 0.1])
    transition = torch.tensor([[1.2, -0.4, 0.2], [-0.5, 0.8, -0.2], [0.1, -0.2, 0.4]])
    first = torch.multinomial(torch.softmax(q0_logits, -1).expand(n, -1), 1).squeeze(-1)
    q1_logits = transition[first]
    second = torch.multinomial(torch.softmax(q1_logits, -1), 1).squeeze(-1)
    q = torch.stack((q0_logits.expand(n, -1), q1_logits), dim=1)
    p = torch.stack((p0.log().expand(n, -1), pnext[first].log(),
                     pnext[second].log()), dim=1)
    result = V.verify_batch(q, p, torch.stack((first, second), dim=1))
    output_first = torch.where(result.accepted == 0, result.bonus, first)
    fallback_second = torch.multinomial(pnext[output_first], 1).squeeze(-1)
    output_second = torch.where(result.accepted == 0, fallback_second,
                                torch.where(result.accepted == 1, result.bonus, second))
    counts = torch.bincount(output_first * 3 + output_second, minlength=9).reshape(3, 3) / n
    torch.testing.assert_close(counts, p0[:, None] * pnext, atol=0.008, rtol=0)


def test_workload_accepts_transformers_mapping_without_string_tokens(tmp_path):
    from collections import UserDict
    import json
    path = Path(__file__).resolve().parents[1] / "experiments/20261101-dspark-integration/bench.py"
    spec = importlib.util.spec_from_file_location("dspark_bench_test", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    (tmp_path / "gsm8k.jsonl").write_text(json.dumps({"turns": ["How many?"]}) + "\n")
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return UserDict({"input_ids": [11, 12, 13]})
    prompts, meta = bench.build_inputs(Tokenizer(), tmp_path, "unused", 8, 1, "math",
                                       wiki=list(range(20)))
    assert prompts == [[0, 1, 2, 3, 4, 11, 12, 13]]
    assert meta[0]["wiki_tokens"] == 5
