"""CPU numerical and independent slot-history checks for DSpark windows."""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy  # load the local OpenMP runtime before CPU PyTorch on Windows
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", pkg)
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.spec_decode.dspark_model import DSparkModel
from nanovllm.spec_decode.dspark_proposer import DSparkProposer
from nanovllm.spec_decode.dspark_window import (begin_owner, commit_mapping, mark_confirmed,
                                                selected_prefill_rows, visible_slots)


def make_model(blocks, window=0, scratch=0):
    torch.manual_seed(22)
    cfg = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=4, attention_bias=False, intermediate_size=24,
                          num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                          markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                          mask_token_id=3, enable_confidence_head=True,
                          confidence_head_with_markov=True, rope_theta=10000)
    model = DSparkModel(cfg, torch.randn(37, 16), torch.randn(37, 16)).eval()
    model.window_tokens = window
    model.bind_cache(blocks, 4, scratch)
    model.cache.zero_()
    return model


def make_seq(length, target_blocks, ring_blocks, W):
    Sequence.block_size = 4
    seq = Sequence(list(range(length)), SamplingParams(max_tokens=1000, ignore_eos=True))
    seq.block_table = list(target_blocks)
    seq.draft_block_table = list(ring_blocks)
    begin_owner(seq, W)
    return seq


def test_window_covering_history_matches_original_full_and_markov():
    model = make_model(5)
    features = torch.randn(5, 32)
    positions = torch.arange(5)
    # Noncontiguous physical blocks help distinguish physical from logical positions.
    slots = torch.tensor([12, 13, 14, 15, 4])
    model.write_context(features, positions, slots)
    anchors = torch.tensor([4])
    absolute = torch.tensor([5])
    original = model.forward_hidden(anchors, absolute, torch.tensor([[3, 1]]), 8)
    model.window_tokens = 8
    windowed = model.forward_hidden(anchors, absolute,
                                    torch.tensor([[12, 13, 14, 15, 4, -1, -1, -1]]), 8)
    torch.testing.assert_close(original, windowed, rtol=1e-5, atol=1e-6)
    predecessor = torch.tensor([11])
    orig_q = torch.nn.functional.linear(original[:, 0], model.lm_head_weight) + model.markov_head(predecessor)
    win_q = torch.nn.functional.linear(windowed[:, 0], model.lm_head_weight) + model.markov_head(predecessor)
    torch.testing.assert_close(orig_q, win_q, rtol=1e-5, atol=1e-6)
    print(f"W>=history max_abs_hidden={(original-windowed).abs().max().item():.9g} "
          f"max_abs_q={(orig_q-win_q).abs().max().item():.9g}")


@pytest.mark.parametrize("W", [4, 8, 12])
def test_ring_wraps_commits_only_confirmed_rows_against_full_cache(W):
    # Independent reference writes only accepted history into a full cache.
    history = 19
    ring = make_model(W // 4, window=W, scratch=8)
    full = make_model(80, window=W)
    full.load_state_dict(ring.state_dict())
    seq = make_seq(history, list(range(80)), list(reversed(range(W // 4))), W)
    reference = make_seq(history, list(range(80)), [], W)
    feats = torch.randn(history, 32)
    labels = {}
    # Exercise selected early/middle/late prefill chunks, including empty selection.
    for start, end in [(0, 3), (3, 10), (10, 16), (16, history)]:
        for obj in (seq, reference):
            obj.num_cached_tokens, obj.num_scheduled_tokens = start, end - start
        rows, dst, spans = selected_prefill_rows([seq], "ring", 4, W)
        rrows, rdst, rspans = selected_prefill_rows([reference], "full_cache", 4, W)
        assert rows == rrows
        if rows:
            absolute = torch.tensor([start + r for r in rows])
            ring.write_context(feats[absolute], absolute, torch.tensor(dst))
            full.write_context(feats[absolute], absolute, torch.tensor(rdst))
            labels.update(zip(dst, absolute.tolist()))
        for owner, a, b in spans + rspans:
            mark_confirmed(owner, a, b, W)
    seq.append_token(1)
    reference.append_token(1)
    anchor = history

    # Several laps, including zero, partial and all seven candidates accepted.
    max_hidden = max_q = 0.0
    for accepted in [0, 3, 7, 2, 7, 0, 5, 7, 1, 7]:
        count = accepted + 1
        proposal = torch.randn(8, 32)
        positions = torch.arange(anchor, anchor + 8)
        before = ring.cache.clone()
        ring.write_context(proposal, positions, None, scratch=True)
        torch.testing.assert_close(ring.cache, before, rtol=0, atol=0)
        rows, dst = commit_mapping(seq, anchor, count, 0, 8, "ring", 4, W)
        ring.commit_scratch(rows, dst)
        labels.update(zip(dst, range(anchor + max(0, count - W), anchor + count)))
        refslots = list(range(anchor, anchor + count))
        full.write_context(proposal[:count], positions[:count], torch.tensor(refslots))
        mark_confirmed(seq, anchor, anchor + count, W)
        mark_confirmed(reference, anchor, anchor + count, W)
        seq.append_tokens([2] * count)
        reference.append_tokens([2] * count)
        anchor += count

        slots = visible_slots(seq, anchor, "ring", 4, W)
        expected_positions = list(range(max(0, anchor - W), anchor))
        assert [labels[s] for s in slots] == expected_positions
        ref_slots = visible_slots(reference, anchor, "full_cache", 4, W)
        torch.testing.assert_close(ring.cache[:, :, slots], full.cache[:, :, ref_slots])
        a = torch.tensor([3])
        pos = torch.tensor([anchor])
        pad = (-len(slots)) % 4
        left = ring.forward_hidden(a, pos, torch.tensor([slots + [-1] * pad]), len(slots) + pad)
        right = full.forward_hidden(a, pos, torch.tensor([ref_slots + [-1] * pad]), len(slots) + pad)
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        max_hidden = max(max_hidden, (left - right).abs().max().item())
        for predecessor in (3, 12):
            prev = torch.tensor([predecessor])
            lq = torch.nn.functional.linear(left[:, 0], ring.lm_head_weight) + ring.markov_head(prev)
            rq = torch.nn.functional.linear(right[:, 0], full.lm_head_weight) + full.markov_head(prev)
            torch.testing.assert_close(lq, rq, rtol=1e-5, atol=1e-6)
            max_q = max(max_q, (lq - rq).abs().max().item())
    print(f"W={W} 10 rounds max_abs_hidden={max_hidden:.9g} max_abs_q={max_q:.9g}")


def test_owner_gaps_duplicate_ring_and_scratch_bounds_fail_closed():
    seq = make_seq(9, [0, 1, 2], [0, 1], 8)
    with pytest.raises(RuntimeError, match="gap"):
        visible_slots(seq, 9, "ring", 4, 8)
    mark_confirmed(seq, 1, 9, 8)
    with pytest.raises(RuntimeError, match="query|scratch"):
        commit_mapping(seq, 9, 3, 4, 6, "ring", 4, 8)
    seq.draft_block_table = [0, 0]
    with pytest.raises(RuntimeError, match="ring block table"):
        visible_slots(seq, 9, "ring", 4, 8)
    seq.dspark_owner = (seq.seq_id, seq.dspark_generation - 1)
    with pytest.raises(RuntimeError, match="owner"):
        visible_slots(seq, 9, "ring", 4, 8)


def test_window_proposer_shapes_padding_and_absolute_rope():
    model = make_model(5, window=8)
    seq = make_seq(11, [0, 1, 2], [3, 4], 8)
    mark_confirmed(seq, 3, 11, 8)
    seq.append_token(5)
    cfg = SimpleNamespace(dspark_window_storage="ring", enforce_eager=True,
                          spec_cuda_graph=False, spec_graph_bs=(1, 2, 4),
                          dspark_split_graph=True, dspark_graph_pad=True, max_num_seqs=4)
    proposer = DSparkProposer(model, cfg)
    anchors, positions, slots, bucket, temps = proposer._inputs([seq])
    assert positions.tolist() == [11] and bucket == 8
    assert slots.tolist() == [visible_slots(seq, 11, "ring", 4, 8)]
    static = (torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
              torch.zeros(4, bucket, dtype=torch.int32))
    proposer._fill_backbone(static, anchors, positions, slots)
    assert static[1].tolist() == [11, 0, 0, 0]
    assert (static[2][1:] == -1).all()
    rng = torch.random.get_rng_state()
    padded = model.base_logits(*static, bucket)
    assert torch.equal(rng, torch.random.get_rng_state())
    actual = model.base_logits(anchors, positions, slots, bucket)
    torch.testing.assert_close(padded[:1], actual, rtol=1e-5, atol=1e-6)


def test_window_graph_capture_static_slot_width_without_cuda(monkeypatch):
    from contextlib import nullcontext
    model = make_model(5, window=8)
    config = SimpleNamespace(spec_graph_bs=(1, 2, 4), max_num_seqs=4)
    proposer = DSparkProposer(model, config)

    class FakeGraph:
        def pool(self):
            return object()

    # Exercise Python shape preparation only; this is not a CUDA graph run.
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **kw: nullcontext())
    _, (anchors, positions, slots), result = proposer._capture_backbone(4, 8)
    assert anchors.shape == positions.shape == (4,)
    assert slots.shape == (4, 8) and (slots == -1).all()
    assert result.shape == (4, 7, model.cfg.vocab_size)
