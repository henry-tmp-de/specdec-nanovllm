"""CPU-only fake-graph checks for actual DSpark prewarm capture order."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy  # load local OpenMP before CPU torch on this Windows host
import torch

from test_dspark_pad import P, make_model, setup_batch


def test_large_shapes_first_preserves_capture_set_real_batch_sampling_and_fallback(monkeypatch):
    model = make_model()
    cfg = SimpleNamespace(enforce_eager=False, spec_cuda_graph=True, spec_graph_bs=(1, 2, 4),
                          max_num_seqs=4, dspark_split_graph=True, dspark_graph_pad=False)
    p = P.DSparkProposer(model, cfg)
    pool_calls = []

    class FakeGraph:
        def __init__(self):
            self.handle = object()

        def pool(self):
            return self.handle

        def replay(self):
            self.callback()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, pool=None: (pool_calls.append(pool) or nullcontext()))
    real_backbone, real_head = p._capture_backbone, p._capture_head
    calls = []

    def capture_backbone(batch, context):
        assert p.prewarm_in_progress and torch.is_inference_mode_enabled()
        calls.append(("backbone", batch, context))
        return real_backbone(batch, context)

    def capture_head(batch):
        assert p.prewarm_in_progress and torch.is_inference_mode_enabled()
        calls.append(("head", batch))
        return real_head(batch)

    monkeypatch.setattr(p, "_capture_backbone", capture_backbone)
    monkeypatch.setattr(p, "_capture_head", capture_head)
    old_graph, old_head = object(), object()
    p.graphs[(1, 16)] = old_graph
    p.head_graphs[1] = old_head
    shapes = [(1, 8), (2, 4), (1, 16), (2, 8), (4, 4), (4, 8), (2, 8)]
    real_batches = [2, 1, 4, 3, 2]
    p.prewarm(shapes, real_batches)

    expected_new = [(4, 8), (4, 4), (2, 8), (2, 4), (1, 8)]
    assert calls == ([('backbone', *s) for s in expected_new] +
                     [('head', b) for b in (4, 3, 2)])
    print(f"mixed fixture capture calls: {calls}")
    assert set(p.graphs) == set(shapes) and set(p.head_graphs) == set(real_batches)
    assert p.prewarmed_shapes == expected_new
    assert p.graphs[(1, 16)] is old_graph and p.head_graphs[1] is old_head
    assert not p.prewarm_in_progress
    assert p.graph_captures == len(expected_new) and p.head_graph_captures == 3
    assert p.graph_runtime_captures == p.head_runtime_captures == 0
    assert pool_calls[0] is None and all(pool is p.backbone_pool for pool in pool_calls[1:5])
    assert pool_calls[5] is None and all(pool is p.head_pool for pool in pool_calls[6:])
    outputs = [p.graphs[key][2] for key in expected_new]
    assert len({t.data_ptr() for t in outputs}) == len(outputs)

    # CPU callbacks emulate replay, but execute the actual split proposal logic.
    # Head callbacks sample exactly real B rows, even when a backbone falls back.
    for batch, context in expected_new:
        graph, (anchors, lengths, tables), base = p.graphs[(batch, context)]
        graph.callback = lambda a=anchors, l=lengths, t=tables, b=base, ctx=context: b.copy_(
            model.base_logits(a, l, t, ctx))
    for batch in (2, 3, 4):
        graph, (base, previous, temperatures), (sampled, logits) = p.head_graphs[batch]
        def sample(b=base, prev=previous, temp=temperatures, out=sampled, q=logits):
            tokens, corrected = model.sample_step(b, prev, temp)
            out.copy_(tokens)
            q.copy_(corrected)
        graph.callback = sample
    monkeypatch.setattr(p, "_graphs_enabled", lambda _: True)
    for batch in (2, 3):
        anchors, lengths, tables = setup_batch(model, batch)
        temperatures = torch.linspace(0.7, 1.0, batch)
        torch.manual_seed(316)
        expected_tokens, expected_q = model.propose(anchors, lengths, tables, 8, temperatures)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(316)
        # Production enters through inference-decorated propose_batch().
        with torch.inference_mode():
            tokens, q = p._split_propose(anchors, lengths, tables, 8, temperatures)
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        torch.testing.assert_close(q, expected_q, rtol=1e-5, atol=1e-5)
    assert p.graph_exact_hits == 1 and p.graph_padded_hits == 0
    assert p.graph_fallbacks == {"batch_shape_not_captured": 1}
    assert p.head_graph_hits == 14
    assert p.graph_runtime_captures == p.head_runtime_captures == 0


def test_prewarm_failure_still_clears_progress_flag():
    model = make_model()
    cfg = SimpleNamespace(enforce_eager=False, spec_cuda_graph=True, spec_graph_bs=(1, 2, 4),
                          max_num_seqs=4, dspark_split_graph=True, dspark_graph_pad=False)
    p = P.DSparkProposer(model, cfg)
    try:
        p.prewarm([(3, 8)], [1])
    except ValueError as exc:
        assert "unsupported prewarm shape" in str(exc)
    else:
        raise AssertionError("invalid shape was not rejected")
    assert not p.prewarm_in_progress and not p.graphs and not p.head_graphs


def test_formal_b8_contexts_are_captured_largest_first(monkeypatch):
    model = make_model()
    cfg = SimpleNamespace(enforce_eager=False, spec_cuda_graph=True,
                          spec_graph_bs=tuple(range(1, 9)), max_num_seqs=8,
                          dspark_split_graph=True, dspark_graph_pad=False)
    p = P.DSparkProposer(model, cfg)
    shapes = [(b, ctx) for b in range(1, 9) for ctx in (4096, 4352)]
    captured_backbone, captured_head = [], []

    def backbone(batch, context):
        assert p.prewarm_in_progress
        captured_backbone.append((batch, context))
        return object()

    def head(batch):
        assert p.prewarm_in_progress
        captured_head.append(batch)
        return object()

    monkeypatch.setattr(p, "_capture_backbone", backbone)
    monkeypatch.setattr(p, "_capture_head", head)
    p.prewarm(shapes, range(1, 9))
    expected = sorted(shapes, key=lambda shape: (shape[0] * shape[1], *shape), reverse=True)
    assert captured_backbone == expected
    assert captured_backbone[:3] == [(8, 4352), (8, 4096), (7, 4352)]
    assert captured_head == list(range(8, 0, -1))
    assert set(p.graphs) == set(shapes) and set(p.head_graphs) == set(range(1, 9))
    print(f"B8 backbone capture order: {captured_backbone}")
    print(f"B8 head capture order: {captured_head}")
