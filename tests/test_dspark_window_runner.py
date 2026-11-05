"""CPU tests of real runner commit decisions and scheduler window lifecycle."""
import importlib
import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

import numpy
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", pkg)


def stub(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    return module


from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.spec_decode.dspark_model import DSparkModel
from nanovllm.spec_decode.dspark_window import begin_owner, mark_confirmed, selected_prefill_rows

stubs = {
    "nanovllm.config": stub("nanovllm.config", Config=object),
    "transformers": stub("transformers", AutoTokenizer=object),
    "nanovllm.models.qwen3": stub("nanovllm.models.qwen3", Qwen3ForCausalLM=object),
    "nanovllm.layers.sampler": stub("nanovllm.layers.sampler", Sampler=object),
    "nanovllm.layers.quant": stub("nanovllm.layers.quant", apply_int8_quant=lambda *a, **kw: None),
    "nanovllm.spec_decode.draft_proposer": stub("nanovllm.spec_decode.draft_proposer", DraftWindow=object,
                                                window_realign_target=lambda *a, **kw: None),
    "nanovllm.utils.loader": stub("nanovllm.utils.loader", load_model=lambda *a, **kw: None),
}
with patch.dict(sys.modules, stubs):
    runner_module = importlib.import_module("nanovllm.engine.model_runner")
    ModelRunner = runner_module.ModelRunner
    Scheduler = importlib.import_module("nanovllm.engine.scheduler").Scheduler
    LLMEngine = importlib.import_module("nanovllm.engine.llm_engine").LLMEngine
    set_context, reset_context = runner_module.set_context, runner_module.reset_context


def cfg(**updates):
    values = dict(max_num_seqs=2, max_num_batched_tokens=32, eos=99,
                  kvcache_block_size=4, num_kvcache_blocks=12,
                  num_draft_blocks=4, draft_window_blocks=2, draft_head_blocks=0,
                  spec_batch_threshold=0, spec_k=7, dspark_window_tokens=8,
                  dspark_window_storage="ring", enable_prefix_cache=False,
                  spec_prefill_chunk_tokens=0, spec_overlap_mixed=False)
    values.update(updates)
    return SimpleNamespace(**values)


def owner(seq_id_offset=0, max_tokens=30, ignore_eos=True):
    Sequence.block_size = 4
    seq = Sequence(list(range(8)), SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))
    seq.block_table = [seq_id_offset + 10, seq_id_offset + 11, seq_id_offset + 12, seq_id_offset + 13]
    seq.draft_block_table = [seq_id_offset, seq_id_offset + 1]
    begin_owner(seq, 8)
    mark_confirmed(seq, 0, 8, 8)
    seq.append_token(1)
    return seq


class FakeScratch:
    def __init__(self):
        self.commits = []

    def commit_scratch(self, rows, slots):
        self.commits.append((list(rows), list(slots)))


def runner():
    obj = ModelRunner.__new__(ModelRunner)
    obj.config, obj.block_size, obj.dspark_model = cfg(), 4, FakeScratch()
    return obj


def test_runner_commits_ragged_real_rows_and_never_dummy():
    r = runner()
    first, second = owner(0), owner(2)
    output = r._dspark_commit_outputs([first, second], [[11], [12, 13, 14, 15]], [0, 2], 10)
    assert output == [[11], [12, 13, 14, 15]]
    assert r.dspark_model.commits == [([0, 2, 3, 4, 5], [0, 8, 9, 10, 11])]
    assert (first.dspark_valid_start, first.dspark_valid_end) == (1, 9)
    assert (second.dspark_valid_start, second.dspark_valid_end) == (4, 12)


def test_runner_truncates_eos_and_limit_before_scheduler_and_does_not_commit_finished():
    r = runner()
    eos = owner(0, ignore_eos=False)
    limited = owner(2, max_tokens=3)
    out = r._dspark_commit_outputs([eos, limited], [[4, 99, 5, 6], [7, 8, 9]], [0, 4], 8)
    assert out == [[4, 99], [7, 8]]
    assert r.dspark_model.commits == [([], [])]
    assert eos.dspark_valid_end == limited.dspark_valid_end == 8


def test_runner_rejects_cross_owner_rows_without_partial_commit():
    r = runner()
    first, second = owner(0), owner(2)
    with pytest.raises(RuntimeError, match="crosses"):
        r._dspark_commit_outputs([first, second], [[1, 2, 3], [4]], [0, 2], 3)
    assert not r.dspark_model.commits and first.dspark_valid_end == 8


def test_scheduler_real_allocation_prefix_disabled_preempt_reentry_and_abort():
    Sequence.block_size = 4
    sched = Scheduler(cfg())
    seqs = [Sequence(list(range(8)), SamplingParams(max_tokens=30, ignore_eos=True)) for _ in range(2)]
    for seq in seqs:
        sched.add(seq)
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == seqs
    assert set(seqs[0].draft_block_table).isdisjoint(seqs[1].draft_block_table)
    rows, slots, spans = selected_prefill_rows(batch, "ring", 4, 8)
    assert rows == list(range(16)) and len(set(slots)) == 16
    for seq, start, end in spans:
        mark_confirmed(seq, start, end, 8)
    sched.postprocess(batch, [1, 2], True)
    assert sched.block_manager.prefix_lookup_count == 0
    assert sched.block_manager.prefix_hit_blocks == 0
    assert not sched.block_manager.hash_to_block_id
    old_owner = seqs[0].dspark_owner
    sched.running.remove(seqs[0])
    sched.preempt(seqs[0])
    assert seqs[0].dspark_owner is None and not seqs[0].draft_block_table
    batch, is_prefill = sched.schedule()
    assert is_prefill and batch == [seqs[0]]
    assert seqs[0].dspark_owner != old_owner
    assert seqs[0].dspark_valid_start == seqs[0].dspark_valid_end == 1
    sched.abort_dspark_step(batch)
    assert seqs[0].status == SequenceStatus.WAITING and seqs[0].dspark_owner is None
    assert not seqs[0].block_table and not seqs[0].draft_block_table


def test_ring_capacity_failure_is_explicit_and_does_not_allocate_target():
    Sequence.block_size = 4
    sched = Scheduler(cfg(num_draft_blocks=0))
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=2))
    sched.add(seq)
    with pytest.raises(RuntimeError, match="cannot admit"):
        sched.schedule()
    assert not seq.block_table and seq in sched.waiting


def test_engine_forward_error_releases_window_and_requeues():
    Sequence.block_size = 4
    scheduler = Scheduler(cfg())
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=10))
    scheduler.add(seq)

    class BrokenRunner:
        def call(self, *args):
            raise RuntimeError("target failed")

    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler, engine.model_runner = scheduler, BrokenRunner()
    with pytest.raises(RuntimeError, match="target failed"):
        engine.step()
    assert seq in scheduler.waiting and seq not in scheduler.running
    assert seq.dspark_owner is None and not seq.draft_block_table and not seq.block_table


def test_engine_postprocess_error_releases_window_and_requeues(monkeypatch):
    Sequence.block_size = 4
    scheduler = Scheduler(cfg())
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=10))
    scheduler.add(seq)

    class StubRunner:
        def call(self, *args):
            return [10]

    def fail(*args):
        raise RuntimeError("postprocess failed")

    monkeypatch.setattr(scheduler, "postprocess", fail)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler, engine.model_runner = scheduler, StubRunner()
    with pytest.raises(RuntimeError, match="postprocess failed"):
        engine.step()
    assert seq in scheduler.waiting and seq.dspark_owner is None
    assert not seq.block_table and not seq.draft_block_table


def test_real_warmup_entry_before_cache_has_no_window_owner(monkeypatch):
    c = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, attention_bias=False, intermediate_size=24,
                        num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                        markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                        mask_token_id=3, enable_confidence_head=True,
                        confidence_head_with_markov=True, rope_theta=10000)
    draft = DSparkModel(c, torch.randn(37, 16), torch.randn(37, 16))
    seen = []

    class Target:
        def __call__(self, ids, positions, layers):
            seen.append(len(ids))
            return torch.zeros(len(ids), 16), torch.zeros(len(ids), 32)

        def compute_logits(self, hidden):
            return hidden

    r = runner()
    r.config.max_model_len = 4
    r.config.max_num_batched_tokens = 8
    r.dspark_model, r.model, r.dspark_target_layers = draft, Target(), (0, 1)
    r.rank, r.draft_model, r.spec_proposer, r.enforce_eager = 0, None, None, True
    r.prepare_sample = lambda seqs: None
    r.sampler = lambda logits, temperatures: torch.tensor([1, 2])

    def prepare(seqs, advance_draft=True):
        assert all(s.dspark_owner is None and not s.block_table for s in seqs)
        count = sum(s.num_scheduled_tokens for s in seqs)
        set_context(True, slot_mapping=torch.empty(0, dtype=torch.int32))
        return torch.zeros(count, dtype=torch.long), torch.arange(count)

    r.prepare_prefill = prepare
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner_module, "_mem_snap", lambda *args: None)
    r.warmup_model()
    assert draft.cache is None and seen == [8]


def test_prefill_selected_rows_reach_direct_window_destination():
    config = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=4, attention_bias=False, intermediate_size=24,
                             num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                             markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                             mask_token_id=3, enable_confidence_head=True,
                             confidence_head_with_markov=True, rope_theta=10000)
    draft = DSparkModel(config, torch.randn(37, 16), torch.randn(37, 16))
    draft.bind_cache(2, 4, scratch_rows=16)
    draft.cache.zero_()
    source = torch.arange(6 * 32).view(6, 32).float()

    class Target:
        def __call__(self, ids, positions, layers, indices):
            return torch.zeros(len(ids), 16), source.index_select(0, indices)

    r = runner()
    r.dspark_model, r.model, r.dspark_target_layers = draft, Target(), (0, 1)
    r._dspark_prefill = (torch.tensor([2, 5]), torch.tensor([1, 6]))
    r.dspark_window_stats = dict(max_feature_tensor_bytes=0, max_projection_tensor_bytes=0)
    set_context(True, slot_mapping=torch.arange(6))
    r._target_forward_hidden(torch.arange(6), torch.arange(6))
    reset_context()
    assert torch.count_nonzero(draft.cache[:, :, 1]) > 0
    assert torch.count_nonzero(draft.cache[:, :, 6]) > 0
    assert torch.count_nonzero(draft.cache[:, :, [0, 2, 3, 4, 5, 7]]) == 0
    assert r.dspark_window_stats["max_feature_tensor_bytes"] == 2 * 32 * 4


def test_ragged_verify_eager_uses_scratch_not_live_ring():
    config = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=4, attention_bias=False, intermediate_size=24,
                             num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                             markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                             mask_token_id=3, enable_confidence_head=True,
                             confidence_head_with_markov=True, rope_theta=10000)
    draft = DSparkModel(config, torch.randn(37, 16), torch.randn(37, 16))
    draft.bind_cache(2, 4, scratch_rows=16)
    draft.cache.zero_()

    class Target:
        def __call__(self, ids, positions, layers):
            return torch.zeros(len(ids), 16), torch.arange(len(ids) * 32).reshape(len(ids), 32).float()

        def compute_logits(self, hidden):
            return hidden

    r = runner()
    r.dspark_model, r.model, r.dspark_target_layers = draft, Target(), (0, 1)
    r.enforce_eager, r.verify_graphs, r._verify_extra = True, {}, None
    ids, pos = torch.tensor([1, 2, 3]), torch.tensor([8, 9, 10])
    set_context(True, slot_mapping=torch.tensor([0, 1, 2]), is_spec_verify=True)
    snapshot = draft.cache.clone()
    logits = r.run_verify_forward(ids, pos)
    reset_context()
    assert logits.shape == (3, 16)
    torch.testing.assert_close(draft.cache, snapshot, rtol=0, atol=0)
    assert torch.count_nonzero(draft.scratch[:, :, :3]) > 0


@pytest.mark.parametrize("graph_batch", [2, 4])
def test_verify_graph_static_mapping_and_commit_with_cpu_fake_replay(graph_batch):
    """Exercise real copy/commit code; the graph itself is an explicit CPU fake."""
    c = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, attention_bias=False, intermediate_size=24,
                        num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                        markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                        mask_token_id=3, enable_confidence_head=True,
                        confidence_head_with_markov=True, rope_theta=10000)
    draft = DSparkModel(c, torch.randn(37, 16), torch.randn(37, 16))
    draft.bind_cache(4, 4, scratch_rows=graph_batch * 8)
    draft.cache.zero_()

    class Target:
        def __call__(self, ids, positions, layers):
            features = ids.float().unsqueeze(1).expand(-1, 32) + positions.float().unsqueeze(1)
            return torch.zeros(len(ids), 16), features

    r = runner()
    r.dspark_model, r.model, r.dspark_target_layers = draft, Target(), (0, 1)
    r.batch_verify_graph, r.graph_pad = True, True
    r.verify_graph_hits, r.verify_graph_padded, r.verify_graph_fallbacks = 0, 0, {}
    Q, K = graph_batch * 8, 7
    gv = dict(input_ids=torch.zeros(Q, dtype=torch.long), positions=torch.zeros(Q, dtype=torch.long),
              slot_mapping=torch.zeros(Q, dtype=torch.int32), cu_seqlens_q=torch.zeros(graph_batch + 1, dtype=torch.int32),
              cu_seqlens_k=torch.zeros(graph_batch + 1, dtype=torch.int32),
              block_tables=torch.zeros(graph_batch, 4, dtype=torch.int32), logits=torch.zeros(Q, 3))

    class CPUFakeGraph:
        def replay(self):
            set_context(True, slot_mapping=gv["slot_mapping"], is_spec_verify=True)
            r._target_forward_hidden(gv["input_ids"], gv["positions"])
            gv["logits"][:, 0] = gv["positions"]

    r.verify_graphs = {(graph_batch, K): (CPUFakeGraph(), gv)}
    real_ids = torch.arange(16, dtype=torch.long)
    real_pos = torch.tensor(list(range(8, 16)) * 2)
    r._verify_extra = dict(n=2, k=K, full_k=True, slot_mapping=torch.arange(16, dtype=torch.int32),
                           cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
                           cu_seqlens_k=torch.tensor([0, 16, 32], dtype=torch.int32),
                           block_tables=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32))
    before = draft.cache.clone()
    logits = r.run_verify_forward(real_ids, real_pos)
    assert logits.shape == (16, 3)
    assert logits[:, 0].tolist() == real_pos.tolist()
    assert r.verify_graph_hits == 1 and r.verify_graph_padded == (graph_batch - 2 > 0)
    if graph_batch > 2:
        assert (gv["slot_mapping"][16:] == -1).all()
    torch.testing.assert_close(before, draft.cache, rtol=0, atol=0)
    first, second = owner(0), owner(2)
    scratch = draft.scratch.clone()
    r._dspark_commit_outputs([first, second], [[10], [11, 12, 13, 14]], [0, 8], 16)
    expected = before.clone()
    expected.index_copy_(2, torch.tensor([0, 8, 9, 10, 11]),
                         scratch.index_select(2, torch.tensor([0, 8, 9, 10, 11])))
    torch.testing.assert_close(draft.cache, expected, rtol=0, atol=0)
    reset_context()


def fake_decode_runner(monkeypatch, max_batch, buckets, with_dspark=True):
    """Run real capture/dispatch methods with CPU-only graph and CUDA stubs."""
    c = SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, attention_bias=False, intermediate_size=24,
                        num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=37,
                        markov_rank=4, target_layer_ids=[0, 1], block_size=7,
                        mask_token_id=3, enable_confidence_head=True,
                        confidence_head_with_markov=True, rope_theta=10000)
    draft = DSparkModel(c, torch.randn(37, 16), torch.randn(37, 16))
    draft.bind_cache(8, 4, scratch_rows=max(max_batch * 8, 8))
    draft.cache.zero_()

    class Target:
        def __call__(self, ids, positions, layers=None):
            values = ids.float() + positions.float() * 0.25
            hidden = values[:, None].expand(-1, 16)
            return hidden if layers is None else (hidden, values[:, None].expand(-1, 32))

        def compute_logits(self, hidden):
            return hidden

    class FakeGraph:
        def pool(self):
            return "fake-pool"

        def replay(self):
            self.callback()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    r = runner()
    r.config = cfg(max_num_seqs=max_batch, max_model_len=32, decode_graph_bs=buckets,
                   hf_config=SimpleNamespace(hidden_size=16))
    r.block_size, r.enforce_eager = 4, False
    r.model, r.dspark_model, r.dspark_target_layers = Target(), (draft if with_dspark else None), (0, 1)
    r.graph_mem = {}
    r.decode_graph_hits = r.decode_graph_exact_hits = r.decode_graph_padded_hits = 0
    r.decode_graph_bucket_histogram, r.decode_graph_fallbacks = {}, {}
    r.capture_cudagraph()
    executed = []
    for bucket, graph in r.graphs.items():
        def replay(b=bucket):
            executed.append(b)
            gv = r.graph_vars
            outer = runner_module.get_context()
            set_context(False, slot_mapping=gv["slot_mapping"][:b],
                        context_lens=gv["context_lens"][:b], block_tables=gv["block_tables"][:b])
            try:
                gv["outputs"][:b].copy_(r._target_forward_hidden(gv["input_ids"][:b], gv["positions"][:b]))
            finally:
                set_context(outer.is_prefill, outer.cu_seqlens_q, outer.cu_seqlens_k,
                            outer.max_seqlen_q, outer.max_seqlen_k, outer.slot_mapping,
                            outer.context_lens, outer.block_tables, outer.is_spec_verify)
        graph.callback = replay
    return r, executed, copy.deepcopy(draft)


@pytest.mark.parametrize("max_batch", [11, 15])
def test_target_decode_explicit_buckets_capture_replay_and_dynamic_scratch(monkeypatch, max_batch):
    r, executed, ref = fake_decode_runner(monkeypatch, max_batch, tuple(range(1, max_batch + 1)))
    assert r.graph_bs == list(range(1, max_batch + 1))
    assert set(r.graphs) == set(r.graph_bs)
    assert r.graph_vars["input_ids"].shape == (max_batch,)
    ring_before = r.dspark_model.cache.clone()
    rng_before = torch.random.get_rng_state()
    for turn, batch in enumerate(list(range(1, max_batch + 1)) + [3, 1, max_batch]):
        ids = torch.arange(batch, dtype=torch.long) + turn
        positions = torch.arange(batch, dtype=torch.long) + turn * 2
        set_context(False, slot_mapping=torch.arange(batch, dtype=torch.int32),
                    context_lens=torch.arange(batch, dtype=torch.int32) + 1,
                    block_tables=torch.zeros(batch, 1, dtype=torch.int32))
        values = ids.float() + positions.float() * 0.25
        expected_features = values[:, None].expand(-1, 32)
        ref.write_context(expected_features, positions, None, scratch=True)
        result = r.run_model(ids, positions, False)
        assert executed[-1] == batch
        torch.testing.assert_close(result, values[:, None].expand(-1, 16))
        torch.testing.assert_close(r.dspark_model.scratch[:, :, :batch], ref.scratch[:, :, :batch])
        torch.testing.assert_close(r.dspark_model.cache, ring_before, rtol=0, atol=0)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert r.decode_graph_hits == r.decode_graph_exact_hits == len(executed)
    assert r.decode_graph_padded_hits == 0
    assert sum(r.decode_graph_bucket_histogram.values()) == len(executed)
    reset_context()


def test_target_decode_default_and_sparse_bucket_policy(monkeypatch):
    r, executed, _ = fake_decode_runner(monkeypatch, 11, ())
    assert r.graph_bs == [1, 2, 4, 8]  # exact original list, including its B11 gap
    assert r.graph_vars["input_ids"].shape == (11,)
    set_context(False, slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
                context_lens=torch.ones(3, dtype=torch.int32),
                block_tables=torch.zeros(3, 1, dtype=torch.int32))
    before = r.dspark_model.cache.clone()
    r.run_model(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), False)
    assert executed == [4] and r.decode_graph_padded_hits == 1
    assert r.graph_vars["slot_mapping"][3].item() == -1
    torch.testing.assert_close(r.dspark_model.cache, before, rtol=0, atol=0)
    with pytest.raises(StopIteration):
        r.run_model(torch.zeros(11, dtype=torch.long), torch.zeros(11, dtype=torch.long), False)

    sparse, calls, _ = fake_decode_runner(monkeypatch, 11, (11, 1, 4))
    assert sparse.graph_bs == [1, 4, 11]
    set_context(False, slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
                context_lens=torch.ones(3, dtype=torch.int32),
                block_tables=torch.zeros(3, 1, dtype=torch.int32))
    sparse.run_model(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), False)
    assert calls == [4] and sparse.decode_graph_padded_hits == 1
    reset_context()


def test_target_decode_eager_fallback_is_counted(monkeypatch):
    r, calls, _ = fake_decode_runner(monkeypatch, 3, (1, 2, 3))
    r.enforce_eager = True
    set_context(False, slot_mapping=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([1], dtype=torch.int32),
                block_tables=torch.zeros(1, 1, dtype=torch.int32))
    result = r.run_model(torch.tensor([2]), torch.tensor([4]), False)
    assert result[0, 0].item() == 3.0
    assert not calls and r.decode_graph_hits == 0
    assert r.decode_graph_fallbacks == {"enforce_eager": 1}
    reset_context()


def test_target_decode_exact_bucket_without_spec_model(monkeypatch):
    r, calls, _ = fake_decode_runner(monkeypatch, 3, (1, 2, 3), with_dspark=False)
    set_context(False, slot_mapping=torch.arange(3, dtype=torch.int32),
                context_lens=torch.ones(3, dtype=torch.int32),
                block_tables=torch.zeros(3, 1, dtype=torch.int32))
    result = r.run_model(torch.tensor([1, 2, 3]), torch.tensor([4, 8, 12]), False)
    assert result[:, 0].tolist() == [2.0, 4.0, 6.0]
    assert calls == [3] and r.decode_graph_exact_hits == 1 and r.decode_graph_padded_hits == 0
    reset_context()
