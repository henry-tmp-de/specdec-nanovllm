"""CPU metadata and fake-CUDA contracts, not GPU correctness or speed tests."""
import importlib
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy  # noqa: F401 -- initialize before local CPU torch
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", pkg)
from nanovllm.engine import layer_prefill as lp
from nanovllm.utils.context import Context, get_context, restore_context


class Seq:
    def __init__(self, sid, start, count, blocks):
        self.seq_id, self.num_cached_tokens, self.num_scheduled_tokens = sid, start, count
        self.block_table = blocks
        self.token_ids = list(range(start + count))
    def __len__(self):
        return len(self.token_ids)


def test_chunk_geometry_257_513_cached_start_and_separate_sequences():
    seqs = [Seq(4, 0, 257, [10, 12]), Seq(7, 256, 257, [30, 31, 33])]
    tasks = lp.plan_chunks(seqs, 128, 256)
    assert [(x.sequence_id, x.start, x.end) for x in tasks] == [
        (4, 0, 128), (4, 128, 256), (4, 256, 257),
        (7, 256, 384), (7, 384, 512), (7, 512, 513)]
    assert [(x.flat_start, x.flat_end) for x in tasks] == [
        (0, 128), (128, 256), (256, 257), (257, 385), (385, 513), (513, 514)]
    assert [x.previous_task for x in tasks] == [None, 0, 1, None, 3, 4]
    assert tasks[2].slots == (12 * 256,)
    assert tasks[3].slots[0] == 31 * 256 and tasks[5].slots == (33 * 256,)
    assert tasks[3].block_table == (30, 31) and tasks[5].block_table == (30, 31, 33)


def test_wavefront_has_both_dependencies_and_no_cross_sequence_edges():
    tasks = lp.plan_chunks([Seq(1, 0, 300, [1, 2]), Seq(2, 0, 130, [3])], 128, 256)
    order = lp.wavefront(tasks, 3, 2)
    assert order[:5] == ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1))
    index = {operation: i for i, operation in enumerate(order)}
    assert len(index) == len(tasks) * 3
    for task in tasks:
        for layer in range(3):
            if layer:
                assert index[(task.task_id, layer - 1)] < index[(task.task_id, layer)]
            if task.previous_task is not None:
                assert tasks[task.previous_task].sequence_id == task.sequence_id
                assert index[(task.previous_task, layer)] < index[(task.task_id, layer)]
    assert tasks[3].sequence_id == 2 and tasks[3].previous_task is None
    active = set()
    for task_id, layer in order:
        if layer == 0:
            active.add(task_id)
        assert len(active) <= 2
        if layer == 2:
            active.remove(task_id)
    assert not active


@pytest.mark.parametrize("seqs", [
    [Seq(1, 0, 1, []),], [Seq(1, 0, 257, [5]),],
    [Seq(1, 0, 1, [5]), Seq(1, 0, 1, [6])],
    [Seq(1, 0, 1, [5]), Seq(2, 0, 1, [5])],
    [Seq(1, 0, 257, [5, 5])],
])
def test_bad_metadata_fails_without_mutation(seqs):
    before = [(s.num_cached_tokens, s.num_scheduled_tokens, list(s.block_table)) for s in seqs]
    with pytest.raises(ValueError):
        lp.plan_chunks(seqs, 128, 256)
    assert before == [(s.num_cached_tokens, s.num_scheduled_tokens, s.block_table) for s in seqs]


class FakeEvent:
    def __init__(self, cuda):
        self.cuda, self.recorded = cuda, False
    def record(self, stream):
        self.recorded = True
        self.cuda.log.append(("record", stream.name, id(self)))


class FakeStream:
    def __init__(self, cuda, name):
        self.cuda, self.name = cuda, name
    def wait_event(self, event):
        assert event.recorded, "waited on an event that had not been recorded"
        self.cuda.log.append(("wait", self.name, id(event)))


class FakeNvtx:
    def __init__(self, cuda):
        self.cuda = cuda
    def range_push(self, name):
        self.cuda.log.append(("nvtx_push", name))
    def range_pop(self):
        self.cuda.log.append(("nvtx_pop",))


class FakeCuda:
    def __init__(self):
        self.log = []
        self.source = FakeStream(self, "source")
        self.current = self.source
        self.created = []
        self.nvtx = FakeNvtx(self)
    def Stream(self, device=None):
        stream = FakeStream(self, f"worker{len(self.created)}")
        self.created.append(stream)
        return stream
    def Event(self):
        return FakeEvent(self)
    def current_stream(self, device=None):
        return self.current
    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous


class FakeLayer:
    def __init__(self, index, observations, fail_on=None):
        self.index, self.observations, self.fail_on = index, observations, fail_on
    def __call__(self, positions, hidden, residual):
        context = get_context()
        self.observations.append((self.index, positions.tolist(), context.cu_seqlens_q.tolist(),
                                  context.cu_seqlens_k.tolist(), context.block_tables.tolist(),
                                  context.slot_mapping.tolist(), residual is None))
        if self.fail_on is not None and int(positions[0]) == self.fail_on:
            raise RuntimeError("fake layer failure")
        if residual is None:
            residual = hidden * 2
        else:
            residual = residual + hidden
        return hidden + (self.index + 1), residual


class FakeCore:
    def __init__(self, layers=3, fail_layer=None, fail_position=None):
        self.observations = []
        self.layers = [FakeLayer(i, self.observations, fail_position if i == fail_layer else None)
                       for i in range(layers)]
    def embed_tokens(self, ids):
        return ids.to(torch.float32).unsqueeze(-1)
    def norm(self, hidden, residual):
        return hidden + residual / 8, residual


class FakeModel:
    def __init__(self, **kwargs):
        self.model = FakeCore(**kwargs)
    def parameters(self):
        return iter(())
    def buffers(self):
        return iter(())


def parent_context(seqs, block_size):
    lengths = [s.num_scheduled_tokens for s in seqs]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    tasks = lp.plan_chunks(seqs, 128, block_size)
    slots = [slot for task in tasks for slot in task.slots]
    context = Context(is_prefill=True, cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
                      slot_mapping=torch.tensor(slots, dtype=torch.int32), block_tables=None)
    restore_context(context)
    return context


def expected_output(ids, layers):
    hidden = ids.float().unsqueeze(-1)
    residual = None
    for i in range(layers):
        residual = hidden * 2 if residual is None else residual + hidden
        hidden = hidden + i + 1
    return hidden + residual / 8


@pytest.mark.parametrize("mode,stream_count", [("serial", 1), ("pipeline", 2)])
def test_fake_cuda_executor_context_residual_output_and_no_host_sync(mode, stream_count, monkeypatch):
    seqs = [Seq(1, 0, 257, [3, 5]), Seq(2, 4, 3, [8])]
    parent = parent_context(seqs, 256)
    ids = torch.arange(260, dtype=torch.long)
    positions = torch.tensor([*range(257), 4, 5, 6])
    fake_cuda, model = FakeCuda(), FakeModel()
    stats = lp.new_stats(mode, 128, 2)
    records = []
    monkeypatch.setattr(lp, "_record_tensor", lambda tensor, stream: records.append((tensor, stream.name)))
    executor = lp.LayerPrefillExecutor(model, torch.empty(1), 256, 128, 2, mode, stats,
                                      cuda=fake_cuda, nvtx_enabled=True)
    output = executor.forward(seqs, ids, positions)
    assert get_context() is parent
    torch.testing.assert_close(output, expected_output(ids, 3))
    assert stats["planned_tasks"] == 4 and stats["layer_ops_enqueued"] == 12
    assert stats["same_chunk_waits"] == 8 and stats["previous_chunk_waits"] == 6
    assert stats["streams_created"] == stream_count and stats["join_waits"] == 4
    assert stats["last_single_stream_reason"] == ("serial_mode" if mode == "serial" else None)
    assert stats["completed_tasks"] == 0 and executor.pending_tasks == 4
    assert sum(x[0] == "nvtx_push" for x in fake_cuda.log) == 12
    assert sum(x[0] == "nvtx_pop" for x in fake_cuda.log) == 12
    assert all(x[0] != "synchronize" for x in fake_cuda.log)
    assert len([x for x in records if x[1] == "source"]) == 4
    for layer, pos, cu_q, cu_k, blocks, slots, first in model.model.observations:
        assert cu_q == [0, len(pos)] and cu_k == [0, pos[-1] + 1]
        assert blocks and slots and len(slots) == len(pos)
        assert first == (layer == 0)
    executor.mark_completed()
    assert stats["completed_steps"] == 1 and stats["completed_tasks"] == 4
    assert executor.pending_tasks == 0 and executor._retained is None


def test_fake_cuda_exception_restores_context_and_fences_workers():
    seqs = [Seq(1, 0, 257, [3, 5])]
    parent = parent_context(seqs, 256)
    fake_cuda = FakeCuda()
    stats = lp.new_stats("pipeline", 128, 2)
    model = FakeModel(fail_layer=1, fail_position=128)
    executor = lp.LayerPrefillExecutor(model, torch.empty(1), 256, 128, 2, "pipeline", stats, cuda=fake_cuda)
    with pytest.raises(RuntimeError, match="fake layer failure"):
        executor.forward(seqs, torch.arange(257), torch.arange(257))
    assert get_context() is parent and executor.failed
    assert not any(x[0] == "nvtx_push" for x in fake_cuda.log)
    assert len([x for x in fake_cuda.log if x[0] == "wait" and x[1] == "source"]) == 2
    assert stats["completed_steps"] == 0
    with pytest.raises(RuntimeError, match="cannot resume"):
        executor.forward(seqs, torch.arange(257), torch.arange(257))


@pytest.mark.parametrize("count,inflight,reason", [(1, 2, "single_chunk"), (129, 1, "max_inflight_one")])
def test_single_stream_reasons_are_not_claimed_as_gpu_overlap(count, inflight, reason):
    seq = Seq(1, 0, count, [3])
    parent_context([seq], 256)
    stats = lp.new_stats("pipeline", 128, inflight)
    executor = lp.LayerPrefillExecutor(FakeModel(), torch.empty(1), 256, 128, inflight,
                                      "pipeline", stats, cuda=FakeCuda())
    executor.forward([seq], torch.arange(count), torch.arange(count))
    assert stats["last_streams_used"] == 1 and stats["last_single_stream_reason"] == reason


class FakeCausalLayer:
    """Small per-layer paged KV simulation to exercise causal dependencies."""
    def __init__(self, index, block_size):
        self.index, self.block_size, self.cache = index, block_size, {}
    def __call__(self, positions, hidden, residual):
        ctx = get_context()
        pos_list = positions.tolist()
        slots = ctx.slot_mapping.tolist()
        blocks = ctx.block_tables.tolist()[0]
        assert ctx.cu_seqlens_k.tolist() == [0, pos_list[-1] + 1]
        for slot, row in zip(slots, hidden):
            self.cache[slot] = row.clone()
        means = []
        for pos in pos_list:
            visible = [blocks[i // self.block_size] * self.block_size + i % self.block_size
                       for i in range(pos + 1)]
            # An unwritten earlier token at this layer raises immediately.
            means.append(torch.stack([self.cache[slot] for slot in visible]).mean(dim=0))
        past = torch.stack(means)
        residual = hidden * 2 if residual is None else residual + hidden
        return hidden + past / 16 + (self.index + 1), residual


def test_fake_per_layer_paged_causal_kv_matches_unchunked_and_serial():
    seqs = [Seq(1, 0, 9, [3, 5, 7]), Seq(2, 0, 5, [9, 12])]
    ids = torch.arange(1, 15)
    positions = torch.tensor([*range(9), *range(5)])
    results, caches = [], []
    for mode, chunk in (("serial", 32), ("serial", 3), ("pipeline", 3)):
        parent_context(seqs, 4)
        model = FakeModel()
        model.model.layers = [FakeCausalLayer(i, 4) for i in range(3)]
        executor = lp.LayerPrefillExecutor(model, torch.empty(1), 4, chunk, 2, mode,
                                          lp.new_stats(mode, chunk, 2), cuda=FakeCuda())
        results.append(executor.forward(seqs, ids, positions))
        caches.append([{key: value.clone() for key, value in layer.cache.items()}
                       for layer in model.model.layers])
    for result in results[1:]:
        torch.testing.assert_close(result, results[0], rtol=0, atol=0)
    for per_layer in caches[1:]:
        for actual, reference in zip(per_layer, caches[0]):
            assert set(actual) == set(reference)
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["serial", "pipeline"])
def test_fake_cached_start_reads_previous_completed_step_kv(mode):
    blocks = [3, 5, 7]
    all_ids = torch.arange(1, 10)
    full_seq = Seq(1, 0, 9, blocks)
    parent_context([full_seq], 4)
    reference_model = FakeModel()
    reference_model.model.layers = [FakeCausalLayer(i, 4) for i in range(3)]
    ref = lp.LayerPrefillExecutor(reference_model, torch.empty(1), 4, 32, 2, "serial",
                                  lp.new_stats("serial", 32, 2), cuda=FakeCuda())
    reference = ref.forward([full_seq], all_ids, torch.arange(9))

    model = FakeModel()
    model.model.layers = [FakeCausalLayer(i, 4) for i in range(3)]
    executor = lp.LayerPrefillExecutor(model, torch.empty(1), 4, 3, 2, mode,
                                      lp.new_stats(mode, 3, 2), cuda=FakeCuda())
    first = Seq(1, 0, 4, blocks)
    parent_context([first], 4)
    executor.forward([first], all_ids[:4], torch.arange(4))
    executor.mark_completed()  # Simulates the existing sampler's completion fence.
    tail = Seq(1, 4, 5, blocks)
    parent_context([tail], 4)
    actual = executor.forward([tail], all_ids[4:], torch.arange(4, 9))
    torch.testing.assert_close(actual, reference[4:], rtol=0, atol=0)
    assert tail.num_cached_tokens == 4 and tail.num_scheduled_tokens == 5
def isolated_config(model_type="qwen3", dtype=torch.bfloat16):
    tf = types.ModuleType("transformers")
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path):
            return SimpleNamespace(max_position_embeddings=4096, model_type=model_type, dtype=dtype)
    tf.AutoConfig = FakeAutoConfig
    quant = types.ModuleType("nanovllm.layers.quant")
    quant.hf_names_to_modules = lambda names: names
    spec = importlib.util.spec_from_file_location("isolated_layer_prefill_config", ROOT / "nanovllm/config.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as mp:
        for name, value in {"transformers": tf, "nanovllm.layers.quant": quant, spec.name: module}.items():
            mp.setitem(sys.modules, name, value)
        spec.loader.exec_module(module)
        yield module.Config


def test_config_defaults_and_enabled_scope(tmp_path):
    for Config in isolated_config():
        old = Config(str(tmp_path))
        assert not old.enable_layer_prefill and old.layer_prefill_chunk_tokens == 128
        assert old.layer_prefill_max_inflight == 2 and old.layer_prefill_mode == "pipeline"
        assert not old.layer_prefill_nvtx
        enabled = Config(str(tmp_path), enable_layer_prefill=True, enable_prefix_cache=False)
        assert enabled.enable_layer_prefill and not enabled.enable_prefix_cache
        serial = Config(str(tmp_path), enable_layer_prefill=True, enable_prefix_cache=False,
                        layer_prefill_mode="serial", layer_prefill_nvtx=True)
        assert serial.layer_prefill_mode == "serial" and serial.layer_prefill_nvtx


@pytest.mark.parametrize("kwargs", [
    {"enable_prefix_cache": True}, {"tensor_parallel_size": 2}, {"spec_overlap_mixed": True},
    {"spec_k": 1}, {"spec_method": "draft"}, {"draft_model": "other"},
    {"quant_weights": "int8"}, {"draft_quant_weights": "int8"},
    {"layer_prefill_chunk_tokens": 0}, {"layer_prefill_chunk_tokens": True},
    {"layer_prefill_max_inflight": 0}, {"layer_prefill_max_inflight": 3},
    {"layer_prefill_mode": "other"}, {"layer_prefill_nvtx": "yes"},
])
def test_unsupported_configurations_fail_early(tmp_path, kwargs):
    for Config in isolated_config():
        values = {"enable_layer_prefill": True, "enable_prefix_cache": False}
        values.update(kwargs)
        with pytest.raises(ValueError):
            Config(str(tmp_path), **values)


def test_auto_detected_int8_target_rejected(tmp_path):
    (tmp_path / "quant_config.json").write_text('{"quant_method":"int8_w8a16","quantized_layers":[]}',
                                                  encoding="utf-8")
    for Config in isolated_config():
        with pytest.raises(ValueError, match="auto-detected"):
            Config(str(tmp_path), enable_layer_prefill=True, enable_prefix_cache=False)


@pytest.mark.parametrize("model_type,dtype", [("other", torch.bfloat16), ("qwen3", torch.float16)])
def test_unsupported_target_family_or_dtype_rejected(tmp_path, model_type, dtype):
    for Config in isolated_config(model_type, dtype):
        with pytest.raises(ValueError, match="BF16 Qwen3"):
            Config(str(tmp_path), enable_layer_prefill=True, enable_prefix_cache=False)


def load_runner(monkeypatch):
    def module(name, **members):
        result = types.ModuleType(name)
        result.__dict__.update(members)
        return result
    stubs = {
        "nanovllm.config": module("nanovllm.config", Config=object),
        "nanovllm.models.qwen3": module("nanovllm.models.qwen3", Qwen3ForCausalLM=object),
        "nanovllm.layers.sampler": module("nanovllm.layers.sampler", Sampler=object),
        "nanovllm.layers.quant": module("nanovllm.layers.quant", apply_int8_quant=lambda *a, **kw: None),
        "nanovllm.spec_decode.draft_proposer": module("nanovllm.spec_decode.draft_proposer",
                                                       DraftWindow=object,
                                                       window_realign_target=lambda *a, **kw: None),
        "nanovllm.utils.loader": module("nanovllm.utils.loader", load_model=lambda *a, **kw: None),
    }
    for name, value in stubs.items():
        monkeypatch.setitem(sys.modules, name, value)
    spec = importlib.util.spec_from_file_location("isolated_layer_prefill_runner", ROOT / "nanovllm/engine/model_runner.py")
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    return result.ModelRunner


def fake_runner(monkeypatch, enabled, seq, *, cache_allocated):
    Runner = load_runner(monkeypatch)
    runner = Runner.__new__(Runner)
    runner.config = SimpleNamespace(enable_layer_prefill=enabled, dspark_window_tokens=0,
                                    layer_prefill_chunk_tokens=128, layer_prefill_max_inflight=2,
                                    layer_prefill_mode="pipeline", layer_prefill_nvtx=False)
    runner.rank, runner.block_size = 0, 256
    runner.draft_model = runner.dspark_model = runner.spec_proposer = None
    runner._draft_prefill_needed = lambda: False
    parent = Context(is_prefill=True, cu_seqlens_q=torch.tensor([0, seq.num_scheduled_tokens]),
                     slot_mapping=torch.arange(seq.num_scheduled_tokens))
    def prepare(seqs, advance_draft):
        restore_context(parent)
        return torch.arange(seq.num_scheduled_tokens), torch.arange(seq.num_scheduled_tokens)
    runner.prepare_prefill = prepare
    runner.prepare_sample = lambda seqs: None
    calls = []
    runner.run_model = lambda *args: calls.append("original") or torch.tensor([[1.0]])
    runner.sampler = lambda logits, temperatures: torch.tensor([42])
    layer = SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(k_cache=torch.ones(1),
                                                                          v_cache=torch.ones(1))))
    def compute_logits(hidden):
        assert get_context() is parent
        calls.append("lm_head")
        return torch.tensor([[2.0]])
    runner.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]), compute_logits=compute_logits)
    if cache_allocated:
        runner.kv_cache = torch.empty(1)
    if enabled:
        runner.layer_prefill_stats = lp.new_stats("pipeline", 128, 2)
        runner.layer_prefill_executor = None
    return runner, calls


def test_runner_off_uses_original_prefill_without_executor(monkeypatch):
    seq = Seq(1, 0, 1, [2])
    runner, calls = fake_runner(monkeypatch, False, seq, cache_allocated=True)
    assert runner.run([seq], True) == [42]
    assert calls == ["original"]
    assert not hasattr(runner, "layer_prefill_executor") and not hasattr(runner, "layer_prefill_stats")
    assert seq.num_cached_tokens == 0 and seq.num_scheduled_tokens == 1


def test_runner_warmup_fallback_only_before_kv_allocation(monkeypatch):
    seq = Seq(1, 0, 1, [])
    runner, calls = fake_runner(monkeypatch, True, seq, cache_allocated=False)
    assert runner.run([seq], True) == [42]
    assert calls == ["original"] and runner.layer_prefill_executor is None
    assert runner.layer_prefill_stats["fallback_reasons"] == {"warmup_no_kv": 1}
    runner.kv_cache = torch.empty(1)
    with pytest.raises(RuntimeError, match="without all target KV"):
        runner.run([seq], True)


def test_runner_real_path_selects_executor_then_parent_lm_head(monkeypatch):
    seq = Seq(1, 0, 1, [2])
    runner, calls = fake_runner(monkeypatch, True, seq, cache_allocated=True)
    class FakeExecutor:
        def forward(self, seqs, ids, positions):
            calls.append("executor")
            return torch.ones(1, 4)
        def mark_completed(self):
            calls.append("completed_after_sampler")
    runner.layer_prefill_executor = FakeExecutor()
    assert runner.run([seq], True) == [42]
    assert calls == ["executor", "lm_head", "completed_after_sampler"]
    assert seq.num_cached_tokens == 0 and seq.num_scheduled_tokens == 1
