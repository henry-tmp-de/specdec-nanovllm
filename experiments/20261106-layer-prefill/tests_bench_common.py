"""CPU-only configuration, input identity and runtime-counter contracts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("layer_prefill_bench_common", HERE / "bench_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_options_keep_same_workload_and_enable_only_selected_mode():
    allopts = [common.options_for(mode, 4, 4096, 1, 128, 0.85) for mode in common.MODES]
    assert [x["enable_layer_prefill"] for x in allopts] == [False, True, True]
    assert [x["layer_prefill_mode"] for x in allopts] == ["pipeline", "serial", "pipeline"]
    for option in allopts:
        assert option["max_num_seqs"] == 4 and option["max_num_batched_tokens"] == 4 * 4096
        assert option["max_model_len"] == 4096 + 1 + 8 and option["decode_graph_bs"] == (1, 2, 3, 4)
        assert option["spec_k"] == 0 and option["draft_model"] == "" and option["enable_prefix_cache"] is False
        assert option["spec_prefill_chunk_tokens"] == 0 and option["spec_overlap_mixed"] is False
    assert common.options_for("pipeline", 1, 1024, 128, 512, 0.70)["max_model_len"] == 1160
    with pytest.raises(ValueError):
        common.options_for("pipeline", 1, 1024, 1, 128, 0.9)


def test_workload_trimming_is_hashed_and_fill_is_not_invented(tmp_path):
    prompt = list(range(1024))
    manifest = {"context_tokens": 1024, "panels": {"code": [
        {"token_ids": prompt, "metadata": {"input_sha256": common.token_hash(prompt), "wiki_tokens": 900}},
        {"token_ids": prompt, "metadata": {"input_sha256": common.token_hash(prompt), "wiki_tokens": 900}}]}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    inputs = common.read_inputs(path, "code", [257, 513])
    assert inputs[0][0] == prompt[-257:] and inputs[1][0] == prompt[-513:]
    assert inputs[0][1]["actual_input_sha256"] == common.token_hash(prompt[-257:])
    assert inputs[0][1]["trimmed_from_left"] == 767
    assert "not inferred" in inputs[0][1]["fill_metadata_note"]
    manifest["panels"]["code"][0]["metadata"]["input_sha256"] = "wrong"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        common.read_inputs(path, "code", [257])


def stats(mode):
    streams = 1 if mode == "serial" else 2
    result = {k: 0 for k in common.COUNTERS}
    result.update({"mode": mode, "chunk_tokens": 128, "max_inflight": 2, "nvtx_enabled": False,
                   "fallback_reasons": {"warmup_no_kv": 1}, "stream_task_counts": {},
                   "streams_created": streams, "stream_ids": [str(i + 100) for i in range(streams)],
                   "last_streams_used": streams, "last_single_stream_reason": "serial_mode" if mode == "serial" else None})
    return result


@pytest.mark.parametrize("mode", ("serial", "pipeline"))
def test_completed_execution_counters_cover_all_dependencies(mode):
    before = stats(mode)
    after = copy.deepcopy(before)
    # Two independent sequences with lengths 257 and 513 -> 3+5 chunks, 4 layers.
    after.update(prefill_steps=1, planned_tasks=8, layer_ops_enqueued=32, same_chunk_waits=24,
                 previous_chunk_waits=24, source_waits=1 if mode == "serial" else 2, join_waits=8,
                 completed_steps=1, completed_tasks=8,
                 stream_task_counts={"0": 8} if mode == "serial" else {"0": 4, "1": 4})
    delta = common.validate_layer_execution(mode, before, after, [257, 513], 4, 128)
    assert delta["completed_tasks"] == 8 and delta["fallback_steps"] == 0
    assert delta["fallback_reasons"] == {}
    broken = copy.deepcopy(after)
    broken["completed_tasks"] -= 1
    with pytest.raises(ValueError, match="counters"):
        common.validate_layer_execution(mode, before, broken, [257, 513], 4, 128)


def test_disabled_mode_has_no_layer_prefill_state():
    assert common.validate_layer_execution("full", None, None, [1024], 36, 128) is None
    with pytest.raises(ValueError, match="disabled"):
        common.validate_layer_execution("full", None, {}, [1024], 36, 128)


def test_single_chunk_diagnostic_is_labeled_one_stream_not_overlap():
    before = stats("pipeline")
    before["chunk_tokens"] = 512
    after = copy.deepcopy(before)
    after.update(prefill_steps=1, planned_tasks=1, layer_ops_enqueued=4, same_chunk_waits=3,
                 previous_chunk_waits=0, source_waits=2, join_waits=1, single_chunk_steps=1,
                 completed_steps=1, completed_tasks=1, stream_task_counts={"0": 1},
                 last_streams_used=1, last_single_stream_reason="single_chunk")
    delta = common.validate_layer_execution("pipeline", before, after, [257], 4, 512)
    assert delta["single_chunk_steps"] == 1 and after["last_streams_used"] == 1
