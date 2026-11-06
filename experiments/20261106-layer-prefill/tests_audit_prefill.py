"""CPU-only audit fixtures: one smoke and one timed sample per mode."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("layer_prefill_audit", HERE / "audit_prefill.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from bench_common import COUNTERS, token_hash, validate_layer_execution
from bench_prefill import validate_result


def snapshots(mode, lengths, layers):
    if mode == "full":
        return None, None, None
    streams = 1 if mode == "serial" else 2
    before = {k: 0 for k in COUNTERS}
    before.update({"mode": mode, "chunk_tokens": 128, "max_inflight": 2, "nvtx_enabled": False,
                   "fallback_reasons": {"warmup_no_kv": 1}, "stream_task_counts": {},
                   "streams_created": streams, "stream_ids": [str(100 + i) for i in range(streams)],
                   "last_streams_used": streams, "last_single_stream_reason": "serial_mode" if mode == "serial" else None})
    after = copy.deepcopy(before)
    tasks = sum(math.ceil(n / 128) for n in lengths)
    counts = {"0": tasks} if streams == 1 else {"0": (tasks + 1) // 2, "1": tasks // 2}
    after.update(prefill_steps=1, planned_tasks=tasks, layer_ops_enqueued=tasks * layers,
                 same_chunk_waits=tasks * (layers - 1), previous_chunk_waits=(tasks - len(lengths)) * layers,
                 source_waits=streams, join_waits=tasks, completed_steps=1, completed_tasks=tasks,
                 stream_task_counts=counts)
    return before, after, validate_layer_execution(mode, before, after, lengths, layers, 128)


def common():
    return {"engine_commit": "commit", "engine_dirty": False, "gpu_uuid": "GPU-test", "budget": 0.85,
            "target_provenance": {"path": "target", "config_sha256": "cfg", "weight_files": [{"name": "w", "bytes": 1}]},
            "chunk": 128, "num_layers": 2, "guard_id": "guard"}


def smoke_rows():
    data = []
    inputs = [{"actual_input_sha256": "a" * 64, "actual_context_tokens": 257},
              {"actual_input_sha256": "b" * 64, "actual_context_tokens": 513}]
    for mode in audit.MODES:
        before, after, delta = snapshots(mode, [257, 513], 2)
        blocks = [{"input_index": i, "physical_blocks": ids, "block_size": 256,
                   "valid_prefix_slots": length, "sample_positions": audit.sample_positions(length),
                   "preimage": "logical prompt slots NaN; page padding zero"}
                  for i, (length, ids) in enumerate(((257, [0, 1]), (513, [2, 3, 4])))]
        data.append({"type": "mode_complete", **common(), "mode": mode, "case": "b2_257_513",
                     "target_kv_blocks": 200, "kv_block_size": 256,
                     "lengths": [257, 513], "input_metadata": inputs, "fixed_history_sha256": ["c" * 64, "d" * 64],
                     "diagnostic_instrumented": True, "prefill_calls": [{"active_b": 2, "scheduled_tokens": [257, 513]}],
                     "sentinel_blocks": blocks, "layer_before": before, "layer_after": after, "layer_delta": delta,
                     "graph_delta": {"decode_graph_exact_hits": 16, "decode_graph_padded_hits": 0, "decode_graph_fallbacks": {}}})
    for left, right, kl_limit, rms_limit in audit.PAIRS:
        for request, length in enumerate((257, 513)):
            for step in range(17):
                data.append({"type": "logit_check", "reference": left, "candidate": right, "case": "b2_257_513",
                             "request_index": request, "step": step, "step_kind": "prefill" if step == 0 else "decode",
                             "finite": True, "pass": True, "kl_direction": "reference || candidate",
                             "kl_limit": kl_limit, "kl": 0.0, "max_abs": 0.0, "reference_argmax": 3,
                             "candidate_argmax": 3, "reference_top1_top2_margin": 1.0, "candidate_top1_top2_margin": 1.0})
            for position in audit.sample_positions(length):
                for kind in ("K", "V"):
                    for layer in range(2):
                        data.append({"type": "kv_check", "reference": left, "candidate": right, "case": "b2_257_513",
                                     "request_index": request, "position": position, "kv": kind, "layer": layer,
                                     "finite": True, "pass": True, "reference_rms": 1.0, "relative_rms": 0.0,
                                     "relative_rms_limit": rms_limit, "max_abs": 0.0})
    checks = len(data) - 3
    data.append({"type": "complete", **common(), "pass": True, "comparisons": checks, "failed_comparisons": 0})
    return [(i + 1, row) for i, row in enumerate(data)]


def perf_rows(mode, wall):
    b, length, output, layers = 4, 4096, 128, 2
    before, after, delta = snapshots(mode, [length] * b, layers)
    identity = {**common(), "mode": mode, "batch": b, "ctx": length, "output_tokens": output, "panel": "code",
                "seed": 20261106, "temperature": 1.0, "diagnostic_instrumented": False,
                "input_metadata": [{"actual_input_sha256": str(i) * 64, "actual_context_tokens": length} for i in range(b)],
                "primary_timing_repeats": 1, "warmup_cohorts": 1}
    requests = []
    for i in range(b):
        ids = list(range(i * output, (i + 1) * output))
        requests.append({"seq_id": i + 1, "arrival_s": 0.0, "finish_s": 127.5,
                         "deliveries": [{"at_s": k + 0.5, "count": 1} for k in range(output)],
                         "output_token_ids": ids, "output_sha256": token_hash(ids)})
    rounds = [{"start_s": k + 0.1, "end_s": k + 0.9, "prefill": k == 0, "active_b": b, "delivered": b,
               "sequences": [{"seq_id": i + 1, "delivered": 1, "proposed": 0, "accepted_prefix": 0}
                             for i in range(b)]} for k in range(output)]
    record = {"type": "run", **identity, "repeat": 0, "requested_batch": b, "arm": "nospec", "ignore_eos": True,
              "wall_s": wall, "requests": requests, "rounds": rounds, "delivered_tokens": b * output,
              "preemptions": 0, "prefix_cached_tokens": 0, "prefix_lookup_count": 0, "prefix_hit_blocks": 0,
              "observed_prefill_lengths": [length] * b, "target_decode_graph": {"exact": 127, "padded": 0, "fallbacks": {}},
              "layer_before": before, "layer_after": after, "layer_delta": delta, "target_kv_blocks": 200,
              "load_initial_graph_s": 10.0, "memory_after_run": {"allocated_bytes": 100}}
    record["metrics"] = json.loads(json.dumps(validate_result(record, [length] * b)))
    warm = {"type": "warmup", **identity, "prefill_calls": [{"scheduled_tokens": [length] * b}],
            "layer_before": before, "layer_after": after, "warmup_s": 1.0}
    end = {"type": "complete", **identity, "repeats_per_panel": 1}
    return [(i + 1, row) for i, row in enumerate((warm, record, end))]


def test_smoke_requires_all_positions_and_preserves_failures():
    rows = smoke_rows()
    result = audit.audit_smoke(rows)
    assert result["valid"], result["errors"]
    assert result["logit_checks"] == 102 and result["kv_checks"] == 144
    broken = copy.deepcopy(rows)
    broken.pop(10)
    assert not audit.audit_smoke(broken)["valid"]
    broken = copy.deepcopy(rows)
    next(row for _, row in broken if row.get("type") == "logit_check")["candidate_argmax"] = 9
    assert not audit.audit_smoke(broken)["valid"]
    broken = copy.deepcopy(rows)
    broken.append((9999, {"type": "failure", "error": "synthetic"}))
    assert audit.audit_smoke(broken)["failures"]


def test_single_run_comparison_requires_valid_smoke_and_same_inputs():
    smoke = audit.audit_smoke(smoke_rows())
    perf = {mode: audit.audit_perf(perf_rows(mode, 200 if mode == "full" else 180), mode) for mode in audit.MODES}
    assert all(value["valid"] for value in perf.values()), perf
    result = audit.summarize(smoke, perf)
    assert result["single_sample_only"] and result["ratios"]["pipeline_over_full"] == pytest.approx(200 / 180)
    perf["pipeline"]["identity"]["input_metadata"][0]["actual_input_sha256"] = "f" * 64
    assert audit.summarize(smoke, perf)["ratios"] is None
    assert audit.summarize({"valid": False}, perf)["ratios"] is None


def test_duplicate_timing_or_mixed_graph_fails_closed():
    rows = perf_rows("pipeline", 180)
    rows.append((4, copy.deepcopy(rows[1][1])))
    assert not audit.audit_perf(rows, "pipeline")["valid"]
    rows = perf_rows("pipeline", 180)
    rows[1][1]["target_decode_graph"]["fallbacks"] = {"missing": 1}
    assert not audit.audit_perf(rows, "pipeline")["valid"]
