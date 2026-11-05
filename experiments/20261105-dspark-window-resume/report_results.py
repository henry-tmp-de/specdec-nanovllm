"""Build a traceable DSpark experiment report from existing JSONL and ledger files.

No GPU, model, downloads, benchmark execution, or historical performance import.
Every reported run keeps its raw file/line and guard identity. Missing work and
failed attempts remain explicit. Primary comparisons require matched conditions
and at least three independently recorded repeats per arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_results as audit

ARMS = ("nospec", "ar_full", "ar_window", "dspark_full", "dspark_ring")
CELL_FIELDS = ("phase", "arm", "mode", "ctx", "requested_batch", "panel", "window", "graph_mode", "gpu_memory_utilization")
COMMON_FIELDS = ("phase", "ctx", "requested_batch", "panel", "seed", "temperature", "output_tokens",
                 "gpu_uuid", "engine_commit", "engine_dirty", "gpu_memory_utilization", "max_num_seqs",
                 "max_model_len", "max_num_batched_tokens", "prefix_lookup", "graph_policy", "mixed_overlap",
                 "ignore_eos", "target_config_sha256", "target_provenance")
METRICS = ("full_request_tps", "decode_span_tps", "full_batch_prefix_tps",
           "ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "tpot_p50_s", "tpot_p95_s", "tpot_p99_s",
           "delivery_gap_p50_s", "delivery_gap_p95_s", "delivery_gap_p99_s",
           "acceptance_fraction", "tau_delivered_per_decode_sequence_round")
WARM_IDENTITY_FIELDS = ("phase", "arm", "mode", "ctx", "requested_batch", "max_num_seqs", "window",
                        "output_tokens", "seed", "temperature", "ignore_eos", "spec_k", "max_model_len",
                        "max_num_batched_tokens", "gpu_memory_utilization", "graph_mode", "graph_policy",
                        "prefix_lookup", "mixed_overlap", "diagnostic_instrumented", "gpu_uuid", "guard_id",
                        "engine_commit", "engine_dirty", "target_config_sha256", "draft_config_sha256",
                        "target_provenance", "draft_provenance")


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_lines(files):
    records, errors = [], []
    for filename in sorted(set(str(Path(p)) for p in files)):
        try:
            lines = Path(filename).read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append({"source": filename, "error": str(exc)})
            continue
        for no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                def reject_constant(token):
                    raise ValueError(f"non-finite JSON number {token}")
                row = json.loads(line, parse_constant=reject_constant)
                if not isinstance(row, dict):
                    raise ValueError("row is not a JSON object")
                records.append((filename, no, row))
            except (ValueError, TypeError) as exc:
                errors.append({"source": filename, "line": no, "error": str(exc)})
    return records, errors


def ledger_index(rows):
    events = defaultdict(dict)
    errors = []
    for source, line, row in rows:
        ident, kind = row.get("id"), row.get("event")
        if not ident or kind not in ("start", "child", "end"):
            errors.append({"source": source, "line": line, "error": "unrecognized ledger event or missing id"})
            continue
        if kind in events[ident]:
            errors.append({"source": source, "line": line, "error": f"duplicate {kind} for {ident}"})
        else:
            events[ident][kind] = (source, line, row)
    return events, errors


def guard_result(ident, ledger):
    events = ledger.get(ident, {}) if ident else {}
    start = events.get("start", (None, None, {}))[2]
    child = events.get("child", (None, None, {}))[2]
    end = events.get("end", (None, None, {}))[2]
    identity_errors = []
    if start and end:
        if start.get("gpu_uuid") != end.get("gpu_uuid") or start.get("kind") != end.get("kind") or start.get("stage") != end.get("stage"):
            identity_errors.append("start/end identity mismatch")
        if type(end.get("elapsed_s")) not in (int, float) or not math.isfinite(end["elapsed_s"]) or end["elapsed_s"] < 0:
            identity_errors.append("invalid elapsed time")
    if child:
        if (not start or child.get("gpu_uuid") != start.get("gpu_uuid") or child.get("kind") != start.get("kind")
                or child.get("stage") != start.get("stage") or type(child.get("pid")) is not int or child["pid"] <= 0
                or type(child.get("start_ticks")) is not int or child["start_ticks"] < 0):
            identity_errors.append("child identity mismatch or missing PID/start ticks")
        if end and end.get("root_identity") != [child.get("pid"), child.get("start_ticks")]:
            identity_errors.append("end root identity differs from child identity")
    if not start:
        status = "missing_start"
    elif not end:
        status = "unfinished"
    elif end.get("outcome") != "ok" or end.get("exit_code") != 0:
        status = "failed"
    elif not child or identity_errors or end.get("ever_owned_compute_observed") is not True:
        status = "invalid"
    else:
        status = "ok"
    return {"guard_id": ident, "stage": start.get("stage"), "kind": start.get("kind"),
            "gpu_uuid": start.get("gpu_uuid"), "status": status,
            "outcome": end.get("outcome"), "exit_code": end.get("exit_code"),
            "elapsed_s": safe_value(end.get("elapsed_s")), "detail": end.get("detail"),
            "owned_compute_observed": end.get("ever_owned_compute_observed"),
            "identity_errors": identity_errors,
            "child_pid": child.get("pid"), "child_start_ticks": child.get("start_ticks"),
            "raw_start": list(events["start"][:2]) if "start" in events else None,
            "raw_end": list(events["end"][:2]) if "end" in events else None}


def raw_ledger(ident, ledger):
    events = ledger.get(ident, {})
    return [events[k][2] for k in ("start", "child", "end") if k in events]


def warm_identity_errors(row, warmups):
    warm = warmups.get(row.get("guard_id"), [])
    if len(warm) != 1:
        return ["warmup identity missing or ambiguous"]
    data = warm[0][2]
    mismatch = [key for key in WARM_IDENTITY_FIELDS if key not in row or key not in data or row[key] != data[key]]
    return ["warmup/run identity mismatch: " + ", ".join(mismatch)] if mismatch else []


def input_hashes(row):
    metadata = row.get("input_metadata")
    if not isinstance(metadata, list):
        return None
    hashes = [m.get("actual_input_sha256", m.get("input_sha256")) for m in metadata if isinstance(m, dict)]
    return hashes if len(hashes) == len(metadata) and all(isinstance(x, str) and x for x in hashes) else None


def provenance_ok(row):
    def file_metadata_ok(model):
        files = model.get("weight_files") if isinstance(model, dict) else None
        return (isinstance(model, dict) and bool(model.get("path")) and bool(model.get("config_sha256"))
                and isinstance(files, list) and bool(files)
                and all(isinstance(f, dict) and isinstance(f.get("name"), str) and bool(f["name"])
                        and type(f.get("bytes")) is int and f["bytes"] > 0 for f in files)
                and len({f["name"] for f in files}) == len(files))
    target = row.get("target_provenance")
    target_ok = file_metadata_ok(target) and row.get("target_config_sha256") == target.get("config_sha256")
    if row.get("arm") == "nospec":
        return target_ok and row.get("draft_provenance") is None
    draft = row.get("draft_provenance")
    return target_ok and file_metadata_ok(draft) and row.get("draft_config_sha256") == draft.get("config_sha256")


def validate_capacity(row, guard, warmups, manifests):
    """Capacity is separate; OOM is retained as failure, never as a small batch."""
    errors = []
    errors.extend(warm_identity_errors(row, warmups))
    try:
        computed = audit.compute_run(row)
        errors.extend(computed["errors"])
    except Exception as exc:
        return [f"malformed raw measurement: {exc}"]
    if row.get("diagnostic_instrumented") is not False or row.get("phase") != "capacity":
        errors.append("capacity record is diagnostic or not capacity phase")
    if guard["status"] != "ok" or guard["kind"] != "capacity" or guard["gpu_uuid"] != row.get("gpu_uuid"):
        errors.append("matching successful capacity guard unavailable")
    if row.get("engine_dirty") is not False or not row.get("engine_commit") or not provenance_ok(row):
        errors.append("clean source or target weight metadata unavailable")
    if (row.get("arm") not in ("dspark_full", "dspark_ring") or row.get("spec_k") != 7
            or row.get("requested_batch") != row.get("max_num_seqs")):
        errors.append("capacity arm or requested batch contract mismatch")
    if (type(row.get("ctx")) is not int or type(row.get("output_tokens")) is not int
            or row.get("max_model_len") != row["ctx"] + row["output_tokens"] + 8):
        errors.append("capacity max_model_len does not cover prompt + output + 8")
    budget = row.get("gpu_memory_utilization")
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 1:
        errors.append("capacity budget unavailable")
    if (row.get("temperature") != 1.0 or row.get("ignore_eos") is not True or row.get("mixed_overlap") is not False
            or row.get("prefix_lookup") is not False or row.get("prefix_lookup_count") != 0 or row.get("prefix_hit_blocks") != 0):
        errors.append("capacity sampling, overlap or prefix policy mismatch")
    b = row.get("requested_batch")
    exact_policy = {"name": "exact_all", "target_decode_padding": False, "target_verify_padding": False,
                    "draft_padding": False, "buckets": list(range(1, b + 1))} if type(b) is int and b > 0 else None
    if row.get("graph_policy") != exact_policy or row.get("graph_mode") != "graph":
        errors.append("capacity graph policy is not exact_all")
    graph = row.get("graph_delta", {})
    target = row.get("target_decode_graph", {})
    if (not isinstance(graph, dict) or not isinstance(target, dict)
            or graph.get("verify_exact_hits", 0) <= 0 or graph.get("graph_exact_hits", 0) <= 0
            or graph.get("verify_padded_hits", 0) or graph.get("graph_padded_hits", 0)
            or graph.get("graph_runtime_captures", 0) or graph.get("head_runtime_captures", 0)
            or target.get("padded", 0) or target.get("fallbacks", {})):
        errors.append("capacity exact graph replay, padding or capture contract mismatch")
    matching = warmups.get(row.get("guard_id"), [])
    if len(matching) != 1:
        errors.append("capacity warmup record missing or ambiguous")
    else:
        warm = matching[0][2]
        if not all(k in warm for k in ("prewarm_s", "prewarm_shapes", "expected_prewarm_shapes", "prewarm_head_batches",
                                        "target_decode_graph_buckets", "target_verify_graph_shapes", "memory_after_prewarm",
                                        "warmup_graph_delta")):
            errors.append("capacity warmup metadata incomplete")
        elif {canon(x) for x in warm["prewarm_shapes"]} != {canon(x) for x in warm["expected_prewarm_shapes"]}:
            errors.append("capacity prewarm shapes differ from expected shapes")
        else:
            try:
                block, ctx, out, w = (int(row[k]) for k in ("kv_block_size", "ctx", "output_tokens", "window"))
                ctx_buckets = {max(block, math.ceil((min(c, w) if w else c) / block) * block)
                               for c in range(ctx, ctx + out + 1)}
                expected = {(n, c) for n in range(1, b + 1) for c in ctx_buckets}
                if ({tuple(x) for x in warm["prewarm_shapes"]} != expected or
                        set(warm["prewarm_head_batches"]) != set(range(1, b + 1)) or
                        set(warm["target_decode_graph_buckets"]) != set(range(1, b + 1)) or
                        {tuple(x) for x in warm["target_verify_graph_shapes"]} != {(n, 7) for n in range(1, b + 1)}):
                    errors.append("capacity prewarm misses an expected exact shape")
                if warm["warmup_graph_delta"].get("graph_runtime_captures", 0) or warm["warmup_graph_delta"].get("head_runtime_captures", 0):
                    errors.append("capacity warmup unexpectedly captured graphs")
            except (ValueError, TypeError, KeyError, ZeroDivisionError, AttributeError):
                errors.append("capacity shape metadata malformed")
    try:
        manifest = manifests[row["ctx"]]
        expected = manifest["panels"][row["panel"]][:row["requested_batch"]]
        hashes = [audit.token_hash(x["token_ids"]) for x in expected]
        if (manifest["context_tokens"] != row["ctx"] or len(expected) != row["requested_batch"]
                or any(len(x["token_ids"]) != row["ctx"] or audit.token_hash(x["token_ids"]) != x["metadata"]["input_sha256"] for x in expected)
                or hashes != input_hashes(row)):
            errors.append("capacity input manifest mismatch")
        for actual, original in zip(row["input_metadata"], expected):
            m = original["metadata"]
            fill = actual.get("wiki_tokens")
            if (type(fill) is not int or not 0 <= fill <= row["ctx"]
                    or type(actual.get("task_truncated_from_left")) is not bool
                    or type(actual.get("wiki_fraction")) not in (int, float)
                    or abs(actual["wiki_fraction"] - fill / row["ctx"]) > 1e-12
                    or any(actual.get(k) != m.get(k) for k in ("wiki_tokens", "task_tokens", "task_truncated_from_left", "dataset", "row"))):
                errors.append("capacity input metadata mismatch")
                break
    except (KeyError, TypeError, AttributeError, ZeroDivisionError):
        errors.append("capacity input manifest unavailable")
    return errors


def validate_primary(source, line, row, warmups, ledger, manifests):
    guard_id = row.get("guard_id")
    warm = warmups.get(guard_id, [])
    ledger_rows = raw_ledger(guard_id, ledger)
    try:
        checked = audit.audit([*warm, (source, line, row)], 1, "formal", None, ledger_rows, manifests)
        if not checked["runs"]:
            return ["primary audit returned no run"]
        errors = list(checked["runs"][0]["metrics"].get("errors", []))
        errors.extend(checked.get("ledger_errors", []))
        errors.extend(warm_identity_errors(row, warmups))
        guard = guard_result(guard_id, ledger)
        if guard["status"] != "ok" or guard["kind"] != row.get("phase") or guard["gpu_uuid"] != row.get("gpu_uuid"):
            errors.append("guard process identity or stage is invalid")
        if not provenance_ok(row):
            errors.append("target path/config/weight file metadata incomplete")
        return errors
    except Exception as exc:
        return [f"primary audit could not validate record: {type(exc).__name__}: {exc}"]


def decode_spans(rounds, requested_batch):
    """Span includes host gaps. Full-B is only the initial consecutive prefix."""
    indexes = [i for i, row in enumerate(rounds) if row.get("prefill") is False]
    if not indexes:
        return {"decode_span_s": None, "decode_span_tps": None, "decode_step_sum_s": None,
                "decode_tokens": None, "full_batch_prefix_span_s": None,
                "full_batch_prefix_tps": None, "interleaved_prefill": None}
    first, last = indexes[0], indexes[-1]
    segment = rounds[first:last + 1]
    decoded = [row for row in segment if row.get("prefill") is False]
    span = rounds[last]["end_s"] - rounds[first]["start_s"]
    step = sum(r["end_s"] - r["start_s"] for r in decoded)
    delivered = sum(r["delivered"] for r in decoded)
    prefix = []
    for row in segment:
        if row.get("prefill") is not False or row.get("active_b") != requested_batch:
            break
        prefix.append(row)
    prefix_span = prefix[-1]["end_s"] - prefix[0]["start_s"] if prefix else None
    prefix_tokens = sum(r["delivered"] for r in prefix)
    return {"decode_span_s": span, "decode_span_tps": delivered / span if span > 0 else None,
            "decode_step_sum_s": step, "decode_tokens": delivered,
            "full_batch_prefix_span_s": prefix_span,
            "full_batch_prefix_tps": prefix_tokens / prefix_span if prefix_span and prefix_span > 0 else None,
            "full_batch_prefix_tokens": prefix_tokens if prefix else None,
            "interleaved_prefill": any(r.get("prefill") is True for r in segment)}


def diff_counter(before, after, key):
    x, y = before.get(key), after.get(key)
    if type(x) in (int, float) and type(y) in (int, float) and math.isfinite(x) and math.isfinite(y) and y >= x:
        return y - x
    return None


def allocator_delta(before, after, key):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    x, y = before.get(key), after.get(key)
    return y - x if type(x) is int and type(y) is int and x >= 0 and y >= 0 else None


def telemetry_errors(row, warmups):
    errors = []
    if (type(row.get("target_kv_blocks")) is not int or row["target_kv_blocks"] <= 0 or
            type(row.get("draft_kv_blocks")) is not int or row["draft_kv_blocks"] < 0 or
            type(row.get("kv_block_size")) is not int or row["kv_block_size"] <= 0):
        errors.append("target or configured draft pool block metadata missing")
    graph = row.get("graph_delta")
    if not isinstance(graph, dict):
        return ["graph telemetry is missing"]
    common = ("verify_hits", "verify_exact_hits", "verify_padded_hits", "verify_fallbacks",
              "decode_graph_hits", "decode_graph_exact_hits", "decode_graph_padded_hits",
              "decode_graph_bucket_histogram", "decode_graph_fallbacks")
    draft = ("graph_hits", "graph_exact_hits", "graph_padded_hits", "graph_captures",
             "graph_runtime_captures", "graph_runtime_capture_s", "graph_runtime_shapes", "graph_fallbacks",
             "active_batch_histogram", "context_bucket_histogram", "head_graph_hits", "head_graph_captures",
             "head_runtime_captures", "head_runtime_capture_s", "head_runtime_batches",
             "input_prepare_calls", "input_prepare_wall_s")
    ar = ("n_graphed", "n_eager", "n_padded", "n_rounds", "n_batch_forwards", "n_catchup_forwards", "n_catchup_tokens")
    required = common + (draft if row.get("arm", "").startswith("dspark_") else ar if row.get("arm", "").startswith("ar_") else ())
    missing = [key for key in required if key not in graph]
    if missing:
        errors.append("graph telemetry fields missing: " + ", ".join(missing))
    mem = row.get("memory_after_run")
    memory_keys = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "named_tensors",
                   "unique_parameter_bytes", "allocator_segment_bytes_by_pool_id", "named_graph_pools", "engine_graph_mem_gb")
    if not isinstance(mem, dict) or any(key not in mem for key in memory_keys):
        errors.append("allocator, tensor, weight or graph memory telemetry missing")
    else:
        quantities = [mem.get(k) for k in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")]
        if (any(type(x) is not int or x < 0 for x in quantities)
                or quantities[1] < quantities[0] or quantities[2] < quantities[0] or quantities[3] < quantities[1]):
            errors.append("allocator byte telemetry is invalid")
        tensors = mem.get("named_tensors") if isinstance(mem.get("named_tensors"), dict) else {}
        needed = ["runner.kv_cache"]
        if row.get("arm", "").startswith("ar_"):
            needed.append("runner.draft_kv_cache")
        elif row.get("arm", "").startswith("dspark_"):
            needed.append("dspark.cache")
            if row.get("arm") == "dspark_ring":
                needed.append("dspark.scratch")
        if any(key not in tensors or not isinstance(tensors[key], dict) or
               type(tensors[key].get("bytes")) is not int or tensors[key]["bytes"] <= 0 or
               not isinstance(tensors[key].get("shape"), list) or not tensors[key].get("dtype") for key in needed):
            errors.append("target/draft KV or required scratch allocation missing")
        pools = mem.get("named_graph_pools") if isinstance(mem.get("named_graph_pools"), dict) else {}
        pool_keys = ["target_decode"]
        if row.get("spec_k", 0):
            pool_keys.append("target_verify")
        if row.get("arm", "").startswith("dspark_"):
            pool_keys.extend(("dspark_backbone", "dspark_head"))
        elif row.get("arm", "").startswith("ar_"):
            pool_keys.append("ar_draft_1")
        if any(key not in pools or not isinstance(pools[key], dict) or type(pools[key].get("reserved_bytes")) is not int or
               pools[key]["reserved_bytes"] < 0 for key in pool_keys):
            errors.append("named graph pool reserved-byte telemetry missing")
        weights = mem.get("unique_parameter_bytes")
        if not isinstance(weights, dict) or any(k not in weights for k in ("model", "draft_model", "dspark_model")):
            errors.append("model weight byte telemetry missing")
        elif (type(weights.get("model")) is not int or weights["model"] <= 0 or
              (row.get("arm", "").startswith("ar_") and (type(weights.get("draft_model")) is not int or weights["draft_model"] <= 0)) or
              (row.get("arm", "").startswith("dspark_") and (type(weights.get("dspark_model")) is not int or weights["dspark_model"] <= 0))):
            errors.append("model or draft weight bytes invalid")
        graph_mem = mem.get("engine_graph_mem_gb")
        if (not isinstance(graph_mem, dict) or "decode_graphs_gb" not in graph_mem or
                (row.get("spec_k", 0) and "verify_graphs_gb" not in graph_mem)):
            errors.append("target graph allocated-delta telemetry missing")
    warm = warmups.get(row.get("guard_id"), [])
    if len(warm) == 1:
        for phase in ("memory_after_init", "memory_after_prewarm", "memory_after_warmup_requests"):
            info = warm[0][2].get(phase)
            if not isinstance(info, dict) or any(k not in info for k in memory_keys):
                errors.append("initial, prewarm or warmup allocator telemetry missing")
                break
    if row.get("arm", "").startswith("dspark_") and type(row.get("window")) is int and row["window"] > 0:
        for field in ("window_stats_before", "window_stats_after"):
            stats = row.get(field)
            if not isinstance(stats, dict) or any(k not in stats for k in ("prefill_target_rows", "prefill_feature_rows",
                    "max_feature_tensor_bytes", "max_projection_tensor_bytes", "commit_calls", "commit_rows", "commit_host_enqueue_s")):
                errors.append("window feature/commit telemetry missing")
    return errors


def summarise_run(source, line, row, warmups, ledger, manifests):
    guard = guard_result(row.get("guard_id"), ledger)
    try:
        metrics = audit.compute_run(row)
        spans = decode_spans(row["rounds"], row["requested_batch"])
    except Exception as exc:
        metrics = {"valid": False, "errors": [f"malformed run: {type(exc).__name__}: {exc}"]}
        spans = {}
    if row.get("phase") == "capacity":
        errors = validate_capacity(row, guard, warmups, manifests)
    else:
        errors = validate_primary(source, line, row, warmups, ledger, manifests)
    if not metrics.get("valid"):
        errors.extend(x for x in metrics.get("errors", []) if x not in errors)
    errors.extend(telemetry_errors(row, warmups))
    if row.get("phase") in ("pilot", "formal") and spans.get("interleaved_prefill") is True:
        errors.append("prefill occurred inside the decode span; fixed-batch decode comparison is not pure")
    if type(row.get("repeat")) is not int or row["repeat"] < 0:
        errors.append("repeat must be a nonnegative integer")
    warm = warmups.get(row.get("guard_id"), [])
    warm_record = warm[0][2] if len(warm) == 1 else {}
    before = row.get("window_stats_before") if isinstance(row.get("window_stats_before"), dict) else {}
    after = row.get("window_stats_after") if isinstance(row.get("window_stats_after"), dict) else {}
    window_instrumented = row.get("arm", "").startswith("dspark_") and type(row.get("window")) is int and row["window"] > 0
    graph_delta = row.get("graph_delta") if isinstance(row.get("graph_delta"), dict) else {}
    values = {"full_request_tps": metrics.get("full_request_tps"),
              "decode_span_tps": spans.get("decode_span_tps"),
              "full_batch_prefix_tps": spans.get("full_batch_prefix_tps"),
              "acceptance_fraction": metrics.get("acceptance_fraction"),
              "tau_delivered_per_decode_sequence_round": metrics.get("tau_delivered_per_decode_round_per_sequence",
                                                                        metrics.get("tau_delivered_per_decode_sequence_round"))}
    for source_key, dest in (("ttft_s", "ttft"), ("tpot_s", "tpot"), ("delivery_batch_gap_s", "delivery_gap")):
        dist = metrics.get(source_key, {})
        for p in (50, 95, 99):
            values[f"{dest}_p{p}_s"] = dist.get(f"p{p}")
    requests = row.get("requests") if isinstance(row.get("requests"), list) else []
    outputs = [{"seq_id": r.get("seq_id"), "output_sha256": r.get("output_sha256"),
                "delivered_tokens": len(r.get("output_token_ids")) if isinstance(r.get("output_token_ids"), list) else None}
               for r in requests if isinstance(r, dict)]
    return {"source": source, "line": line, "guard": guard, "valid": not errors, "validation_errors": errors,
            "repeat": row.get("repeat"), "input_hashes": input_hashes(row),
            "input_metadata": row.get("input_metadata"), "output_hashes": outputs,
            "metrics": values, "request_distributions": {k: metrics.get(k) for k in ("ttft_s", "tpot_s", "delivery_batch_gap_s")},
            "time": {"full_wall_s": row.get("wall_s"), **spans,
                     "prefill_step_sum_s": metrics.get("prefill_s"),
                     "definition": "decode span is first decode start to last decode end, including intervening host gaps"},
            "tokens": {k: metrics.get(k) for k in ("output_tokens", "proposed_tokens", "accepted_prefix_tokens",
                                                       "tail_overproposal_sequence_rounds", "active_batch_rounds")},
            "host": {"draft_input_prepare_s": graph_delta.get("input_prepare_wall_s"),
                     "draft_input_prepare_calls": graph_delta.get("input_prepare_calls"),
                     "commit_enqueue_s": diff_counter(before, after, "commit_host_enqueue_s") if window_instrumented else None,
                     "commit_calls": diff_counter(before, after, "commit_calls") if window_instrumented else None,
                     "definition": "CPU-side input construction and commit enqueue; not GPU kernel durations"},
            "feature_rows": {"target_rows": diff_counter(before, after, "prefill_target_rows") if window_instrumented else None,
                             "selected_rows": diff_counter(before, after, "prefill_feature_rows") if window_instrumented else None,
                             "max_feature_tensor_bytes_process": after.get("max_feature_tensor_bytes") if window_instrumented else None,
                             "projection_tensor_bytes_formula_max_process": after.get("max_projection_tensor_bytes") if window_instrumented else None,
                             "note": "feature tensor bytes use numel*dtype; projection bytes are a shape formula, not allocator peaks; W0 uninstrumented"},
            "graph": {"runtime": row.get("graph_delta"), "target_decode": row.get("target_decode_graph"),
                      "prewarm_s": warm_record.get("prewarm_s"), "prewarm_shapes": warm_record.get("prewarm_shapes"),
                      "expected_shapes": warm_record.get("expected_prewarm_shapes"),
                      "initial_graph_shapes": warm_record.get("target_verify_graph_shapes"),
                      "warmup_graph_delta": warm_record.get("warmup_graph_delta"),
                      "prewarm_process_allocated_delta_bytes": allocator_delta(warm_record.get("memory_after_init"),
                          warm_record.get("memory_after_prewarm"), "allocated_bytes"),
                      "prewarm_process_reserved_delta_bytes": allocator_delta(warm_record.get("memory_after_init"),
                          warm_record.get("memory_after_prewarm"), "reserved_bytes"),
                      "prewarm_delta_note": "whole-process allocator difference across prewarm, not per-graph attribution"},
            "memory": {"after_init": warm_record.get("memory_after_init"),
                       "after_prewarm": warm_record.get("memory_after_prewarm"),
                       "after_warmup_requests": warm_record.get("memory_after_warmup_requests"),
                       "after_run": row.get("memory_after_run"),
                       "target_kv_blocks": row.get("target_kv_blocks"),
                       "configured_draft_blocks_raw": row.get("draft_kv_blocks"),
                       "window_stats_before": before or None, "window_stats_after": after or None,
                       "note": "allocator totals and named tensor/pool bytes overlap; configured draft blocks 0 can mean shared target-block count, not zero allocated KV"},
            "diagnostic_gpu_events_ms": row.get("gpu_event_ms") if row.get("diagnostic_instrumented") else None}


def scoped(row):
    return {k: row.get(k) for k in CELL_FIELDS}


def variant(row):
    value = {k: row.get(k) for k in audit.GROUP_FIELDS}
    value["input_hashes"] = input_hashes(row)
    return value


def common(row):
    value = {k: row.get(k) for k in COMMON_FIELDS}
    value["input_hashes"] = input_hashes(row)
    value["input_sources"] = [{"dataset": m.get("dataset"), "row": m.get("row")}
                              for m in row.get("input_metadata", []) if isinstance(m, dict)]
    return value


def aggregation(values):
    clean = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return {"values": values, "n_available": len(clean),
            "median": statistics.median(clean) if clean else None,
            "min": min(clean) if clean else None, "max": max(clean) if clean else None}


def make_planned(plans):
    result, issues = {}, []
    for name, plan in plans:
        if not isinstance(plan, dict):
            issues.append({"source": name, "error": "plan is not a JSON object"})
            continue
        phase = plan.get("phase")
        jobs = plan.get("jobs")
        if not isinstance(jobs, list):
            issues.append({"source": name, "error": "plan has no jobs list"})
            continue
        budget = plan.get("gpu_memory_utilization")
        if budget is None:
            recorded = [j.get("gpu_memory_utilization") for j in jobs if isinstance(j, dict)]
            budgets = set(recorded) if all(type(v) in (int, float) for v in recorded) else set()
            budget = next(iter(budgets)) if len(budgets) == 1 else None
        if (type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 1
                or (phase in ("pilot", "formal") and budget not in (0.70, 0.85))):
            issues.append({"source": name, "error": "plan does not declare a valid single GPU budget"})
        local_cells = set()
        for job in jobs:
            try:
                if not isinstance(job, dict):
                    raise TypeError("job is not an object")
                if job.get("phase") != phase:
                    issues.append({"source": name, "error": "job phase differs from plan phase"})
                if job.get("gpu_memory_utilization") != budget:
                    issues.append({"source": name, "error": "job GPU budget differs from plan"})
                if type(job.get("repeats")) is not int or job["repeats"] < 1:
                    issues.append({"source": name, "error": "job repeats must be a positive integer"})
                for panel in job["panels"]:
                    cell = {"phase": phase, "arm": job["arm"], "mode": job["mode"], "ctx": job["ctx"],
                            "requested_batch": job["batch"], "panel": panel, "window": job["window"],
                            "graph_mode": "graph", "gpu_memory_utilization": job["gpu_memory_utilization"]}
                    key = canon(cell)
                    local_cells.add(key)
                    if key in result:
                        issues.append({"source": name, "error": f"duplicate planned cell: {key}"})
                    else:
                        result[key] = {"scope": cell, "repeats": job.get("repeats"), "source": name,
                                       "expected_output_tokens": job.get("output_tokens", 256),
                                       "expected_max_num_batched_tokens": job.get("max_num_batched_tokens", job["batch"] * job["ctx"])}
            except (KeyError, TypeError) as exc:
                issues.append({"source": name, "error": f"malformed plan job: {exc}"})
        if "cells" in plan:
            if not isinstance(plan["cells"], list):
                issues.append({"source": name, "error": "plan cells is not a list"})
            else:
                listed = set()
                for cell in plan["cells"]:
                    try:
                        value = {"phase": phase, **{k: cell[k] for k in audit.CELL_FIELDS},
                                 "gpu_memory_utilization": cell.get("gpu_memory_utilization", budget)}
                        if value["gpu_memory_utilization"] != budget:
                            issues.append({"source": name, "error": "cell GPU budget differs from plan"})
                        key = canon(value)
                        if key in listed:
                            issues.append({"source": name, "error": "duplicate plan cell"})
                        listed.add(key)
                    except (KeyError, TypeError, AttributeError) as exc:
                        issues.append({"source": name, "error": f"malformed plan cell: {exc}"})
                if listed != local_cells:
                    issues.append({"source": name, "error": "plan cells and jobs describe different workloads"})
        if phase == "formal" and plan.get("matrix_scope") == "full minimum fixed-batch matrix":
            w = plan.get("window_tokens")
            required = set()
            for ctx in (1024, 4096):
                for b in (1, 2, 4, 8, 16):
                    for panel in ("code", "math", "general"):
                        for arm, mode in (("nospec", "full"), ("ar_full", "full"), ("ar_window", "block_aligned"),
                                          ("dspark_full", "full"), ("dspark_ring", "ring")):
                            required.add(canon({"phase": "formal", "arm": arm, "mode": mode, "ctx": ctx,
                                                "requested_batch": b, "panel": panel,
                                                "window": w if arm in ("ar_window", "dspark_ring") else 0,
                                                "graph_mode": "graph", "gpu_memory_utilization": budget}))
            if (type(w) is not int or w <= 0 or w % 256 or local_cells != required
                    or any(not isinstance(job, dict) or type(job.get("repeats")) is not int or job["repeats"] < 3 for job in jobs)):
                issues.append({"source": name, "error": "claimed full matrix lacks required cells, settings or repeats"})
    return result, issues


def failure_matches(scope, row):
    if (row.get("phase") not in ("pilot", "formal", "capacity") or
            any(row.get(k) is None for k in ("arm", "mode", "ctx", "requested_batch", "window", "gpu_memory_utilization"))):
        return False
    for key in ("phase", "arm", "mode", "ctx", "requested_batch", "window", "graph_mode", "gpu_memory_utilization"):
        if row.get(key) is not None and row.get(key) != scope.get(key):
            return False
    if row.get("panel") is not None:
        return row["panel"] == scope["panel"]
    return scope["panel"] in row.get("panels", [])


def build_report(records, ledger_records, manifests, plans, min_repeats=3, input_errors=None):
    if min_repeats < 3:
        raise ValueError("comparisons require at least three repeats")
    ledger, ledger_errors = ledger_index(ledger_records)
    warmups, observations, failures, diagnostic_events = defaultdict(list), [], [], []
    for source, line, row in records:
        kind = row.get("type")
        if kind == "warmup":
            warmups[row.get("guard_id")].append((source, line, row))
            if row.get("diagnostic_instrumented") is True:
                diagnostic_events.append({"source": source, "line": line, "type": kind,
                                          "guard": guard_result(row.get("guard_id"), ledger), "record": row})
        elif kind == "failure":
            failures.append({"source": source, "line": line, "record": row,
                             "guard": guard_result(row.get("guard_id"), ledger)})
        elif kind == "run" and row.get("phase") in ("pilot", "formal", "capacity") and row.get("diagnostic_instrumented") is False:
            observations.append((source, line, row))
        elif kind is not None:
            # Retain diagnostic probes and failures without promoting timings.
            diagnostic_events.append({"source": source, "line": line, "type": kind,
                                      "guard": guard_result(row.get("guard_id"), ledger), "record": row})
    planned, plan_issues = make_planned(plans)
    grouped = defaultdict(lambda: defaultdict(list))
    for source, line, row in observations:
        cell_key, variant_key = canon(scoped(row)), canon(variant(row))
        summary = summarise_run(source, line, row, warmups, ledger, manifests)
        grouped[cell_key][variant_key].append({"summary": summary, "row": row})
    cells = []
    for key in sorted(set(planned) | set(grouped)):
        plan = planned.get(key)
        variants = grouped.get(key, {})
        scope = plan["scope"] if plan else json.loads(key)
        associated = [f for f in failures if failure_matches(scope, f["record"])]
        variant_rows = []
        for variant_key, items in sorted(variants.items()):
            items = sorted(items, key=lambda x: (type(x["summary"]["repeat"]) is not int,
                                                 x["summary"]["repeat"] if type(x["summary"]["repeat"]) is int else 0,
                                                 x["summary"]["source"], x["summary"]["line"]))
            runs = [x["summary"] for x in items]
            repeats = [r["repeat"] for r in runs]
            expected_repeats = plan.get("repeats") if plan else None
            repeat_valid = all(type(r) is int and r >= 0 for r in repeats)
            complete = (repeat_valid and type(expected_repeats) is int and expected_repeats > 0
                        and len(set(repeats)) == expected_repeats
                        and set(repeats) == set(range(expected_repeats)) and len(repeats) == expected_repeats)
            variant_rows.append({"configuration": json.loads(variant_key), "runs": runs,
                                 "repeats": repeats, "repeat_set_complete": complete,
                                 "metrics": {m: aggregation([r["metrics"].get(m) for r in runs]) for m in METRICS},
                                 "all_valid": all(r["valid"] for r in runs)})
        if associated:
            status = "failed_attempt_present"
        elif not variant_rows:
            status = "missing"
        elif len(variant_rows) > 1:
            status = "inconsistent_conditions"
        elif not variant_rows[0]["all_valid"]:
            status = "unverified_or_invalid"
        elif plan and any(item["row"].get("output_tokens") != plan["expected_output_tokens"] or
                          item["row"].get("max_num_batched_tokens") != plan["expected_max_num_batched_tokens"]
                          for items in variants.values() for item in items):
            status = "plan_condition_mismatch"
        elif (all(type(r) is int for r in variant_rows[0]["repeats"]) and
              len(set(variant_rows[0]["repeats"])) != len(variant_rows[0]["repeats"])):
            status = "duplicate_repeat"
        elif not plan:
            status = "unplanned"
        elif not variant_rows[0]["repeat_set_complete"]:
            status = "partial"
        else:
            status = "complete"
        cells.append({"scope": scope, "status": status, "planned_repeats": plan.get("repeats") if plan else None,
                      "expected_output_tokens": plan.get("expected_output_tokens") if plan else None,
                      "expected_max_num_batched_tokens": plan.get("expected_max_num_batched_tokens") if plan else None,
                      "plan_source": plan.get("source") if plan else None, "variants": variant_rows,
                      "failures": [{"source": f["source"], "line": f["line"], "error": f["record"].get("error"),
                                    "message": f["record"].get("message"),
                                    "failure_phase": f["record"].get("failure_phase"),
                                    "request_subphase": f["record"].get("request_subphase"),
                                    "guard": f["guard"]} for f in associated]})
    ledger_summary = [guard_result(ident, ledger) for ident in sorted(ledger)]
    completed_s = [r["elapsed_s"] for r in ledger_summary if type(r["elapsed_s"]) in (int, float)
                   and math.isfinite(r["elapsed_s"]) and r["elapsed_s"] >= 0]
    unfinished = [r["guard_id"] for r in ledger_summary if r["status"] in ("unfinished", "missing_start", "invalid") or r["identity_errors"]]
    smoke_s = [r["elapsed_s"] for r in ledger_summary if r["kind"] == "smoke"
               and type(r["elapsed_s"]) in (int, float) and math.isfinite(r["elapsed_s"]) and r["elapsed_s"] >= 0]
    if sum(completed_s) > 14400:
        ledger_errors.append({"error": "completed GPU stages exceeded the 14400 second total budget", "elapsed_s": sum(completed_s)})
    if sum(smoke_s) > 1800:
        ledger_errors.append({"error": "completed smoke stages exceeded the 1800 second smoke budget", "elapsed_s": sum(smoke_s)})
    comparison_errors = list(plan_issues) + list(input_errors or []) + list(ledger_errors)
    if unfinished:
        comparison_errors.append({"error": "ledger has unfinished or invalid identities", "guard_ids": unfinished})
    comparisons = compare_cells(cells, grouped, min_repeats, comparison_errors)
    capacity_comparisons = compare_capacity(cells, grouped, comparison_errors)
    curves = build_curves(cells, comparison_errors)
    full_formal_plan = any(isinstance(p, dict) and p.get("phase") == "formal" and p.get("matrix_scope") == "full minimum fixed-batch matrix"
                           for _, p in plans)
    plan_summaries = []
    for name, p in plans:
        p = p if isinstance(p, dict) else {}
        related = [c for c in cells if c.get("plan_source") == name]
        budget = p.get("gpu_memory_utilization")
        if budget is None and related:
            budget = related[0]["scope"].get("gpu_memory_utilization")
        unexpected = [c for c in cells if c.get("plan_source") is None and c["scope"].get("phase") == p.get("phase")
                      and c["scope"].get("gpu_memory_utilization") == budget]
        coverage = {"planned": len(related), "complete": sum(c["status"] == "complete" for c in related),
                    "missing": sum(c["status"] == "missing" for c in related),
                    "partial_or_failed": sum(c["status"] not in ("complete", "missing") for c in related),
                    "unexpected": len(unexpected)}
        complete_full = bool(p.get("phase") == "formal" and p.get("matrix_scope") == "full minimum fixed-batch matrix"
                             and related and all(c["status"] == "complete" for c in related)
                             and not unexpected and not comparison_errors)
        plan_summaries.append({"source": name, "phase": p.get("phase"), "gpu_memory_utilization": budget,
                               "matrix_scope": p.get("matrix_scope"), "graph_policy": p.get("graph_policy"),
                               "coverage": coverage, "full_matrix_complete": complete_full})
    return {"version": 1, "definitions": {
                "full_request_tps": "real delivered output tokens / wall from enqueue to all requests finished",
                "decode_span_tps": "decode-delivered tokens / (last decode end - first decode start), including host gaps",
                "full_batch_prefix_tps": "tokens / span of initial consecutive decode rounds with requested B",
                "delivery_gap": "one interval between real delivery batches, not SSE and no per-token interpolation",
                "tau": "real delivered tokens per decode sequence-round; separate from accepted/proposed fraction",
                "diagnostic": "diagnostic instruments and forced acceptance are never primary throughput",
                "weights": "model provenance uses path, config hash, file names and sizes; not a full weight-content hash",
                "memory": "allocator totals and named tensors/graph pools may overlap; do not sum them"},
            "plans": plan_summaries,
            "plan_issues": plan_issues, "input_errors": input_errors or [], "ledger_errors": ledger_errors,
            "ledger": {"stages": ledger_summary, "completed_stage_elapsed_s": sum(completed_s) if completed_s else None,
                       "completed_smoke_elapsed_s": sum(smoke_s) if smoke_s else None,
                       "total_budget_s": 14400, "smoke_budget_s": 1800,
                       "unfinished_guard_ids": unfinished,
                       "total_is_complete": bool(ledger_summary) and not unfinished and not ledger_errors},
            "cells": cells, "comparisons": comparisons, "capacity_comparisons": capacity_comparisons, "curves": curves,
            "failures": failures, "diagnostic_events": diagnostic_events,
            "summary": {"planned_cells": len(planned), "observed_run_records": len(observations),
                        "missing_cells": sum(c["status"] == "missing" for c in cells),
                        "failed_or_invalid_cells": sum(c["status"] in ("failed_attempt_present", "unverified_or_invalid", "inconsistent_conditions", "duplicate_repeat", "plan_condition_mismatch") for c in cells),
                        "complete_cells": sum(c["status"] == "complete" for c in cells),
                        "diagnostics": len(diagnostic_events), "failures": len(failures),
                        "planned_coverage_available": bool(plans), "full_formal_plan_declared": full_formal_plan,
                        "primary_summary_is_complete": any(p["full_matrix_complete"] for p in plan_summaries)}}


def compare_cells(cells, grouped, minimum, global_errors=()):
    lookup = defaultdict(list)
    for cell in cells:
        s = cell["scope"]
        if (s.get("phase") in ("pilot", "formal") and type(s.get("ctx")) is int
                and type(s.get("requested_batch")) is int and isinstance(s.get("panel"), str)
                and isinstance(s.get("arm"), str) and type(s.get("gpu_memory_utilization")) in (int, float)):
            lookup[(s.get("phase"), s.get("ctx"), s.get("requested_batch"), s.get("panel"),
                    s.get("gpu_memory_utilization"), s.get("arm"))].append(cell)
    results = []
    for cell in cells:
        s = cell["scope"]
        candidate = s.get("arm")
        if (candidate not in ("dspark_ring", "dspark_full", "ar_window") or s.get("phase") not in ("pilot", "formal")
                or type(s.get("ctx")) is not int or type(s.get("requested_batch")) is not int
                or not isinstance(s.get("panel"), str) or type(s.get("gpu_memory_utilization")) not in (int, float)):
            continue
        baselines = (("nospec", "ar_full", "ar_window", "dspark_full", "dspark_window_full") if candidate == "dspark_ring"
                     else ("nospec", "ar_full", "ar_window") if candidate == "dspark_full" else ("ar_full",))
        for baseline in baselines:
            base = lookup.get((s["phase"], s["ctx"], s["requested_batch"], s["panel"],
                               s["gpu_memory_utilization"], baseline), [])
            if candidate == "dspark_ring" and baseline in ("ar_window", "dspark_window_full"):
                base = [x for x in base if x["scope"].get("window") == s.get("window")]
            result = {"scope": s, "candidate_arm": candidate, "baseline_arm": baseline, "status": None, "reasons": [],
                      "full_request_ratio": None, "output_hash_comparison": [],
                      "interpretation": "T=1 output hash differences are observations, not correctness failures; AR and DSpark use different k and window semantics"}
            if global_errors:
                result["reasons"].append("input, plan or ledger contains an unresolved error")
            if cell["status"] != "complete":
                result["reasons"].append(f"candidate cell is {cell['status']}")
            if len(base) != 1:
                result["reasons"].append("baseline absent" if not base else "baseline ambiguous")
            elif base[0]["status"] != "complete":
                result["reasons"].append(f"baseline cell is {base[0]['status']}")
            if result["reasons"]:
                result["status"] = "not_comparable"
                results.append(result)
                continue
            left_rows = next(iter(grouped[canon(s)].values()))
            right_rows = next(iter(grouped[canon(base[0]["scope"])].values()))
            left = {x["summary"]["repeat"]: x for x in left_rows}
            right = {x["summary"]["repeat"]: x for x in right_rows}
            if len(left) < minimum or len(right) < minimum or set(left) != set(right):
                result["status"] = "not_comparable"
                result["reasons"].append(f"fewer than {minimum} matched repeat IDs")
                results.append(result)
                continue
            if any(canon(common(left[k]["row"])) != canon(common(right[k]["row"])) for k in left):
                result["status"] = "not_comparable"
                result["reasons"].append("input, source, GPU, target, workload or runtime policy mismatch")
                results.append(result)
                continue
            same_family = ((candidate.startswith("dspark_") and baseline.startswith("dspark_")) or
                           (candidate.startswith("ar_") and baseline.startswith("ar_")))
            if same_family and any(
                    canon(left[k]["row"].get("draft_provenance")) != canon(right[k]["row"].get("draft_provenance"))
                    for k in left):
                result["status"] = "not_comparable"
                result["reasons"].append("same-family draft provenance differs")
                results.append(result)
                continue
            ratios = []
            for repeat in sorted(left):
                a, b = left[repeat]["summary"], right[repeat]["summary"]
                av, bv = a["metrics"]["full_request_tps"], b["metrics"]["full_request_tps"]
                if av is None or bv is None or bv <= 0:
                    result["reasons"].append(f"invalid throughput at repeat {repeat}")
                    break
                ratios.append(av / bv)
                aa = [x["output_sha256"] for x in a["output_hashes"]]
                bb = [x["output_sha256"] for x in b["output_hashes"]]
                if len(aa) != len(bb):
                    result["reasons"].append(f"output count mismatch at repeat {repeat}")
                    break
                result["output_hash_comparison"].append({"repeat": repeat, "requests": len(aa),
                                                         "matching": sum(x == y for x, y in zip(aa, bb)),
                                                         "different": sum(x != y for x, y in zip(aa, bb)),
                                                         "same_per_request": [x == y for x, y in zip(aa, bb)]})
            if result["reasons"]:
                result["status"] = "not_comparable"
                result["output_hash_comparison"] = []
            else:
                result["status"] = "matched"
                result["repeat_ids"] = sorted(left)
                result["full_request_ratio"] = aggregation(ratios)
            results.append(result)
    return results


def compare_capacity(cells, grouped, global_errors=()):
    """Pair full/ring capacity only when both runs and their metadata match."""
    result = []
    for ring in cells:
        s = ring["scope"]
        if s.get("phase") != "capacity" or s.get("arm") != "dspark_ring":
            continue
        matches = [c for c in cells if c["scope"].get("phase") == "capacity"
                   and c["scope"].get("arm") == "dspark_full"
                   and all(c["scope"].get(k) == s.get(k) for k in
                           ("ctx", "requested_batch", "panel", "gpu_memory_utilization", "graph_mode"))]
        pair = {"scope": s, "status": "not_comparable", "reasons": [], "observations": [],
                "interpretation": "fixed total budget; different target-pool block counts are not a same-target-pool measurement"}
        if global_errors:
            pair["reasons"].append("input, plan or ledger contains an unresolved error")
        if ring["status"] != "complete":
            pair["reasons"].append(f"ring cell is {ring['status']}")
        if len(matches) != 1:
            pair["reasons"].append("full-cache baseline absent" if not matches else "full-cache baseline ambiguous")
        elif matches[0]["status"] != "complete":
            pair["reasons"].append(f"full-cache cell is {matches[0]['status']}")
        if pair["reasons"]:
            result.append(pair)
            continue
        left_rows = next(iter(grouped[canon(s)].values()))
        right_rows = next(iter(grouped[canon(matches[0]["scope"])].values()))
        left = {x["summary"]["repeat"]: x for x in left_rows}
        right = {x["summary"]["repeat"]: x for x in right_rows}
        if not left or set(left) != set(right):
            pair["reasons"].append("capacity repeat IDs do not match")
        elif any(canon(common(left[k]["row"])) != canon(common(right[k]["row"])) or
                 canon(left[k]["row"].get("draft_provenance")) != canon(right[k]["row"].get("draft_provenance"))
                 for k in left):
            pair["reasons"].append("capacity input, source, GPU, model, budget or policy mismatch")
        if pair["reasons"]:
            result.append(pair)
            continue
        for repeat in sorted(left):
            a, b = left[repeat]["summary"], right[repeat]["summary"]
            ai, bi = a["memory"], b["memory"]
            blocks_a, blocks_b = ai.get("target_kv_blocks"), bi.get("target_kv_blocks")
            pair["observations"].append({"repeat": repeat, "ring_memory": ai, "full_memory": bi,
                                         "target_pool_blocks_equal": (blocks_a == blocks_b
                                             if type(blocks_a) is int and type(blocks_b) is int else None),
                                         "ring_full_wall_s": a["time"].get("full_wall_s"),
                                         "full_full_wall_s": b["time"].get("full_wall_s")})
        pair["status"] = "matched"
        result.append(pair)
    return result


def build_curves(cells, global_errors=()):
    groups = defaultdict(dict)
    for cell in cells:
        s = cell["scope"]
        if s.get("phase") not in ("pilot", "formal") or type(s.get("requested_batch")) is not int:
            continue
        key = canon({k: s.get(k) for k in ("phase", "arm", "mode", "ctx", "panel", "window", "graph_mode", "gpu_memory_utilization")})
        groups[key][s.get("requested_batch")] = cell
    curves = []
    for key, by_batch in sorted(groups.items()):
        points, segments, current = [], [], []
        previous_lineage, previous_inputs = None, None
        for batch in (1, 2, 4, 8, 16):
            cell = by_batch.get(batch)
            point = {"batch": batch, "status": cell["status"] if cell else "not_planned",
                     "median_full_request_tps": None, "repeat_values": None}
            if cell and global_errors:
                point["status"] = "blocked_global_error"
            lineage, hashes = None, None
            if cell and not global_errors and cell["status"] == "complete" and len(cell["variants"]) == 1:
                group = cell["variants"][0]
                stats = group["metrics"]["full_request_tps"]
                if stats["n_available"] >= 3 and len(stats["values"]) == stats["n_available"]:
                    point["median_full_request_tps"] = stats["median"]
                    point["repeat_values"] = stats["values"]
                    config = group["configuration"]
                    policy = config.get("graph_policy")
                    if isinstance(policy, dict):
                        policy = {k: v for k, v in policy.items() if k != "buckets"}
                    lineage = canon({k: config.get(k) for k in ("phase", "engine_commit", "engine_dirty", "gpu_uuid",
                                    "target_provenance", "target_config_sha256", "draft_provenance", "draft_config_sha256",
                                    "spec_k", "seed", "temperature", "output_tokens", "gpu_memory_utilization",
                                    "prefix_lookup", "mixed_overlap", "ignore_eos") } | {"graph_policy": policy})
                    hashes = config.get("input_hashes")
            connectable = (point["median_full_request_tps"] is not None and isinstance(hashes, list)
                           and (not current or (lineage == previous_lineage and isinstance(previous_inputs, list)
                                                and hashes[:len(previous_inputs)] == previous_inputs)))
            if not connectable and current:
                segments.append(current)
                current = []
            if point["median_full_request_tps"] is not None and isinstance(hashes, list):
                current.append(batch)
                previous_lineage, previous_inputs = lineage, hashes
            else:
                previous_lineage = previous_inputs = None
            points.append(point)
        if current:
            segments.append(current)
        curves.append({"configuration": json.loads(key), "points": points, "segments": segments,
                       "note": "null points and changed source/target/GPU or mismatched prompt prefixes break the line"})
    return curves


def fmt(value, digits=3):
    if value is None:
        return "未采集"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def values_text(metric):
    if not metric:
        return "未采集"
    values = ", ".join(fmt(v) for v in metric.get("values", []))
    return f"[{values}]；中位 {fmt(metric.get('median'))}；范围 {fmt(metric.get('min'))}–{fmt(metric.get('max'))}"


def metric_triplet(run, name):
    d = run.get("request_distributions", {}).get(name)
    if not isinstance(d, dict):
        return "未采集"
    return "/".join(fmt(d.get(f"p{p}"), 6) for p in (50, 95, 99)) + f" (n={d.get('n')})"


def compact_json(value):
    if value is None:
        return "未采集"
    return canon(value).replace("|", "\\|")


def markdown(report):
    lines = ["# DSpark 实验报告", "",
             "本报告由 raw 逐请求/逐轮和守护台账离线生成；诊断和 capacity 单列。缺测不是 0，失败不会被过滤。",
             "这是当前 nano-vLLM 分支内的模型与缓存策略对照，不是 vLLM、SGLang 或其他推理引擎之间的横向测试。DSpark 滑窗实验关闭 target prefix lookup，在线 mixed 流量不受支持；结果不外推到启用前缀复用或在线混批。",
             "本报告不将历史 prefill、量化、kernel 或混批数字归入本轮 DSpark 结果；matched 对照只说明其列出的模型、负载、批量和条件，不能单凭这一报告推导普遍加速或任务质量。AR k=6、DSpark k=7；AR 块对齐窗口与 DSpark 最近 W 个历史 token 语义不同。W0 为原路径对照，不能将 W0 结果归因为新滑窗工程收益。",
             "",
             "完整请求吞吐为真实交付 tokens / 入队至全部完成墙钟；decode span 是首个 decode 起点至末个 decode 终点，包含轮间 host 间隙。满 B 前缀只含最初连续满批轮。TTFT、TPOT 和交付批次间隔为每次运行内的 P50/P95/P99；批间隔不等于 SSE，也不插值成逐 token 间隔。",
             "",
             f"计划单元 {report['summary']['planned_cells']}，观察到 {report['summary']['observed_run_records']} 次主/容量运行，完整 {report['summary']['complete_cells']}，缺测 {report['summary']['missing_cells']}，失败或无效 {report['summary']['failed_or_invalid_cells']}。",
             "若未提供完整计划，覆盖范围未知；少于三次的单元只展示观察值，不计算置信区间或作为简历对照。", "",
             ]
    fill_4k = [m.get("wiki_fraction") for cell in report["cells"] if cell["scope"].get("ctx") == 4096
               for group in cell["variants"] for run in group["runs"]
               if run.get("valid") is True
               for m in (run.get("input_metadata") if isinstance(run.get("input_metadata"), list) else [])
               if isinstance(m, dict) and type(m.get("wiki_fraction")) in (int, float)
               and math.isfinite(m["wiki_fraction"]) and 0 <= m["wiki_fraction"] <= 1]
    if fill_4k:
        boundary = "，已记录请求均超过95%" if min(fill_4k) > 0.95 else ""
        lines.extend([f"已核验的4K请求 Wiki 补齐比例为 {min(fill_4k):.2%}–{max(fill_4k):.2%}{boundary}；这是合成固定长度吞吐负载，不是自然4K任务或任务质量测量。", ""])
    else:
        lines.extend(["4K 输入的填充比例尚无已核验记录；该补齐面板是合成吞吐负载，不是自然4K任务或任务质量测量。", ""])
    lines += [
             "## 各预算计划覆盖", "",
             "| 计划 | phase | GPU预算 | 完整/计划 | 缺测 | 部分或失败 | 完整正式矩阵 |",
             "|---|---|---:|---|---:|---:|---|"]
    for plan in report["plans"]:
        c = plan["coverage"]
        lines.append("| " + " | ".join(fmt(x) for x in
                     (plan["source"], plan["phase"], plan["gpu_memory_utilization"],
                      f"{c['complete']}/{c['planned']}", c["missing"], c["partial_or_failed"], plan["full_matrix_complete"])) + " |")
    if not report["plans"]:
        lines.append("| 未提供计划 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未知 |")
    lines += ["",
             "## 单元与吞吐", "",
             "| phase | 预算 | ctx | B | panel | arm | W | 状态 | repeat | 完整请求 tokens/s：全部值、中位、范围 | decode span tokens/s：全部值、中位、范围 | 满B前缀 tokens/s：全部值、中位、范围 |",
             "|---|---:|---:|---:|---|---|---:|---|---|---|---|---|"]
    for cell in report["cells"]:
        s = cell["scope"]
        groups = cell["variants"] or [None]
        for group in groups:
            metrics = group["metrics"] if group else {}
            repeats = group["repeats"] if group else []
            lines.append("| " + " | ".join(fmt(x) for x in (s.get("phase"), s.get("gpu_memory_utilization"), s.get("ctx"), s.get("requested_batch"),
                         s.get("panel"), s.get("arm"), s.get("window"), cell["status"], repeats,
                         values_text(metrics.get("full_request_tps")), values_text(metrics.get("decode_span_tps")),
                         values_text(metrics.get("full_batch_prefix_tps")))) + " |")
    lines += ["", "## 条件与负载边界", "",
              "target/draft 身份以路径、配置哈希和权重文件名/大小记录；不表示本轮重新哈希权重内容。填充比例逐请求列出，实际任务截断数按 manifest 记录。", "",
              "invalid 或未验证的原始条件仍列出，明确标注状态；它们不进入上面的已核验填充范围。", "",
              "| phase/ctx/B/panel/arm/W | 审计状态 | k / 输出 / T | GPU / 源码提交 | target / draft 路径 | 图策略 / prefix / 显存预算 | 逐次 Wiki 比例 / 任务截断数 |",
              "|---|---|---|---|---|---|---|"]
    for cell in report["cells"]:
        s = cell["scope"]
        for group in cell["variants"]:
            cfg = group["configuration"]
            input_details = []
            for run in group["runs"]:
                rows = run.get("input_metadata")
                metadata_ok = isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
                input_details.append({"repeat": run.get("repeat"), "valid": run.get("valid"),
                                      "wiki_fractions": [r.get("wiki_fraction") for r in rows] if metadata_ok else None,
                                      "task_truncated": sum(r.get("task_truncated_from_left") is True for r in rows) if metadata_ok else None})
            target = cfg.get("target_provenance") if isinstance(cfg.get("target_provenance"), dict) else {}
            draft = cfg.get("draft_provenance") if isinstance(cfg.get("draft_provenance"), dict) else {}
            label = f"{s.get('phase')}/{s.get('ctx')}/{s.get('requested_batch')}/{s.get('panel')}/{s.get('arm')}/{s.get('window')}"
            data = (label, cell["status"], f"{fmt(cfg.get('spec_k'))} / {fmt(cfg.get('output_tokens'))} / {fmt(cfg.get('temperature'))}",
                    f"{fmt(cfg.get('gpu_uuid'))} / {fmt(cfg.get('engine_commit'))}",
                    f"{fmt(target.get('path'))} / {fmt(draft.get('path'))}",
                    f"{compact_json(cfg.get('graph_policy'))} / {fmt(cfg.get('prefix_lookup'))} / {fmt(cfg.get('gpu_memory_utilization'))}",
                    compact_json(input_details))
            lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in data) + " |")
    lines += ["", "## 逐次延迟与内存", "",
              "单位：延迟秒；缓存和 allocator 字节为采集值，具名图池 reserved 可能共享；模型权重文件大小不能证明权重内容一致。详情和全量图/窗口统计见 JSON。", "",
              "| phase/预算/ctx/B/panel/arm/repeat | TTFT P50/P95/P99 | TPOT P50/P95/P99 | 真实交付批间隔 P50/P95/P99 | accepted/proposed / tau | KV / scratch bytes | peak allocated / reserved bytes | graph allocated GiB / graph pool reserved bytes / 预热全进程 allocΔ/resΔ bytes | graph 命中/回退 / 预热秒 | feature目标行/选中行 | host prepare / commit enqueue 秒 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in report["cells"]:
        s = cell["scope"]
        for group in cell["variants"]:
            for run in group["runs"]:
                value = run["memory"].get("after_run")
                mem = value if isinstance(value, dict) else {}
                tensors = mem.get("named_tensors") if isinstance(mem.get("named_tensors"), dict) else {}
                named = {k: v.get("bytes") for k, v in tensors.items()
                         if isinstance(v, dict) and ("cache" in k or "scratch" in k)}
                graph_pools = mem.get("named_graph_pools") if isinstance(mem.get("named_graph_pools"), dict) else {}
                pools = {k: v.get("reserved_bytes") for k, v in graph_pools.items()
                         if isinstance(v, dict)}
                label = f"{s.get('phase')}/{s.get('gpu_memory_utilization')}/{s.get('ctx')}/{s.get('requested_batch')}/{s.get('panel')}/{s.get('arm')}/{run.get('repeat')}"
                row = (label, metric_triplet(run, "ttft_s"), metric_triplet(run, "tpot_s"),
                       metric_triplet(run, "delivery_batch_gap_s"),
                       f"{fmt(run['tokens'].get('accepted_prefix_tokens'))}/{fmt(run['tokens'].get('proposed_tokens'))} / {fmt(run['metrics'].get('tau_delivered_per_decode_sequence_round'))}",
                       compact_json(named) if named else "未采集",
                       f"{fmt(mem.get('peak_allocated_bytes'))} / {fmt(mem.get('peak_reserved_bytes'))}",
                       f"{compact_json(mem.get('engine_graph_mem_gb'))} / {compact_json(pools) if pools else '未采集'} / {fmt(run['graph'].get('prewarm_process_allocated_delta_bytes'))}/{fmt(run['graph'].get('prewarm_process_reserved_delta_bytes'))}",
                       f"{compact_json(run['graph'].get('target_decode'))} / {compact_json(run['graph'].get('runtime'))} / {fmt(run['graph'].get('prewarm_s'), 6)}",
                       f"{fmt(run['feature_rows'].get('target_rows'))} / {fmt(run['feature_rows'].get('selected_rows'))}",
                       f"{fmt(run['host'].get('draft_input_prepare_s'), 6)} / {fmt(run['host'].get('commit_enqueue_s'), 6)}")
                lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    lines += ["", "## 匹配对照", "",
              "仅相同输入、代码提交、GPU、target 元数据和运行条件且至少三次配对时计算比值；输出哈希差异本身不是 T=1 正确性判断。", "",
              "| 预算/ctx/B/panel/W | 候选 / 对照 | 状态 | 候选/对照吞吐比：全部值、中位、范围 | 每次输出哈希不同的请求数 |",
              "|---|---|---|---|---|"]
    for c in report["comparisons"]:
        s = c["scope"]
        diffs = [x["different"] for x in c["output_hash_comparison"]]
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in
                     (f"{s.get('gpu_memory_utilization')}/{s.get('ctx')}/{s.get('requested_batch')}/{s.get('panel')}/{s.get('window')}",
                      f"{c['candidate_arm']} / {c['baseline_arm']}", c["status"] + (": " + "; ".join(c["reasons"]) if c["reasons"] else ""),
                      values_text(c["full_request_ratio"]), str(diffs) if diffs else "未采集")) + " |")
    lines += ["", "## 容量对照（独立口径）", "",
              "| ctx/B/panel/W/显存预算 | full vs ring | 相同 target KV 块数 |",
              "|---|---|---|"]
    for c in report["capacity_comparisons"]:
        s = c["scope"]
        same = [x["target_pool_blocks_equal"] for x in c["observations"]]
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in
                     (f"{s.get('ctx')}/{s.get('requested_batch')}/{s.get('panel')}/{s.get('window')}/{s.get('gpu_memory_utilization')}",
                      c["status"] + (": " + "; ".join(c["reasons"]) if c["reasons"] else ""),
                      str(same) if same else "未采集")) + " |")
    if not report["capacity_comparisons"]:
        lines.append("| 未提供容量计划或 ring 记录 | 未采集 | 未采集 |")
    lines += ["", "## 失败、缺测和诊断", ""]
    if report["failures"]:
        for failure in report["failures"]:
            row = failure["record"]
            lines.append(f"- {failure['source']}:{failure['line']}：{fmt(row.get('phase'))}/{fmt(row.get('arm'))}，"
                         f"阶段 {fmt(row.get('failure_phase'))}/{fmt(row.get('request_subphase'))}，"
                         f"{fmt(row.get('error'))}: {fmt(row.get('message'))}。")
    else:
        lines.append("- 输入中没有失败记录；未输入的任务状态未知。")
    for stage in report["ledger"]["stages"]:
        if stage["status"] != "ok":
            lines.append(f"- 台账 {fmt(stage['stage'])}: {stage['status']}，outcome={fmt(stage.get('outcome'))}，"
                         f"elapsed={fmt(stage.get('elapsed_s'))} 秒。")
    for issue in report["input_errors"] + report["plan_issues"] + report["ledger_errors"]:
        lines.append(f"- 输入/计划/台账问题：{compact_json(issue)}")
    lines += ["", "| 诊断来源 | 类型/B/策略 | 记录的 KL / KV RMS / 门槛 | 合同结果 |",
              "|---|---|---|---|"]
    relevant = {"target_probe", "target_verify_probe", "target_decode_probe", "draft_probe",
                "padding_contract", "exact_owner_contract", "fixed_replay"}
    shown = 0
    for event in report["diagnostic_events"]:
        if event["type"] not in relevant:
            continue
        row = event["record"]
        numeric = {k: row[k] for k in ("eager_exact_kl", "eager_padded_kl", "target_logit_max_kl",
                   "q_eager_exact_kl", "q_eager_padded_kl", "target_kv_eager_exact_rms",
                   "target_kv_eager_padded_rms", "scratch_eager_exact_rms", "scratch_eager_padded_rms",
                   "target_kv_relative_rms", "dspark_kv_or_scratch_relative_rms",
                   "q_and_target_kl_threshold", "kv_relative_rms_threshold") if k in row}
        identity = f"{event['type']}/{fmt(row.get('batch', row.get('requested_batch')))}/{fmt(row.get('graph_policy', row.get('graph_mode', row.get('graph'))))}"
        outcome = ("通过" if row.get("pass") is True else "失败" if row.get("pass") is False else "未单独判定")
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in
                     (f"{event['source']}:{event['line']}", identity, compact_json(numeric) if numeric else "见 JSON", outcome)) + " |")
        shown += 1
    if not shown:
        lines.append("| 未输入 | 未采集 | 未采集 | 未单独判定 |")
    lines += ["", "曲线数据位于 JSON 的 `curves`：每个缺测/失败/不足三次的点为 null，`segments` 在缺口或来源变化处断开。",
              "诊断 probe 原始记录及未匹配失败均保留在 JSON 的 `diagnostic_events` / `failures`；容量失败不等于 target KV 本身装不下。", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="raw JSONL，可重复")
    p.add_argument("--input-glob", action="append", default=[], help="raw JSONL 的明确模式，可重复")
    p.add_argument("--ledger", required=True, help="共享 GPU ledger JSONL")
    p.add_argument("--manifest", action="append", default=[], help="输入 manifest；多上下文重复传入")
    p.add_argument("--plan", action="append", default=[], help="pilot/formal/capacity 计划；可重复")
    p.add_argument("--out-json", required=True, help="新 JSON 文件，不覆盖")
    p.add_argument("--out-md", required=True, help="新 Markdown 文件，不覆盖")
    p.add_argument("--min-repeats", type=int, default=3, help="对照至少3次，不能小于3")
    a = p.parse_args()
    if a.min_repeats < 3:
        p.error("min-repeats 不能小于 3")
    files = set(a.input)
    for pattern in a.input_glob:
        matches = glob.glob(pattern)
        if not matches:
            p.error(f"input-glob 未匹配：{pattern}")
        files.update(matches)
    if not files:
        p.error("需要至少一个 --input 或 --input-glob")
    records, errors = load_lines(files)
    ledger_rows, ledger_errors = load_lines([a.ledger])
    errors.extend(ledger_errors)
    manifests = {}
    for filename in a.manifest:
        try:
            item = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
            ctx = item["context_tokens"]
            if ctx in manifests:
                p.error(f"重复的 manifest context: {ctx}")
            manifests[ctx] = item
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"source": filename, "error": str(exc)})
    plans = []
    for filename in a.plan:
        try:
            plans.append((filename, json.loads(Path(filename).read_text(encoding="utf-8-sig"))))
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"source": filename, "error": str(exc)})
    report = build_report(records, ledger_rows, manifests, plans, a.min_repeats, errors)
    paths = (Path(a.out_json), Path(a.out_md))
    if paths[0].resolve() == paths[1].resolve() or any(path.exists() for path in paths):
        p.error("输出文件必须不同且尚不存在")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    with paths[1].open("x", encoding="utf-8") as out:
        out.write(markdown(report))
    print("报告已生成；完整={}，缺测={}，失败/无效={}，原始失败={}".format(
          report["summary"]["complete_cells"], report["summary"]["missing_cells"],
          report["summary"]["failed_or_invalid_cells"], report["summary"]["failures"]))


if __name__ == "__main__":
    main()
