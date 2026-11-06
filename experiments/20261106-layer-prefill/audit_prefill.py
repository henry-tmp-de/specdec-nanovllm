"""CPU audit for one mixed-B2 smoke and at most one B4/4K run per mode.

No GPU work is performed. Smoke checks validate the recorded per-position
metrics and coverage; full logits/KV are not stored and cannot be recomputed
from JSONL. Performance recomputes request/round accounting from raw events.
GPU ownership and the shared ledger are separately reviewed by the coordinator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_common import MODES, validate_layer_execution
from bench_prefill import validate_result
from gpu_compare_prefill import PAIRS, ZERO_REF_MAX_ABS, DECODE_STEPS, sample_positions


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_rows(path):
    result = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number_, line in enumerate(stream, 1):
            if line.strip():
                def bad_constant(value):
                    raise ValueError(f"nonfinite JSON constant {value}")
                result.append((number_, json.loads(line, parse_constant=bad_constant)))
    return result


def same_source(a, b):
    fields = ("engine_commit", "engine_dirty", "gpu_uuid", "target_provenance", "budget", "chunk")
    return all(a.get(k) == b.get(k) and a.get(k) is not None for k in fields)


def audit_smoke(rows):
    errors, failures, modes, logits, kv, completions = [], [], {}, {}, {}, []
    for line, row in rows:
        if not isinstance(row, dict):
            errors.append(f"line {line}: not an object")
            continue
        kind = row.get("type")
        if kind == "failure":
            failures.append({"line": line, "error": row.get("error"), "message": row.get("message")})
        elif kind == "complete":
            completions.append(row)
        elif kind == "mode_complete":
            key = row.get("mode")
            if key in modes:
                errors.append(f"line {line}: duplicate mode")
            modes[key] = row
        elif kind in ("logit_check", "kv_check"):
            keys = ("reference", "candidate", "case", "request_index", "step") if kind == "logit_check" else (
                "reference", "candidate", "case", "request_index", "position", "kv", "layer")
            key = tuple(row.get(k) for k in keys)
            dest = logits if kind == "logit_check" else kv
            if key in dest:
                errors.append(f"line {line}: duplicate {kind}")
            dest[key] = row
        else:
            errors.append(f"line {line}: unknown record type {kind!r}")
    if failures:
        errors.append("smoke contains a failure record")
    if set(modes) != set(MODES) or len(completions) != 1:
        errors.append("smoke must contain all three modes and one completion")
    if errors:
        return {"valid": False, "errors": errors, "failures": failures, "logit_checks": len(logits), "kv_checks": len(kv)}
    baseline = modes["full"]
    lengths = [257, 513]
    layers = baseline.get("num_layers")
    if type(layers) is not int or layers <= 0:
        errors.append("invalid layer count")
        return {"valid": False, "errors": errors, "failures": failures, "logit_checks": len(logits), "kv_checks": len(kv)}
    inputs = baseline.get("input_metadata")
    histories = baseline.get("fixed_history_sha256")
    if (not isinstance(inputs, list) or len(inputs) != 2 or not isinstance(histories, list) or len(histories) != 2
            or any(not hash_string(x) for x in histories) or
            any(not isinstance(item, dict) or item.get("actual_context_tokens") != lengths[i]
                or not hash_string(item.get("actual_input_sha256")) for i, item in enumerate(inputs))):
        errors.append("smoke input or fixed-history hashes are missing")
    for mode in MODES:
        row = modes[mode]
        if (not same_source(row, baseline) or row.get("engine_dirty") is not False or row.get("case") != "b2_257_513"
                or row.get("lengths") != lengths or row.get("input_metadata") != inputs
                or row.get("fixed_history_sha256") != histories or row.get("num_layers") != layers
                or row.get("diagnostic_instrumented") is not True or row.get("chunk") != 128
                or not row.get("guard_id") or row.get("guard_id") != baseline.get("guard_id")
                or type(row.get("target_kv_blocks")) is not int or row["target_kv_blocks"] <= 0
                or row.get("kv_block_size") != 256):
            errors.append(f"{mode}: source, fixed inputs/history or scope differ")
        calls, blocks = row.get("prefill_calls"), row.get("sentinel_blocks")
        if (not isinstance(calls, list) or len(calls) != 1 or calls[0].get("active_b") != 2
                or calls[0].get("scheduled_tokens") != lengths or not isinstance(blocks, list) or len(blocks) != 2):
            errors.append(f"{mode}: mixed-batch prefill coverage missing")
        else:
            seen = set()
            pool_blocks = row.get("target_kv_blocks") if type(row.get("target_kv_blocks")) is int else 0
            indexed = {entry["input_index"]: entry for entry in blocks if isinstance(entry, dict)
                       and type(entry.get("input_index")) is int}
            if set(indexed) != {0, 1}:
                errors.append(f"{mode}: prefill request mapping missing or duplicated")
            for i in range(2):
                entry = indexed.get(i, {})
                block_ids = entry.get("physical_blocks") if isinstance(entry, dict) else None
                if (not isinstance(block_ids, list) or len(block_ids) != math.ceil(lengths[i] / 256)
                        or any(type(x) is not int or x < 0 or x >= pool_blocks for x in block_ids)
                        or len(set(block_ids)) != len(block_ids) or set(block_ids) & seen
                        or entry.get("valid_prefix_slots") != lengths[i] or entry.get("block_size") != 256
                        or entry.get("sample_positions") != sample_positions(lengths[i])
                        or entry.get("preimage") != "logical prompt slots NaN; page padding zero"):
                    errors.append(f"{mode}: page ownership/preimage metadata mismatch")
                if isinstance(block_ids, list) and all(type(x) is int for x in block_ids):
                    seen.update(block_ids)
        try:
            delta = validate_layer_execution(mode, row.get("layer_before"), row.get("layer_after"), lengths, layers, 128)
            if delta != row.get("layer_delta"):
                errors.append(f"{mode}: layer counters do not match snapshots")
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"{mode}: layer counters: {exc}")
        graph = row.get("graph_delta")
        if (not isinstance(graph, dict) or graph.get("decode_graph_exact_hits", 0) < DECODE_STEPS
                or graph.get("decode_graph_padded_hits", 0) or graph.get("decode_graph_fallbacks", {})):
            errors.append(f"{mode}: ordinary decode graph telemetry invalid")
    expected_logits, expected_kv = set(), set()
    for reference, candidate, kl_limit, rms_limit in PAIRS:
        for request, length in enumerate(lengths):
            for step in range(DECODE_STEPS + 1):
                key = (reference, candidate, "b2_257_513", request, step)
                expected_logits.add(key)
                row = logits.get(key, {})
                if (row.get("finite") is not True or row.get("pass") is not True or
                        row.get("kl_direction") != "reference || candidate" or row.get("kl_limit") != kl_limit or
                        not number(row.get("kl")) or not -1e-6 <= row["kl"] < kl_limit or
                        type(row.get("reference_argmax")) is not int or type(row.get("candidate_argmax")) is not int or
                        row.get("reference_argmax") != row.get("candidate_argmax") or
                        not number(row.get("reference_top1_top2_margin")) or not number(row.get("candidate_top1_top2_margin")) or
                        row["reference_top1_top2_margin"] < 0 or row["candidate_top1_top2_margin"] < 0 or
                        not number(row.get("max_abs")) or row["max_abs"] < 0 or
                        row.get("step_kind") != ("prefill" if step == 0 else "decode")):
                    errors.append(f"invalid or missing logit check: {key}")
            for position in sample_positions(length):
                for kind in ("K", "V"):
                    for layer in range(layers):
                        key = (reference, candidate, "b2_257_513", request, position, kind, layer)
                        expected_kv.add(key)
                        row = kv.get(key, {})
                        ref, rel, maximum = row.get("reference_rms"), row.get("relative_rms"), row.get("max_abs")
                        good = (row.get("finite") is True and row.get("pass") is True and number(ref) and ref >= 0
                                and number(maximum) and maximum >= 0 and row.get("relative_rms_limit") == rms_limit
                                and ((ref > 0 and number(rel) and 0 <= rel < rms_limit) or
                                     (ref == 0 and rel is None and maximum <= ZERO_REF_MAX_ABS and
                                      row.get("zero_reference_max_abs_limit") == ZERO_REF_MAX_ABS)))
                        if not good:
                            errors.append(f"invalid or missing KV check: {key}")
    if set(logits) != expected_logits or set(kv) != expected_kv:
        errors.append("unexpected or incomplete smoke check coverage")
    end = completions[0]
    if (end.get("pass") is not True or end.get("failed_comparisons") != 0 or
            end.get("comparisons") != len(expected_logits) + len(expected_kv) or not same_source(end, baseline)
            or end.get("guard_id") != baseline.get("guard_id")):
        errors.append("smoke completion or identity mismatch")
    return {"valid": not errors, "errors": errors, "failures": failures, "logit_checks": len(logits),
            "kv_checks": len(kv), "modes": {m: {k: modes[m].get(k) for k in
                     ("target_kv_blocks", "kv_block_size", "load_initial_graph_s", "layer_delta", "graph_delta")}
                     for m in MODES}, "identity": {k: baseline.get(k) for k in
                     ("engine_commit", "gpu_uuid", "target_provenance", "budget", "chunk")}}


def audit_perf(rows, expected_mode):
    errors, failures = [], []
    records = {kind: [row for _, row in rows if isinstance(row, dict) and row.get("type") == kind]
               for kind in ("warmup", "run", "complete", "failure")}
    unknown = [row for _, row in rows if not isinstance(row, dict) or row.get("type") not in records]
    if unknown or len(records["warmup"]) != 1 or len(records["run"]) != 1 or len(records["complete"]) != 1:
        errors.append("expected exactly one warmup, one run and one completion")
    failures.extend({"error": x.get("error"), "message": x.get("message")} for x in records["failure"])
    if failures:
        errors.append("performance raw contains failure record(s)")
    if errors:
        return {"mode": expected_mode, "valid": False, "errors": errors, "failures": failures}
    warm, run, end = records["warmup"][0], records["run"][0], records["complete"][0]
    if (run.get("mode") != expected_mode or run.get("batch") != 4 or run.get("ctx") != 4096
            or run.get("output_tokens") != 128 or run.get("chunk") != 128 or run.get("repeat") != 0
            or run.get("primary_timing_repeats") != 1 or run.get("warmup_cohorts") != 1
            or run.get("temperature") != 1.0 or run.get("diagnostic_instrumented") is not False
            or run.get("engine_dirty") is not False or end.get("repeats_per_panel") != 1
            or any(not same_source(run, x) or run.get("guard_id") != x.get("guard_id") for x in (warm, end))):
        errors.append("performance source, scope, guard or single-run identity mismatch")
    if any(any(run.get(k) != other.get(k) for k in ("mode", "batch", "ctx", "output_tokens", "panel", "seed", "input_metadata"))
           for other in (warm, end)):
        errors.append("warmup/run/complete input identity differs")
    inputs = run.get("input_metadata")
    if (not isinstance(inputs, list) or len(inputs) != 4 or
            any(not isinstance(item, dict) or item.get("actual_context_tokens") != 4096
                or not hash_string(item.get("actual_input_sha256")) for item in inputs)):
        errors.append("performance input hashes or lengths missing")
    calls = warm.get("prefill_calls")
    if not isinstance(calls, list) or len(calls) != 1 or calls[0].get("scheduled_tokens") != [4096] * 4:
        errors.append("warmup did not admit full B4 prompt")
    try:
        warm_delta = validate_layer_execution(expected_mode, warm.get("layer_before"), warm.get("layer_after"),
                                              [4096] * 4, run["num_layers"], 128)
        delta = validate_layer_execution(expected_mode, run.get("layer_before"), run.get("layer_after"),
                                         [4096] * 4, run["num_layers"], 128)
        if delta != run.get("layer_delta"):
            errors.append("measured layer counter delta mismatch")
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"layer counter validation failed: {exc}")
        warm_delta = delta = None
    try:
        metrics = validate_result(run, [4096] * 4)
        if not metrics["valid"]:
            errors.extend(metrics["errors"])
        if json.loads(json.dumps(metrics, allow_nan=False)) != run.get("metrics"):
            errors.append("recorded per-request metrics differ from raw recomputation")
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        errors.append(f"malformed performance run: {exc}")
        metrics = None
    return {"mode": expected_mode, "valid": not errors, "errors": errors, "failures": failures,
            "identity": {k: run.get(k) for k in ("engine_commit", "engine_dirty", "gpu_uuid", "guard_id",
                         "target_provenance", "budget", "chunk", "ctx", "batch", "output_tokens", "panel", "seed", "input_metadata")},
            "metrics": metrics, "layer_delta": delta, "warm_layer_delta": warm_delta,
            "target_decode_graph": run.get("target_decode_graph"), "target_kv_blocks": run.get("target_kv_blocks"),
            "load_initial_graph_s": run.get("load_initial_graph_s"), "warmup_s": warm.get("warmup_s"),
            "memory_after_run": run.get("memory_after_run")}


def summarize(smoke, perf):
    result = {"smoke": smoke, "performance": perf, "single_sample_only": True, "guard_ledger_checked": False,
              "note": "one timed run per mode; no CI; smoke raw has metrics but not full logits/KV tensors",
              "ratios": None}
    if not smoke.get("valid") or set(perf) != set(MODES) or not all(perf[m].get("valid") for m in MODES):
        return result
    first = perf["full"]["identity"]
    fields = ("engine_commit", "engine_dirty", "gpu_uuid", "target_provenance", "budget", "chunk", "ctx", "batch",
              "output_tokens", "panel", "seed", "input_metadata")
    if any(any(perf[mode]["identity"].get(k) != first.get(k) for k in fields) for mode in MODES):
        result["comparison_error"] = "modes do not share the same inputs/source/conditions"
        return result
    # The smoke and benchmark must use the same source/card/model/budget, while
    # their prompt shapes intentionally differ.
    if any(first.get(k) != smoke.get("identity", {}).get(k) for k in
           ("engine_commit", "gpu_uuid", "target_provenance", "budget", "chunk")):
        result["comparison_error"] = "smoke and performance provenance differ"
        return result
    tps = {mode: perf[mode]["metrics"]["full_request_output_tps"] for mode in MODES}
    if any(not number(x) or x <= 0 for x in tps.values()):
        result["comparison_error"] = "throughput missing or nonpositive"
        return result
    result["ratios"] = {"serial_over_full": tps["serial"] / tps["full"],
                        "pipeline_over_full": tps["pipeline"] / tps["full"],
                        "pipeline_over_serial": tps["pipeline"] / tps["serial"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", required=True)
    for mode in MODES:
        parser.add_argument("--" + mode)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    smoke = audit_smoke(read_rows(args.smoke))
    perf = {mode: audit_perf(read_rows(getattr(args, mode)), mode) for mode in MODES if getattr(args, mode)}
    result = summarize(smoke, perf)
    out = Path(args.out_dir)
    if any((out / name).exists() for name in ("summary.json", "performance.csv", "report.md")):
        parser.error("output exists; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    cols = ("mode", "valid", "wall_s", "output_tps", "ttft_p50_s", "tpot_p50_s", "decode_span_s",
            "target_kv_blocks", "load_initial_graph_s", "warmup_s", "errors")
    with (out / "performance.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        for mode in MODES:
            entry = perf.get(mode, {"mode": mode, "valid": False, "errors": ["not supplied"]})
            metrics = entry.get("metrics") or {}
            writer.writerow({"mode": mode, "valid": entry.get("valid"), "wall_s": metrics.get("wall_s"),
                             "output_tps": metrics.get("full_request_output_tps"),
                             "ttft_p50_s": (metrics.get("ttft_s") or {}).get("p50"),
                             "tpot_p50_s": (metrics.get("tpot_s") or {}).get("p50"),
                             "decode_span_s": metrics.get("decode_span_s"), "target_kv_blocks": entry.get("target_kv_blocks"),
                             "load_initial_graph_s": entry.get("load_initial_graph_s"), "warmup_s": entry.get("warmup_s"),
                             "errors": json.dumps(entry.get("errors", []), ensure_ascii=False)})
    lines = ["# 层间 prefill：单次诊断与吞吐", "", "只测 B4/4K/output128，每臂一条计时，无重复范围或置信区间。",
             "正确性记录仅含逐位置指标，不含完整 logits/KV；本工具验证覆盖和门槛，GPU guard ledger 单独核验。",
             "B 为相同对角DAG单流，C 为相同DAG双流；A/B 差含分块、发射、事件与元数据。",
             f"smoke有效：{smoke.get('valid')}；logit检查 {smoke.get('logit_checks')}，KV检查 {smoke.get('kv_checks')}。", "",
             "| 模式 | 有效 | 墙钟秒 | 输出token/s | target KV块 | 失败/缺测 |", "|---|---|---:|---:|---:|---|"]
    for mode in MODES:
        entry = perf.get(mode, {"valid": False, "errors": ["not supplied"]})
        metrics = entry.get("metrics") or {}
        fmt = lambda x: "—" if x is None else f"{x:.6g}" if type(x) in (int, float) else str(x)
        lines.append(f"| {mode} | {entry['valid']} | {fmt(metrics.get('wall_s'))} | {fmt(metrics.get('full_request_output_tps'))} | {fmt(entry.get('target_kv_blocks'))} | {'; '.join(entry.get('errors', [])) or '—'} |")
    lines += ["", f"单次吞吐比：{json.dumps(result['ratios'], ensure_ascii=False) if result['ratios'] is not None else '未满足比较条件'}。",
              "", "同预算不代表同 KV 容量；没有 profile 不能把原因或 GPU 跨流重叠写成已测结论。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"smoke有效={smoke.get('valid')}，primary有效臂={sum(x.get('valid', False) for x in perf.values())}，配对比={'有' if result['ratios'] else '无'}")


if __name__ == "__main__":
    main()
