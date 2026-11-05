"""Audit raw fixed-batch records and recompute all delivery metrics on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import glob
import json
import math
from pathlib import Path
import statistics


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent / 100 * len(ordered)) - 1)]


def distribution(values):
    return {"n": len(values), **{f"p{p}": percentile(values, p) for p in (50, 95, 99)}}


def compute_run(run):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    requests, rounds = run["requests"], run["rounds"]
    wall = float(run["wall_s"])
    check(math.isfinite(wall) and wall > 0, "invalid wall duration")
    check(len(requests) == run["requested_batch"], "request count differs from requested batch")
    ids = [r["seq_id"] for r in requests]
    check(len(set(ids)) == len(ids), "duplicate request ID")
    request_map = {r["seq_id"]: r for r in requests}
    ttft, tpot, tbt, out_count = [], [], [], 0
    for r in requests:
        sid = r["seq_id"]
        deliveries = r["deliveries"]
        counts = [d["count"] for d in deliveries]
        ts = [d["at_s"] for d in deliveries]
        arrival, finish = r["arrival_s"], r["finish_s"]
        check(bool(deliveries), f"request {sid}: no delivered tokens")
        check(all(isinstance(n, int) and n > 0 for n in counts), f"request {sid}: invalid delivery batch")
        check(0 <= arrival <= finish <= wall + 1e-4, f"request {sid}: arrival/finish outside wall")
        check(all(arrival <= t <= finish + 1e-6 for t in ts), f"request {sid}: delivery outside lifetime")
        check(all(b >= a for a, b in zip(ts, ts[1:])), f"request {sid}: delivery clock reversed")
        n = sum(counts)
        out_count += n
        check(n == len(r["output_token_ids"]), f"request {sid}: token count differs from deliveries")
        check(n == run["output_tokens"], f"request {sid}: actual output differs from cap")
        check(token_hash(r["output_token_ids"]) == r["output_sha256"], f"request {sid}: output hash mismatch")
        if ts:
            check(abs(finish - ts[-1]) < 1e-6, f"request {sid}: finish is not last actual delivery")
            ttft.append(ts[0] - arrival)
            if n > 1:
                tpot.append((ts[-1] - ts[0]) / (n - 1))
            # One observation per real delivery event gap; no interpolated tokens.
            tbt.extend(b - a for a, b in zip(ts, ts[1:]))
    check(out_count == run["delivered_tokens"], "recorded total differs from request deliveries")
    per_request = Counter()
    proposed = accepted = decode_tokens = decode_round_rows = 0
    decode_s = prefill_s = 0.0
    full_batch_s = tail_batch_s = 0.0
    full_batch_tokens = tail_batch_tokens = 0
    tail_overproposal = 0
    active = Counter()
    previous_end = 0.0
    first_decode = None
    for i, row in enumerate(rounds):
        begin, end = row["start_s"], row["end_s"]
        check(0 <= begin <= end <= wall + 1e-4, f"round {i}: outside wall")
        check(begin >= previous_end - 1e-6, f"round {i}: overlapping or reversed")
        previous_end = end
        seqs = row["sequences"]
        check(row["active_b"] == len(seqs), f"round {i}: active batch mismatch")
        check(len({x["seq_id"] for x in seqs}) == len(seqs), f"round {i}: duplicate request")
        actual_delivered = 0
        for seq in seqs:
            sid, d, k, a = seq["seq_id"], seq["delivered"], seq["proposed"], seq["accepted_prefix"]
            check(sid in request_map, f"round {i}: unknown request {sid}")
            check(all(type(n) is int and n >= 0 for n in (d, k, a)), f"round {i}: negative/noninteger counts")
            check(a <= k, f"round {i}: accepted > proposed")
            if not row["prefill"] and run.get("ignore_eos") is True:
                verified = seq.get("verified_output_tokens", a + 1 if k else 1)
                remaining = seq.get("remaining_before")
                check(verified == (a + 1 if k else 1),
                      f"round {i}: verified output disagrees with accepted prefix plus bonus")
                check(d == (min(verified, remaining) if type(remaining) is int else verified),
                      f"round {i}: delivered count disagrees with verified output and cap")
                if run.get("arm") in ("dspark", "dspark_full", "dspark_window_full", "dspark_ring"):
                    check(type(remaining) is int and k <= max(0, remaining - 1),
                          f"round {i}: DSpark proposed beyond remaining output budget")
                if type(remaining) is int and k > max(0, remaining - 1):
                    tail_overproposal += 1
            per_request[sid] += d
            actual_delivered += d
            if not row["prefill"]:
                proposed += k
                accepted += a
                decode_tokens += d
                decode_round_rows += 1
        check(actual_delivered == row["delivered"], f"round {i}: delivered count mismatch")
        if row["prefill"]:
            prefill_s += end - begin
        else:
            decode_s += end - begin
            active[row["active_b"]] += 1
            if row["active_b"] == run["requested_batch"]:
                full_batch_s += end - begin
                full_batch_tokens += actual_delivered
            else:
                tail_batch_s += end - begin
                tail_batch_tokens += actual_delivered
            if first_decode is None:
                first_decode = row["active_b"]
    for r in requests:
        check(per_request[r["seq_id"]] == len(r["output_token_ids"]),
              f"request {r['seq_id']}: rounds do not account for all output")
    check(first_decode == run["requested_batch"], "first decode did not reach requested B")
    check(run["preemptions"] == 0, "preemption occurred; capacity condition not met")
    check(run["prefix_cached_tokens"] == 0, "prefix reuse detected")
    return {"valid": not errors, "errors": errors, "wall_s": wall,
            "full_request_tps": out_count / wall if wall > 0 else None,
            "prefill_s": prefill_s, "decode_s": decode_s,
            "decode_tps": decode_tokens / decode_s if decode_s else None,
            "full_batch_decode_tps": full_batch_tokens / full_batch_s if full_batch_s else None,
            "tail_batch_decode_tps": tail_batch_tokens / tail_batch_s if tail_batch_s else None,
            "ttft_s": distribution(ttft), "tpot_s": distribution(tpot),
            "delivery_batch_gap_s": distribution(tbt),
            "delivery_gap_definition": "one gap between actual delivery batches; not SSE or interpolated tokens",
            "proposed_tokens": proposed, "accepted_prefix_tokens": accepted,
            "acceptance_fraction": accepted / proposed if proposed else None,
            "tau_delivered_per_decode_sequence_round": decode_tokens / decode_round_rows if decode_round_rows else None,
            "tail_overproposal_sequence_rounds": tail_overproposal,
            "active_batch_rounds": dict(active), "output_tokens": out_count}


CELL_FIELDS = ("arm", "mode", "ctx", "requested_batch", "panel", "window", "graph_mode")
GROUP_FIELDS = CELL_FIELDS + ("phase", "seed", "temperature", "output_tokens", "gpu_uuid", "engine_commit",
                               "engine_dirty", "gpu_memory_utilization", "max_num_seqs", "max_model_len",
                               "max_num_batched_tokens", "prefix_lookup", "graph_policy", "mixed_overlap", "spec_k",
                               "ignore_eos", "target_config_sha256", "draft_config_sha256")
FORMAL_REQUIRED = GROUP_FIELDS + ("guard_id", "diagnostic_instrumented", "ignore_eos", "spec_k",
                                  "input_metadata", "graph_delta", "prefix_lookup_count", "prefix_hit_blocks")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def audit(rows, min_repeats, mode="diagnostic", plan=None, ledger=None, manifest=None):
    """Diagnostic auditing is deliberately not a formal performance decision."""
    if mode not in ("diagnostic", "formal"):
        raise ValueError("unknown audit mode")
    groups = defaultdict(list)
    failures, invalid, runs, raw = [], [], [], []
    ledger_records, ledger_starts, ledger_children, ledger_errors = {}, {}, {}, []
    if ledger is not None:
        for entry in ledger:
            key = entry.get("id")
            dest = {"start": ledger_starts, "child": ledger_children, "end": ledger_records}.get(entry.get("event"))
            if not key or dest is None or key in dest:
                ledger_errors.append("invalid or duplicate guard event")
            else:
                dest[key] = entry
        if ledger_starts.keys() != ledger_records.keys():
            ledger_errors.append("guard ledger has unfinished jobs")
        if not ledger_children.keys() <= ledger_starts.keys():
            ledger_errors.append("guard ledger contains an unowned child")
        try:
            total = sum(float(e["elapsed_s"]) for e in ledger_records.values())
            smoke = sum(float(e["elapsed_s"]) for e in ledger_records.values() if e["kind"] == "smoke")
            if any(float(e["elapsed_s"]) < 0 for e in ledger_records.values()) or total > 14400 or smoke > 1800:
                ledger_errors.append("guard ledger budget is invalid or exceeded")
        except (KeyError, TypeError, ValueError):
            ledger_errors.append("guard ledger timing is malformed")
    warmups = defaultdict(list)
    for _, _, row in rows:
        if row.get("type") == "warmup":
            warmups[row.get("guard_id")].append(row)
    for source, line_no, row in rows:
        if row.get("type") == "failure":
            failures.append({"source": source, "line": line_no, "record": row})
            continue
        if row.get("type") != "run":
            continue
        try:
            result = compute_run(row)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            result = {"valid": False, "errors": [f"malformed raw record: {exc}"]}
        errors = result["errors"]
        if mode == "diagnostic" and row.get("diagnostic_instrumented") is not True:
            errors.append("diagnostic audit requires explicitly instrumented diagnostic records")
        if mode == "formal":
            missing = [key for key in FORMAL_REQUIRED if key not in row]
            if missing:
                errors.append("formal fields missing: " + ", ".join(missing))
            if row.get("diagnostic_instrumented") is not False:
                errors.append("diagnostic or unknown instrumentation cannot be used as formal throughput")
            if row.get("engine_dirty") is not False:
                errors.append("formal run requires a recorded clean engine commit")
            if row.get("phase") not in ("pilot", "formal"):
                errors.append("capacity diagnostics are separate from fixed-batch performance")
            if not row.get("engine_commit") or not row.get("gpu_uuid") or not row.get("target_config_sha256"):
                errors.append("formal provenance requires engine commit, GPU UUID, target config hash")
            if row.get("gpu_memory_utilization") != 0.70:
                errors.append("fixed-batch primary comparison requires the common 0.70 GPU budget")
            if row.get("graph_mode") != "graph":
                errors.append("fixed-batch primary comparison requires graph-on")
            arm = row.get("arm")
            expected_k = 0 if arm == "nospec" else 6 if arm in ("ar_full", "ar_window") else 7 if arm in ("dspark_full", "dspark_window_full", "dspark_ring") else None
            if expected_k is None or row.get("spec_k") != expected_k:
                errors.append("arm does not declare its expected candidate count")
            if row.get("prefix_lookup") is not False or row.get("prefix_lookup_count") != 0 or row.get("prefix_hit_blocks") != 0:
                errors.append("formal prefix lookup/hit policy is not disabled")
            if row.get("temperature") != 1.0 or row.get("ignore_eos") is not True:
                errors.append("formal sampling policy is not T=1 and ignore_eos")
            if row.get("mixed_overlap") is not False:
                errors.append("unsupported mixed overlap or missing setting")
            if row.get("max_model_len", 0) != row.get("ctx", 0) + row.get("output_tokens", 0) + 8:
                errors.append("formal arms require common max_model_len = prompt + output + 8")
            if row.get("graph_delta", {}).get("graph_runtime_captures", 0) or row.get("graph_delta", {}).get("head_runtime_captures", 0):
                errors.append("unexpected measured-phase graph captures")
            if "target_decode_graph" not in row:
                errors.append("target decode graph telemetry unavailable")
            elif row.get("arm") == "nospec" and not (row["target_decode_graph"].get("exact", 0) + row["target_decode_graph"].get("padded", 0)):
                errors.append("no-spec target decode graph was not observed")
            if row.get("spec_k", 0) > 0:
                if row.get("graph_delta", {}).get("verify_hits", 0) <= 0:
                    errors.append("target verify graph was not observed")
                if row.get("arm", "").startswith("dspark_") and row.get("graph_delta", {}).get("graph_hits", 0) <= 0:
                    errors.append("DSpark draft graph was not observed")
                if row.get("arm", "").startswith("ar_") and row.get("graph_delta", {}).get("n_graphed", 0) <= 0:
                    errors.append("AR draft graph was not observed")
            guard = ledger_records.get(row.get("guard_id"))
            start = ledger_starts.get(row.get("guard_id"))
            child = ledger_children.get(row.get("guard_id"))
            if (not guard or not start or not child or guard.get("outcome") != "ok" or guard.get("exit_code") != 0
                    or guard.get("ever_owned_compute_observed") is not True
                    or guard.get("gpu_uuid") != row.get("gpu_uuid") or guard.get("kind") != row.get("phase")
                    or start.get("gpu_uuid") != row.get("gpu_uuid")
                    or child.get("gpu_uuid") != row.get("gpu_uuid")):
                errors.append("matching successful guard end/UUID unavailable")
            warm = warmups.get(row.get("guard_id"), [])
            if len(warm) != 1 or not all(x in warm[0] for x in
                                         ("prewarm_s", "prewarm_shapes", "expected_prewarm_shapes", "memory_after_prewarm",
                                          "prewarm_head_batches", "target_decode_graph_buckets", "target_verify_graph_shapes",
                                          "ar_draft_graph_buckets", "warmup_graph_delta")):
                errors.append("matching explicit prewarm record is unavailable")
            elif {canonical(x) for x in warm[0]["prewarm_shapes"]} != {canonical(x) for x in warm[0]["expected_prewarm_shapes"]}:
                errors.append("prewarmed shapes differ from expected shapes")
            elif row.get("arm", "").startswith("dspark_"):
                try:
                    batch, ctx, out, window = (int(row[k]) for k in ("requested_batch", "ctx", "output_tokens", "window"))
                    block = int(row["kv_block_size"])
                    allowed = [b for b in (1, 2, 4, 8, 16) if b <= row["max_num_seqs"]]
                    batches = {next(b for b in allowed if b >= actual) for actual in range(1, batch + 1)}
                    contexts = {max(block, math.ceil((min(c, window) if window else c) / block) * block)
                                for c in range(ctx, ctx + out + 1)}
                    expected = {(b, c) for b in batches for c in contexts}
                    if {tuple(v) for v in warm[0]["prewarm_shapes"]} != expected:
                        errors.append("DSpark prewarm does not cover the independently computed shape range")
                    if set(warm[0].get("prewarm_head_batches", [])) != set(range(1, batch + 1)):
                        errors.append("DSpark head prewarm misses real tail batch sizes")
                except (KeyError, TypeError, ValueError, StopIteration, ZeroDivisionError):
                    errors.append("DSpark prewarm coverage metadata malformed")
            if len(warm) == 1:
                record = warm[0]
                required_batch_buckets = {b for b in (1, 2, 4, 8, 16) if b <= row.get("max_num_seqs", 0)}
                if not required_batch_buckets <= set(record.get("target_decode_graph_buckets", [])):
                    errors.append("target decode graph prewarm missing batch buckets")
                if row.get("spec_k", 0) and not {(b, row["spec_k"]) for b in required_batch_buckets} <= {tuple(x) for x in record.get("target_verify_graph_shapes", [])}:
                    errors.append("target verify graph prewarm missing batch buckets")
                if row.get("arm", "").startswith("ar_") and not required_batch_buckets <= set(record.get("ar_draft_graph_buckets", [])):
                    errors.append("AR draft graph prewarm missing batch buckets")
                warm_delta = record.get("warmup_graph_delta", {})
                if warm_delta.get("graph_runtime_captures", 0) or warm_delta.get("head_runtime_captures", 0):
                    errors.append("warmup requests caused an unplanned graph capture")
            if manifest is None:
                errors.append("formal audit requires input manifest")
            else:
                try:
                    source = manifest if "context_tokens" in manifest else manifest[row["ctx"]]
                    expected = source["panels"][row["panel"]][:row["requested_batch"]]
                    hashes = [token_hash(e["token_ids"]) for e in expected]
                    recorded = [x.get("actual_input_sha256", x.get("input_sha256")) for x in row["input_metadata"]]
                    if (source["context_tokens"] != row["ctx"] or len(expected) != row["requested_batch"]
                            or any(len(e["token_ids"]) != row["ctx"] or token_hash(e["token_ids"]) != e["metadata"]["input_sha256"] for e in expected)
                            or hashes != recorded):
                        errors.append("actual input hashes do not match the fixed workload manifest")
                    for actual, original in zip(row["input_metadata"], expected):
                        meta = original["metadata"]
                        n = actual.get("wiki_tokens")
                        if (type(n) is not int or not 0 <= n <= row["ctx"]
                                or type(actual.get("task_truncated_from_left")) is not bool
                                or not isinstance(actual.get("wiki_fraction"), (int, float))
                                or abs(actual["wiki_fraction"] - n / row["ctx"]) > 1e-12
                                or any(actual.get(k) != meta.get(k) for k in
                                       ("wiki_tokens", "task_tokens", "task_truncated_from_left", "dataset", "row"))):
                            errors.append("workload filling, task, or truncation metadata mismatch")
                            break
                except (KeyError, TypeError):
                    errors.append("malformed workload or input metadata")
        result["valid"] = not errors
        identity = {k: row.get(k) for k in CELL_FIELDS + ("repeat", "seed", "gpu_uuid")}
        item = {"source": source, "line": line_no, "identity": identity, "metrics": result}
        runs.append(item)
        if not result["valid"]:
            invalid.append(item)
            continue
        group_config = {k: row.get(k) for k in GROUP_FIELDS}
        group_config["input_hashes"] = [x.get("actual_input_sha256", x.get("input_sha256"))
                                        for x in row.get("input_metadata", [])]
        key = canonical(group_config)
        groups[key].append((row.get("repeat"), result["full_request_tps"], row))
        raw.append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        repeats = [x[0] for x in values]
        summary = {"configuration": json.loads(key), "n": len(values), "repeats": repeats,
                   "unique_repeats": len(set(repeats)) == len(repeats),
                   "enough_repeats": len(set(repeats)) >= min_repeats}
        if mode == "formal":
            tps = [x[1] for x in values]
            summary.update(full_request_tps_all=tps, median=statistics.median(tps), min=min(tps), max=max(tps))
        summaries.append(summary)
    result = {"audit_mode": mode, "runs": runs, "failures": failures, "invalid_runs": len(invalid),
              "groups": summaries, "min_repeats": min_repeats}
    data_valid = bool(runs) and not failures and not invalid and not ledger_errors
    if mode == "diagnostic":
        result.update(data_valid=data_valid, performance_eligible=False,
                      interpretation="raw diagnostic audit only; this does not pass GPU numerical or performance gates")
        return result

    expected_cells, plan_errors = set(), []
    if plan is None or not isinstance(plan.get("cells"), list) or not plan["cells"]:
        plan_errors.append("explicit non-empty expected matrix plan is required")
    else:
        if plan.get("phase") and any(r.get("phase") != plan["phase"] for r in raw):
            plan_errors.append("run phase differs from planned phase")
        for cell in plan["cells"]:
            if any(k not in cell for k in CELL_FIELDS):
                plan_errors.append("planned cell missing required fields")
                continue
            key = canonical({k: cell[k] for k in CELL_FIELDS})
            if key in expected_cells:
                plan_errors.append("duplicate planned cell")
            expected_cells.add(key)
        if plan.get("matrix_scope") == "full minimum fixed-batch matrix":
            selected_w = plan.get("window_tokens")
            required = set()
            for ctx in (1024, 4096):
                for batch in (1, 2, 4, 8, 16):
                    for panel in ("code", "math", "general"):
                        for arm, mode in (("nospec", "full"), ("ar_full", "full"),
                                          ("ar_window", "block_aligned"), ("dspark_full", "full"),
                                          ("dspark_ring", "ring")):
                            window = selected_w if arm in ("ar_window", "dspark_ring") else 0
                            required.add(canonical({"arm": arm, "mode": mode, "ctx": ctx,
                                                    "requested_batch": batch, "panel": panel,
                                                    "window": window, "graph_mode": "graph"}))
            if type(selected_w) is not int or selected_w <= 0 or selected_w % 256 or expected_cells != required or min_repeats < 3:
                plan_errors.append("claimed full matrix does not match the required cells/window/repeats")
    cell_groups = defaultdict(list)
    for group in summaries:
        cell_groups[canonical({k: group["configuration"].get(k) for k in CELL_FIELDS})].append(group)
    missing = sorted(expected_cells - cell_groups.keys())
    unexpected = sorted(cell_groups.keys() - expected_cells)
    split_configs = [key for key, same in cell_groups.items() if len(same) != 1]
    pair = defaultdict(list)
    for r in raw:
        key = (r["ctx"], r["requested_batch"], r["panel"], r["repeat"])
        workload = [x.get("actual_input_sha256", x.get("input_sha256")) for x in r["input_metadata"]]
        paired = {k: r.get(k) for k in ("seed", "temperature", "output_tokens", "gpu_uuid", "engine_commit",
                                       "engine_dirty", "gpu_memory_utilization", "prefix_lookup", "graph_policy",
                                       "target_config_sha256", "max_num_batched_tokens", "max_model_len", "phase")}
        pair[key].append(canonical({"settings": paired, "input_hashes": workload}))
    mismatched_pairs = [list(key) for key, vals in pair.items() if len(set(vals)) != 1]
    sufficient = all(g["enough_repeats"] and g["unique_repeats"] for g in summaries)
    result.update(expected_cell_count=len(expected_cells), missing_cells=[json.loads(x) for x in missing],
                  unexpected_cells=[json.loads(x) for x in unexpected],
                  inconsistent_cell_configs=[json.loads(x) for x in split_configs],
                  mismatched_pairs=mismatched_pairs, plan_errors=plan_errors, ledger_errors=ledger_errors)
    result["matrix_scope"] = plan.get("matrix_scope", "explicit subset") if isinstance(plan, dict) else "unspecified"
    result["pass"] = bool(data_valid and plan is not None and expected_cells and not plan_errors and
                          not missing and not unexpected and not split_configs and not mismatched_pairs and sufficient)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="JSONL；可重复传入")
    p.add_argument("--input-glob", action="append", default=[], help="显式 JSONL 模式；可重复传入")
    p.add_argument("--out", required=True, help="新 JSON 文件，不覆盖")
    p.add_argument("--mode", choices=("diagnostic", "formal"), default="diagnostic")
    p.add_argument("--ledger", help="共享 GPU ledger，formal 必填")
    p.add_argument("--manifest", action="append", help="逐请求固定输入 manifest，formal 必填；多长度时重复传入")
    p.add_argument("--plan", help="期望矩阵 JSON: {cells:[{arm,mode,ctx,requested_batch,panel,window,graph_mode}]}，formal 必填")
    p.add_argument("--min-repeats", type=int, default=1, help="正式矩阵应设为 3")
    a = p.parse_args()
    if a.min_repeats < 1:
        p.error("min-repeats 必须为正")
    rows = []
    filenames = set(a.input)
    for pattern in a.input_glob:
        matches = glob.glob(pattern)
        if not matches:
            p.error(f"input-glob 未匹配任何文件: {pattern}")
        filenames.update(matches)
    if not filenames:
        p.error("需要 --input 或 --input-glob")
    for filename in sorted(filenames):
        for n, line in enumerate(Path(filename).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows.append((filename, n, json.loads(line)))
    ledger = [json.loads(x) for x in Path(a.ledger).read_text(encoding="utf-8").splitlines() if x.strip()] if a.ledger else None
    manifest = None
    if a.manifest:
        manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in a.manifest]
        manifest = {x["context_tokens"]: x for x in manifests}
        if len(manifest) != len(manifests):
            p.error("manifest context length duplicated")
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
    report = audit(rows, a.min_repeats, a.mode, plan, ledger, manifest)
    with Path(a.out).open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    passed = report.get("pass", report.get("data_valid", False))
    print(f"审计完成：{len(report['runs'])} 次运行，{report['invalid_runs']} 次无效，{len(report['failures'])} 条失败；记录有效={passed}；模式={a.mode}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
