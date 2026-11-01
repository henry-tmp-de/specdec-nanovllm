"""Recompute matrix statistics from individual JSONL records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="formal")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    a = ap.parse_args()
    root = Path(a.results)
    report = {"cells": {}, "comparisons": {}, "invalid": [], "failures": []}
    by = {}
    for B in a.batches:
        for arm in ("nospec", "ar", "dspark"):
            path = root / f"{a.tag}-{arm}-b{B}.jsonl"
            if not path.exists():
                report["failures"].append({"batch": B, "arm": arm, "error": "missing"})
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            failures = [r for r in rows if r["type"] == "failure"]
            for row in failures:
                report["failures"].append({"batch": B, "arm": arm, "error": row.get("message")})
            for panel in ("code", "math", "general"):
                samples = sorted((r for r in rows if r["type"] == "run" and r["panel"] == panel),
                                 key=lambda r: r["repeat"])
                if len(samples) != 3 or [s["repeat"] for s in samples] != [0, 1, 2]:
                    report["invalid"].append(f"{arm}/B{B}/{panel}: expected exactly 3 repeats")
                    continue
                if any(s["requested_batch"] != B or s["delivered_tokens"] != B * 256 for s in samples):
                    report["invalid"].append(f"{arm}/B{B}/{panel}: batch or output count mismatch")
                    continue
                hashes = [[x["input_sha256"] for x in s["inputs"]] for s in samples]
                if not all(h == hashes[0] for h in hashes):
                    report["invalid"].append(f"{arm}/B{B}/{panel}: input mismatch across repeats")
                by[(B, panel, arm)] = samples
                def values(key):
                    return [s[key] for s in samples]
                def stats(key):
                    v = values(key)
                    return {"values": v, "median": statistics.median(v), "min": min(v), "max": max(v)}
                cell = {"full_request_tps": stats("full_request_tps"), "decode_tps": stats("decode_tps"),
                        "full_batch_decode_tps": stats("full_batch_decode_tps"),
                        "ttft_median_s": stats("ttft_median_s"), "tpot_median_s": stats("tpot_median_s"),
                        "prefill_s": stats("prefill_s"), "decode_s": stats("decode_s"),
                        "proposed": values("proposed_tokens"), "accepted": values("accepted_prefix_tokens"),
                        "acceptance_fraction": values("acceptance_prefix_fraction"),
                        "mean_delivered_per_round": values("mean_delivered_per_decode_round_per_sequence"),
                        "active_b_rounds": values("active_b_rounds"), "graph_delta": values("graph_delta"),
                        "gpu_stream_span_ms": values("gpu_stream_span_ms"),
                        "postprocess_s": values("postprocess_s"),
                        "peak_allocated_bytes": values("peak_allocated_bytes"),
                        "kv_blocks": values("kv_blocks"), "preemptions": values("preemptions"),
                        "prefix_cached_tokens": values("prefix_cached_tokens"),
                        "input_hashes": hashes[0]}
                report["cells"][f"{arm}/B{B}/{panel}"] = cell
        for panel in ("code", "math", "general"):
            target = by.get((B, panel, "dspark"))
            for base_arm in ("nospec", "ar"):
                control = by.get((B, panel, base_arm))
                if target is None or control is None:
                    continue
                for lhs, rhs in zip(target, control):
                    if [x["input_sha256"] for x in lhs["inputs"]] != [x["input_sha256"] for x in rhs["inputs"]]:
                        report["invalid"].append(f"B{B}/{panel}: {base_arm}/dspark input mismatch")
                t = [x["full_request_tps"] for x in target]
                c = [x["full_request_tps"] for x in control]
                ratios = [x / y for x, y in zip(t, c)]
                report["comparisons"][f"B{B}/{panel}/dspark_vs_{base_arm}"] = {
                    "paired_ratios": ratios, "median_paired_ratio": statistics.median(ratios),
                    "dspark_min_exceeds_control_max": min(t) > max(c)}
    report["decision"] = "incomplete_or_invalid"
    if not report["failures"] and not report["invalid"]:
        keys = [f"B{b}/{panel}/dspark_vs_{arm}" for b in (8, 16)
                for panel in ("code", "math", "general") for arm in ("nospec", "ar")]
        if all(k in report["comparisons"] for k in keys):
            report["decision"] = ("speed_gate_passed_review_still_required"
                                  if all(report["comparisons"][k]["dspark_min_exceeds_control_max"] for k in keys)
                                  else "speed_gate_not_met_keep_branch")
        else:
            report["decision"] = "partial_batches_only"
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "cells": len(report["cells"]),
                      "invalid": report["invalid"], "failures": report["failures"]}, indent=2))


if __name__ == "__main__":
    main()
