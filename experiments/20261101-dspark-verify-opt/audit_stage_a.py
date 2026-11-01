"""Recompute stage A summaries and check paired workloads/tail accounting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


MODES = ("nospec-pad", "ar-pad", "dspark-original", "split-none", "split-target", "split-both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16])
    args = ap.parse_args()
    root = Path(args.results)
    manifest = json.loads(Path(args.manifest).read_text())
    errors, warnings, cells, groups = [], [], {}, {}
    warmups = {}
    for batch in args.batches:
        for arm in ("nospec-pad", "ar-pad", "dspark-original", "dspark-ablations"):
            path = root / f"stage-a-{arm}-b{batch}.jsonl"
            if not path.is_file():
                errors.append(f"{path.name}: missing")
                continue
            all_rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            warm = [x for x in all_rows if x.get("type") == "warmup"]
            bad = [x for x in all_rows if x.get("type") == "failure"]
            runs = [x for x in all_rows if x.get("type") == "run"]
            if bad or len(warm) != 1 or len(runs) != (27 if arm == "dspark-ablations" else 9):
                errors.append(f"{path.name}: unexpected warmup/run/failure counts")
                continue
            warmups[f"{arm}/B{batch}"] = warm[0]
            for mode in (MODES[3:] if arm == "dspark-ablations" else (arm,)):
                for panel in ("code", "math", "general"):
                    group = sorted((r for r in runs if r["mode"] == mode and r["panel"] == panel),
                                   key=lambda r: r["repeat"])
                    where = f"B{batch}/{panel}/{mode}"
                    if len(group) != 3 or [r["repeat"] for r in group] != [0, 1, 2]:
                        errors.append(f"{where}: expected three repeats")
                        continue
                    expected_hashes = [x["metadata"]["input_sha256"]
                                       for x in manifest["panels"][panel][:batch]]
                    for r in group:
                        label = f"{where}/r{r['repeat']}"
                        if (r["requested_batch"] != batch or r["ctx"] != 1024 or r["output_tokens"] != 256
                                or r["seed"] != 20261101 or r["gpu_memory_utilization"] != 0.70
                                or r["delivered_tokens"] != batch * 256 or r["preemptions"]
                                or r["prefix_cached_tokens"]):
                            errors.append(f"{label}: settings, capacity or output mismatch")
                        if [x["input_sha256"] for x in r["inputs"]] != expected_hashes:
                            errors.append(f"{label}: input hash mismatch")
                        if len(r["output_sha256"]) != batch:
                            errors.append(f"{label}: output hash count")
                        expected_flags = {"nospec-pad": (True, False, False),
                                          "ar-pad": (True, False, False),
                                          "dspark-original": (False, False, False),
                                          "split-none": (False, True, False),
                                          "split-target": (True, True, False),
                                          "split-both": (True, True, True)}[mode]
                        flags = (r["target_verify_graph_pad"], r["dspark_split_graph"], r["dspark_draft_graph_pad"])
                        if flags != expected_flags or (mode == "ar-pad" and not r["ar_draft_graph_pad"]):
                            errors.append(f"{label}: padding flags mismatch {flags}")
                        rounds, trace = r["rounds"], r["sequence_rounds"]
                        if len(rounds) != len(trace):
                            errors.append(f"{label}: unaligned round/sequence trace")
                            continue
                        next_remaining = {}
                        for i, (step, seqs) in enumerate(zip(rounds, trace)):
                            entries = seqs["sequences"]
                            if (seqs["prefill"] != step["prefill"] or len(entries) != step["active_b"]
                                    or sum(e["proposed"] for e in entries) != step["proposed"]
                                    or sum(e["accepted_prefix"] for e in entries) != step["accepted"]
                                    or len({e["seq_id"] for e in entries}) != len(entries)
                                    or any(e["remaining_before"] <= 0 or e["remaining_before"] > 256
                                           or e["accepted_prefix"] > e["proposed"] for e in entries)):
                                errors.append(f"{label}: bad sequence accounting in step {i}")
                                break
                            delivered_step = 0
                            for e in entries:
                                seq_id, remaining = e["seq_id"], e["remaining_before"]
                                if seq_id in next_remaining and remaining != next_remaining[seq_id]:
                                    errors.append(f"{label}: remaining token mismatch in step {i}")
                                delivered = min(remaining, 1 + e["accepted_prefix"])
                                delivered_step += delivered
                                next_remaining[seq_id] = remaining - delivered
                            if delivered_step != step["delivered"]:
                                errors.append(f"{label}: per-sequence delivery mismatch in step {i}")
                        dec = [x for x in rounds if not x["prefill"]]
                        hist = Counter(str(x["active_b"]) for x in dec)
                        if (not dec or dec[0]["active_b"] != batch or dict(hist) != r["active_b_rounds"]
                                or sum(x["delivered"] for x in rounds) != r["delivered_tokens"]
                                or sum(x["accepted"] for x in dec) != r["accepted_prefix_tokens"]
                                or sum(x["proposed"] for x in dec) != r["proposed_tokens"]):
                            errors.append(f"{label}: round accounting mismatch")
                        gd = r["graph_delta"]
                        if gd.get("graph_runtime_captures", 0) or gd.get("head_runtime_captures", 0):
                            warnings.append(f"{label}: runtime capture {gd}")
                    groups[(batch, panel, mode)] = group
                    values = [r["full_request_tps"] for r in group]
                    cells[where] = {"full_request_tps": values, "median": statistics.median(values),
                                    "min": min(values), "max": max(values),
                                    "decode_tps": [r["decode_tps"] for r in group],
                                    "full_batch_decode_tps": [r["full_batch_decode_tps"] for r in group],
                                    "ttft_median_s": [r["ttft_median_s"] for r in group],
                                    "tpot_median_s": [r["tpot_median_s"] for r in group],
                                    "acceptance_prefix_fraction": [r["acceptance_prefix_fraction"] for r in group],
                                    "output_hashes_stable_within_group": all(
                                        x["output_sha256"] == group[0]["output_sha256"] for x in group),
                                    "graph_deltas": [r["graph_delta"] for r in group],
                                    "active_b_rounds": [r["active_b_rounds"] for r in group],
                                    "peak_reserved_bytes": [r["peak_reserved_bytes"] for r in group]}
        original = warmups.get(f"dspark-original/B{batch}")
        split = warmups.get(f"dspark-ablations/B{batch}")
        if original and split:
            if original["prewarm_shapes"] != split["prewarm_shapes"]:
                errors.append(f"B{batch}: original and split backbone prewarm shapes differ")
            if len(split["prewarm_head_batches"]) != batch:
                errors.append(f"B{batch}: split head prewarm is incomplete")

    comparisons = {}
    for batch in args.batches:
        for panel in ("code", "math", "general"):
            for lhs, rhs in (("split-target", "split-none"), ("split-both", "split-target"),
                             ("split-both", "dspark-original"), ("split-both", "ar-pad"),
                             ("split-both", "nospec-pad")):
                l, r = groups.get((batch, panel, lhs)), groups.get((batch, panel, rhs))
                if not l or not r:
                    continue
                ratios = [x["full_request_tps"] / y["full_request_tps"] for x, y in zip(l, r)]
                comparisons[f"B{batch}/{panel}/{lhs}_vs_{rhs}"] = {
                    "paired_ratios": ratios, "median_ratio": statistics.median(ratios),
                    "all_repeat_outputs_match": [x["output_sha256"] == y["output_sha256"]
                                                 for x, y in zip(l, r)],
                    "min_lhs_exceeds_max_rhs": min(x["full_request_tps"] for x in l) >
                                               max(y["full_request_tps"] for y in r)}
    result = {"pass": not errors, "errors": errors, "warnings": warnings,
              "cells": cells, "comparisons": comparisons, "warmups": warmups}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "cell_count": len(cells), "errors": errors,
                      "warnings": warnings}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
