"""Validate raw benchmark accounting and paired input IDs, without a GPU."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.results)
    manifest = json.loads(Path(a.manifest).read_text())
    errors, counts, output_stability = [], {}, {}
    expected = [("formal", b, arm) for b in (1, 2, 4, 8, 16) for arm in ("nospec", "ar", "dspark")]
    expected += [("memory70", 16, arm) for arm in ("nospec", "ar", "dspark")]
    for tag, batch, arm in expected:
        key = f"{tag}/{arm}/B{batch}"
        path = root / f"{tag}-{arm}-b{batch}.jsonl"
        if not path.exists():
            errors.append(f"{key}: missing file")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        runs = [r for r in rows if r.get("type") == "run"]
        failures = [r for r in rows if r.get("type") == "failure"]
        if tag == "formal" and batch == 16 and arm == "dspark":
            if len(failures) != 1 or "out of memory" not in failures[0]["message"].lower() or runs:
                errors.append(f"{key}: original capacity failure missing or altered")
            counts[key] = {"runs": 0, "expected_failure": True}
            continue
        if failures or len(runs) != 9:
            errors.append(f"{key}: expected 9 successful runs and no failures")
            continue
        counts[key] = {"runs": len(runs), "expected_failure": False}
        for panel in ("code", "math", "general"):
            group = sorted((r for r in runs if r.get("panel") == panel), key=lambda r: r["repeat"])
            if len(group) != 3 or [r["repeat"] for r in group] != [0, 1, 2]:
                errors.append(f"{key}/{panel}: repeat count")
                continue
            expected_ids = [r["metadata"]["input_sha256"] for r in manifest["panels"][panel][:batch]]
            output_stability[f"{key}/{panel}"] = all(
                x["output_sha256"] == group[0]["output_sha256"] for x in group)
            for r in group:
                where = f"{key}/{panel}/{r['repeat']}"
                if (r["ctx"] != 1024 or r["output_tokens"] != 256 or r["requested_batch"] != batch
                        or r["seed"] != 20261101 or r["k"] != {"nospec": 0, "ar": 6, "dspark": 7}[arm]):
                    errors.append(f"{where}: settings mismatch")
                if tag == "memory70" and r.get("gpu_memory_utilization") != 0.70:
                    errors.append(f"{where}: memory budget mismatch")
                if [x["input_sha256"] for x in r["inputs"]] != expected_ids:
                    errors.append(f"{where}: prompt fingerprint mismatch")
                rounds = r["rounds"]
                dec = [x for x in rounds if not x["prefill"]]
                if (r["preemptions"] or r["prefix_cached_tokens"] or not dec or dec[0]["active_b"] != batch
                        or r["delivered_tokens"] != batch * 256
                        or sum(x["delivered"] for x in rounds) != r["delivered_tokens"]
                        or sum(x["proposed"] for x in dec) != r["proposed_tokens"]
                        or sum(x["accepted"] for x in dec) != r["accepted_prefix_tokens"]
                        or r["accepted_prefix_tokens"] > r["proposed_tokens"]):
                    errors.append(f"{where}: batch, progress or token accounting mismatch")
                actual_hist = {}
                for x in dec:
                    k = str(x["active_b"])
                    actual_hist[k] = actual_hist.get(k, 0) + 1
                if actual_hist != r["active_b_rounds"]:
                    errors.append(f"{where}: active batch histogram mismatch")
                if arm != "nospec" and r["graph_delta"].get("verify_hits", 0) <= 0:
                    errors.append(f"{where}: no target verify graph hits")
                draft_hits = (r["graph_delta"].get("graph_hits", 0) if arm == "dspark"
                              else r["graph_delta"].get("n_graphed", 0))
                if arm != "nospec" and draft_hits <= 0:
                    errors.append(f"{where}: no draft graph hits")
    result = {"pass": not errors, "errors": errors, "files": counts,
              "outputs_identical_across_repeats": output_stability,
              "note": "output identity across methods is not a correctness requirement"}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "files": len(counts), "errors": errors,
                      "stable_repeat_groups": sum(output_stability.values()),
                      "total_repeat_groups": len(output_stability)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
