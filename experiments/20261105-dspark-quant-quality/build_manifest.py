"""Construct the fixed teacher-forced DSpark quality panel from local evidence.

The manifest contains token IDs and must remain in the ignored data directory.
The separate summary contains hashes and counts only.  This process does not
load a model or access a GPU. Source trajectories came from the historical native
target at T=1; a same-weight HF reference is not claimed numerically identical.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "experiments" / "20261105-dspark-window-resume" / "audit_results.py"
spec = importlib.util.spec_from_file_location("quality_previous_raw_audit", AUDIT)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

PANELS = ("code", "math", "general")
OFFSETS = (0, 64, 128, 192)
CTX = 1024
WINDOW = 512
BLOCK = 7
BATCH = 8
OUTPUT = 256
SEED = 20261105
SCHEMA = "dspark-quant-quality-manifest-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_hash(tokens):
    return hashlib.sha256(",".join(str(x) for x in tokens).encode("utf-8")).hexdigest()


def token_list(tokens, expected, what):
    if (not isinstance(tokens, list) or len(tokens) != expected or
            any(type(x) is not int or x < 0 for x in tokens)):
        raise ValueError(f"{what}: expected {expected} nonnegative integer tokens")
    return tokens


def workload_index(workload):
    if workload.get("context_tokens") != CTX or set(workload.get("panels", {})) != set(PANELS):
        raise ValueError("workload must contain the three 1024-token panels")
    result = {}
    for panel in PANELS:
        entries = workload["panels"][panel]
        if not isinstance(entries, list) or len(entries) < BATCH:
            raise ValueError(f"{panel}: too few workload entries")
        selected = {}
        for entry in entries:
            tokens = token_list(entry.get("token_ids"), CTX, f"{panel} prompt")
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{panel}: missing workload metadata")
            digest = token_hash(tokens)
            if digest != metadata.get("input_sha256"):
                raise ValueError(f"{panel}: workload prompt hash mismatch")
            if digest in selected:
                raise ValueError(f"{panel}: ambiguous duplicate prompt hash")
            selected[digest] = (tokens, metadata)
        result[panel] = selected
    return result


def selected_runs(raw_rows):
    picked = {}
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("type") != "run" or row.get("repeat") != 0:
            continue
        if row.get("panel") not in PANELS:
            continue
        if (row.get("arm") != "nospec" or row.get("ctx") != CTX or row.get("requested_batch") != BATCH
                or row.get("output_tokens") != OUTPUT or row.get("seed") != SEED
                or row.get("temperature") != 1.0 or row.get("ignore_eos") is not True
                or row.get("graph_mode") != "graph" or row.get("engine_dirty") is not False):
            raise ValueError("source run does not match target-only fixed settings")
        panel = row["panel"]
        if panel in picked:
            raise ValueError(f"duplicate repeat-0 source panel {panel}")
        try:
            raw_audit = old.compute_run(row)
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise ValueError(f"invalid source run {panel}: {exc}") from exc
        if not raw_audit["valid"]:
            raise ValueError(f"source run {panel} failed delivery audit: {raw_audit['errors']}")
        picked[panel] = row
    if set(picked) != set(PANELS):
        raise ValueError("expected one target-only repeat-0 run per panel")
    fields = ("engine_commit", "gpu_uuid", "target_config_sha256", "target_provenance", "graph_policy")
    for field in fields:
        values = [old.canonical(picked[panel].get(field)) for panel in PANELS]
        if len(set(values)) != 1 or picked[PANELS[0]].get(field) is None:
            raise ValueError(f"source run provenance inconsistent or missing: {field}")
    return picked


def build(workload, raw_rows, source):
    index = workload_index(workload)
    picked = selected_runs(raw_rows)
    cases, per_panel = [], {}
    for panel in PANELS:
        row = picked[panel]
        metadata, requests = row.get("input_metadata"), row.get("requests")
        if not isinstance(metadata, list) or not isinstance(requests, list) or len(metadata) != BATCH or len(requests) != BATCH:
            raise ValueError(f"{panel}: source inputs/requests do not have batch size {BATCH}")
        ids = [request.get("seq_id") for request in requests]
        if any(type(x) is not int for x in ids) or len(set(ids)) != BATCH or ids != sorted(ids):
            raise ValueError(f"{panel}: request IDs must be unique and in input order")
        used_prompts = set()
        wiki_fractions, datasets = [], set()
        for request, input_meta in zip(requests, metadata):
            if not isinstance(input_meta, dict):
                raise ValueError(f"{panel}: missing input metadata")
            prompt_hash = input_meta.get("actual_input_sha256")
            if prompt_hash in used_prompts or prompt_hash not in index[panel]:
                raise ValueError(f"{panel}: prompt hash missing, ambiguous or repeated")
            used_prompts.add(prompt_hash)
            prompt, workload_meta = index[panel][prompt_hash]
            for field in ("dataset", "row", "input_sha256", "task_tokens", "wiki_tokens", "task_truncated_from_left"):
                if field not in workload_meta or input_meta.get(field) != workload_meta[field]:
                    raise ValueError(f"{panel}: input metadata differs from workload: {field}")
            if input_meta.get("actual_context_tokens") != CTX or input_meta.get("source_context_tokens") != CTX:
                raise ValueError(f"{panel}: source/actual prompt length differs")
            wiki = input_meta.get("wiki_tokens")
            if type(wiki) is not int or not 0 <= wiki <= CTX:
                raise ValueError(f"{panel}: invalid Wiki fill count")
            fraction = input_meta.get("wiki_fraction")
            if type(fraction) not in (int, float) or not math.isfinite(fraction) or abs(fraction - wiki / CTX) > 1e-12:
                raise ValueError(f"{panel}: Wiki fill fraction differs from metadata")
            wiki_fractions.append(fraction)
            datasets.add(input_meta["dataset"])
            output = token_list(request.get("output_token_ids"), OUTPUT, f"{panel} output")
            if token_hash(output) != request.get("output_sha256"):
                raise ValueError(f"{panel}: target output hash mismatch")
            for offset in OFFSETS:
                prefix = prompt + output[:offset]
                anchor = output[offset]
                labels = output[offset + 1:offset + 1 + BLOCK]
                if len(prefix) != CTX + offset or len(labels) != BLOCK:
                    raise ValueError("teacher-forced case construction is out of bounds")
                cases.append({"case_id": f"{panel}-r{request['seq_id']}-o{offset:03d}",
                              "panel": panel, "request_id": request["seq_id"], "offset": offset,
                              "context_tokens": len(prefix), "prefix_ids": prefix, "anchor_id": anchor,
                              "label_ids": labels, "prompt_sha256": prompt_hash,
                              "output_sha256": request["output_sha256"], "prefix_sha256": token_hash(prefix),
                              "input_metadata": input_meta})
        per_panel[panel] = {"trajectories": len(requests), "cases": len(requests) * len(OFFSETS),
                            "scored_positions": len(requests) * len(OFFSETS) * BLOCK,
                            "datasets": sorted(datasets), "wiki_fraction_min": min(wiki_fractions),
                            "wiki_fraction_max": max(wiki_fractions)}
    if len(cases) != 96 or len({x["case_id"] for x in cases}) != len(cases):
        raise ValueError("expected 96 uniquely identified cases")
    representative = picked[PANELS[0]]
    provenance = {**source, "source_engine_commit": representative["engine_commit"],
                  "source_target_config_sha256": representative["target_config_sha256"],
                  "source_target_provenance": representative["target_provenance"],
                  "source_gpu_uuid": representative["gpu_uuid"], "source_graph_mode": representative["graph_mode"],
                  "source_graph_policy": representative["graph_policy"], "source_temperature": 1.0,
                  "source_sampling": "historical native target-only autoregressive samples; T=1, repeat 0",
                  "reference_boundary": "HF target uses the same weights; numerical identity to native target is not assumed"}
    manifest = {"schema": SCHEMA, "seed": SEED, "context_tokens": CTX, "window_tokens": WINDOW,
                "block_size": BLOCK, "offsets": list(OFFSETS), "provenance": provenance, "cases": cases}
    summary = {"schema": SCHEMA, "seed": SEED, "context_tokens": CTX, "window_tokens": WINDOW,
               "block_size": BLOCK, "offsets": list(OFFSETS), "provenance": provenance,
               "trajectories": len(PANELS) * BATCH, "cases": len(cases), "scored_positions": len(cases) * BLOCK,
               "offset_counts": dict(Counter(str(case["offset"]) for case in cases)), "panels": per_panel,
               "interpretation": "96 cases are four correlated slices of each of 24 target-sampled trajectories; "
                                 "Wiki-filled prompts, not human gold or standard language-model perplexity"}
    return manifest, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True, help="original 1K workload JSON")
    p.add_argument("--raw", required=True, help="native target-only B8 raw JSONL")
    p.add_argument("--out", required=True, help="new token-bearing manifest, keep under ignored data/")
    p.add_argument("--summary", required=True, help="new token-free summary JSON")
    args = p.parse_args()
    workload_path, raw_path = Path(args.workload), Path(args.raw)
    workload = json.loads(workload_path.read_text(encoding="utf-8-sig"))
    with raw_path.open(encoding="utf-8-sig") as stream:
        raw_rows = [json.loads(line) for line in stream if line.strip()]
    source = {"source_workload_path": str(workload_path.resolve()), "source_workload_sha256": sha256_file(workload_path),
              "source_raw_path": str(raw_path.resolve()), "source_raw_sha256": sha256_file(raw_path)}
    manifest, summary = build(workload, raw_rows, source)
    out, summary_out = Path(args.out), Path(args.summary)
    if out.exists() or summary_out.exists() or out.resolve() == summary_out.resolve():
        p.error("output files must be distinct and not already exist")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with out.open("xb") as stream:
        stream.write(encoded)
    summary["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    with summary_out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"已构造 {summary['trajectories']} 条轨迹、{summary['cases']} cases、{summary['scored_positions']} 评分位置；token IDs 仅存 {out}")


if __name__ == "__main__":
    main()
