"""Synthetic CPU contracts for complete quality reports and failure handling."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_report", HERE / "report_quality.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_manifest():
    cases = []
    for pi, panel in enumerate(report.PANELS):
        for request in range(8):
            prompt = [pi * 100000 + request * 2000 + i for i in range(1024)]
            for offset in report.OFFSETS:
                prefix = prompt + list(range(offset))
                cases.append({"case_id": f"{panel}-{request}-{offset}", "panel": panel, "request_id": request,
                              "offset": offset, "context_tokens": len(prefix), "prefix_ids": prefix,
                              "anchor_id": offset, "label_ids": list(range(offset + 1, offset + 8)),
                              "prompt_sha256": report.token_hash(prompt), "prefix_sha256": report.token_hash(prefix),
                              "output_sha256": "c" * 64})
    target = {"path": "synthetic", "config_sha256": "config", "weight_files": [{"name": "w", "bytes": 10}]}
    return {"schema": report.SCHEMA, "seed": report.SEED, "context_tokens": report.CTX,
            "window_tokens": report.WINDOW, "block_size": report.BLOCK, "offsets": list(report.OFFSETS),
            "provenance": {"source_target_provenance": target}, "cases": cases}


def fixture_raw(manifest, manifest_sha, phase):
    limit = 2 if phase == "smoke" else 96
    identity = {"guard_id": "guard", "gpu_uuid": "gpu", "engine_commit": "commit", "engine_dirty": False,
                "phase": phase, "manifest_sha256": manifest_sha, "manifest_provenance": manifest["provenance"]}
    variants = {key: sorted(values) for key, values in report.expected_modules().items()}
    header = {"type": "header", **identity, "limit": limit, "manifest_schema": report.SCHEMA,
              "seed": report.SEED, "temperature": 1.0, "window_tokens": 512,
              "target_layer_ids": [1, 9, 17, 25, 33], "hf_attention_implementation": "sdpa",
              "target_provenance": manifest["provenance"]["source_target_provenance"],
              "dspark_provenance": {"path": "synthetic", "config_sha256": "draft", "weight_files": [{"name": "d", "bytes": 3}]},
              "variants": variants}
    raw = [{"type": "start", **identity, "limit": limit, "case_count_expected": limit,
            "variant_count_expected": 15}, header]
    for case in manifest["cases"][:limit]:
        common = {key: case[key] for key in ("case_id", "panel", "request_id", "offset", "context_tokens", "prefix_sha256")}
        common["feature_sha256"] = "d" * 64
        for variant, names in variants.items():
            delta = 0.0 if variant == "none" else -0.1 if variant == "q_o_down" else 0.5
            steps = []
            for position in range(1, 8):
                baseline = 1.0 + position / 10
                steps.append({"position": position, "nll_quant": baseline + delta, "nll_bf16": baseline,
                              "nll_target": 0.8 + position / 20, "kl_bf16_quant": 0.0 if variant == "none" else 0.01,
                              "kl_target_quant": 0.3, "overlap_target_quant": 0.8,
                              "argmax_agree_bf16": True, "argmax_agree_target": False})
            raw.append({"type": "score", **common, "variant": variant, "quantized_modules": len(names),
                        "cache_slots": [0, 511], "cache_absolute_positions": [case["context_tokens"] - 512,
                                                                                case["context_tokens"] - 1],
                        "steps": steps, "diagnostics": {"logits_finite": True, "scores_finite": True,
                                                          "target_norm_max_abs": 0.0, "bf16_norm_max_abs": 0.0,
                                                          "quant_norm_max_abs": 0.0}})
        raw.append({"type": "replay_check", **common, "pass": True, "nll_max_abs_diff": 0.0, "kl_max_abs": 0.0,
                    "nll_atol": 1e-4, "kl_atol": 1e-6,
                    "diagnostics": {"logits_finite": True, "scores_finite": True, "target_norm_max_abs": 0.0,
                                    "bf16_norm_max_abs": 0.0, "quant_norm_max_abs": 0.0}})
    raw.append({"type": "complete", **identity, "case_count": limit, "score_rows": limit * 15,
                "locations_per_variant": limit * 7, "variants": 15})
    return [(index + 1, item) for index, item in enumerate(raw)]


def test_full_groups_use_token_weighting_and_include_speed_candidate():
    manifest = fixture_manifest()
    result = report.analyze(manifest, "e" * 64, fixture_raw(manifest, "e" * 64, "full"), "full")
    assert result["valid_raw"], result["errors"]
    assert result["counts"] == {"cases": 96, "trajectories": 24, "variants": 15, "scores": 1440,
                                "replays": 96, "positions_per_variant": 672}
    assert len(result["groups"]) == 15 * (1 + 3 + 7 + 3 * 7)
    overall = next(g for g in result["groups"] if g["scope"] == "overall" and g["variant"] == "ffn")
    assert overall["tokens"] == 672 and overall["mean_nll"] == pytest.approx(1.9)
    assert overall["conditional_nll_exp"] == pytest.approx(math.exp(1.9))
    assert overall["relative_exp_delta_nll"] == pytest.approx(math.exp(0.5))
    candidate = next(g for g in result["groups"] if g["scope"] == "panel_position" and g["panel"] == "math"
                     and g["position"] == 7 and g["variant"] == "q_o_down")
    assert candidate["tokens"] == 32 and candidate["trajectories"] == 8
    assert candidate["relative_exp_delta_nll"] == pytest.approx(math.exp(-0.1))


@pytest.mark.parametrize("problem", ("feature", "duplicate", "missing", "norm", "replay", "source", "failure"))
def test_smoke_never_summarizes_incomplete_or_mismatched_raw(problem):
    manifest = fixture_manifest()
    rows = fixture_raw(manifest, "e" * 64, "smoke")
    if problem == "feature":
        rows[3][1]["feature_sha256"] = "e" * 64
    elif problem == "duplicate":
        rows.append((999, copy.deepcopy(rows[2][1])))
    elif problem == "missing":
        rows.pop(2)
    elif problem == "norm":
        rows[2][1]["diagnostics"]["quant_norm_max_abs"] = 0.1
    elif problem == "replay":
        next(row for _, row in rows if row.get("type") == "replay_check")["pass"] = False
    elif problem == "source":
        rows[1][1]["target_provenance"] = {"config_sha256": "different", "weight_files": [{"name": "w", "bytes": 10}]}
    else:
        rows.append((1000, {"type": "failure", "message": "synthetic failure"}))
    result = report.analyze(manifest, "e" * 64, rows, "smoke")
    assert not result["valid_raw"] and result["errors"] and result["groups"] == []
    if problem == "failure":
        assert len(result["failures"]) == 1


def test_cli_writes_token_free_smoke_reports(tmp_path):
    manifest = fixture_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for _, row in fixture_raw(manifest, digest, "smoke")), encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-B", str(HERE / "report_quality.py"), "--manifest", str(manifest_path),
                           "--raw", str(raw), "--phase", "smoke", "--out-dir", str(out)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    summary_text = (out / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["valid_raw"] and summary["counts"]["scores"] == 30
    assert "prefix_ids" not in summary_text and "label_ids" not in summary_text
    assert "单步分布重叠" in (out / "report.md").read_text(encoding="utf-8")
