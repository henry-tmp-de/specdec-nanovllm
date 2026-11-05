"""Synthetic CPU tests for selected native preset identity and pairing."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_selected_audit", HERE / "audit_selected.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def inventory(preset, selected=None, n=16):
    chosen = audit.EXPECTED[preset] if selected is None else selected
    modules, loading = [], []
    for name in sorted(audit.ALL_NAMES):
        quantized = name in chosen
        k = 8
        weight_bytes = n * k * (1 if quantized else 2)
        scale_bytes = n * 2 if quantized else 0
        item = {"name": name, "shape": [n, k], "quantized": quantized,
                "weight_dtype": "torch.int8" if quantized else "torch.bfloat16", "weight_bytes": weight_bytes,
                "scale_dtype": "torch.bfloat16" if quantized else None,
                "scale_shape": [n] if quantized else None, "scale_bytes": scale_bytes,
                "total_bytes": weight_bytes + scale_bytes}
        modules.append(item)
        if quantized:
            loading.append({"name": name, "shape": [n, k], "weight_dtype_before": "bfloat16",
                            "weight_dtype_after": "int8", "scale_dtype": "bfloat16", "scale_shape": [n],
                            "weight_bytes": weight_bytes, "scale_bytes": scale_bytes,
                            "bytes_before": n * k * 2, "bytes_after": weight_bytes + scale_bytes})
    return {"preset": preset, "quantize_host_wall_s": 1.0, "quantized_modules": len(chosen),
            "candidate_modules": modules, "candidate_total_bytes": sum(x["total_bytes"] for x in modules),
            "loading_stats": loading,
            "noncandidate_parameters": [{"name": name, "dtype": "torch.bfloat16"} for name in
                                        ("fc.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight")],
            "shared_target_weights": [{"name": name, "dtype": "torch.bfloat16", "aliases_target": True}
                                      for name in ("embedding_weight", "lm_head_weight")]}


def run(preset="none", repeat=0, wall=300.0, batch=1):
    requests = []
    for i in range(batch):
        ids = list(range(i * 128, (i + 1) * 128))
        requests.append({"seq_id": i + 1, "arrival_s": 0.0, "finish_s": 127.5,
                         "deliveries": [{"at_s": k + 0.5, "count": 1} for k in range(128)],
                         "output_token_ids": ids, "output_sha256": audit.prior.old.token_hash(ids)})
    rounds = [{"start_s": k + 0.1, "end_s": k + 0.9, "prefill": k == 0, "active_b": batch,
               "delivered": batch, "sequences": [{"seq_id": i + 1, "delivered": 1,
                            "proposed": 0, "accepted_prefix": 0, "remaining_before": 128 - k}
                           for i in range(batch)]} for k in range(128)]
    policy = {"name": "exact_all", "target_decode_padding": False, "target_verify_padding": False,
              "draft_padding": False, "buckets": list(range(1, batch + 1))}
    return {"type": "run", "arm": "dspark_ring", "mode": "ring", "ctx": 1024,
            "requested_batch": batch, "panel": "code", "repeat": repeat, "output_tokens": 128,
            "window": 512, "graph_mode": "graph", "phase": "pilot", "seed": 20261105,
            "temperature": 1.0, "ignore_eos": True, "spec_k": 7,
            "gpu_uuid": "GPU-test", "engine_commit": "commit", "engine_dirty": False,
            "gpu_memory_utilization": 0.85, "max_num_seqs": batch, "max_model_len": 1160,
            "max_num_batched_tokens": 1024 * batch, "prefix_lookup": False, "mixed_overlap": False,
            "prefix_lookup_count": 0, "prefix_hit_blocks": 0, "graph_policy": policy,
            "target_config_sha256": "target", "draft_config_sha256": "draft",
            "target_provenance": {"path": "target"}, "draft_provenance": {"path": "draft"},
            "guard_id": f"{preset}-guard", "diagnostic_instrumented": False,
            "input_metadata": [{"actual_input_sha256": f"{i + 1:064x}"} for i in range(batch)],
            "dspark_quant_preset": preset, "dspark_quant": inventory(preset),
            "target_kv_blocks": 300 if preset == "none" else 280, "load_initial_graph_s": 10.0,
            "graph_delta": {"graph_hits": 5, "graph_exact_hits": 5, "graph_padded_hits": 0,
                            "verify_hits": 3, "verify_exact_hits": 3, "verify_padded_hits": 0,
                            "head_graph_hits": 35, "graph_runtime_captures": 0, "head_runtime_captures": 0,
                            "graph_fallbacks": {}, "verify_fallbacks": {"ragged_draft_len": 2}},
            "target_decode_graph": {"exact": 0, "padded": 0, "fallbacks": {}, "bucket_histogram": {}},
            "wall_s": wall, "delivered_tokens": 128 * batch, "preemptions": 0, "prefix_cached_tokens": 0,
            "requests": requests, "rounds": rounds}


def rows(records):
    return [("fixture", index + 1, row) for index, row in enumerate(records)]


def test_all_presets_match_three_repeats_despite_different_actual_kv():
    data = [run(preset, i, 300 if preset == "none" else 150)
            for preset in audit.PRESETS for i in range(3)]
    result = audit.prior.summarize(rows(data))
    assert len(result["groups"]) == 4 and not result["invalid"]
    assert all(group["three_repeats_valid"] for group in result["groups"])
    pairs = audit.pair_selected(result)
    assert {p["configuration"]["dspark_quant_preset"] for p in pairs} == set(audit.PRESETS[1:])
    assert all(p["summary"]["median"] == 2.0 for p in pairs)
    assert {len(g["configuration"]["quantized_module_names"]) for g in result["groups"]} == {0, 5, 10, 15}
    assert all(g["runs"][0]["graph_delta"]["verify_fallbacks"] == {"ragged_draft_len": 2} for g in result["groups"])


def test_negative_speed_and_actual_selected_names_cannot_be_faked():
    data = [run("none", i) for i in range(3)] + [run("q_o", i, 450) for i in range(3)]
    result = audit.prior.summarize(rows(data))
    assert audit.pair_selected(result)[0]["summary"]["median"] == 300 / 450
    wrong = {f"layers.{layer}.{suffix}" for layer in range(5)
             for suffix in ("self_attn.q_proj", "self_attn.k_proj")}
    data[-1]["dspark_quant"] = inventory("q_o", selected=wrong)
    result = audit.prior.summarize(rows(data))
    assert any("actual selected layer names" in e for invalid in result["invalid"] for e in invalid["errors"])
    assert all(p["summary"] is None for p in audit.pair_selected(result))


def test_bytes_unexpected_graph_fallback_and_scope_fail_closed():
    data = [run("none", i) for i in range(3)] + [run("down", i, 150) for i in range(3)]
    data[-1]["dspark_quant"]["candidate_total_bytes"] += 1
    result = audit.prior.summarize(rows(data))
    assert any("candidate_total_bytes" in e for item in result["invalid"] for e in item["errors"])
    data[-1] = run("down", 2, 150)
    data[-1]["graph_delta"]["verify_fallbacks"]["uncaptured_shape"] = 1
    data[-2]["gpu_memory_utilization"] = 0.70
    result = audit.prior.summarize(rows(data))
    assert len(result["invalid"]) == 2
    assert all(p["summary"] is None for p in audit.pair_selected(result))


def test_input_and_inventory_splits_do_not_merge():
    data = [run("none", i) for i in range(3)] + [run("q_o_down", i, 150) for i in range(3)]
    data[-1]["input_metadata"][0]["actual_input_sha256"] = "b" * 64
    result = audit.prior.summarize(rows(data))
    assert all(p["summary"] is None for p in audit.pair_selected(result))
    data[-1] = run("q_o_down", 2, 150)
    data[-1]["dspark_quant"] = inventory("q_o_down", n=32)
    result = audit.prior.summarize(rows(data))
    assert len(result["groups"]) == 3 and not result["invalid"]
    assert all(p["summary"] is None for p in audit.pair_selected(result))
    for row in data[3:]:
        row["dspark_quant"] = inventory("q_o_down", n=32)
    result = audit.prior.summarize(rows(data))
    pair = audit.pair_selected(result)[0]
    assert len(result["groups"]) == 2 and not result["invalid"]
    assert pair["summary"] is None and "35层形状不一致" in pair["status"]


def test_failure_blocks_guard_and_missing_metadata_is_retained():
    data = [run("none", i) for i in range(3)] + [run("down", i, 150) for i in range(3)]
    data.append({"type": "failure", "guard_id": "down-guard", "dspark_quant_preset": "down", "message": "OOM"})
    result = audit.prior.summarize(rows(data))
    assert len(result["failures"]) == 1 and audit.pair_selected(result)[0]["summary"] is None
    data[-2]["dspark_quant"].pop("candidate_modules")
    result = audit.prior.summarize(rows(data))
    assert result["invalid"] and len(result["failures"]) == 1


def test_quality_records_cannot_mix_into_native_performance_report():
    data = [run("none", i) for i in range(3)] + [run("down", i, 150) for i in range(3)]
    data.append({"type": "score", "variant": "down", "steps": []})
    result = audit.prior.summarize(rows(data))
    assert len(result["other"]) == 1 and result["invalid"]
    assert all(not group["three_repeats_valid"] for group in result["groups"])
    assert all(pair["summary"] is None for pair in audit.pair_selected(result))


@pytest.mark.parametrize("problem", ("loading", "excluded", "shared", "decode"))
def test_malformed_inventory_and_fallback_are_reported_without_crash(problem):
    row = run("down")
    if problem == "loading":
        row["dspark_quant"]["loading_stats"][0]["name"] = []
    elif problem == "excluded":
        row["dspark_quant"]["noncandidate_parameters"][0]["name"] = []
    elif problem == "shared":
        row["dspark_quant"]["shared_target_weights"][0]["name"] = []
    else:
        row["target_decode_graph"]["fallbacks"] = {"bad": None}
    result = audit.prior.summarize(rows([row]))
    assert len(result["invalid"]) == 1 and not result["groups"][0]["three_repeats_valid"]


def test_cli_uses_real_preset_labels_and_exports_unchanged_pair_stats(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for preset in audit.PRESETS:
        with (raw / f"{preset}.jsonl").open("w", encoding="utf-8") as stream:
            for repeat in range(3):
                stream.write(json.dumps(run(preset, repeat, 300 if preset == "none" else 150)) + "\n")
    out = tmp_path / "summary"
    proc = subprocess.run([sys.executable, "-B", str(HERE / "audit_selected.py"), "--input", str(raw / "*.jsonl"),
                           "--out-dir", str(out)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out / "formal-summary.json").read_text(encoding="utf-8"))
    assert len(report["groups"]) == 4 and len(report["pairs"]) == 3 and not report["invalid"]
    assert {p["configuration"]["dspark_quant_preset"] for p in report["pairs"]} == set(audit.PRESETS[1:])
    assert all(p["summary"]["median"] == 2.0 for p in report["pairs"])
    markdown = (out / "formal-report.md").read_text(encoding="utf-8")
    assert "ragged_draft_len" in markdown and "不是置信区间" in markdown
