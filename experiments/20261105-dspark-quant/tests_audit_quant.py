"""Synthetic CPU fixtures for preset isolation, paired inputs and raw metrics."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_cpu_summary", HERE / "audit_quant.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def run(preset="none", repeat=0, wall=10.0, input_hash="a" * 64):
    ids = [7, 8, 9, 10]
    return {"type": "run", "arm": "dspark_ring", "mode": "ring", "ctx": 1024,
            "requested_batch": 1, "panel": "code", "repeat": repeat, "output_tokens": 4,
            "window": 512, "graph_mode": "graph", "phase": "pilot", "seed": 3,
            "temperature": 1.0, "ignore_eos": True, "spec_k": 7,
            "gpu_uuid": "GPU-test", "engine_commit": "commit", "engine_dirty": False,
            "gpu_memory_utilization": 0.85, "max_num_seqs": 1, "max_model_len": 1036,
            "max_num_batched_tokens": 1036, "prefix_lookup": False, "mixed_overlap": False,
            "graph_policy": {"name": "exact_all", "buckets": [1]},
            "target_config_sha256": "target", "draft_config_sha256": "draft",
            "target_provenance": {"path": "target"}, "draft_provenance": {"path": "draft"},
            "guard_id": f"{preset}-guard", "diagnostic_instrumented": False,
            "input_metadata": [{"actual_input_sha256": input_hash}],
            "dspark_quant_preset": preset, "dspark_quant": {"preset": preset, "candidate_total_bytes": 100,
                                                              "quantize_host_wall_s": 6.5},
            "target_kv_blocks": 20, "load_initial_graph_s": 8.0,
            "wall_s": wall, "delivered_tokens": 4, "preemptions": 0, "prefix_cached_tokens": 0,
            "requests": [{"seq_id": 1, "arrival_s": 0.1, "finish_s": 4.0,
                          "deliveries": [{"at_s": 1.0, "count": 1}, {"at_s": 3.0, "count": 2},
                                         {"at_s": 4.0, "count": 1}],
                          "output_token_ids": ids, "output_sha256": audit.old.token_hash(ids)}],
            "rounds": [{"start_s": 0.2, "end_s": 1.1, "prefill": True, "active_b": 1, "delivered": 1,
                        "sequences": [{"seq_id": 1, "delivered": 1, "proposed": 0, "accepted_prefix": 0}]},
                       {"start_s": 1.2, "end_s": 3.1, "prefill": False, "active_b": 1, "delivered": 2,
                        "sequences": [{"seq_id": 1, "delivered": 2, "proposed": 2, "accepted_prefix": 1, "remaining_before": 3}]},
                       {"start_s": 3.2, "end_s": 4.1, "prefill": False, "active_b": 1, "delivered": 1,
                        "sequences": [{"seq_id": 1, "delivered": 1, "proposed": 0, "accepted_prefix": 0, "remaining_before": 1}]}]}


def rows(records):
    return [("fixture", index + 1, record) for index, record in enumerate(records)]


def test_presets_and_three_repeat_pairing_keep_inputs_separate():
    data = [run("none", i) for i in range(3)] + [run("ffn", i, wall=5.0) for i in range(3)]
    result = audit.summarize(rows(data))
    assert len(result["groups"]) == 2
    assert all(group["three_repeats_valid"] for group in result["groups"])
    pair = audit.pair_formal(result)[0]
    assert pair["summary"] == {"values": [2.0, 2.0, 2.0], "median": 2.0, "min": 2.0, "max": 2.0}
    candidate = next(group for group in result["groups"] if group["configuration"]["dspark_quant_preset"] == "ffn")
    metrics = candidate["runs"][0]["metrics"]
    assert metrics["accepted_prefix_tokens"] == 1 and metrics["proposed_tokens"] == 2
    assert abs(metrics["decode_span_s"] - 2.9) < 1e-10
    assert abs(metrics["decode_s"] - 2.8) < 1e-10
    data[-1]["input_metadata"][0]["actual_input_sha256"] = "b" * 64
    changed = audit.pair_formal(audit.summarize(rows(data)))
    assert all(item["summary"] is None for item in changed)


def test_mismatched_preset_and_failed_guard_cannot_pass_pairing():
    data = [run("none", i) for i in range(3)] + [run("ffn", i, wall=5.0) for i in range(3)]
    data[-1]["dspark_quant"]["preset"] = "all"
    result = audit.summarize(rows(data))
    assert result["invalid"]
    assert all(item["summary"] is None for item in audit.pair_formal(result))
    data[-1]["dspark_quant"]["preset"] = "ffn"
    data.append({"type": "failure", "guard_id": "ffn-guard", "dspark_quant_preset": "ffn", "message": "OOM"})
    result = audit.summarize(rows(data))
    assert len(result["failures"]) == 1
    assert audit.pair_formal(result)[0]["pairs"] == []


def test_micro_ranges_keep_slowdowns_and_failure():
    row = {"type": "micro", "correctness_pass": True, "bf16_ms_per_call": [1.0, 1.0, 1.0],
           "int8_ms_per_call": [2.0, 2.0, 2.0]}
    report = audit.summarize_micro(rows([row, {"type": "failure", "error": "OOM"}]))
    assert report["measurements"][0]["paired_bf16_over_int8"]["median"] == 0.5
    assert len(report["failures"]) == 1
    bad = copy.deepcopy(row)
    bad["correctness_pass"] = False
    assert audit.summarize_micro(rows([bad]))["measurements"][0]["paired_bf16_over_int8"] is None


def test_missing_actual_hash_or_bad_repeat_never_joins_a_pair():
    data = [run("none", i) for i in range(3)] + [run("ffn", i, wall=5.0) for i in range(3)]
    del data[-1]["input_metadata"][0]["actual_input_sha256"]
    data[-2]["repeat"] = []
    result = audit.summarize(rows(data))
    assert len(result["invalid"]) == 2
    assert all(item["summary"] is None for item in audit.pair_formal(result))


def test_cli_reads_glob_and_writes_formal_reports(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for preset, wall in (("none", 10.0), ("ffn", 5.0)):
        with (raw / f"{preset}.jsonl").open("w", encoding="utf-8") as stream:
            for repeat in range(3):
                stream.write(json.dumps(run(preset, repeat, wall)) + "\n")
    output = tmp_path / "summary"
    result = subprocess.run([sys.executable, "-B", str(HERE / "audit_quant.py"), "--kind", "formal",
                             "--input", str(raw / "*.jsonl"), "--out-dir", str(output)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "formal-summary.json").read_text(encoding="utf-8"))
    assert len(report["groups"]) == 2 and report["pairs"][0]["summary"]["median"] == 2.0
    assert (output / "formal-runs.csv").is_file() and (output / "formal-pairs.csv").is_file()
    markdown = (output / "formal-report.md").read_text(encoding="utf-8")
    assert "不是置信区间" in markdown and "| 6.5 | 8 | 20 | 100 |" in markdown
