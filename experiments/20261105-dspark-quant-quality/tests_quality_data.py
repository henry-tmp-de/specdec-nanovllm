"""CPU-only manifest tests using synthetic trajectories, never real token IDs."""
import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_manifest_builder", HERE / "build_manifest.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture_data():
    workload = {"context_tokens": 1024, "panels": {}}
    runs = []
    for pi, panel in enumerate(builder.PANELS):
        entries, metadata, requests = [], [], []
        for i in range(8):
            prompt = [pi * 1_000_000 + i * 10_000 + k for k in range(1024)]
            base_meta = {"dataset": panel, "row": i, "input_sha256": builder.token_hash(prompt),
                         "task_tokens": 512, "wiki_tokens": 512, "task_truncated_from_left": False}
            entries.append({"token_ids": prompt, "metadata": base_meta})
            metadata.append({**base_meta, "source_context_tokens": 1024, "actual_context_tokens": 1024,
                             "actual_input_sha256": base_meta["input_sha256"], "wiki_fraction": 0.5})
            out = [pi * 10_000 + i * 300 + k for k in range(256)]
            requests.append({"seq_id": i + 1, "arrival_s": 0.0, "finish_s": 255.5,
                             "deliveries": [{"at_s": k + 0.5, "count": 1} for k in range(256)],
                             "output_token_ids": out, "output_sha256": builder.token_hash(out)})
        rounds = [{"start_s": k + 0.1, "end_s": k + 0.9, "prefill": k == 0, "active_b": 8, "delivered": 8,
                   "sequences": [{"seq_id": i + 1, "delivered": 1, "proposed": 0, "accepted_prefix": 0}
                                 for i in range(8)]} for k in range(256)]
        workload["panels"][panel] = entries
        runs.append({"type": "run", "arm": "nospec", "ctx": 1024, "requested_batch": 8,
                     "output_tokens": 256, "seed": 20261105, "temperature": 1.0, "ignore_eos": True,
                     "graph_mode": "graph", "engine_dirty": False, "repeat": 0, "panel": panel,
                     "engine_commit": "source-commit", "gpu_uuid": "source-gpu", "target_config_sha256": "config",
                     "target_provenance": {"path": "target"}, "graph_policy": {"name": "exact_all"},
                     "input_metadata": metadata, "requests": requests, "rounds": rounds,
                     "wall_s": 257.0, "delivered_tokens": 2048, "preemptions": 0, "prefix_cached_tokens": 0})
    return workload, runs


def test_manifest_has_exact_prefix_anchor_and_seven_labels():
    workload, runs = fixture_data()
    manifest, summary = builder.build(workload, runs, {"source_raw_sha256": "synthetic"})
    assert (summary["trajectories"], summary["cases"], summary["scored_positions"]) == (24, 96, 672)
    assert all(summary["panels"][p]["cases"] == 32 for p in builder.PANELS)
    for panel in builder.PANELS:
        for offset in builder.OFFSETS:
            case = next(c for c in manifest["cases"] if c["panel"] == panel and c["request_id"] == 1 and c["offset"] == offset)
            prompt = workload["panels"][panel][0]["token_ids"]
            output = next(r for r in runs if r["panel"] == panel)["requests"][0]["output_token_ids"]
            assert case["prefix_ids"] == prompt + output[:offset]
            assert case["anchor_id"] == output[offset]
            assert case["label_ids"] == output[offset + 1:offset + 8]
            assert case["context_tokens"] == 1024 + offset
            assert case["prefix_sha256"] == builder.token_hash(case["prefix_ids"])
    assert "prefix_ids" not in repr(summary) and "label_ids" not in repr(summary)


def test_workload_matching_uses_actual_hash_not_input_index():
    workload, runs = fixture_data()
    workload["panels"]["code"].reverse()
    manifest, _ = builder.build(workload, runs, {})
    case = next(c for c in manifest["cases"] if c["panel"] == "code" and c["request_id"] == 1 and c["offset"] == 0)
    assert case["prefix_ids"] == workload["panels"]["code"][-1]["token_ids"]
    runs[0]["input_metadata"][0]["actual_input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prompt hash"):
        builder.build(workload, runs, {})


@pytest.mark.parametrize("problem", ("source_setting", "output_hash", "duplicate_seq", "metadata", "wiki_fraction"))
def test_bad_source_is_not_silently_accepted(problem):
    workload, runs = fixture_data()
    if problem == "source_setting":
        runs[0]["temperature"] = 0.1
    elif problem == "output_hash":
        runs[0]["requests"][0]["output_sha256"] = "bad"
    elif problem == "duplicate_seq":
        runs[0]["requests"][1]["seq_id"] = runs[0]["requests"][0]["seq_id"]
    elif problem == "metadata":
        runs[0]["input_metadata"][0]["row"] = 7
    else:
        runs[0]["input_metadata"][0]["wiki_fraction"] = 0.2
    with pytest.raises(ValueError):
        builder.build(workload, runs, {})
