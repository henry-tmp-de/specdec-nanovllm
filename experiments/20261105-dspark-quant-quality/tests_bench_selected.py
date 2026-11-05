"""CPU-only fixed-weight benchmark configuration and provenance contracts."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import bench_selected as selected


@pytest.mark.parametrize("preset", selected.PRESETS)
@pytest.mark.parametrize("batch", (1, 8))
def test_fixed_native_configuration_matches_requested_preset(preset, batch):
    prior = selected.load_existing_wrapper()
    bench = prior.load_bench()
    selected.install_wrappers(bench, prior, preset)
    result = bench.options_for("dspark_ring", batch, 1024, 128, 512,
                               "target", "ar", "draft", batch * 1024, 0.85)
    assert result["dspark_quant_preset"] == preset
    assert result["spec_method"] == "dspark" and result["spec_k"] == 7
    assert result["dspark_window_tokens"] == 512 and result["dspark_window_storage"] == "ring"
    assert result["gpu_memory_utilization"] == 0.85 and result["max_model_len"] == 1160
    assert result["spec_cuda_graph"] and result["dspark_split_graph"]
    assert result["spec_graph_bs"] == result["decode_graph_bs"] == tuple(range(1, batch + 1))
    assert not result["spec_graph_pad"] and not result["dspark_graph_pad"]
    assert not result["spec_overlap_mixed"] and not result["enable_prefix_cache"]


def test_scope_and_budget_are_rejected_if_changed():
    prior = selected.load_existing_wrapper()
    bench = prior.load_bench()
    selected.install_wrappers(bench, prior, "q_o")
    for arm, batch, ctx, out, window, budget in (
        ("dspark_ring", 16, 1024, 128, 512, 0.85),
        ("dspark_ring", 1, 4096, 128, 512, 0.85),
        ("dspark_ring", 1, 1024, 256, 512, 0.85),
        ("dspark_ring", 1, 1024, 128, 1024, 0.85),
        ("dspark_full", 1, 1024, 128, 0, 0.85),
        ("dspark_ring", 1, 1024, 128, 512, 0.70),
    ):
        with pytest.raises(ValueError):
            bench.options_for(arm, batch, ctx, out, window, "target", "ar", "draft", 8192, budget)


def test_cli_fixed_panels_repeats_and_no_diagnostic_timing():
    arguments = ["--dspark-quant-preset", "q_o_down", "--batch", "8", "--target", "target",
                 "--dspark-model", "dspark", "--manifest", "workload", "--out", "out.jsonl"]
    parsed = selected.parse_args(arguments)
    argv = selected.native_argv(parsed)
    assert argv[argv.index("--panels") + 1:argv.index("--repeats")] == ["code", "math", "general"]
    assert argv[argv.index("--repeats") + 1] == "3"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.85"
    assert argv[argv.index("--phase") + 1] == "pilot"
    assert "--diagnostic-cost" not in argv and "--repeat" not in argv
    for bad in (arguments + ["--panel", "code"],
                arguments + ["--gpu-memory-utilization", "0.70"],
                [*arguments[:1], "ffn", *arguments[2:]],
                [*arguments[:3], "16", *arguments[4:]]):
        with pytest.raises(SystemExit):
            selected.parse_args(bad)


def test_actual_inventory_is_attached_only_after_loading():
    bench = SimpleNamespace(options_for=lambda *a, **kw: {"gpu_memory_utilization": 0.85},
                            run_once=lambda *a, **kw: {"wall_s": 2.0})
    seen = []
    prior = SimpleNamespace(runtime_inventory=lambda runner, preset:
                            seen.append((runner, preset)) or {"preset": preset, "quantized_modules": 15})
    records = selected.install_wrappers(bench, prior, "q_o_down")
    failure = json.loads(records.dumps({"type": "failure", "message": "load failed"}))
    assert failure["dspark_quant_preset"] == "q_o_down" and "dspark_quant" not in failure
    runner = object()
    result = bench.run_once(SimpleNamespace(model_runner=runner))
    assert seen == [(runner, "q_o_down")]
    for kind in ("warmup", "run", "failure"):
        row = json.loads(records.dumps({"type": kind, **result}))
        assert row["dspark_quant_preset"] == row["dspark_quant"]["preset"] == "q_o_down"
        assert row["dspark_quant"]["quantized_modules"] == 15


def test_main_dispatches_to_native_harness_with_fixed_arguments(monkeypatch):
    seen = []
    bench = SimpleNamespace(options_for=lambda *a, **kw: {"gpu_memory_utilization": 0.85},
                            run_once=lambda *a, **kw: {"wall_s": 2.0},
                            main=lambda: seen.append(list(sys.argv)))
    prior = SimpleNamespace(load_bench=lambda: bench,
                            runtime_inventory=lambda *args: {"preset": "down"})
    monkeypatch.setattr(selected, "load_existing_wrapper", lambda: prior)
    monkeypatch.setattr(sys, "argv", ["bench_selected.py", "--dspark-quant-preset", "down", "--batch", "1",
                                     "--target", "target", "--dspark-model", "draft", "--manifest", "data",
                                     "--out", "out"])
    selected.main()
    assert len(seen) == 1
    assert seen[0][0] == "bench_selected.py"
    assert seen[0][seen[0].index("--batch") + 1] == "1"
    assert seen[0][seen[0].index("--repeats") + 1] == "3"
    assert bench.options_for("dspark_ring", 1, 1024, 128, 512)["dspark_quant_preset"] == "down"
