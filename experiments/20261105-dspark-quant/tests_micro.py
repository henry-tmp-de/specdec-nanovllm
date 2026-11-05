"""CPU-only microbenchmark contracts; no CUDA or real-model claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy  # noqa: F401 -- initialize before local CPU torch
import pytest
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("dspark_micro_cpu", HERE / "gpu_micro.py")
micro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(micro)


def guard_env():
    gpu = "GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91"
    return {"DS_GUARD_ID": "41f55249-ff2f-47f4-8301-270cd9c302b9", "DS_GPU_UUID": gpu,
            "CUDA_VISIBLE_DEVICES": gpu, "DS_RUN_DEADLINE_UNIX": "200"}


def test_guard_identity_and_deadline_fail_closed():
    env = guard_env()
    assert micro.guard_environment(env, now=100)["gpu_uuid"] == env["DS_GPU_UUID"]
    for key, value in (("CUDA_VISIBLE_DEVICES", "0"), ("DS_GPU_UUID", "GPU-00000000-0000-0000-0000-000000000000"),
                       ("DS_RUN_DEADLINE_UNIX", "nan"), ("DS_RUN_DEADLINE_UNIX", "100"),
                       ("DS_GUARD_ID", "not-a-uuid")):
        bad = {**env, key: value}
        with pytest.raises(RuntimeError):
            micro.guard_environment(bad, now=100)
    del env["DS_GUARD_ID"]
    with pytest.raises(RuntimeError):
        micro.guard_environment(env, now=100)


def test_checkpoint_shape_relationships():
    valid = {"gate": [12, 8], "down": [8, 12], "q": [16, 8], "k": [4, 8], "o": [8, 16]}
    micro.validate_shapes(valid)
    with pytest.raises(ValueError):
        micro.validate_shapes({**valid, "down": [12, 8]})
    with pytest.raises(ValueError):
        micro.validate_shapes({key: value for key, value in valid.items() if key != "q"})


def test_same_q_reference_uses_post_dot_scale_and_nonfinite_fails():
    x = torch.tensor([[0.87890625, 0.734375, -0.6875]], dtype=torch.bfloat16)
    q = torch.tensor([[127, -121, 83]], dtype=torch.int8)
    scale = torch.tensor([0.017], dtype=torch.bfloat16)
    ref = micro.quantized_reference(x, q, scale, torch)
    expected = (x.float() @ q.float().T) * scale.float()
    predequant = x.float() @ (q.to(torch.bfloat16) * scale[:, None]).float().T
    assert torch.equal(ref, expected)
    assert not torch.equal(ref, predequant)
    assert micro.errors(ref, expected, torch)["relative_rms"] == 0
    bad = ref.clone()
    bad[0, 0] = float("nan")
    assert not micro.errors(bad, expected, torch)["finite"]
    assert micro.QUANT_RELATIVE_RMS_LIMIT == 0.01
    assert (micro.GRAPH_RTOL, micro.GRAPH_ATOL) == (0.016, 1e-5)


def test_cpu_row_chunk_quantizer_matches_engine(monkeypatch):
    # Import the real quantizer with only Triton decorators stubbed: no kernel is run.
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    quant_spec = importlib.util.spec_from_file_location("micro_real_quantizer", ROOT / "nanovllm/layers/quant.py")
    quant = importlib.util.module_from_spec(quant_spec)
    quant_spec.loader.exec_module(quant)
    weight = torch.randn(7, 9, generator=torch.Generator().manual_seed(121), dtype=torch.bfloat16)
    q, scale = micro.quantize_cpu(weight, quant.quantize_weight, torch, chunk_rows=3)
    expected_q, expected_scale = quant.quantize_weight(weight, "per_channel")
    assert torch.equal(q, expected_q) and torch.equal(scale, expected_scale)
    assert q.dtype == torch.int8 and scale.dtype == torch.bfloat16
    assert micro.tensor_sha256(q, torch) == hashlib.sha256(q.numpy().tobytes()).hexdigest()
    weight[1, 2] = float("inf")
    with pytest.raises(ValueError):
        micro.quantize_cpu(weight, quant.quantize_weight, torch, chunk_rows=3)


def test_event_timing_batches_calls_and_rotates_order():
    counts = {key: 0 for key in (("eager", "bf16"), ("eager", "int8"),
                                 ("graph", "bf16"), ("graph", "int8"))}
    syncs = []
    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
        def record(self):
            pass
        def synchronize(self):
            syncs.append(1)
        def elapsed_time(self, other):
            return 12.0
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event))
    def operation(key):
        def call():
            counts[key] += 1
        return call
    samples, orders = micro.measure(fake, {key: operation(key) for key in counts}, time.time() + 10)
    assert set(counts.values()) == {300}
    assert len(syncs) == 12
    assert all(values == [0.12, 0.12, 0.12] for values in samples.values())
    assert orders[0] != orders[1] != orders[2]


def test_unguarded_entry_fails_before_cuda(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("DS_") and key != "CUDA_VISIBLE_DEVICES"}
    result = subprocess.run([sys.executable, "-B", str(HERE / "gpu_micro.py"), "--draft", "unused",
                             "--out", str(tmp_path / "failure.jsonl")], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    row = json.loads((tmp_path / "failure.jsonl").read_text(encoding="utf-8"))
    assert row["type"] == "failure" and "gpu_control.py" in row["error"]
