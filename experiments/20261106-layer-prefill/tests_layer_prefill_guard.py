"""CPU-only checks of this experiment's guard budget; no GPU is queried."""
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("layer_prefill_guard_wrapper", HERE / "gpu_control.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

GPU = "GPU-f1619d21-c909-7c21-34f1-05c39a7b0c91"


def ledger(path, elapsed, kind):
    rows = [{"event": "start", "id": "job", "kind": kind, "gpu_uuid": GPU},
            {"event": "end", "id": "job", "kind": kind, "gpu_uuid": GPU,
             "elapsed_s": elapsed, "outcome": "timeout"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_wrapper_sets_new_limits_without_gpu_access():
    guard = wrapper.load_guard()
    assert guard.TOTAL_LIMIT_S == 600 and guard.SMOKE_LIMIT_S == 300
    assert guard.CLEANUP_RESERVE_S == 45
    assert guard.normalize_gpu_uuid(GPU.lower()) == GPU
    with patch.object(wrapper, "load_guard", side_effect=AssertionError("must fail before loading guard")):
        with pytest.raises(SystemExit):
            wrapper.main(["--stage-timeout-s", "601"])


@pytest.mark.parametrize("elapsed,previous_kind,next_kind", [(555, "pilot", "pilot"), (255, "smoke", "smoke")])
def test_exhausted_budget_stops_before_gpu_query(tmp_path, elapsed, previous_kind, next_kind):
    guard = wrapper.load_guard()
    path = tmp_path / "ledger.jsonl"
    ledger(path, elapsed, previous_kind)
    argv = ["--gpu", "0", "--uuid", GPU, "--stage", "budget-check", "--kind", next_kind,
            "--ledger", str(path), "--log", str(tmp_path / "unused.log"), "--stage-timeout-s", "100",
            "--", "unused-command"]
    fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
    with patch.object(guard.sys, "platform", "linux"), \
         patch.object(guard.os, "pidfd_open", create=True), \
         patch.object(guard.signal, "pidfd_send_signal", create=True), \
         patch.dict(sys.modules, {"fcntl": fake_fcntl}), \
         patch.object(guard, "gpu_state", side_effect=AssertionError("must not query GPU")):
        with pytest.raises(SystemExit, match="预算不足"):
            guard.main(argv)
    assert not (tmp_path / "unused.log").exists()


def test_identity_wrapper_preserves_strict_uuid_check():
    spec = importlib.util.spec_from_file_location("layer_prefill_identity_wrapper", HERE / "gpu_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.normalize_gpu_uuid(GPU) == GPU
    with pytest.raises((TypeError, ValueError)):
        module.normalize_gpu_uuid([0] * 15)


def test_full_guard_argv_is_forwarded_without_abbreviation_or_child_reparse():
    argv = ["--gpu", "0", "--uuid", GPU, "--stage", "layer-smoke-01", "--kind", "smoke",
            "--ledger", "ledger.jsonl", "--log", "stage.log", "--stage-timeout-s", "300", "--",
            "python", "worker.py", "--stage-timeout-s", "999"]
    fake = types.SimpleNamespace(main=lambda values: calls.append(values))
    calls = []
    with patch.object(wrapper, "load_guard", return_value=fake):
        wrapper.main(argv)
    assert calls == [argv]
