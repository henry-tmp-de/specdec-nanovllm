"""Use the existing single-GPU ownership guard with this experiment's limits.

All new stages share one ledger. No GPU is queried when this module is imported;
only the coordinator launches guarded workloads on an explicitly checked card.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

OLD = Path(__file__).resolve().parents[1] / "20261105-dspark-window-resume"
TOTAL_LIMIT_S = 1800
SMOKE_LIMIT_S = 300


def load_guard():
    sys.path.insert(0, str(OLD))
    spec = importlib.util.spec_from_file_location("layer_prefill_previous_gpu_guard", OLD / "gpu_control.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.TOTAL_LIMIT_S = TOTAL_LIMIT_S
    guard.SMOKE_LIMIT_S = SMOKE_LIMIT_S
    return guard


if __name__ == "__main__":
    load_guard().main()
