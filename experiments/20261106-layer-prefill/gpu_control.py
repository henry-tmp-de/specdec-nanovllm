"""Use the existing single-GPU ownership guard with this experiment's limits.

All new stages share one ledger. No GPU is queried when this module is imported;
only the coordinator launches guarded workloads on an explicitly checked card.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

OLD = Path(__file__).resolve().parents[1] / "20261105-dspark-window-resume"
TOTAL_LIMIT_S = 600
SMOKE_LIMIT_S = 300


def load_guard():
    sys.path.insert(0, str(OLD))
    spec = importlib.util.spec_from_file_location("layer_prefill_previous_gpu_guard", OLD / "gpu_control.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.TOTAL_LIMIT_S = TOTAL_LIMIT_S
    guard.SMOKE_LIMIT_S = SMOKE_LIMIT_S
    return guard


def main(argv=None):
    # The reused guard enforces the overridden limit but its old error text
    # says 14400. Fail early with this experiment's actual limit instead.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--stage-timeout-s", type=float)
    values = list(sys.argv[1:] if argv is None else argv)
    outer = values[:values.index("--")] if "--" in values else values
    args, _ = parser.parse_known_args(outer)
    if args.stage_timeout_s is not None and not 45 < args.stage_timeout_s <= TOTAL_LIMIT_S:
        parser.error("stage-timeout-s must be greater than 45 and at most 600")
    load_guard().main(argv)


if __name__ == "__main__":
    main()
