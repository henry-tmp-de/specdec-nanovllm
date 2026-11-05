"""Run quality and DSpark contracts on CPU, recording commands and outcomes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cpu-evidence"
SUITES = {
    "quality_contracts": ["experiments/20261105-dspark-quant-quality/tests_quality_eval.py",
                          "experiments/20261105-dspark-quant-quality/tests_quality_data.py",
                          "experiments/20261105-dspark-quant-quality/tests_quality_metrics.py"],
    "existing_dspark_contracts": ["tests/test_dspark_quant.py", "tests/test_dspark_contract.py",
                                  "tests/test_dspark_window_config.py", "tests/test_dspark_window_runner.py",
                                  "tests/test_dspark_window.py", "tests/test_dspark_pad.py"],
}


def main():
    OUT.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    records = []
    for name, paths in SUITES.items():
        code = f"import numpy, pytest; raise SystemExit(pytest.main(['-q'] + {paths!r}))"
        command = [sys.executable, "-B", "-c", code]
        started = perf_counter()
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        seconds = perf_counter() - started
        (OUT / f"{name}.log").write_text(proc.stdout, encoding="utf-8")
        records.append({"name": name, "command": command, "exit_code": proc.returncode,
                        "seconds": round(seconds, 3), "log": f"cpu-evidence/{name}.log"})
        print(f"{name}: exit={proc.returncode}, seconds={seconds:.2f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(record["exit_code"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
