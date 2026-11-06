"""Run isolated CPU layer-prefill and existing engine regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cpu-evidence"
CASES = [
    ("layer_prefill", "pytest", ["tests/test_layer_prefill.py"]),
    ("dspark_window_and_quant", "pytest", ["tests/test_dspark_quant.py", "tests/test_dspark_contract.py",
                                            "tests/test_dspark_window_config.py", "tests/test_dspark_window_runner.py",
                                            "tests/test_dspark_window.py", "tests/test_dspark_pad.py"]),
    ("prefill_chunk_gate", "script", "tests/test_prefill_chunk_gate.py"),
    ("prefill_multiseq", "script", "tests/test_prefill_multiseq.py"),
    ("block_alloc", "script", "tests/test_block_alloc.py"),
    ("graph_pad", "script", "tests/test_graph_pad.py"),
]


def main():
    OUT.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    records = []
    for name, kind, paths in CASES:
        code = (f"import numpy,pytest; raise SystemExit(pytest.main(['-q'] + {paths!r}))" if kind == "pytest" else
                f"import numpy,runpy; runpy.run_path({paths!r}, run_name='__main__')")
        command = [sys.executable, "-B", "-c", code]
        start = perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = perf_counter() - start
        (OUT / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        records.append({"name": name, "command": command, "exit_code": result.returncode,
                        "seconds": round(elapsed, 3), "log": f"cpu-evidence/{name}.log"})
        print(f"{name}: exit={result.returncode}, seconds={elapsed:.2f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(record["exit_code"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
