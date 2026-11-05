"""Record CPU-only checks for selectable ordinary target decode graph buckets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("decode-graph-cpu")
OUT.mkdir(exist_ok=True)
tests = [
    ("new_joint", "pytest", ["tests/test_dspark_feature_rows.py", "tests/test_dspark_window.py",
                               "tests/test_dspark_window_config.py", "tests/test_dspark_window_runner.py"]),
    ("dspark_contract", "pytest", ["tests/test_dspark_contract.py"]),
    ("dspark_pad", "pytest", ["tests/test_dspark_pad.py"]),
    ("graph_pad", "script", "tests/test_graph_pad.py"),
    ("overlap_mixed", "script", "tests/test_overlap_mixed.py"),
    ("mixed_preemption", "script", "experiments/20261105-dspark-window-resume/cpu_mixed_preemption_repro.py"),
]
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
records = []
for name, kind, target in tests:
    if kind == "pytest":
        code = f"import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s'] + {target!r}))"
    else:
        code = f"import numpy, runpy; runpy.run_path({target!r}, run_name='__main__')"
    command = [sys.executable, "-B", "-c", code]
    started = perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    records.append(dict(name=name, command=command, exit_code=result.returncode,
                        seconds=round(perf_counter() - started, 3), log=f"decode-graph-cpu/{name}.log"))
    print(f"{name}: exit={result.returncode}", flush=True)
(OUT / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if any(result["exit_code"] for result in records):
    sys.exit(1)
