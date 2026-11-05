"""Run isolated CPU-only regressions and save exact output and exit codes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("cpu-evidence")
OUT.mkdir(exist_ok=True)
PY = sys.executable
tests = [
    ("window", "pytest", "tests/test_dspark_window.py"),
    ("runner", "pytest", "tests/test_dspark_window_runner.py"),
    ("features", "pytest", "tests/test_dspark_feature_rows.py"),
    ("config", "pytest", "tests/test_dspark_window_config.py"),
    ("existing_dspark", "pytest", "tests/test_dspark_contract.py"),
    ("existing_dspark_pad", "pytest", "tests/test_dspark_pad.py"),
    ("existing_graph_pad", "script", "tests/test_graph_pad.py"),
    ("overlap_mixed", "script", "tests/test_overlap_mixed.py"),
    ("block_alloc", "script", "tests/test_block_alloc.py"),
    ("prefill_chunk", "script", "tests/test_prefill_chunk_gate.py"),
    ("prefill_multiseq", "script", "tests/test_prefill_multiseq.py"),
    ("spec_gate", "script", "tests/test_spec_gate.py"),
    ("prefix_hash", "script", "tests/test_prefix_hash.py"),
    ("mixed_preemption", "script", "experiments/20261105-dspark-window-resume/cpu_mixed_preemption_repro.py"),
    ("decode_slot", "slot", "experiments/20261102-dspark-window/cpu_decode_slot_repro.py"),
]
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
records = []
for name, kind, path in tests:
    if kind == "pytest":
        code = f"import numpy, pytest; raise SystemExit(pytest.main(['-q', '-s', {path!r}]))"
    elif kind == "slot":
        code = ("import sys, runpy; sys.argv = [" + repr(path) + ", '--output', "
                + repr(str(OUT / 'decode-slot.json')) + "]; "
                + f"runpy.run_path({path!r}, run_name='__main__')")
    else:
        code = f"import numpy, runpy; runpy.run_path({path!r}, run_name='__main__')"
    cmd = [PY, "-B", "-c", code]
    start = perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, encoding="utf-8",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = perf_counter() - start
    (OUT / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    records.append(dict(name=name, command=cmd, exit_code=result.returncode,
                        seconds=round(elapsed, 3), log=f"cpu-evidence/{name}.log"))
    print(f"{name}: exit={result.returncode} {elapsed:.1f}s", flush=True)
(OUT / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if any(r["exit_code"] for r in records):
    sys.exit(1)
