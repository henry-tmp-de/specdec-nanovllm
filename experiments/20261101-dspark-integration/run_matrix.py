"""Sequential fixed-batch matrix; GPU checks and accounting live in gpu_control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_control import gpu_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ar-model", required=True)
    ap.add_argument("--dspark-model", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 1, 2, 4])
    ap.add_argument("--tag", default="formal")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = ap.parse_args()
    resultdir = Path(a.results)
    resultdir.mkdir(parents=True, exist_ok=True)
    ledger = resultdir / "gpu_budget.jsonl"
    # Prioritize the user's B8/B16 decision, then complete the full curve.
    for b in a.batches:
        for arm in ("nospec", "ar", "dspark"):
            stem = f"{a.tag}-{arm}-b{b}"
            out = resultdir / f"{stem}.jsonl"
            if out.exists():
                records = [json.loads(x) for x in out.read_text().splitlines()]
                if sum(r.get("type") == "run" for r in records) == 9:
                    print(json.dumps({"stage": stem, "status": "already_complete"}), flush=True)
                    continue
                raise RuntimeError(f"partial {out}; audit before any rerun")
            # nvidia-smi utilization may lag briefly after our previous process.
            # Memory or a compute PID belonging to anyone else immediately stops.
            for _ in range(15):
                state = gpu_state(a.gpu)
                if state["used_mib"] > 100 or state["compute_processes"]:
                    raise RuntimeError(f"GPU no longer idle: {state}")
                if state["util_pct"] == 0:
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"GPU utilization remained nonzero: {state}")
            command = [a.python, str(HERE / "gpu_control.py"), "--gpu", str(a.gpu), "--stage", stem,
                       "--ledger", str(ledger), "--log", str(resultdir / f"{stem}.log"), "--",
                       a.python, "-u", str(HERE / "bench.py"), "--arm", arm, "--batch", str(b),
                       "--target", a.target, "--ar-model", a.ar_model, "--dspark-model", a.dspark_model,
                       "--datasets", a.datasets, "--wiki", a.wiki, "--manifest", a.manifest,
                       "--out", str(out), "--ctx", "1024", "--output-tokens", "256", "--repeats", "3",
                       "--gpu-memory-utilization", str(a.gpu_memory_utilization)]
            print(json.dumps({"stage": stem, "status": "starting", "gpu_before": state}), flush=True)
            ret = subprocess.run(command, check=False)
            if ret.returncode:
                rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
                failures = [r for r in rows if r.get("type") == "failure"]
                msg = failures[-1].get("message", "") if failures else ""
                capacity = "capacity insufficient" in msg.lower() or "out of memory" in msg.lower()
                print(json.dumps({"stage": stem, "status": "capacity_failure" if capacity else "stopped",
                                  "returncode": ret.returncode, "message": msg}), flush=True)
                if not capacity:
                    raise SystemExit("stopped on non-capacity failure; inspect the raw log")
    print(json.dumps({"status": "matrix_finished"}), flush=True)


if __name__ == "__main__":
    main()
