"""Approved stage A only: serial jobs, live idle/port checks, stop on failure."""
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
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--uuid", default="GPU-e0671215-034e-dd14-093f-b3755e543b6f")
    ap.add_argument("--results", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ar-model", required=True)
    ap.add_argument("--dspark-model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port", type=int, default=24719)
    args = ap.parse_args()
    outdir = Path(args.results)
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = outdir / "gpu-budget.jsonl"
    # Rotate external controls across B; split modes are interleaved by
    # benchmark.py within each panel/repeat.
    orders = [(8, ("dspark-original", "dspark-ablations", "ar-pad", "nospec-pad")),
              (16, ("nospec-pad", "ar-pad", "dspark-ablations", "dspark-original"))]
    for batch, arms in orders:
        for arm in arms:
            stage = f"stage-a-{arm}-b{batch}"
            out = outdir / f"{stage}.jsonl"
            if out.exists():
                raise RuntimeError(f"output exists; audit before restarting: {out}")
            for _ in range(15):
                state = gpu_state(args.gpu)
                if state["uuid"] != args.uuid or state["used_mib"] > 100 or state["compute_processes"]:
                    raise RuntimeError(f"GPU no longer confirmed idle: {state}")
                if state["util_pct"] == 0:
                    break
                time.sleep(1)  # utilization may lag after our own process
            else:
                raise RuntimeError(f"GPU utilization remained nonzero: {state}")
            port = subprocess.check_output(["ss", "-H", "-ltn", f"sport = :{args.port}"], text=True)
            if port.strip():
                raise RuntimeError(f"port {args.port} is occupied: {port.strip()}")
            timeout = 600 if arm == "dspark-ablations" else 300
            command = [args.python, str(HERE / "gpu_control.py"), "--gpu", str(args.gpu),
                       "--uuid", args.uuid, "--stage", stage, "--budget-s", "2700",
                       "--stage-timeout-s", str(timeout), "--ledger", str(ledger),
                       "--log", str(outdir / f"{stage}.log"), "--", args.python, "-u",
                       str(HERE / "benchmark.py"), "--arm", arm, "--batch", str(batch),
                       "--target", args.target, "--ar-model", args.ar_model,
                       "--dspark-model", args.dspark_model, "--manifest", args.manifest,
                       "--ctx", "1024", "--output-tokens", "256", "--repeats", "3",
                       "--out", str(out)]
            env = os.environ.copy()
            env["NANOVLLM_PORT"] = str(args.port)
            print(json.dumps({"stage": stage, "status": "starting", "gpu_before": state,
                              "port": args.port, "timeout_s": timeout}), flush=True)
            result = subprocess.run(command, env=env, check=False)
            if result.returncode:
                # No automatic retry: preserves OOM, timeout and correctness failures.
                raise SystemExit(f"{stage} failed ({result.returncode}); inspect raw records and stop")
    print(json.dumps({"status": "stage_a_finished"}), flush=True)


if __name__ == "__main__":
    main()
