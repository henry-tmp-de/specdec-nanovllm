"""Run one owned GPU process after a live idle check and enforce total budget."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_state(index):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = [x.strip().split(", ") for x in raw.strip().splitlines()]
    matches = [r for r in rows if int(r[0]) == index]
    if len(matches) != 1:
        raise RuntimeError("requested GPU index not found")
    i, uuid, name, used, total, util = matches[0]
    raw_procs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                        "--format=csv,noheader,nounits"], text=True)
    pids = [r.strip() for r in raw_procs.splitlines() if uuid in r]
    return {"index": int(i), "uuid": uuid, "name": name, "used_mib": int(used),
            "total_mib": int(total), "util_pct": int(util), "compute_processes": pids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--budget-s", type=int, default=2700)
    p.add_argument("--stage-timeout-s", type=int, default=300)
    p.add_argument("--uuid", default="GPU-e0671215-034e-dd14-093f-b3755e543b6f")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command:
        p.error("a child command is required")
    ledger = Path(a.ledger)
    previous = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    starts = {r["id"] for r in previous if r["event"] == "start"}
    ends = {r["id"] for r in previous if r["event"] == "end"}
    if starts != ends:
        raise RuntimeError("ledger contains unfinished owned job; audit it before continuing")
    spent = sum(r["elapsed_s"] for r in previous if r["event"] == "end")
    remaining = a.budget_s - spent
    if remaining <= 15:
        raise RuntimeError("GPU budget exhausted")
    state = gpu_state(a.gpu)
    if (state["uuid"] != a.uuid or "RTX 3090" not in state["name"] or state["used_mib"] > 100
            or state["util_pct"] or state["compute_processes"]):
        raise RuntimeError(f"GPU is not confirmed idle: {state}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    timeout = min(remaining - 15, a.stage_timeout_s)
    env["DS_RUN_DEADLINE_UNIX"] = str(time.time() + timeout)
    env.setdefault("NANOVLLM_PORT", "24719")
    ident = f"{a.stage}-{time.time_ns()}"
    append(ledger, {"event": "start", "id": ident, "stage": a.stage,
                    "unix": time.time(), "remaining_s": remaining, "stage_timeout_s": timeout,
                    "gpu_before": state, "command": command})
    start = time.monotonic()
    exitcode = None
    timed_out = False
    try:
        with open(a.log, "w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                exitcode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    exitcode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    exitcode = process.wait()
    finally:
        elapsed = time.monotonic() - start
        after = gpu_state(a.gpu)
        append(ledger, {"event": "end", "id": ident, "stage": a.stage,
                        "unix": time.time(), "elapsed_s": elapsed, "exit_code": exitcode,
                        "timed_out": timed_out, "gpu_after": after})
    print(json.dumps({"stage": a.stage, "exit_code": exitcode, "elapsed_s": elapsed,
                      "total_spent_s": spent + elapsed, "remaining_s": a.budget_s - spent - elapsed,
                      "gpu_after": after}), flush=True)
    sys.exit(exitcode if exitcode is not None else 1)


if __name__ == "__main__":
    main()
