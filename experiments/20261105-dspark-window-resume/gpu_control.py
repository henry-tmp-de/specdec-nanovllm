"""Single-GPU, single-command guard. Linux only; no GPU work happens on import.

All stages in one experiment must share one ledger. The guard never reuses
historical PIDs: a process is owned only when its (PID, /proc start ticks) was
observed in the newly launched process tree. An unfinished ledger requires
manual audit, not automatic cleanup. A polling guard cannot prevent another
user starting between observations, so it stops as soon as a foreign process is
observed and never signals that process.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

TOTAL_LIMIT_S = 14400
SMOKE_LIMIT_S = 1800
CLEANUP_RESERVE_S = 45
POLL_S = 0.5


def parse_proc_stat(text):
    # comm may contain spaces and parentheses; split after its last ')'.
    end = text.rfind(")")
    if end < 0:
        raise ValueError("invalid /proc stat")
    pid = int(text[:text.find(" ")])
    fields = text[end + 2:].split()  # starts with field 3 (state)
    if len(fields) < 20:
        raise ValueError("short /proc stat")
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
            "session": int(fields[3]), "start_ticks": int(fields[19])}


def process_snapshot():
    result = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            p = parse_proc_stat(Path("/proc", name, "stat").read_text())
            result[p["pid"]] = p
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            # Processes can vanish while /proc is being scanned.
            continue
    return result


def resolve_tgid(tid):
    for line in Path("/proc", str(int(tid)), "status").read_text().splitlines():
        if line.startswith("Tgid:"):
            return int(line.split()[1])
    raise RuntimeError(f"cannot resolve TID {tid} to Tgid")


def identity(p):
    return (p["pid"], p["start_ticks"])


def extend_owned(snapshot, owned):
    """Expand from identities already observed; no stale PID alone is trusted."""
    current_owned = {pid for pid, p in snapshot.items() if identity(p) in owned}
    while True:
        added = {pid for pid, p in snapshot.items()
                 if p["ppid"] in current_owned and pid not in current_owned}
        if not added:
            break
        current_owned.update(added)
    owned.update(identity(snapshot[pid]) for pid in current_owned)
    return current_owned


def gpu_state(index):
    def query(fields, which):
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-{which}={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=3)
        return [list(map(str.strip, row)) for row in csv.reader(out.splitlines()) if row]
    matches = [r for r in query("index,uuid,name,memory.used,memory.total,utilization.gpu", "gpu")
               if int(r[0]) == index]
    if len(matches) != 1 or len(matches[0]) != 6:
        raise RuntimeError("selected GPU index unavailable or ambiguous")
    i, dev_uuid, name, used, total, util = matches[0]
    procs = []
    for row in query("pid,gpu_uuid,used_gpu_memory", "compute-apps"):
        if len(row) != 3:
            raise RuntimeError(f"unrecognized compute process record: {row!r}")
        if row[1] == dev_uuid:
            procs.append({"reported_pid_or_tid": int(row[0]), "gpu_uuid": row[1],
                          "used_mib": row[2]})
    return {"index": int(i), "uuid": dev_uuid, "name": name,
            "used_mib": int(used), "total_mib": int(total),
            "util_pct": int(util), "compute_processes": procs}


def classify_compute(state, snapshot, owned):
    current_owned = extend_owned(snapshot, owned)
    observed = []
    for p in state["compute_processes"]:
        tid = p["reported_pid_or_tid"]
        try:
            tgid = resolve_tgid(tid)
            process = snapshot.get(tgid)
            if process is None:
                raise RuntimeError("Tgid absent from process snapshot")
            is_owned = tgid in current_owned and identity(process) in owned
            observed.append({**p, "tgid": tgid, "start_ticks": process["start_ticks"],
                             "owned": is_owned})
        except (OSError, ValueError, RuntimeError) as exc:
            # A disappearing TID is not evidence of ownership.
            observed.append({**p, "owned": False, "resolution_error": str(exc)})
    return observed


def signal_owned(snapshot, owned, sig):
    """Signal via pidfd after validating the live identity; never use old PIDs."""
    extend_owned(snapshot, owned)
    sent = []
    for pid, ticks in sorted(owned, reverse=True):
        fd = None
        try:
            fd = os.pidfd_open(pid, 0)
            current = parse_proc_stat(Path("/proc", str(pid), "stat").read_text())
            if current["start_ticks"] != ticks or current["state"] == "Z":
                continue
            signal.pidfd_send_signal(fd, sig)
            sent.append({"pid": pid, "start_ticks": ticks, "signal": sig})
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            if fd is not None:
                os.close(fd)
    return sent


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_ledger(path, expected_uuid):
    if not path.exists():
        return 0.0, 0.0
    starts, children, ends = {}, {}, {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        key = row.get("id")
        if not key or row.get("gpu_uuid") != expected_uuid:
            raise RuntimeError(f"ledger line {number}: missing identity or GPU UUID mismatch")
        dest = {"start": starts, "child": children, "end": ends}.get(row.get("event"))
        if dest is None or key in dest:
            raise RuntimeError(f"ledger line {number}: invalid or duplicate event")
        dest[key] = row
    if starts.keys() != ends.keys():
        raise RuntimeError("ledger has unfinished jobs; audit live processes and ledger before resuming")
    if not children.keys() <= starts.keys():
        raise RuntimeError("ledger contains child events without a start")
    total = smoke = 0.0
    for key, end in ends.items():
        elapsed = float(end["elapsed_s"])
        if not 0 <= elapsed < float("inf") or end.get("kind") != starts[key].get("kind"):
            raise RuntimeError("invalid ledger timing or stage kind")
        total += elapsed
        if end["kind"] == "smoke":
            smoke += elapsed
    return total, smoke


def main(argv=None):
    p = argparse.ArgumentParser(description="仅守护一次已授权的单卡任务；所有阶段共用 ledger。")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--uuid", required=True, help="现场检查的 GPU UUID")
    p.add_argument("--stage", required=True)
    p.add_argument("--kind", choices=("smoke", "pilot", "formal", "capacity"), required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--log", required=True, help="必须是不存在的新文件")
    p.add_argument("--stage-timeout-s", type=float, required=True, help="包括清理的阶段上限")
    p.add_argument("--max-idle-mib", type=int, default=100, choices=range(0, 101), metavar="0..100")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        p.error("-- 后必须给出子进程命令")
    if not sys.platform.startswith("linux"):
        p.error("GPU 守护仅支持有 /proc 的 Linux；CPU 单测可跨平台运行")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        p.error("守护器要求 Linux pidfd_open 与 pidfd_send_signal，无法安全清理时不启动")
    if not CLEANUP_RESERVE_S < args.stage_timeout_s <= TOTAL_LIMIT_S:
        p.error("stage-timeout-s 必须大于 45 且不超过 14400")
    import fcntl
    ledger, log_path = Path(args.ledger), Path(args.log)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger.with_name(ledger.name + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("同一 ledger 已有运行中的任务")
        spent, smoke_spent = load_ledger(ledger, args.uuid)
        available = min(TOTAL_LIMIT_S - spent, args.stage_timeout_s)
        if args.kind == "smoke":
            available = min(available, SMOKE_LIMIT_S - smoke_spent)
        if available <= CLEANUP_RESERVE_S:
            raise SystemExit("剩余总预算或 smoke 预算不足以运行并清理")
        state = gpu_state(args.gpu)
        if (state["uuid"] != args.uuid or state["used_mib"] > args.max_idle_mib
                or state["util_pct"] != 0 or state["compute_processes"]):
            raise SystemExit(f"GPU 未确认空闲: {state}")
        # Exclusive log creation before ledger start, never overwrite evidence.
        log = log_path.open("x", encoding="utf-8")
        stage_id = str(uuid.uuid4())
        env = os.environ.copy()
        # CUDA's ordinal enumeration can differ from nvidia-smi's ordering.
        env["CUDA_VISIBLE_DEVICES"] = args.uuid
        env["DS_GPU_UUID"] = args.uuid
        env["DS_GUARD_ID"] = stage_id
        deadline = time.time() + available - CLEANUP_RESERVE_S
        env["DS_RUN_DEADLINE_UNIX"] = str(deadline)
        start = time.monotonic()
        append(ledger, {"event": "start", "id": stage_id, "stage": args.stage,
                        "kind": args.kind, "gpu_uuid": args.uuid, "unix": time.time(),
                        "gpu_before": state, "stage_limit_s": available, "command": command,
                        "guard_pid": os.getpid(),
                        "guard_start_ticks": parse_proc_stat(Path(f"/proc/{os.getpid()}/stat").read_text())["start_ticks"]})
        proc = None
        owned = set()
        root_identity = None
        outcome = "error"
        detail = ""
        signals = []
        exit_code = None
        last_state = state
        observed = []
        ever_owned_compute = False
        stop_requested = [None]
        def stop_on_signal(signum, frame):
            # Do not raise inside Popen, between spawning and recording identity.
            stop_requested[0] = signum
        old_handlers = {sig: signal.signal(sig, stop_on_signal)
                        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            root = parse_proc_stat(Path(f"/proc/{proc.pid}/stat").read_text())
            root_identity = identity(root)
            owned.add(root_identity)
            append(ledger, {"event": "child", "id": stage_id, "stage": args.stage,
                            "kind": args.kind, "gpu_uuid": args.uuid, "unix": time.time(),
                            "pid": root["pid"], "start_ticks": root["start_ticks"]})
            print(f"启动 {args.stage}: GPU {args.uuid}, PID {proc.pid}, start_ticks {root['start_ticks']}", flush=True)
            while True:
                if stop_requested[0] is not None:
                    outcome, detail = "interrupted", f"guard received signal {stop_requested[0]}"
                    break
                snap = process_snapshot()
                extend_owned(snap, owned)
                state = gpu_state(args.gpu)
                last_state = state
                if state["uuid"] != args.uuid:
                    outcome, detail = "gpu_identity_changed", "GPU UUID changed"
                    break
                observed = classify_compute(state, snap, owned)
                ever_owned_compute |= any(x["owned"] for x in observed)
                foreign = [x for x in observed if not x["owned"]]
                if foreign:
                    outcome, detail = "foreign_process", json.dumps(foreign, ensure_ascii=False)
                    break
                rc = proc.poll()
                if rc is not None:
                    exit_code = rc
                    # Do not hide orphaned descendants even if they have released CUDA.
                    live_owned = [pid for pid in extend_owned(snap, owned)
                                  if pid != proc.pid and snap[pid]["state"] != "Z"]
                    outcome = ("ok" if rc == 0 and not live_owned and ever_owned_compute
                               else "no_owned_compute_observed" if rc == 0 and not live_owned
                               else "child_failed_or_lingering")
                    break
                if time.monotonic() - start >= available - CLEANUP_RESERVE_S:
                    outcome, detail = "timeout", "reserved time for owned-process cleanup"
                    break
                time.sleep(POLL_S)
        except BaseException as exc:
            outcome, detail = "guard_exception", f"{type(exc).__name__}: {exc}"
        finally:
            # Further signals must not interrupt cleanup or ledger completion.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            if proc is not None:
                try:
                    signals += signal_owned(process_snapshot(), owned, signal.SIGTERM)
                    stop_at = time.monotonic() + 5
                    while time.monotonic() < stop_at:
                        snap = process_snapshot()
                        live = [pid for pid in extend_owned(snap, owned)
                                if snap[pid]["state"] != "Z"]
                        if not live:
                            break
                        time.sleep(0.25)
                    signals += signal_owned(process_snapshot(), owned, signal.SIGKILL)
                    try:
                        exit_code = proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        outcome = "owned_process_cleanup_failed"
                except BaseException as exc:
                    outcome, detail = "owned_process_cleanup_failed", f"{type(exc).__name__}: {exc}"
            log.close()
            try:
                last_state = gpu_state(args.gpu)
                observed = classify_compute(last_state, process_snapshot(), owned)
                if any(x["owned"] for x in observed):
                    outcome = "owned_gpu_process_remains"
                elif any(not x["owned"] for x in observed):
                    outcome = "foreign_process"
                elif (last_state["uuid"] != args.uuid or last_state["used_mib"] > args.max_idle_mib
                      or last_state["util_pct"] != 0):
                    # Utilization can lag; still account the settling interval.
                    settle_until = min(start + available, time.monotonic() + 5)
                    while time.monotonic() < settle_until:
                        time.sleep(0.5)
                        last_state = gpu_state(args.gpu)
                        observed = classify_compute(last_state, process_snapshot(), owned)
                        if observed or last_state["uuid"] != args.uuid:
                            outcome = "gpu_not_idle_after_run"
                            break
                        if last_state["used_mib"] <= args.max_idle_mib and last_state["util_pct"] == 0:
                            break
                    else:
                        outcome = "gpu_not_idle_after_run"
            except BaseException as exc:
                outcome, detail = "final_gpu_check_failed", f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - start
            if elapsed > available:
                outcome = "budget_overrun"
            row = {"event": "end", "id": stage_id, "stage": args.stage, "kind": args.kind,
                   "gpu_uuid": args.uuid, "unix": time.time(), "elapsed_s": elapsed,
                   "exit_code": exit_code, "outcome": outcome, "detail": detail,
                   "root_identity": list(root_identity) if root_identity else None,
                   "ever_owned_compute_observed": ever_owned_compute,
                   "observed_owned_identities": [list(x) for x in sorted(owned)],
                   "signals": signals, "gpu_after": last_state, "compute_after": observed}
            append(ledger, row)
            print(f"结束 {args.stage}: {outcome}, 用时 {elapsed:.2f}s, 累计 {spent + elapsed:.2f}s", flush=True)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        if outcome != "ok" or exit_code != 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
