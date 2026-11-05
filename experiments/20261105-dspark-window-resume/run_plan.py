"""Emit a reviewable rotated experiment plan; never launches processes or GPUs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FORMAL_ARMS = ("nospec", "ar_full", "ar_window", "dspark_full", "dspark_ring")
PANELS = ("code", "math", "general")


def mode_of(arm):
    return "ring" if arm == "dspark_ring" else "full_cache" if arm == "dspark_window_full" else "block_aligned" if arm == "ar_window" else "full"


def build(phase, window, panels, repeats, gpu_memory_utilization=None):
    if window <= 0 or window % 256 or len(set(panels)) != len(panels) or not panels or repeats < 1:
        raise ValueError("invalid window, panels or repeats")
    if phase == "formal" and repeats < 3:
        raise ValueError("formal cells require at least three repeats")
    budget = (0.85 if phase == "capacity" else 0.70) if gpu_memory_utilization is None else gpu_memory_utilization
    if not math.isfinite(budget) or not 0 < budget <= 1:
        raise ValueError("GPU memory utilization must be finite and in (0, 1]")
    if phase in ("pilot", "formal") and budget not in (0.70, 0.85):
        raise ValueError("pilot/formal budget must be 0.70 or 0.85")
    jobs = []
    if phase == "formal":
        for i, batch in enumerate((1, 2, 4, 8, 16)):
            for j, ctx in enumerate((1024, 4096)):
                shift = (i * 2 + j) % len(FORMAL_ARMS)
                arms = FORMAL_ARMS[shift:] + FORMAL_ARMS[:shift]
                for arm in arms:
                    w = window if arm in ("ar_window", "dspark_ring") else 0
                    jobs.append({"phase": phase, "arm": arm, "mode": mode_of(arm),
                                 "batch": batch, "ctx": ctx, "window": w,
                                 "gpu_memory_utilization": budget,
                                 "panels": list(panels), "repeats": repeats})
    elif phase == "pilot":
        for i, batch in enumerate((2, 8)):
            grid = [("dspark_full", 0)] + [(arm, w) for w in (512, 1024, 2048)
                                                for arm in ("dspark_window_full", "dspark_ring")]
            shift = i % len(grid)
            for arm, w in grid[shift:] + grid[:shift]:
                jobs.append({"phase": phase, "arm": arm, "mode": mode_of(arm),
                             "batch": batch, "ctx": 4096, "window": w,
                             "gpu_memory_utilization": budget,
                             "panels": list(panels), "repeats": repeats})
    elif phase == "capacity":
        for arm in ("dspark_full", "dspark_ring"):
            jobs.append({"phase": phase, "arm": arm, "mode": mode_of(arm),
                         "batch": 4, "ctx": 16384, "window": window if arm == "dspark_ring" else 0,
                         "panels": list(panels), "repeats": repeats, "max_num_batched_tokens": 4096,
                         "gpu_memory_utilization": budget,
                         "capacity_policy": "fixed total GPU budget; target pool block count may differ"})
    else:
        raise ValueError("unknown phase")
    cells = [{"arm": j["arm"], "mode": j["mode"], "ctx": j["ctx"], "requested_batch": j["batch"],
              "panel": panel, "window": j["window"], "graph_mode": "graph",
              "gpu_memory_utilization": j["gpu_memory_utilization"]}
             for j in jobs for panel in j["panels"]]
    full = phase == "formal" and tuple(panels) == PANELS and repeats >= 3
    return {"phase": phase, "execution": "plan only; no command is launched", "window_tokens": window,
            "gpu_memory_utilization": budget,
            "graph_policy": "exact_all: target decode, draft, and target verify capture every real batch size",
            "matrix_scope": "full minimum fixed-batch matrix" if full else "pilot or explicit subset",
            "model_lifecycle": "one process per arm/B/ctx; prewarm once, rotate panel order across repeats",
            "ordering_note": "arm order rotates between B/ctx groups, repeats remain within each arm process",
            "jobs": jobs, "cells": cells}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("pilot", "formal", "capacity"), required=True)
    p.add_argument("--window", type=int, required=True, help="formal/capacity 所选窗口；pilot 保留该参数用于记录")
    p.add_argument("--panels", nargs="+", choices=PANELS)
    p.add_argument("--repeats", type=int)
    p.add_argument("--gpu-memory-utilization", type=float,
                   help="pilot/formal 默认0.70，可显式0.85；capacity默认0.85")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    panels = tuple(a.panels) if a.panels else (PANELS if a.phase == "formal" else ("code",))
    repeats = a.repeats if a.repeats is not None else (3 if a.phase == "formal" else 1)
    try:
        plan = build(a.phase, a.window, panels, repeats, a.gpu_memory_utilization)
    except ValueError as exc:
        p.error(str(exc))
    with Path(a.out).open("x", encoding="utf-8") as out:
        json.dump(plan, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(f"仅生成计划：{len(plan['jobs'])} 个进程、{len(plan['cells']) * repeats} 次测量；未启动 GPU")


if __name__ == "__main__":
    main()
