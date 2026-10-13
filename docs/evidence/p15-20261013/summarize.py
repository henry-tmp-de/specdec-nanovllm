import json, os, statistics, sys

D = os.path.dirname(os.path.abspath(__file__))

def rows(scen):
    out = []
    with open(os.path.join(D, f"scen{scen}.log"), encoding="utf-8", errors="replace") as f:
        for l in f:
            if "@@SROW@@ " in l:
                out.append(json.loads(l.split("@@SROW@@ ", 1)[1]))
    return out

def agg(vals):
    if not vals:
        return None
    return "[%.2f, %.2f] ~%2d" % (min(vals), max(vals), len(vals))

def med(vals):
    return statistics.median(vals) if vals else float("nan")

specs = {
    "A": [("wall_s", "%.2f"), ("ttft_long_ms", "%.0f"), ("itl_p50_ms", "%.1f"),
          ("itl_p99_ms", "%.1f"), ("itl_max_ms", "%.1f"), ("itl_win_p50_ms", "%.1f"),
          ("itl_win_max_ms", "%.1f"), ("itl_win_n", "%.0f")],
    "B": [("wall_s", "%.2f"), ("ttft_p50_ms", "%.0f"), ("ttft_p99_ms", "%.0f"),
          ("dec_itl_p50_ms", "%.1f")],
    "C": [("wall_s", "%.2f"), ("tok_per_s", "%.2f"), ("delivered_tokens", "%.0f")],
}

for scen in ("A", "B", "C"):
    rs = rows(scen)
    modes = []
    for r in rs:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    print("=" * 78)
    print(f"场景 {scen}   (每条件 n={len(rs)//len(modes)} 次重复，三档在同一进程内交替采样)")
    print("=" * 78)
    for key, fmt in specs[scen]:
        print(f"  {key}")
        base = None
        for m in modes:
            vals = [r[key] for r in rs if r["mode"] == m and r.get(key) is not None]
            if not vals:
                continue
            v = med(vals)
            if base is None and m in ("yield", "exclusive"):
                base = v
            rel = "" if base in (None, 0) else f"   ({100*(v-base)/base:+.1f}% vs 第一档)"
            print(f"    {m:11s} med={fmt % v:>9s}  [{fmt % min(vals)}, {fmt % max(vals)}]"
                  f"  n={len(vals)}{rel}")
    # 每 run 明细（B 场景 TTFT 是三条各自的）
    if scen == "B":
        print("  逐 run 的三条 TTFT / e2e (ms):")
        for r in rs:
            print(f"    {r['mode']:11s} run={r['run']}  ttft={r['ttft_ms']}  e2e={r['e2e_ms']}")
    if scen == "C":
        print("  逐 run:")
        for r in rs:
            print(f"    {r['mode']:11s} run={r['run']}  wall={r['wall_s']}s"
                  f"  tokens={r['delivered_tokens']}  tok/s={r['tok_per_s']}")
    # 混合 step 计数
    print("  混合 step 计数（concurrent/serial 档）:")
    for m in modes:
        ss = [r.get("mixed_steps") for r in rs if r["mode"] == m and r.get("mixed_steps")]
        if ss and any(x.get("concurrent_steps") or x.get("serial_steps") for x in ss):
            print(f"    {m:11s} {ss}")
    print()
