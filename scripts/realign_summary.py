#!/usr/bin/env python
"""realign_summary.py —— 汇总 out/matrix/*.json（engine_alpha 输出）成一张表。

每个 cell = (align, p, m, ctx)，把该 cell 的所有 run 聚合成 mean / SE(run-to-run)。
同时从 logs/matrix/<tag>.log 里抓 draft 池 / target 池块数。
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "matrix")
LOGS = os.path.join(ROOT, "logs", "matrix")


def parse_tag(tag):
    # a{align}_p{p}m{m}_c{ctx}_r{n} | w0_c{ctx}_r{n}
    m = re.match(r"^a(\d)_p(\d)m(\d+)_c(\d+)_r(\d+)$", tag)
    if m:
        return dict(align=int(m[1]), p=int(m[2]), m=int(m[3]), ctx=int(m[4]), r=int(m[5]))
    m = re.match(r"^w0_c(\d+)_r(\d+)$", tag)
    if m:
        return dict(align=None, p=0, m=0, ctx=int(m[1]), r=int(m[2]))
    return None


def pools(tag):
    p = os.path.join(LOGS, tag + ".log")
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding="utf-8", errors="ignore").read()
    d = {}
    mo = re.search(r"draft 池 (\d+) 块", txt)
    if mo:
        d["draft_blocks"] = int(mo.group(1))
    mo = re.search(r"target 池 (\d+) 块", txt)
    if mo:
        d["target_blocks"] = int(mo.group(1))
    return d


def main():
    cells = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        tag = d.get("tag") or os.path.basename(f)[:-5]
        key = parse_tag(tag)
        if key is None:
            continue
        ck = (key["ctx"], key["align"], key["p"], key["m"])
        cells.setdefault(ck, {"alphas": [], "frees": [],
                              "tags": [], "pools": {}})
        cells[ck]["alphas"].append(d["alpha_step"])
        cells[ck]["frees"].append(d.get("alpha_free"))
        cells[ck]["tags"].append(tag)
        if not cells[ck]["pools"]:
            cells[ck]["pools"] = pools(tag)

    def lab(k):
        ctx, align, p, m = k
        if m == 0:
            return f"w0  (ctx={ctx})"
        return f"A{align} P={p} M={m:2d} (ctx={ctx})"

    rows = []
    for k in sorted(cells, key=lambda x: (x[0], 0 if x[2] == 0 and x[3] == 0 else 1,
                                          x[1] if x[1] is not None else 0, x[2], x[3])):
        c = cells[k]
        a = c["alphas"]
        mean = statistics.mean(a)
        se = (statistics.stdev(a) / len(a) ** 0.5) if len(a) > 1 else float("nan")
        fr = [x for x in c["frees"] if x is not None]
        rows.append((lab(k), a, mean, se,
                     statistics.mean(fr) if fr else float("nan"), c["pools"],
                     round(max(a) - min(a), 4) if len(a) > 1 else 0.0))

    print(f"{'config':26s} {'mean a':>8s} {'SE':>7s} {'spread':>7s} {'a_free':>7s} "
          f"{'draft':>5s} {'tgt':>5s}  runs")
    for r in rows:
        pools_ = r[5]
        print(f"{r[0]:26s} {r[2]:8.4f} {r[3]:7.4f} {r[6]:7.4f} {r[4]:7.4f} "
              f"{pools_.get('draft_blocks',''):>5} {pools_.get('target_blocks',''):>5}  "
              f"{[round(x,4) for x in r[1]]}")

    # ---- 关键对比 ----
    print("\n=== 关键对比 ===")
    def g(ctx, align, p, m):
        return cells.get((ctx, align, p, m), {}).get("alphas")
    for ctx in (8192, 16384):
        w0 = g(ctx, None, 0, 0)
        print(f"\n-- ctx={ctx} --  w0 mean={statistics.mean(w0):.4f}" if w0 else "")
        for m in (5, 12):
            for align in (0, 1):
                a0 = g(ctx, align, 0, m)
                a2 = g(ctx, align, 2, m)
                if a0 and a2:
                    d = statistics.mean(a2) - statistics.mean(a0)
                    sa = (statistics.stdev(a0)/len(a0)**0.5) if len(a0) > 1 else 0
                    sb = (statistics.stdev(a2)/len(a2)**0.5) if len(a2) > 1 else 0
                    sig = abs(d) / ((sa**2 + sb**2) ** 0.5) if (sa or sb) else float("inf")
                    print(f"   M={m} ALIGN={align}: P=0 {statistics.mean(a0):.4f} vs "
                          f"P=2 {statistics.mean(a2):.4f}  Δ={d:+.4f} "
                          f"(pooled SE {(sa**2+sb**2)**0.5:.4f}, {sig:.1f}σ)")


if __name__ == "__main__":
    main()
