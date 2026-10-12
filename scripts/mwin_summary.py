#!/usr/bin/env python
"""mwin_summary.py —— 把 M 扫描 / P7-D 二维网格的 acceptance_matrix JSON 汇总成表。

用法：
  python scripts/mwin_summary.py out/mw/mw_*.json
  # 或给定目录，自动收 *.json
输出：
  · 每个 (P, M) 的容量 / 接受率 / tok/s(maxB) / ms 每轮(maxB) / TTFT 表
  · ★ 「要求接受率 >= 阈值时，draft 池最小的那个 (P,M)」的明确答案
"""
import glob
import json
import os
import sys


def load(paths):
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                out = json.load(f)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        b = out.get("blocks", {})
        M = b.get("draft_window_blocks", out.get("window", 0) // 256)
        P = b.get("draft_head_blocks", out.get("head", 0))
        for cell in out.get("cells", []):
            v = cell["variants"].get("L") or next(iter(cell["variants"].values()))
            rows.append(dict(
                group=out["group"], P=P, M=M, B=cell["B"], ctx=cell["ctx"],
                target=b.get("num_kvcache_blocks"), draft_pool=b.get("num_draft_blocks"),
                win=out.get("window"),
                alpha=(v.get("accept_rate") or {}).get("median"),
                tok_s_maxB=(v.get("tok_per_s_maxB") or {}).get("median"),
                ms_maxB=(v.get("ms_per_round_maxB") or {}).get("median"),
                ttft=(v.get("ttft_ms") or {}).get("median"),
                tok_per_round=(v.get("tok_per_round") or {}).get("median"),
                maxB=v.get("maxB"), nf=(v.get("noise_floor_maxB_pct")),
                out_md5=v.get("out_md5"),
            ))
    return rows


def main():
    args = sys.argv[1:]
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, "*.json")))
        else:
            paths += sorted(glob.glob(a))
    rows = load(paths)
    if not rows:
        print("no rows")
        return
    ctxs = sorted({r["ctx"] for r in rows})
    for ctx in ctxs:
        for B in sorted({r["B"] for r in rows if r["ctx"] == ctx}):
            rs = [r for r in rows if r["ctx"] == ctx and r["B"] == B]
            rs.sort(key=lambda r: (r["draft_pool"] if r["draft_pool"] is not None else 1e9))
            print(f"\n=== ctx={ctx} B={B} (natural long text, wikitext2, out={rows[0].get('out','?')}) ===")
            print("| group | P | M | draft池 | target池 | 容量(token) | 受理率 | tok/s(maxB) | ms/轮(maxB) | TTFT ms | 噪声地板 | out_md5 |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for r in rs:
                cap = (r["target"] * 256) if r["target"] else None
                a = f'{r["alpha"]:.4f}' if r["alpha"] is not None else "-"
                print(f'| {r["group"]} | {r["P"]} | {r["M"]} | {r["draft_pool"]} | {r["target"]} | '
                      f'{cap} | {a} | {r["tok_s_maxB"]:.1f} | {r["ms_maxB"]:.2f} | '
                      f'{r["ttft"]:.1f} | {r["nf"]:.2f}% | {str(r["out_md5"])[:8]} |')

    # ---- ★ 关键问题：受理率 >= 阈值时 draft 池最小的那个角 ----
    for thr in (0.75, 0.70, 0.60):
        print(f"\n### 受理率 >= {thr} 的候选（按 draft 池升序）")
        cand = [r for r in rows if r["alpha"] is not None and r["alpha"] >= thr]
        # 同一 (P,M,ctx,B) 只留一个
        seen = {}
        for r in cand:
            seen[(r["P"], r["M"], r["ctx"], r["B"])] = r
        cand = sorted(seen.values(), key=lambda r: (r["draft_pool"], r["ctx"], r["B"]))
        if not cand:
            print("  （无：没有任何配置达到该受理率）")
        for r in cand[:12]:
            print(f'  ctx={r["ctx"]} B={r["B"]} P={r["P"]} M={r["M"]}  draft池={r["draft_pool"]} '
                  f'target={r["target"]} α={r["alpha"]:.4f} TTFT={r["ttft"]:.1f}ms '
                  f'tok/s(maxB)={r["tok_s_maxB"]:.1f}')


if __name__ == "__main__":
    main()
