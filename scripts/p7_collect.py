"""B 步结果汇总：W 曲线表 + 「默认档是否与 1c1d907 逐字节等价」的对拍。

用法:
  python p7_collect.py curve  <p7-runs 目录>          # W × 负载 的曲线
  python p7_collect.py equiv  <equiv 目录>            # base(1c1d907) vs curr
"""
import os
import sys
import json
import glob

import torch


def load(d, prefix=""):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        if prefix and not os.path.basename(f).startswith(prefix):
            continue
        for line in open(f, errors="ignore"):
            if line.startswith("@@B@@"):
                r = json.loads(line[5:])
                r["_file"] = os.path.basename(f)
                recs.setdefault((r.get("workload", r.get("mode")), r["W"] if "W" in r else None),
                                []).append(r)
    return recs


def rng(xs, fmt="%.3f"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    xs = sorted(xs)
    m = xs[len(xs) // 2]
    if xs[0] == xs[-1]:
        return fmt % m
    return (fmt % m) + " [%s~%s]" % (fmt % xs[0], fmt % xs[-1])


def curve(d):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*", "W*.log"))):
        for line in open(f, errors="ignore"):
            if line.startswith("@@B@@"):
                r = json.loads(line[5:])
                recs.setdefault((r["workload"], r["W"]), []).append(r)
    if not recs:
        print("没有数据:", d)
        return 1
    wls = sorted({k[0] for k in recs})
    Ws = sorted({k[1] for k in recs})
    for wl in wls:
        base = recs.get((wl, 0), [])
        btp = sorted(r["output_tok_per_s"] for r in base)
        bmed = btp[len(btp) // 2] if btp else None
        bar = sorted(r["accept_rate"] for r in base if r["accept_rate"])
        barmed = bar[len(bar) // 2] if bar else None
        print(f"\n===== 负载 {wl} =====  (W=0 = 全上下文 draft = 1c1d907 行为)")
        print(f"{'W':>6} {'接受率':>22} {'吞吐 tok/s':>24} {'TTFT ms':>16} "
              f"{'轮/序列产出':>16} {'target块':>8} {'draft块':>7} {'KV总GB':>8} "
              f"{'KB/token':>9} {'容量tok':>9} {'峰显存GB':>8} {'回退':>4} {'rep':>3}")
        for W in Ws:
            rs = recs.get((wl, W))
            if not rs:
                continue
            g = lambda k, f="%.3f": rng([r.get(k) for r in rs], f)
            ar = rng([r["accept_rate"] for r in rs], "%.4f")
            tp = rng([r["output_tok_per_s"] for r in rs], "%.2f")
            tt = rng([r["ttft_median_ms"] for r in rs], "%.1f")
            rp = rng([r["out_tokens_per_round_per_seq"] for r in rs], "%.3f")
            nb = rng([r["num_kvcache_blocks"] for r in rs], "%d")
            nd = rng([r["num_draft_blocks"] for r in rs], "%d")
            kv = rng([r["kv_total_gb"] for r in rs], "%.3f")
            kpt = rng([r["kv_bytes_per_token"] for r in rs], "%.1f")
            cap = rng([r["target_capacity_tokens"] for r in rs], "%d")
            pk = rng([r["mem_alloc_peak_gb"] for r in rs], "%.2f")
            fb = rng([r["window_fallbacks"] for r in rs], "%d")
            print(f"{W:>6} {ar:>22} {tp:>24} {tt:>16} {rp:>16} {nb:>8} {nd:>7} "
                  f"{kv:>8} {kpt:>9} {cap:>9} {pk:>8} {fb:>4} {len(rs):>3}")

        # ---- 门槛判定 ----
        print(f"\n  [{wl}] 门槛判定（相对 W=0）")
        for W in Ws:
            if W == 0 or (wl, W) not in recs or not base:
                continue
            rs = recs[(wl, W)]
            tp = sorted(r["output_tok_per_s"] for r in rs)[len(rs) // 2]
            reg = (tp - bmed) / bmed * 100
            capb = base[0]["target_capacity_tokens"]
            capn = rs[0]["target_capacity_tokens"]
            arn = sorted(r["accept_rate"] for r in rs if r["accept_rate"])[len(rs) // 2]
            okv = "OK " if reg >= -3.0 else "✗  "
            print(f"    W={W:>5}: 容量 {capb}->{capn} (+{100*(capn-capb)/capb:+.1f}%) | "
                  f"接受率 {barmed:.4f}->{arn:.4f} (Δ{arn-barmed:+.4f}) | "
                  f"吞吐 {bmed:.2f}->{tp:.2f} ({reg:+.2f}%) [{okv}≤3% 回退]")
    return 0


def equiv(d):
    """base(1c1d907) vs curr(spec_draft_window=0)：必须逐字节等价。"""
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        for line in open(f, errors="ignore"):
            if line.startswith("@@B@@"):
                r = json.loads(line[5:])
                key = (os.path.basename(f).split("_")[0], r["mode"], r["rep"])
                recs[key] = r
                r["_file"] = os.path.basename(f)
    modes = sorted({k[1] for k in recs})
    reps = sorted({k[2] for k in recs})
    print("== 默认档逐字节等价：base=1c1d907  vs  curr=当前代码(spec_draft_window=0) ==")
    print(f"{'mode':>5} {'rep':>4} {'base省略':>9} {'curr省略':>9} {'base接受率':>11} "
          f"{'curr接受率':>11} {'base补齐':>9} {'curr补齐':>9} {'logits argmax':>13}")
    allok = True
    for m in modes:
        for rep in reps:
            b, c = recs.get(("base", m, rep)), recs.get(("curr", m, rep))
            if not b or not c:
                continue
            same = b["out_md5"] == c["out_md5"]
            ar_same = abs((b["accept_rate"] or -1) - (c["accept_rate"] or -1)) < 1e-9
            lg_txt, lg_ok = "-", True
            pb, pc = b.get("draft_logits_saved"), c.get("draft_logits_saved")
            if pb and pc and os.path.exists(pb) and os.path.exists(pc):
                lb, lc = torch.load(pb, map_location="cpu"), torch.load(pc, map_location="cpu")
                ag = (lb.argmax(-1) == lc.argmax(-1)).float().mean().item()
                dmax = (lb - lc).abs().max().item()
                lg_txt = f"{ag:.6f} (Δmax {dmax:.4f})"
                lg_ok = ag == 1.0
            ok = same and ar_same and lg_ok
            allok = allok and ok
            print(f"{m:>5} {rep:>4} {b['out_md5'][:8]:>9} {c['out_md5'][:8]:>9} "
                  f"{b['accept_rate']:>11} {c['accept_rate']:>11} "
                  f"{b['draft_counters']['catchup_tokens']:>9} "
                  f"{c['draft_counters']['catchup_tokens']:>9} {lg_txt:>13} "
                  f"[{'PASS' if ok else 'FAIL'}]")
    print("\n" + ("✓ 默认档与 1c1d907 逐字节等价（输出 md5 / 接受率 / draft logits argmax 全等）"
                  if allok else "✗ 存在不一致"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(curve(sys.argv[2]) if sys.argv[1] == "curve" else equiv(sys.argv[2]))
