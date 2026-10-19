#!/usr/bin/env python
"""p17_merge_curve.py —— 把 6 个 outdir（ctx × group）合并成一张总表。

背景：P17 的批量速度曲线拆成 6 个并行进程跑（2 ctx × 3 group），
每个进程只写自己那 6 个 cell（B ∈ {1,2,4,8,16,32}）。
本脚本把 36 个 cell 读回来，渲染成与 `p17_batch_curve.py --all` 同样口径的三张表：
  ① 主表：tok/s(maxB) / active B / TTFT / ms/轮 / tok/轮 / ms/token / α / 噪声地板
  ② 相对提升表：S_nospec/G、S6/G、S6/S_nospec、S6/S6_nospec_win
  ③ 每 token 成本拆 + 配置表（max_num_seqs / M / window / KV 准入）

用法：
    python scripts/p17_merge_curve.py --root out/p17r --out out/p17r/COMBINED.md
"""
import argparse
import glob
import json
import os

LADDER = ["U", "G", "S_nospec", "S6", "S6_nospec_win"]
GROUP_OF = {"U": "base", "G": "base", "S_nospec": "s6", "S6": "s6",
            "S6_nospec_win": "nowin"}
# ★ ctx=4096 的 nowin 有两份：`ctx4k_nowin` 落在 GPU 0（挂 Xorg、SM 卡在 1395 MHz 的坏卡），
#   `ctx4k_nowin_good` 是补跑到一张好卡（GPU 6）上的干净版。正式表用后者；
#   前者只用来量「坏卡惩罚」（见文档 §环境状态）。
DIRS = ["ctx1k_base", "ctx1k_s6", "ctx1k_nowin",
        "ctx4k_base", "ctx4k_s6", "ctx4k_nowin_good"]


def load(root):
    """→ {(ctx, B, group): cell}"""
    cells = {}
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(root, d, "*.json"))):
            if os.path.basename(f) == "merged.json":
                continue
            with open(f) as fh:
                c = json.load(fh)
            cells[(c["ctx"], c["B"], c["group"])] = c
    return cells


def m(d, key):
    """取 median；缺就 None。"""
    if not d:
        return None
    x = d.get(key)
    if x is None:
        return None
    return x["median"] if isinstance(x, dict) else x


def nf(d, key):
    """取噪声地板 range_pct。"""
    if not d:
        return None
    x = d.get(key)
    if x is None or not isinstance(x, dict):
        return None
    return x.get("range_pct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/p17r")
    ap.add_argument("--out", default="out/p17r/COMBINED.md")
    a = ap.parse_args()

    cells = load(a.root)
    if not cells:
        raise SystemExit("没找到任何 cell，检查 --root")
    ctxs = sorted({c for c, _, _ in cells})
    bs = sorted({b for _, b, _ in cells})

    def get(ctx, B, v):
        return cells.get((ctx, B, GROUP_OF[v]))

    L = []
    L.append("# P17 批量速度曲线（合并表）\n")
    L.append("cells=%d，ctx=%s，B=%s，档=%s\n" % (len(cells), ctxs, bs, LADDER))

    # ---------------- ① 主表 ----------------
    L.append("\n## ① 主表：每档绝对数（`tok/s(maxB)` 为主指标）\n")
    L.append("| ctx | B | 档 | tok/s(maxB) | active B | TTFT ms | ms/轮(maxB) | "
             "tok/轮(maxB) | ms/token | α | 噪声地板 | 轮数@maxB | 总轮数 |")
    L.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ctx in ctxs:
        for B in bs:
            for v in LADDER:
                c = get(ctx, B, v)
                if c is None:
                    L.append("| %d | %d | %s | **跑不了** | - | - | - | - | - | - | - | - | - |"
                             % (ctx, B, v))
                    continue
                x = c["variants"][v]
                ar = m(x, "accept_rate")
                nfl = nf(x, "ms_per_round_maxB")
                L.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    ctx, B, v,
                    "%.1f" % m(x, "tok_per_s_maxB"),
                    x["maxB"],
                    "%.1f" % m(x, "ttft_ms"),
                    "%.3f" % m(x, "ms_per_round_maxB"),
                    "%.3f" % m(x, "tok_per_round_maxB"),
                    "%.3f" % m(x, "ms_per_token_maxB"),
                    "-" if ar is None else "%.4f" % ar,
                    "-" if nfl is None else "%.2f%%" % nfl,
                    x["rounds_maxB"], int(m(x, "rounds"))))
            L.append("")

    # ---------------- ② 相对提升表 ----------------
    L.append("\n## ② 相对提升表（比值都基于 `tok/s(maxB)`；`G` = nano-vllm 默认配置）\n")
    L.append("| ctx | B | U | **G** | S_nospec | **S6** | S6_nowin | "
             "**S6/G** | **S_nospec/G** | **S6/S_nospec** | **S6_nowin/G** | **S6/S6_nowin** |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def tok(ctx, B, v):
        c = get(ctx, B, v)
        if c is None:
            return None
        return m(c["variants"][v], "tok_per_s_maxB")

    def r(a_, b_):
        return "-" if (a_ is None or b_ is None) else "**%.3f×**" % (a_ / b_)

    for ctx in ctxs:
        for B in bs:
            vals = {v: tok(ctx, B, v) for v in LADDER}
            L.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                ctx, B,
                "-" if vals["U"] is None else "%.1f" % vals["U"],
                "-" if vals["G"] is None else "%.1f" % vals["G"],
                "-" if vals["S_nospec"] is None else "%.1f" % vals["S_nospec"],
                "-" if vals["S6"] is None else "%.1f" % vals["S6"],
                "-" if vals["S6_nospec_win"] is None else "%.1f" % vals["S6_nospec_win"],
                r(vals["S6"], vals["G"]),
                r(vals["S_nospec"], vals["G"]),
                r(vals["S6"], vals["S_nospec"]),
                r(vals["S6_nospec_win"], vals["G"]),
                r(vals["S6"], vals["S6_nospec_win"])))
        L.append("")

    # ---------------- ③ 每 token 成本拆 ----------------
    L.append("\n## ③ 每 token 成本拆（`ms/轮(maxB)` ÷ `tok/轮(maxB)`）\n")
    L.append("| ctx | B | G ms/tok | G tok/轮 | S6 ms/tok | S6 tok/轮 | nowin ms/tok | "
             "nowin tok/轮 | S6 每轮比 G 贵 | S6 每轮多吐 | S6 α | nowin α |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ctx in ctxs:
        for B in bs:
            g, s, n = get(ctx, B, "G"), get(ctx, B, "S6"), get(ctx, B, "S6_nospec_win")
            if not (g and s and n):
                L.append("| %d | %d | - | - | - | - | - | - | - | - | - | - |" % (ctx, B))
                continue
            gx, sx, nx = g["variants"]["G"], s["variants"]["S6"], n["variants"]["S6_nospec_win"]
            L.append("| %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f× | %.2f× | %s | %s |" % (
                ctx, B,
                m(gx, "ms_per_token_maxB"), m(gx, "tok_per_round_maxB"),
                m(sx, "ms_per_token_maxB"), m(sx, "tok_per_round_maxB"),
                m(nx, "ms_per_token_maxB"), m(nx, "tok_per_round_maxB"),
                m(sx, "ms_per_round_maxB") / m(gx, "ms_per_round_maxB"),
                m(sx, "tok_per_round_maxB") / m(gx, "tok_per_round_maxB"),
                "%.4f" % (m(sx, "accept_rate") or float("nan")),
                "%.4f" % (m(nx, "accept_rate") or float("nan"))))
        L.append("")

    # ---------------- ④ 配置表 / KV 准入 ----------------
    L.append("\n## ④ 配置表与 KV 准入\n")
    L.append("| ctx | B | group | max_num_seqs | window(M×256) | draft 私有池 | "
             "target 池块数 | 每序列需要块 | 理论最大并发 | 实际 active B |")
    L.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for ctx in ctxs:
        for B in bs:
            for gname in ("base", "s6", "nowin"):
                c = cells.get((ctx, B, gname))
                if c is None:
                    continue
                bl = c["blocks"]
                per_seq = ctx // 256 + 1
                tb = bl["num_kvcache_blocks"]
                L.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    ctx, B, gname, c["max_num_seqs"], c["window"], bl["num_draft_blocks"],
                    tb, per_seq, tb // per_seq,
                    "/".join(str(c["variants"][v]["maxB"]) for v in c["variants"])))
        L.append("")

    txt = "\n".join(L)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n@@WROTE@@" + a.out)


if __name__ == "__main__":
    main()
