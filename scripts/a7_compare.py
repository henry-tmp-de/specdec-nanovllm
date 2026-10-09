"""A 步正确性对拍的分析器：读 a7_verify.py 的日志 + 落盘的 draft logits。

结论口径
--------
  · 复用（NEW）必须复现全量补齐（OLD）与无复用参考（REF）：
      接受率 / 提出数 / 接受数 / 输出 token 完全一致
      第一个 propose 的 draft logits 差异在数值噪声量级（并报 argmax 一致率）
  · 负对照（NEG）必须【仍然回退到补齐】（catchup_tokens > 0）并复现 REF；
  · 反事实（NEGX，假装 target 命中就等于 draft 有效）必须【明显跑偏】——
    这一条证明了那道闸是承重的（否则负对照根本测不出东西）。

用法: python a7_compare.py <verify 输出目录> [基准模式=NEW]
"""
import os
import sys
import json
import glob

import torch


def load(d):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        for line in open(f, errors="ignore"):
            if line.startswith("@@B@@"):
                r = json.loads(line[5:])
                r["_file"] = os.path.basename(f)
                recs.setdefault(r["mode"], []).append(r)
    for m in recs:
        recs[m].sort(key=lambda r: r["rep"])
    return recs


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def fmt(rng):
    lo, hi = min(rng), max(rng)
    return f"{lo:.4f}" if abs(hi - lo) < 1e-9 else f"{lo:.4f}~{hi:.4f}"


def main():
    d = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "NEW"
    recs = load(d)
    if not recs:
        print("没有找到 @@B@@ 记录"); return 1
    order = [m for m in ("REF", "NEW", "OLD", "NEG", "NEGX") if m in recs]
    order += [m for m in sorted(recs) if m not in order]

    print("== 逐模式汇总（中位数 + 组间范围）==")
    print(f"{'mode':6} {'rep':>3} {'接受率':>18} {'提出':>6} {'接受':>6} "
          f"{'补齐tok':>9} {'补齐fwd':>8} {'TTFTms':>9} {'wall s':>9}  out_md5")
    for m in order:
        rs = recs[m]
        ar = [r["accept_rate"] for r in rs if r["accept_rate"] is not None]
        pr = [r["proposed"] for r in rs]
        ac = [r["accepted"] for r in rs]
        ct = [r["draft_counters"]["catchup_tokens"] for r in rs]
        cf = [r["draft_counters"]["catchup_forwards"] for r in rs]
        tt = [r["ttft_ms"] for r in rs]
        wl = [r["wall_s"] for r in rs]
        md5s = sorted({r["out_md5"][:8] for r in rs})
        print(f"{m:6} {len(rs):>3} {fmt(ar):>18} {fmt(pr):>6} {fmt(ac):>6} "
              f"{fmt(ct):>9} {fmt(cf):>8} {fmt(tt):>9} {fmt(wl):>9}  {','.join(md5s)}")

    # ---- draft logits 对拍（用 rep 最小的那一次）----
    print("\n== 第一个 propose 的原始 logits 对拍（相对 %s，rep 最小的一次）==" % base)
    base_rec = recs.get(base, [None])[0]
    if base_rec is None:
        print("缺少基准模式"); return 1
    bl = torch.load(base_rec["draft_logits_saved"], map_location="cpu")
    ok = True
    for m in order:
        r = recs[m][0]
        p = r.get("draft_logits_saved")
        if not p or not os.path.exists(p):
            print(f"  {m:5} 无 logits"); continue
        l = torch.load(p, map_location="cpu")
        if l.shape != bl.shape:
            print(f"  {m:5} ★ shape 不同 {list(l.shape)} vs {list(bl.shape)}"); ok = False; continue
        dmax = (l - bl).abs().max().item()
        rel = dmax / max(1e-9, bl.abs().max().item())
        agree = (l.argmax(-1) == bl.argmax(-1)).float().mean().item()
        flag = "OK " if (agree >= 0.99 and dmax < 0.5) else "★DIFF"
        print(f"  {m:5} max|Δ|={dmax:9.5f}  max|base|={bl.abs().max().item():9.3f}  "
              f"rel={rel:.2e}  argmax 一致率={agree:.6f}  [{flag}]")
        if flag.strip() == "★DIFF":
            ok = False

    # ---- 判定 ----
    print("\n== 判定 ==")
    def same_out(a, b):
        ra, rb = recs.get(a, []), recs.get(b, [])
        if not ra or not rb:
            return None
        return len({r["out_md5"] for r in ra} | {r["out_md5"] for r in rb}) == 1
    verdicts = []
    if "REF" in recs and "NEW" in recs:
        v = same_out("REF", "NEW")
        verdicts.append(("① 复用路径复现无复用参考的输出", v))
    if "NEW" in recs and "OLD" in recs:
        v = same_out("NEW", "OLD")
        ra = med([r["accept_rate"] for r in recs["NEW"]])
        rb = med([r["accept_rate"] for r in recs["OLD"]])
        verdicts.append(("② 复用路径复现全量补齐路径的输出", v))
        verdicts.append((f"③ 接受率一致（NEW={ra:.4f} OLD={rb:.4f} Δ={abs(ra-rb):.2e}）",
                         abs(ra - rb) < 1e-4))
    if "NEW" in recs:
        v = med([r["draft_counters"]["catchup_tokens"] for r in recs["NEW"]])
        verdicts.append((f"④ 修复生效：NEW 的补齐量 = {v}", v == 0))
    if "NEG" in recs:
        v = med([r["draft_counters"]["catchup_tokens"] for r in recs["NEG"]])
        verdicts.append((f"⑤ 负对照仍回退到补齐（NEG 补齐 = {v} > 0）", v > 0))
        verdicts.append(("⑥ 负对照输出 == 参考（闸门拦住了脏 KV）", same_out("NEG", "REF")))
    if "NEGX" in recs and "NEG" in recs:
        ra = med([r["accept_rate"] for r in recs["NEGX"] if r["accept_rate"]])
        rb = med([r["accept_rate"] for r in recs["NEG"] if r["accept_rate"]])
        v = same_out("NEGX", "NEG")
        verdicts.append((f"⑦ 承重性：跳过补齐会跑偏（NEGX 接受率 {ra:.4f} vs NEG {rb:.4f}）",
                         (v is False) or (abs(ra - rb) > 0.01)))
    allok = True
    for name, v in verdicts:
        mark = "PASS" if v else "FAIL"
        if not v:
            allok = False
        print(f"  [{mark}] {name}")
    print("\n" + ("✓ A 步正确性全部通过" if allok and ok else "✗ 有未通过项"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
