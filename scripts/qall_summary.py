#!/usr/bin/env python
"""qall_summary.py —— 把 bench_draft_int8.py 的若干份 JSON 汇总成报告用的表

算的东西（都是配对的，同一 run 内 int8 vs bf16）：
  · ms/轮 配对比值（中位数 + 9 组里 int8 更快的组数 + 最小/最大）
  · 每次 draft 前向省下的时间 = (ms/轮 bf16 − ms/轮 int8)/K  → 与微基准预测相除得「捕获率」
  · 投机 / 普通 的比值（tok/s 口径）在 bf16 与 int8 两行各是多少
  · 空转噪声地板（S_bf16 自身组间范围）

用法: python scripts/qall_summary.py <json> [<json> ...]
"""
import json
import sys
import statistics


def one(r):
    K = r["k"]
    b = r["S_bf16"]["raw_ms_per_round"]
    q = r["S_int8"]["raw_ms_per_round"]
    ratio = [q[i] / b[i] for i in range(len(b))]
    wins = sum(1 for x in ratio if x < 1)
    dsav = (statistics.median(b) - statistics.median(q)) / K * 1e3      # µs / draft 前向
    tot_b = r["draft_total_bytes_bf16"]
    tot_q = r["draft_total_bytes_int8"]
    per_fwd_saved_gb = (tot_b - tot_q) / 2 ** 30
    return {
        "load": f"{r['prompts']} B={r['B']}/ctx={r['ctx']}",
        "scope": r["draft_scope"],
        "U": r["U"]["tok_per_s"]["median"],
        "S_bf16": r["S_bf16"]["tok_per_s"]["median"],
        "S_int8": r["S_int8"]["tok_per_s"]["median"],
        "msr_bf16": r["S_bf16"]["ms_per_round"]["median"],
        "msr_int8": r["S_int8"]["ms_per_round"]["median"],
        "pair_med": statistics.median(ratio), "pair_min": min(ratio), "pair_max": max(ratio),
        "wins": f"{wins}/{len(ratio)}",
        "rounds_bf16": r["S_bf16"]["rounds"]["median"],
        "rounds_int8": r["S_int8"]["rounds"]["median"],
        "acc_bf16": (r["S_bf16"]["accept_rate"] or {}).get("median"),
        "acc_int8": (r["S_int8"]["accept_rate"] or {}).get("median"),
        "floor_pct": r["noise_floor_ms_per_round"]["range_pct"],
        "spec_over_plain_bf16": r["S_bf16"]["tok_per_s"]["median"] / r["U"]["tok_per_s"]["median"],
        "spec_over_plain_int8": r["S_int8"]["tok_per_s"]["median"] / r["U"]["tok_per_s"]["median"],
        "us_saved_per_fwd": dsav,
        "gb_saved_per_fwd": per_fwd_saved_gb,
        "draft_bytes_bf16_GB": tot_b / 2 ** 30, "draft_bytes_int8_GB": tot_q / 2 ** 30,
    }


def main():
    rows = []
    for f in sys.argv[1:]:
        with open(f) as fh:
            for r in json.load(fh):
                rows.append(one(r))
    hdr = (f"{'负载':>18} {'U tok/s':>8} {'S_bf16':>7} {'S_int8':>7} | "
           f"{'ms/轮 bf16':>10} {'int8':>8} | {'配对比值':>8} {'更快':>5} "
           f"{'区间':>17} | {'轮数':>11} {'接受率':>15} | {'地板%':>6} "
           f"| {'投机/普通 bf16':>7} {'int8':>6} | {'省µs/前向':>9}")
    print(hdr)
    for r in rows:
        acc = (f"{r['acc_bf16']:.4f}->{r['acc_int8']:.4f}"
               if r["acc_bf16"] is not None else "-")
        print(f"{r['load']:>18} {r['U']:>8.1f} {r['S_bf16']:>7.1f} {r['S_int8']:>7.1f} | "
              f"{r['msr_bf16']:>10.3f} {r['msr_int8']:>8.3f} | "
              f"{r['pair_med']:>8.4f} {r['wins']:>5} "
              f"{'[%.3f,%.3f]' % (r['pair_min'], r['pair_max']):>17} | "
              f"{r['rounds_bf16']:>5.0f}->{r['rounds_int8']:<5.0f} {acc:>15} | "
              f"{r['floor_pct']:>6.2f} | {r['spec_over_plain_bf16']:>7.3f} "
              f"{r['spec_over_plain_int8']:>6.3f} | {r['us_saved_per_fwd']:>9.1f}")
    print()
    print("JSON:")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
