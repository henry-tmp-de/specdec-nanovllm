"""Render reviewed report summaries as standalone Matplotlib PNG/PDF figures.

The plotter does not read benchmark raw data or recompute statistics. It uses
only report_results' exported medians, min/max values, matched comparisons and
precomputed curve segments. Missing/failed points are omitted and break lines.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

BATCHES = (1, 2, 4, 8, 16)
CONTEXTS = (1024, 4096)
PANELS = (("code", "代码"), ("math", "数学"), ("general", "通用"))
ARMS = {
    "nospec": {"label": "无投机", "color": "#374151", "marker": "o", "window": 0, "mode": "full"},
    "ar_full": {"label": "AR 全上下文（k=6）", "color": "#2563a6", "marker": "s", "window": 0, "mode": "full"},
    "ar_window": {"label": "AR 窗口 512（块对齐）", "color": "#0b8276", "marker": "^", "window": 512, "mode": "block_aligned"},
    "dspark_full": {"label": "DSpark 全上下文（k=7）", "color": "#c46a15", "marker": "D", "window": 0, "mode": "full"},
    "dspark_ring": {"label": "DSpark 环形窗口 512", "color": "#ad3044", "marker": "P", "window": 512, "mode": "ring"},
}


def choose_font(explicit=None):
    candidates = ([Path(explicit)] if explicit else
                  [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
                   Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                   Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")])
    for candidate in candidates:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            return font_manager.FontProperties(fname=str(candidate)).get_name(), str(candidate)
    if explicit:
        raise ValueError(f"指定字体不存在：{explicit}")
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei")
    available = {font.name: font.fname for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name, available[name]
    raise RuntimeError("本机未找到中文字体；请用 --font 指定已有字体文件")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def usable_stat(stats):
    if not isinstance(stats, dict) or stats.get("n_available") != 3:
        return False
    values = stats.get("values")
    if not isinstance(values, list) or len(values) != 3 or not all(finite_number(v) for v in values):
        return False
    lo, mid, hi = stats.get("min"), stats.get("median"), stats.get("max")
    return all(finite_number(v) for v in (lo, mid, hi)) and lo <= mid <= hi


def scope_key(phase, ctx, panel, arm, batch, budget):
    setting = ARMS[arm]
    return (phase, ctx, panel, arm, batch, setting["window"], setting["mode"], budget)


def select_cells(report, args):
    indexed = {}
    for cell in report.get("cells", []):
        scope = cell.get("scope", {})
        arm = scope.get("arm")
        if (scope.get("phase") != "formal" or arm not in ARMS or scope.get("ctx") not in CONTEXTS
                or scope.get("panel") not in {p for p, _ in PANELS} or scope.get("requested_batch") not in BATCHES
                or scope.get("gpu_memory_utilization") != args.budget or scope.get("graph_mode") != "graph"):
            continue
        expected = ARMS[arm]
        if scope.get("window") != expected["window"] or scope.get("mode") != expected["mode"]:
            continue
        key = scope_key("formal", scope["ctx"], scope["panel"], arm, scope["requested_batch"], args.budget)
        if key in indexed:
            raise ValueError(f"报告同一单元出现多条记录：{key}")
        indexed[key] = cell
    selected, status, commits, targets = {}, {}, set(), set()
    draft_models = {"ar": set(), "dspark": set()}
    for ctx in CONTEXTS:
        for panel, _ in PANELS:
            for arm in ARMS:
                for batch in BATCHES:
                    key = scope_key("formal", ctx, panel, arm, batch, args.budget)
                    cell = indexed.get(key)
                    reason = "未计划或未记录" if cell is None else str(cell.get("status", "状态缺失"))
                    groups = cell.get("variants", []) if cell else []
                    if cell and cell.get("status") == "complete" and len(groups) == 1:
                        group = groups[0]
                        config = group.get("configuration", {})
                        commit = config.get("engine_commit")
                        if (not isinstance(commit, str) or not commit.startswith(args.engine_commit)
                                or config.get("gpu_uuid", "").casefold() != args.gpu_uuid.casefold()
                                or config.get("seed") != args.seed):
                            reason = "源码/GPU/seed不匹配"
                        elif not group.get("all_valid") or not group.get("repeat_set_complete") or len(group.get("runs", [])) != 3:
                            reason = "未通过审计或非3次"
                        elif (config.get("engine_dirty") is not False or config.get("temperature") != 1.0
                              or config.get("output_tokens") != 256 or config.get("prefix_lookup") is not False
                              or config.get("mixed_overlap") is not False or config.get("ignore_eos") is not True
                              or config.get("gpu_memory_utilization") != args.budget):
                            reason = "运行条件不匹配"
                        elif (not isinstance(config.get("graph_policy"), dict)
                              or config["graph_policy"].get("name") != "exact_all"
                              or any(config["graph_policy"].get(k) is not False for k in
                                     ("target_decode_padding", "target_verify_padding", "draft_padding"))):
                            reason = "图策略不匹配"
                        else:
                            target = config.get("target_provenance")
                            draft = config.get("draft_provenance")
                            if not isinstance(target, dict) or not target.get("config_sha256") or not target.get("path"):
                                reason = "target身份未记录"
                            elif arm != "nospec" and (not isinstance(draft, dict) or not draft.get("config_sha256") or not draft.get("path")):
                                reason = "draft身份未记录"
                            else:
                                commits.add(commit)
                                targets.add(json.dumps(target, sort_keys=True, ensure_ascii=False))
                                if arm.startswith("ar_"):
                                    draft_models["ar"].add(json.dumps(draft, sort_keys=True, ensure_ascii=False))
                                elif arm.startswith("dspark_"):
                                    draft_models["dspark"].add(json.dumps(draft, sort_keys=True, ensure_ascii=False))
                                selected[key] = group
                                reason = "已审/3次"
                    status[key] = reason
    if not selected:
        raise ValueError("无匹配的正式3次已审单元；请检查报告、源码、GPU、预算和seed")
    if len(commits) != 1 or len(targets) != 1 or any(len(x) > 1 for x in draft_models.values()):
        raise ValueError("所选数据混入多个源码、target或同系列draft身份；拒绝绘图")
    return selected, status, next(iter(commits)), json.loads(next(iter(targets))), draft_models


def curve_index(report, args):
    curves = {}
    for curve in report.get("curves", []):
        c = curve.get("configuration", {})
        arm = c.get("arm")
        if (c.get("phase") != "formal" or arm not in ARMS or c.get("ctx") not in CONTEXTS
                or c.get("panel") not in {p for p, _ in PANELS} or c.get("gpu_memory_utilization") != args.budget
                or c.get("graph_mode") != "graph" or c.get("window") != ARMS[arm]["window"]
                or c.get("mode") != ARMS[arm]["mode"]):
            continue
        key = (c["ctx"], c["panel"], arm)
        if key in curves:
            raise ValueError(f"报告出现重复曲线：{key}")
        curves[key] = curve
    return curves


def comparison_index(report, args):
    out = {}
    for comp in report.get("comparisons", []):
        s = comp.get("scope", {})
        arm = comp.get("candidate_arm")
        if (arm not in ("dspark_full", "dspark_ring") or comp.get("baseline_arm") != "nospec"
                or s.get("phase") != "formal" or s.get("ctx") not in CONTEXTS
                or s.get("panel") not in {p for p, _ in PANELS} or s.get("requested_batch") not in BATCHES
                or s.get("gpu_memory_utilization") != args.budget or s.get("window") != ARMS[arm]["window"]):
            continue
        key = (s["ctx"], s["panel"], arm, s["requested_batch"])
        if key in out:
            raise ValueError(f"报告出现重复相对nospec对照：{key}")
        out[key] = comp
    return out


def series_points(ctx, panel, arm, metric, selected, curves, args, comparisons=None):
    """Use exported medians/ranges and segments verbatim; only filter missing points."""
    curve = curves.get((ctx, panel, arm))
    if not isinstance(curve, dict):
        return [], {}
    segments = curve.get("segments", [])
    point_map = {x.get("batch"): x for x in curve.get("points", []) if isinstance(x, dict)}
    measured, values = set(), {}
    for batch in BATCHES:
        key = scope_key("formal", ctx, panel, arm, batch, args.budget)
        group = selected.get(key)
        point = point_map.get(batch, {})
        if not group or point.get("median_full_request_tps") is None:
            continue
        stats = group.get("metrics", {}).get(metric)
        if comparisons is not None:
            base_key = scope_key("formal", ctx, panel, "nospec", batch, args.budget)
            comp = comparisons.get((ctx, panel, arm, batch), {})
            if base_key not in selected or comp.get("status") != "matched":
                continue
            stats = comp.get("full_request_ratio")
        elif metric == "full_request_tps" and isinstance(stats, dict) and stats.get("median") != point.get("median_full_request_tps"):
            raise ValueError("报告曲线中位数与已审单元中位数不一致")
        if usable_stat(stats):
            measured.add(batch)
            values[batch] = {"median": stats["median"], "min": stats["min"], "max": stats["max"]}
    filtered = []
    # For ratios, a segment is valid only where both the candidate and baseline
    # report segments contain the same adjacent batch edge.
    base_edges = None
    if comparisons is not None:
        base_curve = curves.get((ctx, panel, "nospec"), {})
        base_edges = {(a, b) for segment in base_curve.get("segments", [])
                      for a, b in zip(segment, segment[1:])}
    for segment in segments:
        current = []
        for batch in segment:
            if batch not in measured or (current and base_edges is not None and (current[-1], batch) not in base_edges):
                if current:
                    filtered.append(current)
                    current = []
            if batch in measured:
                current.append(batch)
        if current:
            filtered.append(current)
    return filtered, values


def draw_series(ax, segments, values, arm):
    style = ARMS[arm]
    for segment in segments:
        xs = [BATCHES.index(b) for b in segment]
        ys = [values[b]["median"] for b in segment]
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], linewidth=1.6,
                markersize=5.1, solid_capstyle="round", zorder=3)
        for x, b in zip(xs, segment):
            # Draw the already exported min/max directly, without recomputing a statistic.
            ax.vlines(x, values[b]["min"], values[b]["max"], color=style["color"], linewidth=1.2, zorder=2)
            ax.plot([x - 0.045, x + 0.045], [values[b]["min"]] * 2, color=style["color"], linewidth=1.0)
            ax.plot([x - 0.045, x + 0.045], [values[b]["max"]] * 2, color=style["color"], linewidth=1.0)


def status_annotation(ax, ctx, panel, status, budget, arms):
    missing, failed = [], []
    for batch in BATCHES:
        labels = [status.get(scope_key("formal", ctx, panel, arm, batch, budget), "未记录") for arm in arms]
        failures = ("failed_attempt_present", "unverified_or_invalid", "inconsistent_conditions",
                    "duplicate_repeat", "plan_condition_mismatch")
        if any(x in failures for x in labels):
            failed.append(str(batch))
        if any(x != "已审/3次" and x not in failures for x in labels):
            missing.append(str(batch))
    text = "部分方案缺测/未完成 B: " + ("、".join(missing) if missing else "无")
    if failed:
        text += "   部分方案失败/无效 B: " + "、".join(failed)
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=7.3,
            color="#4b5563", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5})


def metric_status(ctx, panel, metric, arms, selected, status, curves, args, comparisons=None):
    displayed = dict(status)
    for arm in arms:
        if comparisons is not None and arm == "nospec":
            continue
        segments, values = series_points(ctx, panel, arm, metric, selected, curves, args, comparisons)
        for b in BATCHES:
            key = scope_key("formal", ctx, panel, arm, b, args.budget)
            if displayed.get(key) == "已审/3次" and not (b in values and any(b in seg for seg in segments)):
                displayed[key] = "指标未导出或对照未匹配"
    return displayed


def style_axis(ax, title, ctx, panel, status, budget, arms):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(range(len(BATCHES)), labels=[str(b) for b in BATCHES])
    ax.set_xlim(-0.18, len(BATCHES) - 0.82)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5, colors="#374151")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.margins(y=0.19)
    status_annotation(ax, ctx, panel, status, budget, arms)


def legend_handles(arms):
    return [Line2D([0], [0], color=ARMS[a]["color"], marker=ARMS[a]["marker"], linewidth=1.6,
                   markersize=5.5, label=ARMS[a]["label"]) for a in arms]


def title(fig, heading, subtitle, arms, badge, tall=False):
    fig.suptitle(heading, fontsize=16, fontweight="bold", x=0.5, y=0.985)
    fig.text(0.5, 0.952 if not tall else 0.961, subtitle, ha="center", va="top", fontsize=9, color="#475569")
    handles = legend_handles(arms)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905 if not tall else 0.925),
               ncol=min(len(handles), 5), frameon=False, fontsize=9)
    fig.text(0.5, 0.042, "nano-vLLM 内部对照；prefix 关闭、在线 mixed 不支持；4K 为 Wiki 补齐吞吐面板，非任务质量测试。",
             ha="center", va="bottom", fontsize=8.5, color="#475569")
    fig.text(0.5, 0.018, "误差线为3次测量的 min–max（不是置信区间）；缺测/失败不绘点、不跨缺口连线。",
             ha="center", va="bottom", fontsize=8.5, color="#475569")
    if badge:
        fig.text(0.99, 0.985, badge, ha="right", va="top", fontsize=9, color="#b91c1c", fontweight="bold")


def save(fig, output, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=200 if suffix == "png" else None,
                    facecolor="white", metadata={"Creator": "Matplotlib"})
    plt.close(fig)


def make_figures(report, selected, status, curves, comparisons, args, outdir, target, commit, badge):
    target_name = str(target["path"]).replace("\\", "/").rstrip("/").split("/")[-1]
    subtitle = (f"{target_name} · 单卡 {args.gpu_uuid} · 显存预算 {args.budget:.2f} · T=1 · "
                f"输出256 · seed {args.seed} · source {commit[:12]}")
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.16, wspace=0.26, hspace=0.37)
    title(fig, "固定批完整请求吞吐", subtitle, tuple(ARMS), badge)
    for i, ctx in enumerate(CONTEXTS):
        for j, (panel, label) in enumerate(PANELS):
            ax = axes[i, j]
            for arm in ARMS:
                seg, vals = series_points(ctx, panel, arm, "full_request_tps", selected, curves, args)
                draw_series(ax, seg, vals, arm)
            display_status = metric_status(ctx, panel, "full_request_tps", tuple(ARMS), selected, status, curves, args)
            style_axis(ax, f"{ctx // 1024}K · {label}", ctx, panel, display_status, args.budget, tuple(ARMS))
            if j == 0:
                ax.set_ylabel("完整请求吞吐 (token/s)", fontsize=9)
            if i == 1:
                ax.set_xlabel("请求批量 B", fontsize=9)
    save(fig, outdir, "throughput")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.16, wspace=0.26, hspace=0.37)
    title(fig, "相对无投机：DSpark 两种策略", subtitle + " · 1×是参照定义，不是插补数据", ("dspark_full", "dspark_ring"), badge)
    for i, ctx in enumerate(CONTEXTS):
        for j, (panel, label) in enumerate(PANELS):
            ax = axes[i, j]
            ax.axhline(1.0, color="#9ca3af", linewidth=1.0, linestyle="--", zorder=1)
            for arm in ("dspark_full", "dspark_ring"):
                seg, vals = series_points(ctx, panel, arm, "full_request_tps", selected, curves, args, comparisons)
                draw_series(ax, seg, vals, arm)
            display_status = metric_status(ctx, panel, "full_request_tps", ("nospec", "dspark_full", "dspark_ring"),
                                           selected, status, curves, args, comparisons)
            style_axis(ax, f"{ctx // 1024}K · {label}", ctx, panel, display_status, args.budget,
                       ("nospec", "dspark_full", "dspark_ring"))
            if j == 0:
                ax.set_ylabel("完整请求吞吐 / 无投机", fontsize=9)
            if i == 1:
                ax.set_xlabel("请求批量 B", fontsize=9)
    save(fig, outdir, "relative-to-nospec")

    fig, axes = plt.subplots(4, 3, figsize=(15.5, 14.2), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.14, wspace=0.26, hspace=0.54)
    title(fig, "请求延迟：TTFT 与 TPOT", subtitle + " · 每次先取请求 P50，再展示3次中位与范围", tuple(ARMS), badge, tall=True)
    for i, (ctx, metric, name) in enumerate(((1024, "ttft_p50_s", "TTFT"), (4096, "ttft_p50_s", "TTFT"),
                                              (1024, "tpot_p50_s", "TPOT"), (4096, "tpot_p50_s", "TPOT"))):
        for j, (panel, label) in enumerate(PANELS):
            ax = axes[i, j]
            for arm in ARMS:
                seg, vals = series_points(ctx, panel, arm, metric, selected, curves, args)
                draw_series(ax, seg, vals, arm)
            display_status = metric_status(ctx, panel, metric, tuple(ARMS), selected, status, curves, args)
            style_axis(ax, f"{ctx // 1024}K · {label} · {name}", ctx, panel, display_status, args.budget, tuple(ARMS))
            if j == 0:
                ax.set_ylabel(f"{name} P50 (秒)", fontsize=9)
            if i == 3:
                ax.set_xlabel("请求批量 B", fontsize=9)
    save(fig, outdir, "latency-ttft-tpot")


def status_files(report, selected, status, curves, comparisons, args, outdir, commit, target, badge, font_path):
    rows = []
    for ctx in CONTEXTS:
        for panel, label in PANELS:
            for arm in ARMS:
                entry = {"ctx": ctx, "panel": panel, "arm": arm, "batches": {}}
                for b in BATCHES:
                    key = scope_key("formal", ctx, panel, arm, b, args.budget)
                    group = selected.get(key)
                    info = {"status": status.get(key, "未记录"), "metrics": {}}
                    for metric in ("full_request_tps", "ttft_p50_s", "tpot_p50_s"):
                        segments, plotted = series_points(ctx, panel, arm, metric, selected, curves, args)
                        info["metrics"][metric] = "已绘" if b in plotted and any(b in x for x in segments) else "缺测/无效"
                    if arm in ("dspark_full", "dspark_ring"):
                        segments, plotted = series_points(ctx, panel, arm, "full_request_tps", selected, curves, args, comparisons)
                        info["metrics"]["relative_to_nospec"] = ("已绘" if b in plotted and any(b in x for x in segments) else "缺测/不匹配")
                    entry["batches"][str(b)] = info
                rows.append(entry)
    data = {"interim": report["ledger"].get("total_is_complete") is not True, "figure_badge": badge,
            "formal_coverage_complete": report["summary"].get("primary_summary_is_complete") is True,
            "input_report": str(Path(args.report).resolve()),
            "source_commit": commit, "gpu_uuid": args.gpu_uuid, "budget": args.budget,
            "seed": args.seed, "target_provenance": target, "font_file": font_path,
            "relative_series": ["dspark_full", "dspark_ring"],
            "note": "statistics and gap segments come from report_results; no interpolation or additional ratios",
            "rows": rows}
    with (outdir / "plot-status.json").open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    lines = ["# 绘图覆盖与缺失", "", "图只呈现已审、匹配指定条件且完整3次的单元。",
             f"状态：{badge or '完整正式矩阵已覆盖'}；完整报告与全阶段GPU账目以终态复核为准。", "",
             "相对无投机图只显示报告已导出的 DSpark full/ring；1×是参考定义，不补算AR比值。", "",
             "| 上下文 | panel | arm | B1 | B2 | B4 | B8 | B16 |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        labels = [row["batches"][str(b)]["status"] for b in BATCHES]
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in
                     (row["ctx"], row["panel"], ARMS[row["arm"]]["label"], *labels)) + " |")
    lines += ["", "细分各metric是否绘出见 plot-status.json。缺测/失败不填零，曲线只沿报告给出的连续 segments 绘制。", ""]
    (outdir / "plot-status.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, help="report_results 生成的 JSON")
    p.add_argument("--out-dir", required=True, help="新输出目录；不得覆盖同名文件")
    p.add_argument("--engine-commit", required=True, help="完整commit或至少7位的唯一前缀")
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--budget", type=float, default=0.85)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--font", help="已有中文字体文件；默认查找本机字体")
    p.add_argument("--allow-interim", action="store_true", help="允许绘制明确标记的内部阶段性快照")
    args = p.parse_args()
    if len(args.engine_commit) < 7:
        p.error("engine-commit 至少需要7位")
    if not finite_number(args.budget) or not 0 < args.budget <= 1:
        p.error("budget 无效")
    report = json.loads(Path(args.report).read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or not all(k in report for k in ("cells", "curves", "comparisons", "summary", "ledger")):
        p.error("输入不是 report_results 导出的报告")
    if any(report.get(k) for k in ("input_errors", "plan_issues", "ledger_errors")):
        p.error("报告存在未解决的输入、计划或台账错误，拒绝绘图")
    ledger_complete = report["ledger"].get("total_is_complete") is True
    coverage_complete = report["summary"].get("primary_summary_is_complete") is True
    badge = ("阶段性内部预览" if not ledger_complete else "正式批量结束 · 覆盖不全" if not coverage_complete else None)
    if not ledger_complete and not args.allow_interim:
        p.error("ledger不是完整终态；内部预览需显式 --allow-interim")
    try:
        selected, status, commit, target, drafts = select_cells(report, args)
        curves = curve_index(report, args)
        comparisons = comparison_index(report, args)
        family, font_path = choose_font(args.font)
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    outdir = Path(args.out_dir)
    names = ("throughput.png", "throughput.pdf", "relative-to-nospec.png", "relative-to-nospec.pdf",
             "latency-ttft-tpot.png", "latency-ttft-tpot.pdf", "plot-status.json", "plot-status.md")
    if any((outdir / name).exists() for name in names):
        p.error("输出目录含同名结果，请使用新目录")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = family
    make_figures(report, selected, status, curves, comparisons, args, outdir, target, commit, badge)
    status_files(report, selected, status, curves, comparisons, args, outdir, commit, target, badge, font_path)
    print(f"已生成3组 PNG/PDF 及覆盖说明；选择{len(selected)}个已审单元；状态={badge or '完整覆盖'}；字体={font_path}")


if __name__ == "__main__":
    main()
