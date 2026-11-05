"""CPU summaries of guarded DSpark quant micro or full-request JSONL records.

Full-request records use the previous compute_run; ratios pair the same repeat,
actual input hashes and complete recorded configuration against preset=none.
These summaries do not replace the guard, graph, numerical or plan audits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import glob
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments" / "20261105-dspark-window-resume" / "audit_results.py"
spec = importlib.util.spec_from_file_location("dspark_previous_raw_audit", OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
PRESETS = ("none", "ffn", "attention", "all")
REPEATS = {0, 1, 2}


def run_identity(row):
    actual = row.get("dspark_quant")
    actual = actual.get("preset") if isinstance(actual, dict) else None
    requested = row.get("dspark_quant_preset")
    metadata = row.get("input_metadata")
    hashes = ([item.get("actual_input_sha256") if isinstance(item, dict) else None
               for item in metadata] if isinstance(metadata, list) else None)
    identity = {field: row.get(field) for field in old.GROUP_FIELDS}
    identity.update(dspark_quant_preset=actual if actual is not None else requested, input_hashes=hashes,
                    diagnostic_instrumented=row.get("diagnostic_instrumented"))
    errors = []
    if actual not in PRESETS or requested != actual:
        errors.append("actual quant preset missing or differs from requested preset")
    if (not isinstance(hashes, list) or len(hashes) != row.get("requested_batch")
            or any(not isinstance(x, str) or len(x) != 64 or
                   any(c not in "0123456789abcdef" for c in x.lower()) for x in hashes)):
        errors.append("actual input hashes missing or incomplete")
    required = old.GROUP_FIELDS + ("guard_id",)
    if any(row.get(key) is None for key in required) or row.get("engine_dirty") is not False:
        errors.append("clean source/GPU/model/seed/config/guard identity missing")
    if row.get("diagnostic_instrumented") is not False:
        errors.append("diagnostic or unknown instrumentation")
    if type(row.get("repeat")) is not int or row["repeat"] not in REPEATS:
        errors.append("repeat must be 0, 1 or 2")
    return identity, errors


def decode_span(row):
    decode = [r for r in row["rounds"] if not r["prefill"]]
    return decode[-1]["end_s"] - decode[0]["start_s"] if decode else None


def stats(values):
    if len(values) != 3 or not all(type(v) in (float, int) and math.isfinite(v) for v in values):
        return None
    return {"values": values, "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(rows):
    groups = defaultdict(list)
    failures, warmups, other, invalid = [], [], [], []
    for source, line, row in rows:
        location = {"source": source, "line": line}
        if not isinstance(row, dict):
            invalid.append({**location, "error": "record is not an object", "record": row})
            continue
        kind = row.get("type")
        if kind == "failure":
            failures.append({**location, "record": row})
            continue
        if kind == "warmup":
            warmups.append({**location, "record": row})
            continue
        if kind != "run":
            other.append({**location, "record": row})
            continue
        identity, problems = run_identity(row)
        try:
            computed = old.compute_run(row)
            computed["decode_span_s"] = decode_span(row)
        except (KeyError, TypeError, ValueError, ArithmeticError, AttributeError) as exc:
            computed = {"valid": False, "errors": [f"malformed raw record: {exc}"], "decode_span_s": None}
        computed["errors"].extend(problems)
        computed["valid"] = not computed["errors"]
        quant = row.get("dspark_quant") if isinstance(row.get("dspark_quant"), dict) else {}
        entry = {**location, "repeat": row.get("repeat"), "guard_id": row.get("guard_id"),
                 "metrics": computed, "quantized_modules": quant.get("quantized_modules"),
                 "quantize_host_wall_s": quant.get("quantize_host_wall_s"),
                 "candidate_total_bytes": quant.get("candidate_total_bytes"),
                 "load_initial_graph_s": row.get("load_initial_graph_s"),
                 "target_kv_blocks": row.get("target_kv_blocks"), "graph_delta": row.get("graph_delta"),
                 "target_decode_graph": row.get("target_decode_graph"),
                 "memory_after_run": row.get("memory_after_run")}
        groups[old.canonical(identity)].append(entry)
        if not computed["valid"]:
            invalid.append({**location, "errors": computed["errors"]})
    result_groups = []
    for key, entries in sorted(groups.items()):
        repeats = [entry["repeat"] for entry in entries]
        complete = len(entries) == 3 and all(type(r) is int for r in repeats) and set(repeats) == REPEATS and all(e["metrics"]["valid"] for e in entries)
        ordered = sorted(entries, key=lambda e: e["repeat"]) if complete else entries
        def values(name, child=None):
            if not complete:
                return None
            raw = [e["metrics"].get(name) for e in ordered]
            return stats([x.get(child) if isinstance(x, dict) else None for x in raw] if child else raw)
        result_groups.append({"configuration": json.loads(key), "runs": entries, "three_repeats_valid": complete,
                              "summary": {"full_request_tps": values("full_request_tps"),
                                          "wall_s": values("wall_s"),
                                          "ttft_p50_s": values("ttft_s", "p50"),
                                          "tpot_p50_s": values("tpot_s", "p50"),
                                          "delivery_batch_gap_p50_s": values("delivery_batch_gap_s", "p50"),
                                          "decode_span_s": values("decode_span_s"),
                                          "acceptance_fraction": values("acceptance_fraction"),
                                          "tau": values("tau_delivered_per_decode_sequence_round")}})
    for group in result_groups:
        for entry in group["runs"]:
            matches = [w["record"] for w in warmups if w["record"].get("guard_id") == entry["guard_id"]]
            entry["prewarm_s"] = matches[0].get("prewarm_s") if len(matches) == 1 else None
    return {"scope": "compute_run summary only; no ledger, graph, warmup, plan or model-correctness audit",
            "performance_eligible": False, "three_repeat_range_is_confidence_interval": False,
            "load_note": "load_initial_graph_s includes initialization, initial graphs and host PTQ; PTQ is not additive",
            "decode_span_definition": "first decode round start to last decode round end, including host gaps",
            "groups": result_groups, "invalid": invalid, "failures": failures, "warmups": warmups, "other": other}


def pair_formal(result):
    """Pair only identical recorded conditions and actual inputs, repeat by repeat."""
    def common(configuration):
        return old.canonical({k: v for k, v in configuration.items() if k != "dspark_quant_preset"})
    bases = {common(group["configuration"]): group for group in result["groups"]
             if group["configuration"]["dspark_quant_preset"] == "none"}
    failed_guards = {item["record"].get("guard_id") for item in result["failures"]
                     if item["record"].get("guard_id")}
    pairs = []
    for group in result["groups"]:
        config = group["configuration"]
        if config["dspark_quant_preset"] not in ("ffn", "attention", "all"):
            continue
        baseline = bases.get(common(config))
        pair = {"configuration": config, "baseline_preset": "none", "pairs": [], "summary": None}
        if baseline is None:
            pair["status"] = "没有同条件 none 基准"
            pairs.append(pair)
            continue
        def by_repeat(entries):
            result = defaultdict(list)
            for entry in entries:
                if type(entry["repeat"]) is int and entry["repeat"] in REPEATS:
                    result[entry["repeat"]].append(entry)
            return result
        a, b = by_repeat(group["runs"]), by_repeat(baseline["runs"])
        for repeat in sorted(REPEATS):
            if len(a[repeat]) != 1 or len(b[repeat]) != 1:
                continue
            candidate, base = a[repeat][0], b[repeat][0]
            if (not candidate["metrics"]["valid"] or not base["metrics"]["valid"]
                    or candidate["guard_id"] in failed_guards or base["guard_id"] in failed_guards):
                continue
            x, y = candidate["metrics"].get("full_request_tps"), base["metrics"].get("full_request_tps")
            if (type(x) not in (float, int) or type(y) not in (float, int)
                    or not math.isfinite(x) or not math.isfinite(y) or y <= 0):
                continue
            pair["pairs"].append({"repeat": repeat, "ratio": x / y,
                                  "candidate_tps": x, "none_tps": y,
                                  "candidate_source": candidate["source"], "candidate_line": candidate["line"],
                                  "none_source": base["source"], "none_line": base["line"]})
        if len(pair["pairs"]) == 3:
            pair["summary"] = stats([p["ratio"] for p in pair["pairs"]])
            pair["status"] = "3对原始记录匹配；尚需独立核对 guard/graph"
        else:
            pair["status"] = f"只有 {len(pair['pairs'])}/3 对；不汇总"
        pairs.append(pair)
    return pairs


def summarize_micro(rows):
    measurements, failures, completions, other = [], [], [], []
    for source, line, row in rows:
        location = {"source": source, "line": line}
        if not isinstance(row, dict):
            other.append({**location, "record": row})
            continue
        kind = row.get("type")
        if kind == "failure":
            failures.append({**location, "record": row})
        elif kind == "complete":
            completions.append({**location, "record": row})
        elif kind == "micro":
            bf, quant = row.get("bf16_ms_per_call"), row.get("int8_ms_per_call")
            valid = (row.get("correctness_pass") is True and isinstance(bf, list) and isinstance(quant, list)
                     and len(bf) == len(quant) == 3 and all(type(x) in (float, int) and math.isfinite(x) and x > 0
                                                            for x in bf + quant))
            ratios = [a / b for a, b in zip(bf, quant)] if valid else []
            measurements.append({**location, "record": row, "timing_complete": valid,
                                 "paired_bf16_over_int8": stats(ratios) if valid else None})
        else:
            other.append({**location, "record": row})
    return {"scope": "layer0 linear micro with synthetic inputs; not full-model speedup",
            "ratio_definition": "per-round BF16 event time / INT8 event time, then median and min-max",
            "three_repeat_range_is_confidence_interval": False,
            "measurements": measurements, "failures": failures, "completions": completions, "other": other}


def csv_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value


def write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in columns})


def formal_rows(result):
    for group in result["groups"]:
        for run in group["runs"]:
            metrics = run["metrics"]
            graph = run.get("graph_delta") if isinstance(run.get("graph_delta"), dict) else {}
            item = {**group["configuration"], **run, "valid": metrics.get("valid"),
                    "errors": metrics.get("errors"), "wall_s": metrics.get("wall_s"),
                    "delivered_tokens": metrics.get("output_tokens"), "full_request_tps": metrics.get("full_request_tps"),
                    "decode_span_s": metrics.get("decode_span_s"),
                    "accepted": metrics.get("accepted_prefix_tokens"), "proposed": metrics.get("proposed_tokens"),
                    "acceptance_fraction": metrics.get("acceptance_fraction"),
                    "tau": metrics.get("tau_delivered_per_decode_sequence_round"),
                    "graph_hits": graph.get("graph_hits"), "graph_fallbacks": graph.get("graph_fallbacks"),
                    "verify_hits": graph.get("verify_hits"), "verify_fallbacks": graph.get("verify_fallbacks"),
                    "decode_graph_hits": graph.get("decode_graph_hits"),
                    "decode_graph_fallbacks": graph.get("decode_graph_fallbacks")}
            for field in ("ttft_s", "tpot_s", "delivery_batch_gap_s"):
                distribution = metrics.get(field) if isinstance(metrics.get(field), dict) else {}
                for percentile in ("p50", "p95", "p99"):
                    item[f"{field}_{percentile}"] = distribution.get(percentile)
            yield item


def show(value):
    return "—" if value is None else f"{value:.4g}" if type(value) is float else str(value)


def formal_markdown(result):
    lines = ["# DSpark W8A16 完整请求：CPU 原始记录汇总", "",
             "只复算请求与轮次记录；guard、实际图命中、计划和模型数值须独立审核，以下配对为暂定。",
             "三次范围是 min–max，不是置信区间。吞吐包含完整请求，解码跨度包含轮间主机间隔。",
             "加载时间包含初始图与 PTQ；host PTQ 是其中一部分，不能相加。同一 guard 的加载记录会在不同 panel 行重复，不是新增样本。",
             "显存预算相同不代表 target KV 块数相同；各来源/GPU/输入哈希详见 CSV 和 JSON。缺失用 —。", "",
             f"记录组 {len(result['groups'])}，无效 {len(result['invalid'])}，失败 {len(result['failures'])}，解析错误 {len(result['parse_errors'])}。", "",
             "| B | ctx | panel | preset | 显存预算 | 3次有效 | 吞吐 median [min,max] token/s | host PTQ 秒 | 加载+初始图 秒 | target KV 块 | 候选层字节 |",
             "|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|"]
    for group in result["groups"]:
        c, st = group["configuration"], group["summary"]["full_request_tps"]
        summary = "—" if st is None else f"{show(st['median'])} [{show(st['min'])}, {show(st['max'])}]"
        def per_run(field):
            values = [entry.get(field) for entry in group["runs"]]
            if not values:
                return "—"
            return show(values[0]) if all(v == values[0] for v in values) else ", ".join(show(v) for v in values)
        lines.append(f"| {c.get('requested_batch')} | {c.get('ctx')} | {c.get('panel')} | {c.get('dspark_quant_preset')} | {show(c.get('gpu_memory_utilization'))} | {group['three_repeats_valid']} | {summary} | {per_run('quantize_host_wall_s')} | {per_run('load_initial_graph_s')} | {per_run('target_kv_blocks')} | {per_run('candidate_total_bytes')} |")
    lines += ["", "| B | ctx | panel | preset 对 none | 匹配重复 | 吞吐比 median [min,max] | 状态 |",
              "|---:|---:|---|---|---:|---|---|"]
    for pair in result["pairs"]:
        c, st = pair["configuration"], pair["summary"]
        summary = "—" if st is None else f"{show(st['median'])} [{show(st['min'])}, {show(st['max'])}]"
        lines.append(f"| {c.get('requested_batch')} | {c.get('ctx')} | {c.get('panel')} | {c.get('dspark_quant_preset')} | {len(pair['pairs'])} | {summary} | {pair['status']} |")
    lines += ["", "逐次 TTFT/TPOT、接受量、真实交付、Graph 计数、KV 块和内存见 formal-runs.csv / formal-summary.json。", ""]
    return "\n".join(lines)


def micro_markdown(result):
    lines = ["# DSpark W8A16 layer0 算子 micro", "",
             "真实 checkpoint 权重、合成 BF16 激活；仅算子时间，不代表整模型或请求加速。",
             "时间为 CUDA Event 每调用均值；BF16/INT8 每轮配对后取中位数及 min–max，范围不是置信区间。", "",
             f"测量行 {len(result['measurements'])}，失败 {len(result['failures'])}，解析错误 {len(result['parse_errors'])}。", "",
             "| 权重 | B | M | 模式 | BF16 ms | INT8 ms | BF16/INT8 median [min,max] | 量化参照 RMS | 数值 |",
             "|---|---:|---:|---|---:|---:|---|---:|---|"]
    for item in result["measurements"]:
        r, st = item["record"], item["paired_bf16_over_int8"]
        ratio = "—" if st is None else f"{show(st['median'])} [{show(st['min'])}, {show(st['max'])}]"
        ref = r.get("quant_reference") if isinstance(r.get("quant_reference"), dict) else {}
        lines.append(f"| {r.get('weight_name')} | {r.get('batch')} | {r.get('rows')} | {r.get('mode')} | {show(r.get('bf16_median_ms'))} | {show(r.get('int8_median_ms'))} | {ratio} | {show(ref.get('relative_rms'))} | {r.get('correctness_pass')} |")
    lines += ["", "原 BF16 权重误差与 graph/eager 对照仅见 micro-summary.json；失败不填零。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("micro", "formal"))
    parser.add_argument("--input", required=True, nargs="+", help="raw JSONL paths or quoted glob patterns")
    parser.add_argument("--out-dir", required=True, help="output directory; never overwrite existing results")
    args = parser.parse_args()
    rows, parse_errors = [], []
    filenames = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parse_errors.append({"source": pattern, "line": None, "error": "no matching input"})
        filenames.extend(matches)
    for filename in dict.fromkeys(filenames):
        path = Path(filename)
        try:
            with path.open(encoding="utf-8") as stream:
                for line, text in enumerate(stream, 1):
                    if not text.strip():
                        continue
                    try:
                        def invalid_constant(value):
                            raise ValueError(f"nonfinite JSON constant {value}")
                        rows.append((str(path.resolve()), line, json.loads(text, parse_constant=invalid_constant)))
                    except (ValueError, json.JSONDecodeError) as exc:
                        parse_errors.append({"source": str(path.resolve()), "line": line, "error": str(exc)})
        except OSError as exc:
            parse_errors.append({"source": str(path), "line": None, "error": str(exc)})
    result = summarize(rows) if args.kind == "formal" else summarize_micro(rows)
    result["parse_errors"] = parse_errors
    if args.kind == "formal":
        result["pairs"] = pair_formal(result)
    output = Path(args.out_dir)
    names = [f"{args.kind}-summary.json", f"{args.kind}-report.md", f"{args.kind}-runs.csv"]
    if args.kind == "formal":
        names.append("formal-pairs.csv")
    if any((output / name).exists() for name in names):
        parser.error("output directory contains files with these names")
    output.mkdir(parents=True, exist_ok=True)
    with (output / names[0]).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if args.kind == "formal":
        cols = ("requested_batch", "ctx", "panel", "repeat", "dspark_quant_preset", "engine_commit", "gpu_uuid",
                "seed", "input_hashes", "source", "line", "guard_id", "valid", "errors", "wall_s",
                "delivered_tokens", "full_request_tps", "decode_span_s", "ttft_s_p50", "ttft_s_p95", "ttft_s_p99",
                "tpot_s_p50", "tpot_s_p95", "tpot_s_p99", "delivery_batch_gap_s_p50", "delivery_batch_gap_s_p95",
                "delivery_batch_gap_s_p99", "accepted", "proposed", "acceptance_fraction", "tau", "graph_hits",
                "graph_fallbacks", "verify_hits", "verify_fallbacks", "decode_graph_hits", "decode_graph_fallbacks",
                "target_kv_blocks", "quantized_modules", "candidate_total_bytes", "quantize_host_wall_s",
                "load_initial_graph_s", "prewarm_s", "memory_after_run")
        write_csv(output / "formal-runs.csv", cols, formal_rows(result))
        pair_cols = ("requested_batch", "ctx", "panel", "dspark_quant_preset", "engine_commit", "gpu_uuid", "seed",
                     "input_hashes", "status", "pair_count", "ratios", "median", "min", "max")
        pair_rows = []
        for pair in result["pairs"]:
            summary = pair["summary"] or {}
            pair_rows.append({**pair["configuration"], "status": pair["status"], "pair_count": len(pair["pairs"]),
                              "ratios": [x["ratio"] for x in pair["pairs"]], **summary})
        write_csv(output / "formal-pairs.csv", pair_cols, pair_rows)
        markdown = formal_markdown(result)
        count = f"{len(result['groups'])} 组，{len(result['pairs'])} 个候选对照，无效 {len(result['invalid'])} 条"
    else:
        cols = ("weight_name", "shape_nk", "batch", "rows", "mode", "engine_commit", "gpu_uuid", "seed",
                "input_sha256", "bf16_weight_sha256", "int8_weight_sha256", "bf16_scale_sha256",
                "bf16_weight_bytes", "int8_weight_bytes", "scale_bytes", "split_k_from_kernel_policy",
                "correctness_pass", "quant_reference", "original_bf16_diagnostic", "graph_vs_eager",
                "bf16_ms_per_call", "int8_ms_per_call", "bf16_median_ms", "int8_median_ms",
                "paired_bf16_over_int8", "source", "line")
        micro_rows = [{**item["record"], "source": item["source"], "line": item["line"],
                       "paired_bf16_over_int8": item["paired_bf16_over_int8"]} for item in result["measurements"]]
        write_csv(output / "micro-runs.csv", cols, micro_rows)
        markdown = micro_markdown(result)
        count = f"{len(result['measurements'])} 个模式测量"
    (output / names[1]).write_text(markdown, encoding="utf-8")
    print(f"已汇总 {count}；失败 {len(result['failures'])} 条，解析错误 {len(parse_errors)} 条；仅为初步 CPU 汇总")
    if parse_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
