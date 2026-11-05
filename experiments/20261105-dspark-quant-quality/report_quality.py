"""Recompute teacher-forced DSpark quality scores from one raw JSONL run.

Smoke (2 cases) and full (96 cases) are processed separately. No vocabulary
probabilities or token IDs are exported. This CPU report verifies the raw
record contract; the GPU guard ledger is reviewed separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_manifest import BLOCK, CTX, OFFSETS, PANELS, SCHEMA, SEED, WINDOW, sha256_file, token_hash
from quality_metrics import POSITION_FIELDS, pooled_nll, paired_nll

SUFFIXES = {
    "none": (), "q": ("self_attn.q_proj",), "k": ("self_attn.k_proj",),
    "v": ("self_attn.v_proj",), "o": ("self_attn.o_proj",),
    "gate": ("mlp.gate_proj",), "up": ("mlp.up_proj",), "down": ("mlp.down_proj",),
    "q_o": ("self_attn.q_proj", "self_attn.o_proj"),
    "q_o_down": ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"),
    "kv": ("self_attn.k_proj", "self_attn.v_proj"),
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "ffn": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    "all": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}
NLL_ATOL = 1e-4
KL_ATOL = 1e-6
NORMALIZATION_ATOL = 1e-5


def expected_modules():
    return {variant: {f"layers.{i}.{suffix}" for i in range(5) for suffix in suffixes}
            for variant, suffixes in SUFFIXES.items()}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def manifest_index(manifest):
    if (manifest.get("schema") != SCHEMA or manifest.get("seed") != SEED or
            manifest.get("context_tokens") != CTX or manifest.get("window_tokens") != WINDOW or
            manifest.get("block_size") != BLOCK or manifest.get("offsets") != list(OFFSETS) or
            not isinstance(manifest.get("provenance"), dict)):
        raise ValueError("manifest protocol differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 96:
        raise ValueError("manifest must contain 96 cases")
    ids, trajectories = set(), defaultdict(set)
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("manifest case is not an object")
        cid, panel, request, offset = (case.get(k) for k in ("case_id", "panel", "request_id", "offset"))
        if (not isinstance(cid, str) or not cid or cid in ids or panel not in PANELS
                or type(request) is not int or offset not in OFFSETS or type(offset) is not int):
            raise ValueError("invalid or duplicate manifest case identity")
        ids.add(cid)
        trajectories[(panel, request)].add(offset)
        prefix, labels, anchor = case.get("prefix_ids"), case.get("label_ids"), case.get("anchor_id")
        if (not isinstance(prefix, list) or len(prefix) != CTX + offset or case.get("context_tokens") != len(prefix)
                or not isinstance(labels, list) or len(labels) != BLOCK or type(anchor) is not int
                or any(type(x) is not int or x < 0 for x in prefix + labels) or anchor < 0
                or token_hash(prefix) != case.get("prefix_sha256")
                or token_hash(prefix[:CTX]) != case.get("prompt_sha256") or not is_hash(case.get("output_sha256"))):
            raise ValueError(f"{cid}: invalid manifest prefix/labels/hash")
    if (len(trajectories) != 24 or any(v != set(OFFSETS) for v in trajectories.values())
            or any(sum(key[0] == p for key in trajectories) != 8 for p in PANELS)):
        raise ValueError("manifest does not contain eight four-offset trajectories per panel")
    return cases


def analyze(manifest, manifest_sha, raw_rows, phase):
    errors, failures = [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
    try:
        manifest_cases = manifest_index(manifest)
    except (ValueError, TypeError, KeyError) as exc:
        return {"phase": phase, "valid_raw": False, "errors": [str(exc)], "failures": [], "groups": []}
    limit = 2 if phase == "smoke" else 96
    cases = {x["case_id"]: x for x in manifest_cases[:limit]}
    starts, headers, completions, scores, replays = [], [], [], {}, {}
    for line, row in raw_rows:
        if not isinstance(row, dict):
            errors.append(f"line {line}: record is not an object")
            continue
        kind = row.get("type")
        if kind == "start":
            starts.append(row)
        elif kind == "header":
            headers.append(row)
        elif kind == "complete":
            completions.append(row)
        elif kind == "failure":
            failures.append({"line": line, "record": row})
        elif kind == "score":
            key = (row.get("case_id"), row.get("variant"))
            if any(not isinstance(value, str) for value in key):
                errors.append(f"line {line}: score case/variant must be strings")
            elif key in scores:
                errors.append(f"line {line}: duplicate score {key}")
            else:
                scores[key] = row
        elif kind == "replay_check":
            key = row.get("case_id")
            if not isinstance(key, str):
                errors.append(f"line {line}: replay case must be a string")
            elif key in replays:
                errors.append(f"line {line}: duplicate BF16 replay {key}")
            else:
                replays[key] = row
        else:
            errors.append(f"line {line}: unknown record type {kind!r}")
    check(len(starts) == 1, "expected exactly one start record")
    check(len(headers) == 1, "expected exactly one header")
    check(len(completions) == 1, "expected exactly one complete record")
    check(not failures, "raw contains failure record(s)")
    if len(headers) != 1 or len(completions) != 1:
        return {"phase": phase, "valid_raw": False, "errors": errors, "failures": failures, "groups": []}
    header, complete = headers[0], completions[0]
    if len(starts) == 1:
        start = starts[0]
        check(start.get("guard_id") == header.get("guard_id") and start.get("phase") == phase
              and start.get("manifest_sha256") == manifest_sha
              and start.get("engine_commit") == header.get("engine_commit")
              and start.get("gpu_uuid") == header.get("gpu_uuid") and start.get("engine_dirty") is False
              and start.get("limit") == limit and start.get("case_count_expected") == limit
              and start.get("variant_count_expected") == len(SUFFIXES), "start identity/coverage differs from header")
    check(header.get("phase") == phase and header.get("limit") == limit, "header phase/case limit mismatch")
    check(header.get("manifest_sha256") == manifest_sha and complete.get("manifest_sha256") == manifest_sha,
          "manifest SHA mismatch")
    check(header.get("manifest_schema") == SCHEMA and header.get("seed") == SEED and header.get("temperature") == 1.0,
          "header schema/seed/temperature mismatch")
    check(header.get("window_tokens") == WINDOW and header.get("engine_dirty") is False,
          "window/source cleanliness mismatch")
    check(all(isinstance(header.get(k), str) and header.get(k) for k in ("guard_id", "gpu_uuid", "engine_commit")),
          "guard/GPU/source identity missing")
    check(complete.get("guard_id") == header.get("guard_id") and complete.get("phase") == phase,
          "header and complete guard/phase differ")
    check(all(complete.get(key) == header.get(key) for key in ("engine_commit", "gpu_uuid", "engine_dirty")),
          "header and complete source/GPU differ")
    check(header.get("manifest_provenance") == manifest.get("provenance")
          and complete.get("manifest_provenance") == manifest.get("provenance"), "manifest provenance differs")
    check(complete.get("case_count") == limit and complete.get("score_rows") == limit * len(SUFFIXES)
          and complete.get("locations_per_variant") == limit * BLOCK and complete.get("variants") == len(SUFFIXES),
          "complete counts differ from planned coverage")
    check(header.get("target_layer_ids") == [1, 9, 17, 25, 33] and header.get("hf_attention_implementation") == "sdpa",
          "target feature layer or attention implementation mismatch")
    source_prov = manifest.get("provenance", {}).get("source_target_provenance", {})
    target_prov, dspark_prov = header.get("target_provenance"), header.get("dspark_provenance")
    check(isinstance(target_prov, dict) and isinstance(source_prov, dict)
          and bool(target_prov.get("config_sha256")) and target_prov.get("config_sha256") == source_prov.get("config_sha256")
          and bool(target_prov.get("weight_files")) and target_prov.get("weight_files") == source_prov.get("weight_files"),
          "HF target config/weight file list differs from trajectory source")
    check(isinstance(dspark_prov, dict) and bool(dspark_prov.get("config_sha256")) and bool(dspark_prov.get("weight_files")),
          "DSpark checkpoint provenance missing")
    expected = expected_modules()
    actual = header.get("variants")
    check(isinstance(actual, dict) and set(actual) == set(expected), "variant inventory differs")
    if isinstance(actual, dict):
        for variant, names in expected.items():
            observed = actual.get(variant)
            check(isinstance(observed, list) and len(observed) == len(names)
                  and all(isinstance(x, str) for x in observed) and set(observed) == names,
                  f"variant module names differ: {variant}")
    check(set(scores) == {(cid, variant) for cid in cases for variant in SUFFIXES}, "score coverage missing or unexpected")
    check(set(replays) == set(cases), "BF16 replay coverage missing or unexpected")

    valid_rows = []
    for cid, case in cases.items():
        feature_hash = None
        first = scores.get((cid, "none"), {}).get("steps")
        for variant, names in expected.items():
            row = scores.get((cid, variant))
            if not isinstance(row, dict):
                continue
            label = f"{cid}/{variant}"
            check(all(row.get(key) == case.get(key) for key in
                      ("panel", "request_id", "offset", "context_tokens", "prefix_sha256")), f"{label}: case identity mismatch")
            check(row.get("quantized_modules") == len(names), f"{label}: quantized module count mismatch")
            digest = row.get("feature_sha256")
            check(is_hash(digest), f"{label}: missing feature SHA")
            if feature_hash is None:
                feature_hash = digest
            check(digest == feature_hash, f"{label}: feature tensor changed across variants")
            check(row.get("cache_slots") == [0, WINDOW - 1] and
                  row.get("cache_absolute_positions") == [case["context_tokens"] - WINDOW, case["context_tokens"] - 1],
                  f"{label}: cache geometry mismatch")
            diag = row.get("diagnostics")
            check(isinstance(diag, dict) and diag.get("logits_finite") is True and diag.get("scores_finite") is True
                  and all(finite(diag.get(k)) and 0 <= diag[k] <= NORMALIZATION_ATOL for k in
                          ("target_norm_max_abs", "bf16_norm_max_abs", "quant_norm_max_abs")),
                  f"{label}: nonfinite or unnormalized distribution")
            steps = row.get("steps")
            if not isinstance(steps, list) or len(steps) != BLOCK:
                errors.append(f"{label}: expected seven positions")
                continue
            for position, step in enumerate(steps, 1):
                where = f"{label}/position{position}"
                if not isinstance(step, dict):
                    errors.append(f"{where}: step not an object")
                    continue
                check(step.get("position") == position, f"{where}: position mismatch")
                if not all(finite(step.get(k)) for k in POSITION_FIELDS[:6]):
                    errors.append(f"{where}: missing/nonfinite score")
                    continue
                check(all(type(step.get(k)) is bool for k in POSITION_FIELDS[6:]), f"{where}: agreement not boolean")
                check(all(step[k] >= -1e-6 for k in ("nll_quant", "nll_bf16", "nll_target")), f"{where}: negative NLL")
                check(step["kl_bf16_quant"] >= -1e-5 and step["kl_target_quant"] >= -1e-5 and
                      -1e-6 <= step["overlap_target_quant"] <= 1 + 1e-6, f"{where}: KL/overlap outside tolerance")
                if isinstance(first, list) and len(first) == BLOCK and isinstance(first[position - 1], dict):
                    ref = first[position - 1]
                    for field in ("nll_bf16", "nll_target"):
                        check(finite(ref.get(field)) and abs(step[field] - ref[field]) <= NLL_ATOL,
                              f"{where}: shared {field} differs from none")
                if variant == "none":
                    check(abs(step["nll_quant"] - step["nll_bf16"]) <= NLL_ATOL
                          and abs(step["kl_bf16_quant"]) <= KL_ATOL and step.get("argmax_agree_bf16") is True,
                          f"{where}: BF16 none self-comparison failed")
                valid_rows.append({"panel": case["panel"], "request_id": case["request_id"],
                                   "case_id": cid, "position": position, "variant": variant, **step})
        replay = replays.get(cid)
        if isinstance(replay, dict):
            check(all(replay.get(key) == case.get(key) for key in
                      ("panel", "request_id", "offset", "context_tokens", "prefix_sha256"))
                  and replay.get("feature_sha256") == feature_hash, f"{cid}: BF16 replay identity/feature mismatch")
            check(replay.get("pass") is True and finite(replay.get("nll_max_abs_diff"))
                  and replay["nll_max_abs_diff"] <= NLL_ATOL and finite(replay.get("kl_max_abs"))
                  and replay["kl_max_abs"] <= KL_ATOL and replay.get("nll_atol") == NLL_ATOL
                  and replay.get("kl_atol") == KL_ATOL, f"{cid}: BF16 first/last replay failed")
            diag = replay.get("diagnostics")
            check(isinstance(diag, dict) and diag.get("logits_finite") is True and diag.get("scores_finite") is True
                  and all(finite(diag.get(k)) and 0 <= diag[k] <= NORMALIZATION_ATOL for k in
                          ("target_norm_max_abs", "bf16_norm_max_abs", "quant_norm_max_abs")),
                  f"{cid}: BF16 replay nonfinite or unnormalized")
    result = {"phase": phase, "valid_raw": not errors, "errors": errors, "failures": failures,
              "guard_ledger_checked": False, "manifest_sha256": manifest_sha,
              "source": {k: header.get(k) for k in ("guard_id", "gpu_uuid", "engine_commit", "target_provenance",
                                                     "dspark_provenance", "manifest_provenance", "hf_attention_implementation")},
              "counts": {"cases": len(cases), "trajectories": len({(x['panel'], x['request_id']) for x in cases.values()}),
                         "variants": len(SUFFIXES), "scores": len(scores), "replays": len(replays),
                         "positions_per_variant": limit * BLOCK}, "groups": []}
    if errors:
        return result
    grouped = defaultdict(list)
    for item in valid_rows:
        for scope, panel, position in (("overall", None, None), ("panel", item["panel"], None),
                                       ("position", None, item["position"]),
                                       ("panel_position", item["panel"], item["position"])):
            grouped[(scope, panel, position, item["variant"])].append(item)
    for (scope, panel, position, variant), values in sorted(grouped.items(), key=lambda x: repr(x[0])):
        losses = paired_nll([x["nll_bf16"] for x in values], [x["nll_quant"] for x in values])
        target = pooled_nll([x["nll_target"] for x in values])
        mean = lambda key: math.fsum(float(x[key]) for x in values) / len(values)
        result["groups"].append({"scope": scope, "panel": panel, "position": position, "variant": variant,
                                 "tokens": len(values), "cases": len({x["case_id"] for x in values}),
                                 "trajectories": len({(x["panel"], x["request_id"]) for x in values}),
                                 "mean_nll": losses["candidate"]["mean_nll"],
                                 "total_nll": losses["candidate"]["total_nll"],
                                 "conditional_nll_exp": losses["candidate"]["exp_mean_nll"],
                                 "bf16_mean_nll": losses["baseline"]["mean_nll"],
                                 "target_mean_nll": target["mean_nll"],
                                 "delta_mean_nll": losses["delta_mean_nll"],
                                 "relative_exp_delta_nll": losses["relative_exp_delta_nll"],
                                 "kl_bf16_quant_mean": mean("kl_bf16_quant"),
                                 "kl_target_quant_mean": mean("kl_target_quant"),
                                 "overlap_target_quant_mean": mean("overlap_target_quant"),
                                 "argmax_agree_bf16_fraction": mean("argmax_agree_bf16"),
                                 "argmax_agree_target_fraction": mean("argmax_agree_target")})
    return result


def report_markdown(result):
    phase, counts = result["phase"], result.get("counts", {})
    lines = [f"# DSpark 条件质量诊断（{phase}）", "",
             "本报告基于固定 target prefix 特征和真实前驱的 teacher forcing；不是独立草稿 LM 的标准 PPL。",
             "条件分数为 exp(总 token NLL / token 数)；相对分数为 exp(量化平均 NLL − BF16 平均 NLL)。",
             "负 ΔNLL 只表示所选样本分数较低，不推断泛化提升；overlap 是同历史单步分布重叠，不是实测 alpha/τ。",
             "q_o_down 是依据历史算子 micro 选出的速度候选；本报告不测模型或请求吞吐，质量分数不证明提速。",
             "历史样本由 native Graph target 以 T=1 采样；HF 同权重参考不假称逐位等价。文件名、字节和 config hash 匹配，权重内容本轮未重哈希。",
             "prompt 含 Wiki 填充，24 条轨迹的四个 offset 相关；不是人类 gold 或 96 个独立样本。",
             "此 CPU 汇总不审 GPU guard ledger；smoke 与 full 独立，不混算。", "",
             f"原始记录有效：{result['valid_raw']}；phase={phase}；counts={json.dumps(counts, ensure_ascii=False)}；错误 {len(result['errors'])}。", ""]
    if not result["valid_raw"]:
        lines += ["原始记录不完整或不一致，没有生成汇总。", ""] + [f"- {e}" for e in result["errors"]]
        return "\n".join(lines) + "\n"
    for scope, heading in (("overall", "整体"), ("panel", "按面板"), ("position", "按预测位置")):
        lines += [f"## {heading}", "", "| 面板 | 位置 | 变体 | token数 | 平均NLL | 条件指数 | ΔNLL | exp(ΔNLL) | KL(BF16‖量化) | KL(target‖量化) | overlap | top1同BF16 | top1同target |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for g in result["groups"]:
            if g["scope"] != scope:
                continue
            n = lambda name: f"{g[name]:.8g}"
            lines.append(f"| {g['panel'] or '整体'} | {g['position'] if g['position'] is not None else '全部'} | {g['variant']} | {g['tokens']} | {n('mean_nll')} | {n('conditional_nll_exp')} | {n('delta_mean_nll')} | {n('relative_exp_delta_nll')} | {n('kl_bf16_quant_mean')} | {n('kl_target_quant_mean')} | {n('overlap_target_quant_mean')} | {n('argmax_agree_bf16_fraction')} | {n('argmax_agree_target_fraction')} |")
        lines.append("")
    lines += ["面板×位置的完整结果、case/轨迹计数及总 NLL 见 summary.csv 与 summary.json。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw", required=True, help="one smoke OR one full JSONL file")
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    manifest_path, raw_path, out = Path(args.manifest), Path(args.raw), Path(args.out_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows, parse_errors = [], []
    with raw_path.open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                try:
                    def invalid_constant(value):
                        raise ValueError(f"nonfinite JSON constant {value}")
                    rows.append((line, json.loads(text, parse_constant=invalid_constant)))
                except ValueError as exc:
                    parse_errors.append(f"line {line}: {exc}")
    result = analyze(manifest, sha256_file(manifest_path), rows, args.phase)
    result["raw_path"], result["raw_sha256"] = str(raw_path.resolve()), sha256_file(raw_path)
    result["errors"].extend(parse_errors)
    if parse_errors:
        result["valid_raw"], result["groups"] = False, []
    names = ("summary.json", "summary.csv", "report.md")
    if any((out / name).exists() for name in names):
        parser.error("output directory contains same-name files")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    columns = ("scope", "panel", "position", "variant", "tokens", "cases", "trajectories", "total_nll", "mean_nll",
               "conditional_nll_exp", "bf16_mean_nll", "target_mean_nll", "delta_mean_nll", "relative_exp_delta_nll",
               "kl_bf16_quant_mean", "kl_target_quant_mean", "overlap_target_quant_mean",
               "argmax_agree_bf16_fraction", "argmax_agree_target_fraction")
    with (out / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["groups"])
    (out / "report.md").write_text(report_markdown(result), encoding="utf-8")
    print(f"{args.phase}: raw有效={result['valid_raw']}，汇总组={len(result['groups'])}，错误={len(result['errors'])}；guard另审")
    if not result["valid_raw"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
