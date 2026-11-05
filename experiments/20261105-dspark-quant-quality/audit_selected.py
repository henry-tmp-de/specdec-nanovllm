"""CPU audit of the selected-preset native DSpark W8A16 speed pilot.

The prior quant audit supplies compute_run, same-input/config grouping, paired
three-repeat statistics, and CSV/JSON/Markdown output. This adapter validates
the real 35-layer inventory and selected preset before handing it the records.
It does not validate the GPU guard ledger or turn quality probes into speed data.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments" / "20261105-dspark-quant" / "audit_quant.py"
spec = importlib.util.spec_from_file_location("selected_previous_quant_audit", PREVIOUS)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

PRESETS = ("none", "q_o", "down", "q_o_down")
SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SELECTED_SUFFIXES = {"none": (), "q_o": ("self_attn.q_proj", "self_attn.o_proj"),
                     "down": ("mlp.down_proj",),
                     "q_o_down": ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")}
EXPECTED = {p: {f"layers.{layer}.{suffix}" for layer in range(5) for suffix in suffixes}
            for p, suffixes in SELECTED_SUFFIXES.items()}
ALL_NAMES = {f"layers.{layer}.{suffix}" for layer in range(5) for suffix in SUFFIXES}
INVENTORY_FIELDS = ("quantized_module_names", "candidate_total_bytes", "candidate_inventory_sha256", "target_kv_blocks")

_prior_identity = prior.run_identity
_prior_summarize = prior.summarize
_prior_pair_formal = prior.pair_formal
_prior_markdown = prior.formal_markdown
prior.PRESETS = PRESETS


def inventory_identity(row):
    """Validate actual replacement names, dtypes and bytes, not just a preset label."""
    problems = []
    def check(condition, detail):
        if not condition:
            problems.append(detail)
    actual = row.get("dspark_quant")
    if not isinstance(actual, dict) or actual.get("preset") not in PRESETS:
        return {}, ["missing selected-preset runtime inventory"]
    preset = actual["preset"]
    check(row.get("dspark_quant_preset") == preset, "requested and actual preset differ")
    check(type(actual.get("quantized_modules")) is int and actual["quantized_modules"] == len(EXPECTED[preset]),
          "actual selected module count differs from preset")
    host_s = actual.get("quantize_host_wall_s")
    check(type(host_s) in (int, float) and math.isfinite(host_s) and host_s >= 0,
          "missing/nonfinite host PTQ time")
    modules = actual.get("candidate_modules")
    if not isinstance(modules, list) or len(modules) != len(ALL_NAMES) or any(not isinstance(x, dict) for x in modules):
        return {}, problems + ["expected 35 candidate layer records"]
    names = [x.get("name") for x in modules]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(ALL_NAMES) or set(names) != ALL_NAMES:
        return {}, problems + ["candidate names missing, duplicated or unexpected"]
    selected = set()
    total = 0
    for module in modules:
        name, shape, flag = module["name"], module.get("shape"), module.get("quantized")
        if (not isinstance(shape, list) or len(shape) != 2 or
                any(type(x) is not int or x <= 0 for x in shape) or type(flag) is not bool):
            problems.append(f"{name}: invalid shape/quantized flag")
            continue
        n, k = shape
        if flag:
            selected.add(name)
        weight_bytes, scale_bytes = n * k * (1 if flag else 2), n * 2 if flag else 0
        check(module.get("weight_dtype") == ("torch.int8" if flag else "torch.bfloat16") and
              module.get("weight_bytes") == weight_bytes and
              module.get("scale_dtype") == ("torch.bfloat16" if flag else None) and
              module.get("scale_shape") == ([n] if flag else None) and
              module.get("scale_bytes") == scale_bytes and module.get("total_bytes") == weight_bytes + scale_bytes,
              f"{name}: actual weight/scale metadata or byte count inconsistent")
        total += weight_bytes + scale_bytes
    check(selected == EXPECTED[preset], "actual selected layer names differ from preset")
    check(type(actual.get("candidate_total_bytes")) is int and actual["candidate_total_bytes"] == total,
          "candidate_total_bytes differs from actual inventory")
    loading = actual.get("loading_stats")
    if not isinstance(loading, list) or any(not isinstance(x, dict) for x in loading):
        problems.append("loading_stats missing or malformed")
    else:
        loading_names = [item.get("name") for item in loading]
        check(all(isinstance(x, str) for x in loading_names) and len(loading_names) == len(EXPECTED[preset])
              and set(loading_names) == EXPECTED[preset], "loading_stats names differ from selected preset")
        by_name = {item["name"]: item for item in modules}
        for item in loading:
            name = item.get("name")
            if not isinstance(name, str) or name not in selected or name not in by_name:
                continue
            module = by_name[name]
            check(item.get("shape") == module.get("shape") and item.get("weight_dtype_before") == "bfloat16"
                  and item.get("weight_dtype_after") == "int8" and item.get("scale_dtype") == "bfloat16"
                  and item.get("scale_shape") == module.get("scale_shape")
                  and item.get("weight_bytes") == module.get("weight_bytes")
                  and item.get("scale_bytes") == module.get("scale_bytes")
                  and item.get("bytes_after") == module.get("total_bytes"),
                  f"{name}: loading statistics differ from live inventory")
    excluded = actual.get("noncandidate_parameters")
    if not isinstance(excluded, list) or any(not isinstance(x, dict) for x in excluded):
        problems.append("unselected BF16 parameter inventory missing")
    else:
        excluded_names = [item.get("name") for item in excluded]
        check(all(isinstance(name, str) for name in excluded_names)
              and {"fc.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight"} <= set(excluded_names)
              and all(item.get("dtype") == "torch.bfloat16" for item in excluded),
              "protected FC/Markov or other unselected parameter is not BF16")
    shared = actual.get("shared_target_weights")
    if not isinstance(shared, list) or any(not isinstance(x, dict) for x in shared):
        problems.append("target-shared weights missing")
    else:
        shared_names = [item.get("name") for item in shared]
        check(len(shared) == 2 and all(isinstance(name, str) for name in shared_names)
              and set(shared_names) == {"embedding_weight", "lm_head_weight"}
              and all(item.get("dtype") == "torch.bfloat16" and item.get("aliases_target") is True for item in shared),
              "embedding or LM head is not BF16 and aliased to target")
    target_blocks = row.get("target_kv_blocks")
    check(type(target_blocks) is int and target_blocks > 0, "actual target KV block count unavailable")
    # Include actual inventory in within-preset grouping. Pairing deliberately
    # retains a fixed total budget but allows different KV counts across presets.
    digest = hashlib.sha256(prior.old.canonical(sorted(modules, key=lambda item: item["name"])).encode("utf-8")).hexdigest()
    identity = {"quantized_module_names": sorted(selected), "candidate_total_bytes": actual.get("candidate_total_bytes"),
                "candidate_inventory_sha256": digest, "target_kv_blocks": target_blocks}
    return identity, problems


def run_identity(row):
    identity, problems = _prior_identity(row)
    inventory, inventory_problems = inventory_identity(row)
    identity.update(inventory)
    problems.extend(inventory_problems)
    batch = row.get("requested_batch")
    exact_policy = {"name": "exact_all", "target_decode_padding": False, "target_verify_padding": False,
                    "draft_padding": False, "buckets": list(range(1, batch + 1))} if type(batch) is int and batch > 0 else None
    if (row.get("arm") != "dspark_ring" or row.get("mode") != "ring" or row.get("phase") != "pilot"
            or batch not in (1, 8) or row.get("max_num_seqs") != batch or row.get("ctx") != 1024
            or row.get("output_tokens") != 128 or row.get("window") != 512 or row.get("seed") != 20261105
            or row.get("gpu_memory_utilization") != 0.85 or row.get("temperature") != 1.0
            or row.get("ignore_eos") is not True or row.get("mixed_overlap") is not False
            or row.get("prefix_lookup") is not False or row.get("prefix_lookup_count") != 0
            or row.get("prefix_hit_blocks") != 0 or row.get("spec_k") != 7 or row.get("graph_mode") != "graph"
            or row.get("graph_policy") != exact_policy or row.get("max_model_len") != 1024 + 128 + 8
            or row.get("panel") not in ("code", "math", "general")):
        problems.append("run does not match selected pilot scope")
    graph = row.get("graph_delta")
    decode = row.get("target_decode_graph")
    if not isinstance(graph, dict) or not isinstance(decode, dict):
        problems.append("native graph telemetry missing")
    else:
        for stem in ("graph", "verify"):
            hits, exact, padded = (graph.get(k) for k in
                                   (("graph_hits", "graph_exact_hits", "graph_padded_hits") if stem == "graph" else
                                    ("verify_hits", "verify_exact_hits", "verify_padded_hits")))
            if not (type(hits) is int and hits > 0 and type(exact) is int and exact == hits and padded == 0):
                problems.append(f"{stem} exact graph was not observed or was padded")
        if (type(graph.get("head_graph_hits")) is not int or graph["head_graph_hits"] <= 0
                or graph.get("graph_runtime_captures") != 0 or graph.get("head_runtime_captures") != 0):
            problems.append("draft head graph missing or a runtime capture occurred")
        draft_fb, verify_fb = graph.get("graph_fallbacks"), graph.get("verify_fallbacks")
        if not isinstance(draft_fb, dict) or any(type(v) is not int or v != 0 for v in draft_fb.values()):
            problems.append("draft graph fallback or malformed fallback telemetry")
        if (not isinstance(verify_fb, dict) or any(k != "ragged_draft_len" or type(v) is not int or v < 0
                                                   for k, v in verify_fb.items())):
            problems.append("unexpected or malformed target verify fallback")
        if (decode.get("padded") != 0 or not isinstance(decode.get("fallbacks"), dict)
                or any(type(v) is not int or v != 0 for v in decode["fallbacks"].values())):
            problems.append("target decode padding/fallback or missing telemetry")
    return identity, problems


def pair_selected(result):
    """Reuse prior exact repeat pairing while allowing budget-matched KV differences."""
    def common(config):
        excluded = set(INVENTORY_FIELDS) | {"dspark_quant_preset"}
        return prior.old.canonical({k: v for k, v in config.items() if k not in excluded})
    groups = defaultdict(list)
    for group in result["groups"]:
        config = group["configuration"]
        groups[(config.get("dspark_quant_preset"), common(config))].append(group)
    pairs = []
    for group in result["groups"]:
        config = group["configuration"]
        preset = config.get("dspark_quant_preset")
        if preset not in PRESETS[1:]:
            continue
        key = common(config)
        baseline, candidate = groups.get(("none", key), []), groups[(preset, key)]
        if len(baseline) != 1 or len(candidate) != 1:
            pairs.append({"configuration": config, "baseline_preset": "none", "pairs": [], "summary": None,
                          "status": "无唯一同条件 none 或存在多种实际库存/KV配置"})
            continue
        def alias(item, label):
            filtered = {k: v for k, v in item["configuration"].items() if k not in INVENTORY_FIELDS}
            filtered["dspark_quant_preset"] = label
            return {**item, "configuration": filtered}
        # Previous code has a literal candidate list; alias only this temporary
        # isolated pair to its existing FFN branch, then restore actual preset.
        paired = _prior_pair_formal({"groups": [alias(baseline[0], "none"), alias(group, "ffn")],
                                    "failures": result["failures"]})
        if len(paired) != 1:
            raise RuntimeError("prior pair implementation did not return one candidate")
        paired[0]["configuration"] = config
        pairs.extend(paired)
    return pairs


def summarize_selected(rows):
    result = _prior_summarize(rows)
    if result["other"]:
        # A quality diagnostic JSONL has start/score/replay/complete records,
        # whereas native benchmark files contain only warmup/run/failure. Never
        # let a mixed glob silently produce speed comparisons.
        message = "non-native record type found; separate quality and performance raw files"
        result["invalid"].append({"source": None, "line": None, "errors": [message]})
        for group in result["groups"]:
            group["three_repeats_valid"] = False
            group["summary"] = {key: None for key in group["summary"]}
            for entry in group["runs"]:
                entry["metrics"]["valid"] = False
                entry["metrics"]["errors"].append(message)
    return result


def selected_markdown(result):
    text = _prior_markdown(result).replace("# DSpark W8A16 完整请求：CPU 原始记录汇总",
                                          "# DSpark W8A16 选层性能：CPU 原始记录汇总", 1)
    note = ("\n预设真实量化层：none=0、q_o=10、down=5、q_o_down=15；其他候选层、target 与共享权重为 BF16。"
            "逐 run 的选中层名、库存 SHA、KV 块数和字节记录在 formal-summary.json。"
            "verify 的 ragged_draft_len 回退单独记录；同预算允许实际 target KV 块数不同。"
            "此报告不验证 GPU guard ledger，亦不是 teacher-forced 质量评分。\n")
    return text + note


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="native benchmark JSONL paths or quoted globs")
    parser.add_argument("--out-dir", required=True, help="new output directory")
    args = parser.parse_args()
    sys.argv = [str(PREVIOUS), "--kind", "formal", "--input", *args.input, "--out-dir", args.out_dir]
    prior.main()


prior.run_identity = run_identity
prior.summarize = summarize_selected
prior.pair_formal = pair_selected
prior.formal_markdown = selected_markdown


if __name__ == "__main__":
    main()
