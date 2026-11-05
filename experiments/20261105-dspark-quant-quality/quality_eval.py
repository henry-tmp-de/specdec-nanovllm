"""Guarded fixed-history DSpark W8A16 conditional distribution diagnostic.

This is not standalone language-model perplexity or an engine speed test.
The target sees the full prefix; the draft receives only its last 512 feature
rows. Every variant rebuilds its own draft KV and conditions Markov correction
on the same true predecessors. Launch only via this experiment's GPU guard.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy  # noqa: F401 -- initialize before local CPU PyTorch when testing
import torch

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/20261105-dspark-window-resume"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))
from gpu_identity import normalize_gpu_uuid, verify_visible_gpu

WINDOW = 512
BLOCK = 7
BASE_CONTEXT = 1024
OFFSETS = (0, 64, 128, 192)
LAYERS = (1, 9, 17, 25, 33)
SUFFIXES = {
    "none": (),
    "q": ("self_attn.q_proj",), "k": ("self_attn.k_proj",),
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
STEP_FIELDS = ("nll_quant", "nll_bf16", "nll_target", "kl_bf16_quant", "kl_target_quant",
               "overlap_target_quant", "argmax_agree_bf16", "argmax_agree_target")
DIAGNOSTIC_FIELDS = ("logits_finite", "scores_finite", "target_norm_max_abs",
                     "bf16_norm_max_abs", "quant_norm_max_abs")
BASELINE_NLL_ATOL = 1e-4
BASELINE_KL_ATOL = 1e-6
NORMALIZATION_ATOL = 1e-5


def token_hash(ids):
    return hashlib.sha256(",".join(str(x) for x in ids).encode("ascii")).hexdigest()


def tensor_hash(tensor):
    raw = tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_cases(manifest):
    if (manifest.get("schema") != "dspark-quant-quality-manifest-v1" or
            manifest.get("seed") != 20261105 or manifest.get("context_tokens") != BASE_CONTEXT or
            manifest.get("window_tokens") != WINDOW or manifest.get("block_size") != BLOCK or
            manifest.get("offsets") != list(OFFSETS) or not isinstance(manifest.get("provenance"), dict) or
            not manifest["provenance"]):
        raise ValueError("manifest protocol, seed or provenance mismatch")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 96:
        raise ValueError("manifest must contain 96 cases from 24 trajectories")
    seen_cases, trajectories = set(), {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        cid, panel, rid, offset = (case.get(k) for k in ("case_id", "panel", "request_id", "offset"))
        if (not isinstance(cid, str) or not cid or cid in seen_cases or
                panel not in ("code", "math", "general") or type(rid) is not int or
                type(offset) is not int or offset not in OFFSETS):
            raise ValueError("invalid or duplicate case identity")
        seen_cases.add(cid)
        key = (panel, rid)
        entries = trajectories.setdefault(key, {})
        if offset in entries:
            raise ValueError("duplicate trajectory offset")
        entries[offset] = case
        prefix, anchor, labels = case.get("prefix_ids"), case.get("anchor_id"), case.get("label_ids")
        if (not isinstance(prefix, list) or len(prefix) != BASE_CONTEXT + offset or
                case.get("context_tokens") != len(prefix) or not isinstance(labels, list) or
                len(labels) != BLOCK or type(anchor) is not int or anchor < 0 or
                any(type(x) is not int or x < 0 for x in prefix + labels)):
            raise ValueError(f"{cid}: invalid prefix, anchor or seven labels")
        if case.get("prefix_sha256") != token_hash(prefix) or case.get("prompt_sha256") != token_hash(prefix[:BASE_CONTEXT]):
            raise ValueError(f"{cid}: prompt/prefix hash mismatch")
        output_hash = case.get("output_sha256")
        if (not isinstance(output_hash, str) or len(output_hash) != 64 or
                any(c not in "0123456789abcdef" for c in output_hash)):
            raise ValueError(f"{cid}: missing output provenance hash")
        if not isinstance(case.get("input_metadata"), dict):
            raise ValueError(f"{cid}: input metadata missing")
    if len(trajectories) != 24 or any(sum(panel == p for panel, _ in trajectories) != 8
                                      for p in ("code", "math", "general")):
        raise ValueError("expected eight distinct trajectories per panel")
    for entries in trajectories.values():
        if set(entries) != set(OFFSETS):
            raise ValueError("trajectory missing an offset")
        ordered = [entries[offset] for offset in OFFSETS]
        if len({x["prompt_sha256"] for x in ordered}) != 1 or len({x["output_sha256"] for x in ordered}) != 1:
            raise ValueError("trajectory provenance differs across offsets")
        # Offset 0's anchor and next 7 labels must appear at the beginning of
        # each later prefix. Other cross-offset continuity follows similarly.
        for prior, later in zip(ordered, ordered[1:]):
            expected = [prior["anchor_id"], *prior["label_ids"]]
            start = BASE_CONTEXT + prior["offset"]
            if later["prefix_ids"][:start] != prior["prefix_ids"] or later["prefix_ids"][start:start + 8] != expected:
                raise ValueError("trajectory offsets do not share the same generated prefix")
    return cases


def causal_inputs(case):
    prefix = list(case["prefix_ids"])
    labels = list(case["label_ids"])
    if len(labels) != BLOCK:
        raise ValueError("teacher forcing requires seven labels")
    return prefix, [case["anchor_id"], *labels], [case["anchor_id"], *labels[:-1]]


def variants_for(all_names):
    names = tuple(all_names)
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("expected exactly 35 distinct candidate linears")
    expected = {f"layers.{i}.{suffix}" for i in range(5) for suffix in SUFFIXES["all"]}
    if set(names) != expected:
        raise ValueError("candidate inventory does not match five DSpark layers")
    variants = {name: tuple(candidate for candidate in names
                            if any(candidate.endswith("." + suffix) for suffix in suffixes))
                for name, suffixes in SUFFIXES.items()}
    if any(len(selected) != len(SUFFIXES[name]) * 5 for name, selected in variants.items()):
        raise ValueError("variant has missing or duplicate layers")
    return variants


def _put(model, name, value):
    parent, leaf = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), leaf, value)


class QuantBank:
    """Temporarily swap prebuilt modules; never capture graphs with this bank."""

    def __init__(self, model, names, original, quantized, variants, stats):
        self.model, self.names = model, names
        self.original, self.quantized, self.variants, self.stats = original, quantized, variants, stats
        self.protected = {name: model.get_submodule(name) for name in
                          ("fc", "markov_head", "confidence_head", "hidden_norm", "norm")}
        self.shared = (model.embedding_weight, model.lm_head_weight)

    def restore(self):
        for name in self.names:
            _put(self.model, name, self.original[name])

    @contextmanager
    def selected(self, variant):
        selected = set(self.variants[variant])
        try:
            for name in self.names:
                _put(self.model, name, self.quantized[name] if name in selected else self.original[name])
            if any(self.model.get_submodule(name) is not module for name, module in self.protected.items()):
                raise RuntimeError("protected DSpark module was replaced")
            if (self.model.embedding_weight is not self.shared[0] or
                    self.model.lm_head_weight is not self.shared[1]):
                raise RuntimeError("target-shared weights were replaced")
            yield
        finally:
            self.restore()


def build_bank(model, expected_names, apply_quant, quant_type):
    names = tuple(expected_names("all"))
    variants = variants_for(names)
    originals = {name: model.get_submodule(name) for name in names}
    try:
        stats = apply_quant(model, "all")
        quantized = {name: model.get_submodule(name) for name in names}
        if (any(not isinstance(quantized[name], quant_type) for name in names) or
                {x["name"] for x in stats} != set(names) or len(stats) != len(names)):
            raise RuntimeError("quantized bank does not cover the exact 35 modules")
    finally:
        for name in names:
            _put(model, name, originals[name])
    return QuantBank(model, names, originals, quantized, variants, stats)


def _cache_length(cache):
    if cache is None or not callable(getattr(cache, "get_seq_length", None)):
        raise RuntimeError("target did not return an inspectable dynamic KV cache")
    return int(cache.get_seq_length())


def target_reference(target, case, window=WINDOW, layer_ids=LAYERS):
    """Return prefix-only last-window features and seven teacher-forced p rows."""
    prefix, suffix, _ = causal_inputs(case)
    length = len(prefix)
    if length < window:
        raise ValueError("prefix shorter than draft window")
    dev = target.get_input_embeddings().weight.device
    prefix_ids = torch.tensor([prefix], dtype=torch.long, device=dev)
    prefix_positions = torch.arange(length, dtype=torch.long, device=dev).unsqueeze(0)
    prefix_mask = torch.ones((1, length), dtype=torch.long, device=dev)
    out = target.model(input_ids=prefix_ids, attention_mask=prefix_mask, position_ids=prefix_positions,
                       use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    if _cache_length(cache) != length:
        raise RuntimeError("target prefix cache does not cover full prefix")
    states = out.hidden_states
    if states is None or len(states) <= max(layer_ids) + 1:
        raise RuntimeError("target did not return all requested hidden states")
    selected = [states[layer + 1][:, -window:, :] for layer in layer_ids]
    if any(x.ndim != 3 or x.shape[:2] != (1, window) or x.dtype != torch.bfloat16 for x in selected):
        raise RuntimeError("unexpected target prefix feature shape or dtype")
    features = torch.cat(selected, dim=-1).squeeze(0).contiguous()
    prefix_feature_hash = tensor_hash(features)
    del out, states, selected
    suffix_ids = torch.tensor([suffix], dtype=torch.long, device=dev)
    suffix_positions = torch.arange(length, length + len(suffix), dtype=torch.long, device=dev).unsqueeze(0)
    suffix_mask = torch.ones((1, length + len(suffix)), dtype=torch.long, device=dev)
    suffix_out = target.model(input_ids=suffix_ids, attention_mask=suffix_mask, position_ids=suffix_positions,
                              past_key_values=cache, use_cache=True, output_hidden_states=False)
    if _cache_length(suffix_out.past_key_values) != length + len(suffix):
        raise RuntimeError("target teacher-forced cache length mismatch")
    target_logits = target.lm_head(suffix_out.last_hidden_state[:, :BLOCK, :]).squeeze(0)
    if target_logits.ndim != 2 or target_logits.shape[0] != BLOCK:
        raise RuntimeError("target did not return seven causal prediction rows")
    if tensor_hash(features) != prefix_feature_hash:
        raise RuntimeError("target suffix forward modified the prefix-only features")
    return features, target_logits, prefix_feature_hash


def draft_logits(model, case, features, window=WINDOW, block_size=256):
    """Rebuild KV and apply Markov to true predecessor IDs at seven positions."""
    prefix, _, previous = causal_inputs(case)
    length, device = len(prefix), model.fc.weight.device
    if window % block_size or features.shape[0] != window or features.dtype != torch.bfloat16:
        raise ValueError("invalid BF16 draft context features or window geometry")
    if model.window_tokens != window:
        raise RuntimeError("draft window configuration differs from the evaluated window")
    model.bind_cache(window // block_size, block_size)
    if model.sentinel != window:
        raise RuntimeError("unexpected draft cache sentinel")
    slots = torch.arange(window, device=device, dtype=torch.long)
    positions = torch.arange(length - window, length, device=device, dtype=torch.long)
    model.write_context(features, positions, slots)
    anchor = torch.tensor([case["anchor_id"]], device=device, dtype=torch.long)
    lens = torch.tensor([length], device=device, dtype=torch.long)
    base = model.base_logits(anchor, lens, slots.to(torch.int32).unsqueeze(0), window)
    if base.ndim != 3 or base.shape[:2] != (1, BLOCK) or base.dtype != torch.bfloat16:
        raise RuntimeError("draft backbone did not return seven BF16 logits rows")
    rows = []
    for index, token in enumerate(previous):
        prior = torch.tensor([token], device=device, dtype=torch.long)
        correction = model.markov_head(prior)
        if correction.dtype != torch.bfloat16:
            raise RuntimeError("Markov correction is not BF16")
        rows.append((base[:, index] + correction).squeeze(0))
    logits = torch.stack(rows, dim=0)
    if logits.dtype != torch.bfloat16 or logits.shape[0] != BLOCK:
        raise RuntimeError("draft BF16 logits contract failed")
    return logits


def guard_environment(env):
    required = ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX", "CUDA_VISIBLE_DEVICES")
    if any(not env.get(key) for key in required):
        raise RuntimeError("must start through this experiment's gpu_control.py")
    try:
        guard = str(uuid.UUID(env["DS_GUARD_ID"]))
        expected = normalize_gpu_uuid(env["DS_GPU_UUID"])
        visible = normalize_gpu_uuid(env["CUDA_VISIBLE_DEVICES"])
        deadline = float(env["DS_RUN_DEADLINE_UNIX"])
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid GPU guard identity or deadline") from exc
    if expected != visible or not math.isfinite(deadline) or time.time() >= deadline:
        raise RuntimeError("GPU identity mismatch or guard deadline expired")
    return {"guard_id": guard, "gpu_uuid": expected, "deadline_unix": deadline}


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("GPU guard deadline reached")


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "nanovllm",
                "experiments/20261105-dspark-quant-quality")
    return {"engine_commit": git("rev-parse", "HEAD"), "engine_dirty": bool(dirty)}


def checkpoint_provenance(directory):
    directory = Path(directory).resolve()
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors checkpoint under {directory}")
    return {"path": str(directory), "config_sha256": file_hash(directory / "config.json"),
            "weight_files": [{"name": p.name, "bytes": p.stat().st_size} for p in weights],
            "weight_hash_note": "sizes only; content hashes not recalculated"}


def check_target_matches_manifest(current, provenance):
    """Match the known source model config and checkpoint file inventory."""
    reference = provenance.get("source_target_provenance")
    if not isinstance(reference, dict) or current.get("config_sha256") != provenance.get("source_target_config_sha256"):
        raise ValueError("target configuration differs from the trajectory source")
    expected = reference.get("weight_files")
    if (not isinstance(expected, list) or not expected or
            current.get("weight_files") != expected or reference.get("config_sha256") != current.get("config_sha256")):
        raise ValueError("target checkpoint file inventory differs from the trajectory source")


def construct_dspark(model_class, cfg, embedding, lm_head):
    """Mirror production's BF16 parameter initialization while keeping RoPE FP32."""
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(embedding.device):
            model = model_class(cfg, embedding, lm_head)
    finally:
        torch.set_default_dtype(original_dtype)
    if (model.fc.weight.dtype != torch.bfloat16 or model.inv_freq.dtype != torch.float32 or
            model.inv_freq.device != embedding.device or model.embedding_weight is not embedding or
            model.lm_head_weight is not lm_head):
        raise RuntimeError("DSpark BF16/FP32 RoPE or shared-target construction mismatch")
    return model


def metric_rows(metrics):
    if set(STEP_FIELDS + DIAGNOSTIC_FIELDS) != set(metrics):
        raise RuntimeError("quality metric fields missing")
    columns = []
    for name in STEP_FIELDS:
        value = metrics[name]
        if not torch.is_tensor(value) or value.shape != (BLOCK,):
            raise RuntimeError(f"quality metric {name} must have seven rows")
        columns.append(value.to(torch.float32))
    data = torch.stack(columns, dim=1).detach().to("cpu")
    if not torch.isfinite(data).all().item():
        raise RuntimeError("non-finite quality metric")
    matrix = data.tolist()
    rows = []
    for index, values in enumerate(matrix):
        row = {"position": index + 1}
        for key, value in zip(STEP_FIELDS, values):
            row[key] = bool(value) if key.startswith("argmax_agree_") else float(value)
        if (row["kl_bf16_quant"] < -1e-5 or row["kl_target_quant"] < -1e-5 or
                not -1e-6 <= row["overlap_target_quant"] <= 1 + 1e-6):
            raise RuntimeError("invalid KL or distribution overlap")
        rows.append(row)
    diagnostics = {}
    for key, value in metrics.items():
        if key in STEP_FIELDS:
            continue
        if not torch.is_tensor(value) or value.numel() != 1:
            raise RuntimeError(f"invalid metric diagnostic: {key}")
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            raise RuntimeError(f"non-finite metric diagnostic: {key}")
        diagnostics[key] = item
        if key.endswith("_norm_max_abs") and not 0 <= float(item) <= NORMALIZATION_ATOL:
            raise RuntimeError(f"probability normalization failure: {key}")
        if "finite" in key and not bool(item):
            raise RuntimeError(f"non-finite logits: {key}")
    return rows, diagnostics


def baseline_replay_check(score_distributions, target_logits, first, replay, labels):
    metrics = score_distributions(target_logits, first, replay, labels)
    rows, diagnostics = metric_rows(metrics)
    nll_diff = max(abs(row["nll_quant"] - row["nll_bf16"]) for row in rows)
    kl = max(abs(row["kl_bf16_quant"]) for row in rows)
    return {"nll_max_abs_diff": nll_diff, "kl_max_abs": kl,
            "nll_atol": BASELINE_NLL_ATOL, "kl_atol": BASELINE_KL_ATOL,
            "pass": nll_diff <= BASELINE_NLL_ATOL and kl <= BASELINE_KL_ATOL,
            "diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dspark-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, choices=(2, 96), default=2,
                        help="2-case smoke by default; 96-case full run uses a separate output")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    ident, stage, current_case, current_variant, bank = {}, "preflight", None, None, None
    with path.open("x", encoding="utf-8") as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        try:
            guard = guard_environment(os.environ)
            ident.update(guard, **git_identity(), phase="smoke" if args.limit == 2 else "full")
            if ident["engine_dirty"]:
                raise RuntimeError("quality source is dirty; commit it before GPU execution")
            check_deadline(guard["deadline_unix"])
            manifest_path = Path(args.manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cases = manifest_cases(manifest)
            ident.update(manifest_sha256=file_hash(manifest_path), manifest_schema=manifest["schema"],
                         manifest_provenance=manifest["provenance"], seed=manifest["seed"])
            write({"type": "start", **ident, "limit": args.limit,
                   "case_count_expected": args.limit, "variant_count_expected": len(SUFFIXES)})
            gpu = verify_visible_gpu(torch, guard["gpu_uuid"])
            torch.cuda.set_device(0)
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM
            from nanovllm.spec_decode.dspark_model import DSparkModel, load_official_weights, validate_config
            from nanovllm.spec_decode.dspark_quant import DSparkInt8Linear, apply_dspark_quant, expected_names
            from quality_metrics import score_distributions

            ident.update(gpu_name=gpu.name, torch_version=torch.__version__, transformers_version=transformers.__version__,
                         target_provenance=checkpoint_provenance(args.target),
                         dspark_provenance=checkpoint_provenance(args.dspark_model))
            check_target_matches_manifest(ident["target_provenance"], manifest["provenance"])
            stage = "load_target"
            target = AutoModelForCausalLM.from_pretrained(
                args.target, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                device_map={"": 0}).eval()
            embedding, lm_head = target.get_input_embeddings().weight, target.get_output_embeddings().weight
            if (embedding.dtype != torch.bfloat16 or lm_head.dtype != torch.bfloat16 or
                    embedding.device.type != "cuda" or lm_head.device != embedding.device):
                raise RuntimeError("target embedding/LM head must both be BF16 on the selected CUDA device")
            if any(p.device != embedding.device or (p.is_floating_point() and p.dtype != torch.bfloat16)
                   for p in target.parameters()):
                raise RuntimeError("target parameters must remain on the same device in BF16")
            actual_attention = getattr(target.config, "_attn_implementation", None)
            if actual_attention != "sdpa":
                raise RuntimeError(f"unexpected HF attention implementation: {actual_attention!r}")
            ident["hf_attention_implementation"] = actual_attention
            vocab = embedding.shape[0]
            if any(max([*case["prefix_ids"], case["anchor_id"], *case["label_ids"]]) >= vocab for case in cases):
                raise ValueError("manifest token outside target vocabulary")
            stage = "load_dspark"
            cfg = AutoConfig.from_pretrained(args.dspark_model, local_files_only=True)
            validate_config(cfg, target.config, BLOCK)
            model = construct_dspark(DSparkModel, cfg, embedding, lm_head)
            ident["dspark_fc_dtype"] = str(model.fc.weight.dtype)
            ident["dspark_inv_freq_dtype"] = str(model.inv_freq.dtype)
            ident["shared_embedding_and_lm_head_alias_target"] = True
            model.window_tokens = WINDOW
            load_official_weights(model, args.dspark_model)
            stage = "prepare_quant_bank"
            check_deadline(guard["deadline_unix"])
            bank_start = time.perf_counter()
            bank = build_bank(model, expected_names, apply_dspark_quant, DSparkInt8Linear)
            bank_seconds = time.perf_counter() - bank_start
            write({"type": "header", **ident, "limit": args.limit, "window_tokens": WINDOW,
                   "target_layer_ids": list(LAYERS), "temperature": 1.0,
                   "target_reference": "HF Qwen3 model; full prefix cache; prefix-only draft features",
                   "draft_cache_policy": "fresh chronological slots for every variant; no CUDA Graph",
                   "markov_policy": "anchor then labels 0..5; BF16 add then FP32 softmax",
                   "variants": {key: list(value) for key, value in bank.variants.items()},
                   "quant_bank_stats": bank.stats, "quant_bank_host_wall_s": bank_seconds,
                   "smoke_and_full_are_separate": True})
            score_count = 0
            with torch.inference_mode():
                for case in cases[:args.limit]:
                    check_deadline(guard["deadline_unix"])
                    current_case, current_variant = case["case_id"], None
                    stage = "target_prefix_and_teacher_forcing"
                    features, target_logits, feature_sha = target_reference(target, case)
                    labels = torch.tensor(case["label_ids"], dtype=torch.long, device=embedding.device)
                    common = {"case_id": current_case, "panel": case["panel"], "request_id": case["request_id"],
                              "offset": case["offset"], "context_tokens": case["context_tokens"],
                              "prefix_sha256": case["prefix_sha256"], "feature_sha256": feature_sha}
                    first = None
                    for variant, names in bank.variants.items():
                        check_deadline(guard["deadline_unix"])
                        stage, current_variant = "draft_variant", variant
                        with bank.selected(variant):
                            logits = draft_logits(model, case, features)
                            if variant == "none":
                                first = logits
                            rows, diagnostics = metric_rows(score_distributions(target_logits, first, logits, labels))
                        if tensor_hash(features) != feature_sha:
                            raise RuntimeError("draft variant modified prefix-only feature tensor")
                        write({"type": "score", **common, "variant": variant, "quantized_modules": len(names),
                               "cache_slots": [0, WINDOW - 1],
                               "cache_absolute_positions": [case["context_tokens"] - WINDOW,
                                                            case["context_tokens"] - 1],
                               "steps": rows, "diagnostics": diagnostics})
                        score_count += 1
                    stage, current_variant = "bf16_replay", "none"
                    with bank.selected("none"):
                        replay = draft_logits(model, case, features)
                    if tensor_hash(features) != feature_sha:
                        raise RuntimeError("BF16 replay modified prefix-only feature tensor")
                    replay_check = baseline_replay_check(score_distributions, target_logits, first, replay, labels)
                    write({"type": "replay_check", **common, **replay_check})
                    if not replay_check["pass"]:
                        raise RuntimeError("first and last BF16 NLL/KL replay differ")
            if score_count != args.limit * len(bank.variants):
                raise RuntimeError("incomplete variant count")
            write({"type": "complete", **ident, "case_count": args.limit, "score_rows": score_count,
                   "locations_per_variant": args.limit * BLOCK, "variants": len(bank.variants)})
        except BaseException as exc:
            write({"type": "failure", **ident, "failure_phase": stage, "case_id": current_case,
                   "variant": current_variant, "error": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc()})
            raise
        finally:
            if bank is not None:
                bank.restore()


if __name__ == "__main__":
    main()
