"""Matched fixed-batch panel/repeats in one process; rotate arms externally.

No forced acceptance or extra q/target forward is performed.  All outputs are
real T=1 deliveries; this script does not implement online mixed traffic. Each
invocation produces a prewarm record plus each panel/repeat's raw request/round
telemetry. Use only through the shared gpu_control.py ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bench_smoke import prewarm_shapes, read_manifest, token_hash
from audit_results import compute_run
from gpu_identity import verify_visible_gpu

ARMS = ("nospec", "ar_full", "ar_window", "dspark_full", "dspark_window_full", "dspark_ring")
BUCKETS = (1, 2, 4, 8, 16)


def validate_budget(phase, utilization):
    if not math.isfinite(utilization) or not 0 < utilization <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if phase in ("pilot", "formal") and utilization not in (0.70, 0.85):
        raise ValueError("pilot/formal budget must be an explicitly planned 0.70 or 0.85")


def options_for(arm, batch, ctx, output_tokens, window, target, ar_model, dspark_model,
                batched_tokens, gpu_memory_utilization=0.70):
    if arm not in ARMS or batch not in BUCKETS or ctx <= 0 or output_tokens < 9:
        raise ValueError("invalid fixed-batch cell")
    window_arms = ("ar_window", "dspark_window_full", "dspark_ring")
    if (arm in window_arms and (window <= 0 or window % 256)) or (arm not in window_arms and window != 0):
        raise ValueError("window arms require a positive block-aligned window; full arms require W=0")
    k = 0 if arm == "nospec" else 6 if arm.startswith("ar_") else 7
    if batched_tokens < max(batch * (k + 1), 1):
        raise ValueError("max_num_batched_tokens cannot hold the fixed decode batch")
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    # Same model-length and block-table capacity for every arm.
    exact_buckets = tuple(range(1, batch + 1))
    options = dict(max_num_seqs=batch, max_model_len=ctx + output_tokens + 8,
                   max_num_batched_tokens=batched_tokens, gpu_memory_utilization=gpu_memory_utilization,
                   tensor_parallel_size=1, enforce_eager=False, spec_cuda_graph=True,
                   spec_graph_bs=exact_buckets, decode_graph_bs=exact_buckets,
                   spec_graph_pad=False, dspark_graph_pad=False, spec_overlap_mixed=False,
                   enable_prefix_cache=False)
    if arm.startswith("ar_"):
        options.update(spec_method="draft", spec_k=k, draft_model=ar_model)
        if arm == "ar_window":
            options.update(spec_draft_window=window, spec_draft_head_blocks=0,
                           spec_draft_win_align=True, spec_draft_win_realign=True)
    elif arm.startswith("dspark_"):
        options.update(spec_method="dspark", spec_k=k, draft_model=dspark_model,
                       dspark_split_graph=True,
                       dspark_window_tokens=window,
                       dspark_window_storage="ring" if arm == "dspark_ring" else "full_cache")
    return options


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_provenance(path):
    if not path:
        return None
    directory = Path(path)
    return {"path": str(directory), "config_sha256": file_hash(directory / "config.json"),
            "weight_files": [{"name": p.name, "bytes": p.stat().st_size}
                             for p in sorted(directory.glob("*.safetensors"))],
            "weight_content_hash_note": "file sizes only; weights not rehashed by this script"}


def git_provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all", "--", "nanovllm",
                     "experiments/20261105-dspark-window-resume"))
    return commit, dirty


def snapshot(runner):
    p = runner.spec_proposer
    data = {"verify_hits": runner.verify_graph_hits,
            "verify_padded_hits": runner.verify_graph_padded,
            "verify_exact_hits": runner.verify_graph_hits - runner.verify_graph_padded,
            "verify_fallbacks": dict(runner.verify_graph_fallbacks)}
    for key in ("decode_graph_hits", "decode_graph_exact_hits", "decode_graph_padded_hits",
                "decode_graph_bucket_histogram", "decode_graph_fallbacks"):
        if hasattr(runner, key):
            value = getattr(runner, key)
            data[key] = value.copy() if isinstance(value, dict) else value
    for key in ("graph_hits", "graph_exact_hits", "graph_padded_hits", "graph_captures",
                "graph_runtime_captures", "graph_runtime_capture_s", "graph_runtime_shapes",
                "graph_fallbacks", "active_batch_histogram", "context_bucket_histogram",
                "head_graph_hits", "head_graph_captures", "head_runtime_captures",
                "head_runtime_capture_s", "head_runtime_batches", "input_prepare_calls", "input_prepare_wall_s",
                "n_graphed", "n_eager", "n_padded", "n_rounds", "n_batch_forwards",
                "n_catchup_forwards", "n_catchup_tokens"):
        if p is not None and hasattr(p, key):
            value = getattr(p, key)
            data[key] = value.copy() if isinstance(value, (dict, list)) else value
    return data


def delta(before, after):
    out = {}
    for key, value in after.items():
        initial = before.get(key, {} if isinstance(value, dict) else [] if isinstance(value, list) else 0)
        if isinstance(value, dict):
            out[key] = {str(k): v - initial.get(k, 0) for k, v in value.items() if v != initial.get(k, 0)}
        elif isinstance(value, list):
            out[key] = value[len(initial):]
        else:
            out[key] = value - initial
    return out


def memory(runner):
    import torch
    tensors = {}
    for label, obj in (("runner", runner), ("dspark", runner.dspark_model)):
        if obj is None:
            continue
        for name, value in vars(obj).items():
            if torch.is_tensor(value) and ("cache" in name or "scratch" in name or "feature" in name):
                tensors[f"{label}.{name}"] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                              "bytes": value.numel() * value.element_size()}
    by_pool = {}
    for segment in torch.cuda.memory_snapshot():
        key = str(tuple(segment.get("segment_pool_id", (0, 0))))
        by_pool[key] = by_pool.get(key, 0) + segment["total_size"]
    named_pools = {"target_decode": getattr(runner, "graph_pool", None)}
    if runner.verify_graphs:
        named_pools["target_verify"] = next(iter(runner.verify_graphs.values()))[0].pool()
    proposer = runner.spec_proposer
    if proposer is not None:
        if hasattr(proposer, "backbone_pool"):
            named_pools["dspark_backbone"] = proposer.backbone_pool
            named_pools["dspark_head"] = proposer.head_pool
        for batch, entry in getattr(proposer, "_graphs", {}).items():
            named_pools[f"ar_draft_{batch}"] = entry[0].pool()
    weights, seen = {}, set()
    for label in ("model", "draft_model", "dspark_model"):
        obj = getattr(runner, label, None)
        n = 0
        if obj is not None:
            for p in obj.parameters():
                key = (str(p.device), p.data_ptr())
                if key not in seen:
                    seen.add(key)
                    n += p.numel() * p.element_size()
        weights[label] = n
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "named_tensors": tensors,
            "unique_parameter_bytes": weights, "allocator_segment_bytes_by_pool_id": by_pool,
            "named_graph_pools": {name: {"pool_id": list(pool) if pool is not None else None,
                                         "reserved_bytes": by_pool.get(str(tuple(pool)), 0) if pool is not None else 0}
                                  for name, pool in named_pools.items()},
            "engine_graph_mem_gb": runner.graph_mem,
            "accounting_note": "allocator pool totals overlap tensor/weight values; unnamed temporaries are included in allocator totals"}


def run_once(llm, prompts, output_tokens, seed, deadline, capture_state, diagnostic_cost=False):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook
    runner, scheduler = llm.model_runner, llm.scheduler
    hook = TokenDeliveryHook()
    llm.token_hook = hook
    scheduler.token_hook = hook
    graph_before = snapshot(runner)
    window_before = dict(getattr(runner, "dspark_window_stats", {}))
    preemptions = []
    old_preempt = scheduler.preempt
    def preempt(seq):
        preemptions.append(seq.seq_id)
        capture_state["preempted_sequence_ids"] = list(preemptions)
        return old_preempt(seq)
    scheduler.preempt = preempt
    event_rows = {"draft": [], "verify": [], "commit": []}
    current_round = [None]
    restore = []
    def install(obj, name, category):
        if obj is None or not hasattr(obj, name):
            return
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            value = original(*args, **kwargs)
            end.record()
            event_rows[category].append((current_round[0], begin, end))
            return value
        setattr(obj, name, wrapped)
        restore.append((obj, name, original))
    if diagnostic_cost:
        install(runner.spec_proposer, "propose_batch", "draft")
        install(runner, "run_verify_forward", "verify")
        if getattr(runner.config, "dspark_window_tokens", 0):
            install(runner, "_dspark_commit_outputs", "commit")
    ar_windows = {}
    dspark_windows = {}
    if diagnostic_cost and getattr(runner.config, "dspark_window_tokens", 0):
        import nanovllm.spec_decode.dspark_window as window_module
        original_visible_slots = window_module.visible_slots
        def track_visible_slots(seq, anchor, storage, block_size, window):
            slots = original_visible_slots(seq, anchor, storage, block_size, window)
            dspark_windows[seq.seq_id] = {"history_start": max(0, anchor - window), "history_end": anchor,
                                         "visible_tokens": len(slots), "first_slot": slots[0] if slots else None,
                                         "last_slot": slots[-1] if slots else None,
                                         "valid_start": seq.dspark_valid_start, "valid_end": seq.dspark_valid_end,
                                         "source": "observed engine visible_slots result"}
            return slots
        window_module.visible_slots = track_visible_slots
        restore.append((window_module, "visible_slots", original_visible_slots))
    if diagnostic_cost and runner.draft_model is not None:
        original = runner._draft_request
        def track_request(seq):
            request = original(seq)
            M = request["window_blocks"]
            bounds = []
            for offset in range(runner.config.spec_k):
                p = request["context_len"] - 1 + offset
                first_block = max(request["head_blocks"], request["valid_from"] // runner.block_size,
                                  p // runner.block_size - M + 1) if M else 0
                start = first_block * runner.block_size
                bounds.append({"position": p, "start": start, "end_inclusive": p,
                               "visible_tokens": p + 1 - start})
            ar_windows[seq.seq_id] = {"block_aligned_steps": bounds,
                                      "catchup_tokens": len(request["catchup_tokens"]),
                                      "valid_from": request["valid_from"]}
            return request
        runner._draft_request = track_request
        restore.append((runner, "_draft_request", original))
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for prompt in prompts:
            llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=output_tokens, ignore_eos=True))
        rounds, completed, seen_prefill = capture_state["rounds"], {}, set()
        prefix_cached = 0
        first_decode = True
        while not llm.is_finished():
            if time.time() >= deadline:
                raise TimeoutError("shared GPU deadline reached")
            current_round[0] = len(rounds)
            ar_windows.clear()
            dspark_windows.clear()
            t0 = time.perf_counter()
            capture_state["current_phase"] = "scheduler"
            seqs, prefill = scheduler.schedule()
            capture_state["current_phase"] = "prefill" if prefill else "decode"
            capture_state["last_active_batch"] = len(seqs)
            capture_state["preempted_sequence_ids"] = list(preemptions)
            if not seqs:
                raise RuntimeError("scheduler did not admit any requested sequence")
            if preemptions:
                raise RuntimeError(f"fixed batch capacity failed; preempted {preemptions}")
            if not prefill and first_decode:
                first_decode = False
                if len(seqs) != len(prompts):
                    raise RuntimeError(f"fixed batch capacity failed: first decode B={len(seqs)}")
            for seq in seqs:
                if prefill and seq.seq_id not in seen_prefill:
                    prefix_cached += seq.num_cached_tokens
                    seen_prefill.add(seq.seq_id)
            if prefix_cached:
                raise RuntimeError("disabled target prefix lookup nevertheless reused tokens")
            before_counts = {s.seq_id: s.num_completion_tokens for s in seqs}
            anchors = {s.seq_id: len(s) - 1 for s in seqs}
            ar_state = {s.seq_id: {"draft_valid_len": s.draft_valid_len,
                                    "draft_win_start": getattr(s, "draft_win_start", 0),
                                    "draft_block_table": list(s.draft_block_table)}
                        for s in seqs} if runner.draft_model is not None else {}
            output = runner.run(seqs, prefill)
            if len(output) != len(seqs):
                raise RuntimeError("output batch differs from scheduled batch")
            entries = []
            for seq, raw in zip(seqs, output):
                k = len(seq.draft_tokens)
                accepted = max(0, len(raw) - 1) if k and isinstance(raw, list) else 0
                C = anchors[seq.seq_id]
                w = getattr(runner.config, "dspark_window_tokens", 0)
                entry = {"seq_id": seq.seq_id, "proposed": k, "accepted_prefix": accepted,
                         "verified_output_tokens": len(raw) if isinstance(raw, list) else 1,
                         "remaining_before": seq.max_tokens - before_counts[seq.seq_id],
                         "anchor_absolute_position": C,
                         "target_cached_tokens": seq.num_cached_tokens,
                         "target_blocks": len(seq.block_table)}
                if runner.dspark_model is not None and not prefill:
                    entry["dspark_window_semantics"] = dspark_windows.get(seq.seq_id,
                        {"history_start": max(0, C - w) if w else 0, "history_end": C,
                         "visible_tokens": min(C, w) if w else C, "anchor_excluded": True,
                         "source": "configured geometry; no per-round input probe"})
                if seq.seq_id in ar_windows:
                    entry["ar_window_observed"] = ar_windows[seq.seq_id]
                if seq.seq_id in ar_state:
                    entry["ar_state_before"] = ar_state[seq.seq_id]
                    entry["ar_state_after_forward"] = {"draft_valid_len": seq.draft_valid_len,
                                                        "draft_win_start": getattr(seq, "draft_win_start", 0),
                                                        "draft_block_table": list(seq.draft_block_table)}
                entries.append(entry)
            delivered = scheduler.postprocess(seqs, output, prefill)
            end = time.perf_counter()
            for seq, entry in zip(seqs, entries):
                entry["delivered"] = seq.num_completion_tokens - before_counts[seq.seq_id]
                if seq.is_finished:
                    completed[seq.seq_id] = list(seq.completion_token_ids)
            rounds.append({"start_s": t0 - started, "end_s": end - started, "prefill": bool(prefill),
                           "active_b": len(seqs), "delivered": delivered, "sequences": entries})
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        requests = []
        for sid, record in sorted(hook.requests.items()):
            output = completed[sid]
            requests.append({"seq_id": sid, "arrival_s": record["arrival"] - started,
                             "finish_s": record["end"] - started,
                             "deliveries": [{"at_s": at - started, "count": n} for at, n in record["batches"]],
                             "output_token_ids": output, "output_sha256": token_hash(output)})
        for category, rows in event_rows.items():
            for index, start, end in rows:
                if index is not None and index < len(rounds):
                    key = category + "_gpu_event_ms"
                    rounds[index][key] = rounds[index].get(key, 0.0) + start.elapsed_time(end)
        window_after = dict(getattr(runner, "dspark_window_stats", {}))
        graph_delta = delta(graph_before, snapshot(runner))
        target_graph = {"exact": graph_delta.get("decode_graph_exact_hits", 0),
                        "padded": graph_delta.get("decode_graph_padded_hits", 0),
                        "fallbacks": graph_delta.get("decode_graph_fallbacks", {}),
                        "bucket_histogram": graph_delta.get("decode_graph_bucket_histogram", {})}
        return {"wall_s": wall, "requests": requests, "rounds": rounds,
                "delivered_tokens": sum(len(x) for x in completed.values()),
                "preemptions": len(preemptions), "prefix_cached_tokens": prefix_cached,
                "prefix_lookup_count": scheduler.block_manager.prefix_lookup_count,
                "prefix_hit_blocks": scheduler.block_manager.prefix_hit_blocks,
                "graph_delta": graph_delta,
                "target_decode_graph": target_graph,
                "gpu_event_ms": ({name: sum(x.elapsed_time(y) for _, x, y in values)
                                  for name, values in event_rows.items()} if diagnostic_cost else None),
                "gpu_event_definition": ("diagnostic stream event spans without per-kernel synchronization"
                                         if diagnostic_cost else "not collected in primary wall-time run"),
                "window_stats_before": window_before, "window_stats_after": window_after}
    finally:
        scheduler.preempt = old_preempt
        llm.token_hook = None
        scheduler.token_hook = None
        for obj, name, original in reversed(restore):
            setattr(obj, name, original)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--phase", choices=("pilot", "formal", "capacity"), required=True)
    ap.add_argument("--batch", type=int, required=True, choices=BUCKETS)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--output-tokens", type=int, default=256)
    panel_args = ap.add_mutually_exclusive_group()
    panel_args.add_argument("--panel", choices=("code", "math", "general"), help="单面板 pilot")
    panel_args.add_argument("--panels", nargs="+", choices=("code", "math", "general"), help="默认三个面板")
    repeat_args = ap.add_mutually_exclusive_group()
    repeat_args.add_argument("--repeat", type=int, help="单次 pilot 的 repeat 编号")
    repeat_args.add_argument("--repeats", type=int, help="重复次数；默认 3")
    ap.add_argument("--repeat-offset", type=int, default=0)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ar-model", required=True)
    ap.add_argument("--dspark-model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-num-batched-tokens", type=int, help="容量实验可显式设置相同 chunked prefill 预算")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70,
                    help="默认 0.70；pilot/formal 可显式 0.85，capacity 按独立计划设置")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--diagnostic-cost", action="store_true",
                    help="额外 CUDA event 与逐轮窗口观测；输出不可作为正式吞吐")
    a = ap.parse_args()
    batched = a.max_num_batched_tokens if a.max_num_batched_tokens is not None else a.batch * a.ctx
    try:
        validate_budget(a.phase, a.gpu_memory_utilization)
        options = options_for(a.arm, a.batch, a.ctx, a.output_tokens, a.window,
                              a.target, a.ar_model, a.dspark_model, batched, a.gpu_memory_utilization)
    except ValueError as exc:
        ap.error(str(exc))
    panels = [a.panel] if a.panel else a.panels if a.panels else ["code", "math", "general"]
    repeats = [a.repeat] if a.repeat is not None else list(range(a.repeat_offset, a.repeat_offset + (a.repeats if a.repeats is not None else 3)))
    if (not panels or len(set(panels)) != len(panels) or not repeats or
            any(r < 0 for r in repeats) or a.repeat_offset < 0 or (a.repeat is not None and a.repeat_offset)):
        ap.error("panel 或 repeat 参数无效")
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    if manifest["context_tokens"] != a.ctx:
        ap.error("formal workload context must match manifest exactly")
    panel_rows = {panel: read_manifest(a.manifest, panel, a.batch, a.ctx) for panel in panels}
    for rows in panel_rows.values():
        for _, metadata in rows:
            if (type(metadata.get("wiki_tokens")) is not int or not 0 <= metadata["wiki_tokens"] <= a.ctx
                    or type(metadata.get("task_truncated_from_left")) is not bool):
                ap.error("manifest must record Wiki filling and task truncation")
            metadata["wiki_fraction"] = metadata["wiki_tokens"] / a.ctx
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("x", encoding="utf-8")
    def write(row):
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    mode = "ring" if a.arm == "dspark_ring" else "full_cache" if a.arm == "dspark_window_full" else "block_aligned" if a.arm == "ar_window" else "full"
    k = 0 if a.arm == "nospec" else 6 if a.arm.startswith("ar_") else 7
    ident = {"arm": a.arm, "mode": mode, "phase": a.phase, "requested_batch": a.batch,
             "max_num_seqs": a.batch, "ctx": a.ctx, "window": a.window,
             "output_tokens": a.output_tokens, "panels": panels, "repeats": repeats,
             "seed": a.seed, "temperature": 1.0, "ignore_eos": True, "spec_k": k,
             "max_num_batched_tokens": batched, "max_model_len": options["max_model_len"],
             "gpu_memory_utilization": options["gpu_memory_utilization"], "graph_mode": "graph",
             "graph_policy": {"name": "exact_all", "target_decode_padding": False,
                              "target_verify_padding": False, "draft_padding": False,
                              "buckets": list(range(1, a.batch + 1))},
             "prefix_lookup": False, "mixed_overlap": False,
             "diagnostic_instrumented": a.diagnostic_cost,
             "instrumentation": ("diagnostic CUDA events and per-round window geometry" if a.diagnostic_cost else
                                 "request delivery hook, CPU round bookkeeping, native graph counters"),
             "guard_id": os.environ.get("DS_GUARD_ID"),
             "gpu_uuid": os.environ.get("DS_GPU_UUID")}
    llm = None
    state = {"rounds": []}
    active_state = state
    stage = "preflight"
    active_cell = None
    try:
        if not all(os.environ.get(x) for x in ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX")):
            raise RuntimeError("must start through gpu_control.py")
        commit, dirty = git_provenance()
        ident.update(engine_commit=commit, engine_dirty=dirty)
        if dirty:
            raise RuntimeError("engine or experiment source is dirty; use a frozen committed clone")
        target_prov = model_provenance(a.target)
        draft_path = a.ar_model if a.arm.startswith("ar_") else a.dspark_model if a.arm.startswith("dspark_") else None
        draft_prov = model_provenance(draft_path)
        ident.update(target_config_sha256=target_prov["config_sha256"],
                     draft_config_sha256=draft_prov["config_sha256"] if draft_prov else None,
                     target_provenance=target_prov, draft_provenance=draft_prov)
        import torch
        gpu = verify_visible_gpu(torch, ident["gpu_uuid"])
        ident["gpu_name"] = gpu.name
        from nanovllm import LLM
        from nanovllm.config import Config
        missing = set(options) - {field.name for field in fields(Config)}
        if missing:
            raise RuntimeError(f"engine config fields unavailable: {sorted(missing)}")
        stage = "model_load_initial_graph_or_warmup"
        init_start = time.perf_counter()
        llm = LLM(a.target, **options)
        runner, prop = llm.model_runner, llm.model_runner.spec_proposer
        if runner.config.max_model_len < options["max_model_len"]:
            raise RuntimeError("target max positions cannot cover prompt + output + speculation")
        if runner.config.quant_weights or runner.config.draft_quant_weights:
            raise RuntimeError("quantized model detected; this experiment requires BF16")
        for label in ("model", "draft_model", "dspark_model"):
            model = getattr(runner, label, None)
            if model is not None and any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
                raise RuntimeError(f"{label} parameters are not exclusively BF16")
        if runner.config.enable_prefix_cache or llm.scheduler.block_manager.enable_prefix_cache:
            raise RuntimeError("target prefix lookup was not disabled")
        for key, expected in options.items():
            if getattr(runner.config, key) != expected:
                raise RuntimeError(f"engine config mismatch for {key}")
        for key in ("decode_graph_hits", "decode_graph_exact_hits", "decode_graph_padded_hits",
                    "decode_graph_bucket_histogram", "decode_graph_fallbacks"):
            if not hasattr(runner, key):
                raise RuntimeError(f"engine missing native target decode graph counter {key}")
        torch.cuda.synchronize()
        ident["load_initial_graph_s"] = time.perf_counter() - init_start
        ident["target_kv_blocks"] = runner.config.num_kvcache_blocks
        ident["draft_kv_blocks"] = runner.config.num_draft_blocks
        ident["kv_block_size"] = runner.block_size
        mem_init = memory(runner)
        stage = "graph_prewarm"
        expected_shapes = []
        before_warm = snapshot(runner)
        warm_start = time.perf_counter()
        if a.arm.startswith("dspark_"):
            expected_shapes = prewarm_shapes(a.batch, a.batch, a.ctx, a.output_tokens,
                                             a.window, runner.block_size, tuple(range(1, a.batch + 1)))
            prop.prewarm(expected_shapes, range(1, a.batch + 1))
        torch.cuda.synchronize()
        prewarm_s = time.perf_counter() - warm_start
        mem_after_prewarm = memory(runner)
        after_prewarm = snapshot(runner)
        stage = "warmup_requests"
        warm_state = {"rounds": []}
        active_state = warm_state
        warm = run_once(llm, [x for x, _ in panel_rows[panels[0]]], min(16, a.output_tokens), a.seed,
                        float(os.environ["DS_RUN_DEADLINE_UNIX"]), warm_state, a.diagnostic_cost)
        warm_graph_delta = delta(after_prewarm, snapshot(runner))
        write({"type": "warmup", **ident, "prewarm_s": prewarm_s,
               "expected_prewarm_shapes": expected_shapes,
               "prewarm_shapes": list(getattr(prop, "prewarmed_shapes", [])) if expected_shapes else [],
               "prewarm_head_batches": list(range(1, a.batch + 1)) if expected_shapes else [],
               "target_decode_graph_buckets": list(runner.graph_bs),
               "target_verify_graph_shapes": [list(x) for x in sorted(runner.verify_graphs)],
               "ar_draft_graph_buckets": sorted(getattr(prop, "_graphs", {}).keys()) if prop is not None else [],
               "prewarm_graph_delta": delta(before_warm, after_prewarm),
               "warmup_request_wall_s": warm["wall_s"], "warmup_graph_delta": warm_graph_delta,
               "memory_after_init": mem_init, "memory_after_prewarm": mem_after_prewarm,
               "memory_after_warmup_requests": memory(runner)})
        if warm_graph_delta.get("graph_runtime_captures", 0) or warm_graph_delta.get("head_runtime_captures", 0):
            raise RuntimeError("unplanned graph capture during warmup requests; inspect recorded shapes")
        print(f"预热完成：{a.phase}/{a.arm}，B={a.batch}，面板={panels}，重复={repeats}", flush=True)
        stage = "measured_requests"
        for position, repeat in enumerate(repeats):
            # Rotate panel ordering across timing repeats; arm ordering is set by the external runner.
            shift = position % len(panels)
            for panel in panels[shift:] + panels[:shift]:
                rows = panel_rows[panel]
                active_cell = {"panel": panel, "repeat": repeat,
                               "input_metadata": [m for _, m in rows]}
                active_state = {"rounds": []}
                result = run_once(llm, [x for x, _ in rows], a.output_tokens, a.seed,
                                  float(os.environ["DS_RUN_DEADLINE_UNIX"]), active_state, a.diagnostic_cost)
                result.update(memory_after_run=memory(runner))
                record = {"type": "run", **ident, **active_cell, **result}
                write(record)
                computed = compute_run(record)
                if not computed["valid"]:
                    raise RuntimeError("raw measurement contract failed: " + "; ".join(computed["errors"]))
                if result["graph_delta"].get("graph_runtime_captures", 0) or result["graph_delta"].get("head_runtime_captures", 0):
                    raise RuntimeError("measured-phase graph capture occurred")
                if llm.scheduler.block_manager.prefix_lookup_count or llm.scheduler.block_manager.prefix_hit_blocks:
                    raise RuntimeError("prefix lookup or hit occurred")
                print(f"完成：{a.phase}/{a.arm} B={a.batch} {panel} repeat={repeat}，真实交付 {result['delivered_tokens']} tokens", flush=True)
    except BaseException as exc:
        write({"type": "failure", **ident, **(active_cell or {}), "failure_phase": stage,
               "request_subphase": active_state.get("current_phase"),
               "last_active_batch": active_state.get("last_active_batch"),
               "preempted_sequence_ids": active_state.get("preempted_sequence_ids", []),
               "error": type(exc).__name__, "message": str(exc),
               "partial_rounds": active_state["rounds"], "traceback": traceback.format_exc()})
        raise
    finally:
        f.close()
        if llm is not None:
            import atexit
            llm.exit()
            atexit.unregister(llm.exit)


if __name__ == "__main__":
    main()
