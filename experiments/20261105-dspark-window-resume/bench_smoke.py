"""Instrumented real DSpark smoke: one mode and one fixed batch per process.

This is NOT a throughput benchmark. It recomputes eager conditional q on the
first real proposal (extra GPU work) and records raw delivery/round telemetry.
Use gpu_control.py for an explicitly assigned GPU. Separate replay controls,
forced rejections, and numeric full_cache/ring parity remain mandatory.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpu_identity import verify_visible_gpu


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def target_decode_bucket(batch):
    """Ordinary target decode uses the engine's existing padded graph buckets."""
    return next(b for b in (1, 2, 4, 8, 16) if b >= batch)


def prewarm_shapes(batch, max_seqs, ctx, output_tokens, window, block_size, graph_buckets):
    allowed = sorted({b for b in graph_buckets if b <= max_seqs})
    if not allowed or allowed[-1] < batch:
        raise ValueError("graph buckets do not cover the requested real batch")
    graph_batches = sorted({next(b for b in allowed if b >= real) for real in range(1, batch + 1)})
    # Conservative range: covers any tail batch and every context bucket through
    # the full output budget. W=0 means unbounded full history.
    def bucket(c):
        visible = min(c, window) if window else c
        return max(block_size, math.ceil(visible / block_size) * block_size)
    contexts = sorted({bucket(c) for c in range(ctx, ctx + output_tokens + 1)})
    return [(b, c) for b in graph_batches for c in contexts]


def read_manifest(path, panel, batch, ctx):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source_ctx = data["context_tokens"]
    if not 0 < ctx <= source_ctx:
        raise ValueError("smoke ctx must fit the supplied manifest")
    rows = data["panels"][panel][:batch]
    if len(rows) != batch:
        raise ValueError("manifest does not contain enough requests")
    result = []
    for row in rows:
        ids, meta = row["token_ids"], row["metadata"]
        if len(ids) != source_ctx or token_hash(ids) != meta["input_sha256"]:
            raise ValueError("manifest length or hash mismatch")
        selected = ids[-ctx:]
        result.append((selected, {**meta, "source_context_tokens": source_ctx,
                                 "actual_context_tokens": ctx, "actual_input_sha256": token_hash(selected)}))
    return result


def snapshot(runner):
    proposer = runner.spec_proposer
    result = {"verify_hits": runner.verify_graph_hits,
              "verify_padded_hits": runner.verify_graph_padded,
              "verify_exact_hits": runner.verify_graph_hits - runner.verify_graph_padded,
              "verify_fallbacks": dict(runner.verify_graph_fallbacks)}
    for name in ("decode_graph_hits", "decode_graph_exact_hits", "decode_graph_padded_hits",
                 "decode_graph_bucket_histogram", "decode_graph_fallbacks"):
        if hasattr(runner, name):
            v = getattr(runner, name)
            result[name] = v.copy() if isinstance(v, (dict, list)) else v
    for name in ("graph_hits", "graph_exact_hits", "graph_padded_hits", "graph_captures",
                 "graph_runtime_captures", "graph_runtime_capture_s", "graph_runtime_shapes",
                 "graph_fallbacks", "active_batch_histogram", "head_graph_hits",
                 "head_graph_captures", "head_runtime_captures", "head_runtime_capture_s",
                 "head_runtime_batches", "context_bucket_histogram", "input_prepare_calls",
                 "input_prepare_wall_s"):
        if hasattr(proposer, name):
            v = getattr(proposer, name)
            result[name] = v.copy() if isinstance(v, (dict, list)) else v
    return result


def delta(before, after):
    result = {}
    for name, value in after.items():
        old = before.get(name, {} if isinstance(value, dict) else [] if isinstance(value, list) else 0)
        if isinstance(value, dict):
            result[name] = {str(k): v - old.get(k, 0) for k, v in value.items() if v != old.get(k, 0)}
        elif isinstance(value, list):
            result[name] = value[len(old):]
        else:
            result[name] = value - old
    return result


def memory(runner):
    import torch
    tensors, weights, pointers = {}, {}, set()
    for label, obj in (("runner", runner), ("dspark", runner.dspark_model)):
        for name, value in vars(obj).items():
            if torch.is_tensor(value) and ("cache" in name or "scratch" in name or "feature" in name):
                tensors[f"{label}.{name}"] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                              "bytes": value.numel() * value.element_size()}
    for label, obj in (("target", runner.model), ("dspark", runner.dspark_model)):
        count = 0
        for par in obj.parameters():
            key = (str(par.device), par.data_ptr())
            if key not in pointers:
                pointers.add(key)
                count += par.numel() * par.element_size()
        weights[label] = count
    pool_segments = {}
    for segment in torch.cuda.memory_snapshot():
        pool = str(tuple(segment.get("segment_pool_id", (0, 0))))
        pool_segments[pool] = pool_segments.get(pool, 0) + segment["total_size"]
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "named_kv_scratch_feature_tensors": tensors, "unique_parameter_bytes": weights,
            "allocator_segment_bytes_by_pool_id": pool_segments,
            "engine_graph_mem_gb": getattr(runner, "graph_mem", None),
            "accounting_note": "allocator pools overlap named tensor/weight amounts; feature temporaries may be unnamed"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="exclusive new JSONL")
    p.add_argument("--panel", choices=("code", "math", "general"), default="code")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--max-seqs", type=int, help="默认 batch；旧补桶 B=3/11/15 需设为 4/16/16")
    p.add_argument("--ctx", type=int, default=252)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--mode", choices=("w0", "full_cache", "ring"), required=True)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--graph", choices=("eager", "exact", "padded"), required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--fixed-anchor", type=int, help="仅对拍时替换 prefill 首 token；之后仍为真实 T=1 采样")
    a = p.parse_args()
    a.max_seqs = a.max_seqs if a.max_seqs is not None else a.batch
    if not 1 <= a.batch <= a.max_seqs <= 16 or a.output_tokens < 9 or a.ctx < 1:
        p.error("batch/ctx/output 长度或 max-seqs 无效")
    if a.fixed_anchor is not None and a.fixed_anchor < 0:
        p.error("fixed-anchor 不能是负数")
    if a.graph == "exact" and a.output_tokens < 10:
        p.error("exact smoke 至少需10个输出，保证第一轮后仍有普通decode对拍机会")
    if (a.mode == "w0") != (a.window == 0):
        p.error("w0 必须 window=0，full_cache/ring 必须 window>0")
    if a.window and a.window % 256:
        p.error("window 必须为 256 的整数倍")
    buckets = (1, 2, 4, 8, 16)
    if a.graph == "padded" and a.batch in buckets:
        p.error("padded smoke 必须使用未精确捕获的 batch，例如 3、11、15")
    if a.graph == "exact":
        # All possible real tail sizes get exact speculative graphs.
        buckets = tuple(range(1, a.batch + 1))
    if a.graph != "eager":
        if a.graph == "padded" and a.max_seqs < target_decode_bucket(a.batch):
            p.error("普通 target decode 桶需要足够 max-seqs；B3 用 4、B11/15 用 16")
        prewarm_shapes(a.batch, a.max_seqs, a.ctx, a.output_tokens, a.window, 256, buckets)
    inputs = read_manifest(a.manifest, a.panel, a.batch, a.ctx)
    output_path = Path(a.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create before importing torch or loading models; do not overwrite evidence.
    raw = output_path.open("x", encoding="utf-8")
    def write(row):
        raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        raw.flush()

    ident = {"type": "run", "arm": "dspark", "mode": a.mode, "window": a.window,
             "graph_mode": a.graph, "requested_batch": a.batch, "max_num_seqs": a.max_seqs,
             "spec_graph_buckets": buckets,
             "decode_graph_buckets": tuple(range(1, a.max_seqs + 1)) if a.graph == "exact" else None,
             "ordinary_target_decode_bucket": (a.batch if a.graph == "exact" else
                                                target_decode_bucket(a.batch) if a.graph == "padded" else None),
             "ctx": a.ctx, "output_tokens": a.output_tokens, "panel": a.panel, "repeat": 0,
             "seed": a.seed, "temperature": 1.0, "ignore_eos": True, "spec_k": 7,
             "prefix_lookup": False, "mixed_overlap": False, "diagnostic_instrumented": True,
             "fixed_anchor": a.fixed_anchor, "input_metadata": [m for _, m in inputs],
             "guard_id": os.environ.get("DS_GUARD_ID"), "gpu_uuid": os.environ.get("DS_GPU_UUID"),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    llm = None
    try:
        if not all(os.environ.get(x) for x in ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX")):
            raise RuntimeError("必须通过本轮 gpu_control.py 在已授权 GPU 上启动 smoke")
        import torch
        verify_visible_gpu(torch, os.environ["DS_GPU_UUID"])
        from nanovllm import LLM, SamplingParams
        from nanovllm.config import Config
        from nanovllm.engine.token_hook import TokenDeliveryHook
        required = {"dspark_window_tokens", "dspark_window_storage", "enable_prefix_cache"}
        if a.graph == "exact":
            required.add("decode_graph_bs")
        missing = required - {f.name for f in fields(Config)}
        if missing:
            raise RuntimeError(f"engine missing window config fields: {sorted(missing)}")
        max_len = a.ctx + a.output_tokens + 7
        options = dict(max_num_seqs=a.max_seqs, max_model_len=max_len,
                       max_num_batched_tokens=max(a.batch * a.ctx, max_len),
                       gpu_memory_utilization=0.70, tensor_parallel_size=1,
                       enforce_eager=a.graph == "eager", spec_cuda_graph=a.graph != "eager",
                       spec_method="dspark", spec_k=7, draft_model=a.draft,
                       spec_graph_bs=buckets, spec_graph_pad=a.graph == "padded",
                       dspark_split_graph=True, dspark_graph_pad=a.graph == "padded",
                       dspark_window_tokens=a.window,
                       dspark_window_storage="ring" if a.mode == "ring" else "full_cache",
                       enable_prefix_cache=False, spec_overlap_mixed=False)
        if a.graph == "exact":
            options["decode_graph_bs"] = tuple(range(1, a.max_seqs + 1))
        started = time.perf_counter()
        llm = LLM(a.target, **options)
        runner, scheduler = llm.model_runner, llm.scheduler
        prop, model = runner.spec_proposer, runner.dspark_model
        if a.fixed_anchor is not None and a.fixed_anchor >= model.cfg.vocab_size:
            raise ValueError("fixed-anchor exceeds model vocabulary")
        for name in required:
            if getattr(runner.config, name) != options[name]:
                raise RuntimeError(f"engine did not retain config {name}")
        if runner.config.max_model_len < max_len:
            raise RuntimeError("target max position capacity cannot cover prompt + output + speculation")
        torch.cuda.synchronize()
        init_s = time.perf_counter() - started
        mem_init = memory(runner)
        warm_shapes = []
        warm_s = 0.0
        if a.graph != "eager":
            warm_shapes = prewarm_shapes(a.batch, a.max_seqs, a.ctx, a.output_tokens,
                                         a.window, runner.block_size, buckets)
            warm_started = time.perf_counter()
            prop.prewarm(warm_shapes, range(1, a.batch + 1))
            torch.cuda.synchronize()
            warm_s = time.perf_counter() - warm_started
        write({"type": "warmup", **{k: v for k, v in ident.items() if k != "type"},
               "init_s": init_s, "prewarm_s": warm_s, "prewarm_shapes": warm_shapes,
               "prewarm_head_batches": list(range(1, a.batch + 1)) if a.graph != "eager" else [],
               "graph_after_prewarm": snapshot(runner), "memory_after_init": mem_init,
               "memory_after_prewarm": memory(runner)})
        print(f"模型加载与预热完成：{a.mode}，B={a.batch}，{a.graph}", flush=True)

        # Extra eager forward checks the same actual conditional predecessor for
        # each of seven real proposal logits; it is diagnostic overhead.
        checks = []
        original_propose = prop.propose_batch
        @torch.inference_mode()
        def checked_propose(seqs):
            chains, logits = original_propose(seqs)
            if checks:
                return chains, logits
            # Preserve actual graph output before any additional eager operation.
            preserved_logits = logits.detach().clone()
            actual = preserved_logits.float()
            anchors, lengths, tables, bucket, temperatures = prop._inputs(seqs)
            base = model.base_logits(anchors, lengths, tables, bucket)
            previous = anchors
            expected = []
            for j in range(7):
                expected.append(base[:, j] + model.markov_head(previous))
                previous = torch.tensor([chain[j] for chain in chains], device=anchors.device)
            expected = torch.stack(expected, dim=1).float()
            logp, logq = torch.log_softmax(expected, -1), torch.log_softmax(actual, -1)
            kls = (logp.exp() * (logp - logq)).sum(-1)
            errors = (expected - actual).abs().amax(-1)
            checks.append({"batch": len(seqs), "context_bucket": bucket,
                           "absolute_context_lengths": lengths.tolist(),
                           "context_table_kind": "physical_slots" if a.window else "target_block_table",
                           "actual_context_table": tables.tolist(),
                           "sampled_candidates": chains, "positions": 7,
                           "max_conditional_kl": float(kls.max()),
                           "max_logit_abs": float(errors.max()),
                           "relative_logit_rms": float((expected - actual).square().mean().sqrt() /
                                                       expected.square().mean().sqrt().clamp_min(1e-12)),
                           "kl_per_sequence_position": kls.tolist()})
            return chains, preserved_logits
        prop.propose_batch = checked_propose

        # Diagnostic comparisons restore identical query-slot/scratch preimages
        # before each path. Confirmed prefix slots are never changed here.
        numeric = {"target_verify": None, "target_decode": None}
        active_seqs = [None]
        if a.graph == "exact":
            for name in ("decode_graph_hits", "decode_graph_exact_hits", "decode_graph_padded_hits"):
                if not hasattr(runner, name):
                    raise RuntimeError(f"engine missing target decode graph counter {name}")
            from nanovllm.utils.context import get_context
            def check_query_mapping(slots, positions, verifying):
                expected_positions, expected_slots = [], []
                owned_blocks = [block for seq in active_seqs[0] for block in seq.block_table]
                if len(set(owned_blocks)) != len(owned_blocks):
                    raise AssertionError("target block tables contain shared or duplicate blocks")
                for seq in active_seqs[0]:
                    count = 1 + len(seq.draft_tokens) if verifying else 1
                    for pos in range(len(seq) - 1, len(seq) - 1 + count):
                        if pos // runner.block_size >= len(seq.block_table):
                            raise AssertionError("target block table does not cover query")
                        expected_positions.append(pos)
                        expected_slots.append(seq.block_table[pos // runner.block_size] * runner.block_size + pos % runner.block_size)
                if slots.tolist() != expected_slots or positions.tolist() != expected_positions:
                    raise AssertionError("target query positions or physical slots differ from sequence block tables")
            def read_query_state(slots):
                slots = slots.long()
                if slots.ndim != 1 or bool((slots < 0).any()) or torch.unique(slots).numel() != slots.numel():
                    raise RuntimeError("invalid or duplicate real target query slots")
                blocks = torch.div(slots, runner.block_size, rounding_mode="floor")
                offsets = slots % runner.block_size
                target = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                if a.mode == "ring":
                    draft = model.scratch[:, :, :slots.numel()].detach().clone()
                else:
                    draft = model.cache.index_select(2, slots).detach().clone()
                return slots, blocks, offsets, target, draft
            def restore_query_state(state):
                slots, blocks, offsets, target, draft = state
                runner.kv_cache[:, :, blocks, offsets] = target
                if a.mode == "ring":
                    model.scratch[:, :, :slots.numel()].copy_(draft)
                else:
                    model.cache.index_copy_(2, slots, draft)
                restored = read_query_state(slots)
                for before, after in ((target, restored[3]), (draft, restored[4])):
                    if not torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)):
                        raise AssertionError("target query or DSpark scratch preimage was not restored bitwise")
            def compare(ref_logits, actual_logits, ref_state, actual_state):
                if ref_logits.shape != actual_logits.shape:
                    raise RuntimeError("target numerical outputs differ in shape")
                kls = []
                for start in range(0, ref_logits.shape[0], 16):
                    p = torch.log_softmax(ref_logits[start:start + 16].float(), -1)
                    q = torch.log_softmax(actual_logits[start:start + 16].float(), -1)
                    kls.append((p.exp() * (p - q)).sum(-1))
                def rms(x, y):
                    x, y = x.float(), y.float()
                    return float((x - y).square().mean().sqrt() / x.square().mean().sqrt().clamp_min(1e-12))
                return {"rows": ref_logits.shape[0], "target_logit_max_kl": float(torch.cat(kls).max()),
                        "target_kv_relative_rms": rms(ref_state[3], actual_state[3]),
                        "dspark_kv_or_scratch_relative_rms": rms(ref_state[4], actual_state[4]),
                        "same_query_preimage_restored": True}
            original_verify = runner.run_verify_forward
            @torch.inference_mode()
            def checked_verify(ids, positions):
                if numeric["target_verify"] is not None:
                    return original_verify(ids, positions)
                slots = runner._verify_extra["slot_mapping"]
                check_query_mapping(slots, positions, verifying=True)
                preimage = read_query_state(slots)
                previous_graphs = runner.verify_graphs
                try:
                    runner.verify_graphs = {}
                    ref = original_verify(ids, positions).detach().clone()
                    ref_state = read_query_state(slots)
                    restore_query_state(preimage)
                finally:
                    runner.verify_graphs = previous_graphs
                before_exact = runner.verify_graph_hits - runner.verify_graph_padded
                actual = original_verify(ids, positions).detach().clone()
                if runner.verify_graph_hits - runner.verify_graph_padded != before_exact + 1:
                    raise AssertionError("target verify exact graph was not actually replayed")
                actual_state = read_query_state(slots)
                numeric["target_verify"] = compare(ref, actual, ref_state, actual_state)
                write({"type": "target_verify_probe", **{k: v for k, v in ident.items() if k != "type"},
                       **numeric["target_verify"]})
                return actual
            runner.run_verify_forward = checked_verify
            original_run_model = runner.run_model
            @torch.inference_mode()
            def checked_run_model(ids, positions, is_prefill):
                if is_prefill or numeric["target_decode"] is not None:
                    return original_run_model(ids, positions, is_prefill)
                slots = get_context().slot_mapping
                check_query_mapping(slots, positions, verifying=False)
                preimage = read_query_state(slots)
                old_eager = runner.enforce_eager
                try:
                    runner.enforce_eager = True
                    ref = original_run_model(ids, positions, False).detach().clone()
                    ref_state = read_query_state(slots)
                    restore_query_state(preimage)
                finally:
                    runner.enforce_eager = old_eager
                before_exact = runner.decode_graph_exact_hits
                actual = original_run_model(ids, positions, False).detach().clone()
                if runner.decode_graph_exact_hits != before_exact + 1:
                    raise AssertionError("target decode exact graph was not actually replayed")
                actual_state = read_query_state(slots)
                numeric["target_decode"] = compare(ref, actual, ref_state, actual_state)
                write({"type": "target_decode_probe", **{k: v for k, v in ident.items() if k != "type"},
                       **numeric["target_decode"]})
                return actual
            runner.run_model = checked_run_model

        events = {"draft": [], "verify": [], "commit": []}
        current_round = [None]
        def install_event(obj, name, category):
            original = getattr(obj, name)
            def wrapped(*args, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(*args, **kwargs)
                end.record()
                events[category].append((current_round[0], start, end))
                return result
            setattr(obj, name, wrapped)
        install_event(prop, "propose_batch", "draft")
        install_event(runner, "run_verify_forward", "verify")
        commit_instrumented = bool(a.window and hasattr(runner, "_dspark_commit_outputs"))
        if commit_instrumented:
            install_event(runner, "_dspark_commit_outputs", "commit")
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        scheduler.token_hook = hook
        preemptions = []
        original_preempt = scheduler.preempt
        def count_preempt(seq):
            preemptions.append(seq.seq_id)
            return original_preempt(seq)
        scheduler.preempt = count_preempt
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()
        before = snapshot(runner)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for prompt, _ in inputs:
            llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=a.output_tokens, ignore_eos=True))
        rounds, done, prefix_cached, seen_prefill = [], {}, 0, set()
        decode_rounds = 0
        forced_ordinary_decode = False
        deadline = float(os.environ.get("DS_RUN_DEADLINE_UNIX", "0"))
        while not llm.is_finished():
            if deadline and time.time() >= deadline:
                raise TimeoutError("GPU guard deadline reached")
            t0 = time.perf_counter()
            current_round[0] = len(rounds)
            seqs, is_prefill = scheduler.schedule()
            if not seqs:
                raise RuntimeError("scheduler did not admit any request")
            if preemptions:
                raise RuntimeError(f"requested B was preempted: {preemptions}")
            for seq in seqs:
                if is_prefill and seq.seq_id not in seen_prefill:
                    seen_prefill.add(seq.seq_id)
                    prefix_cached += seq.num_cached_tokens
            if prefix_cached:
                raise RuntimeError(f"disabled prefix lookup nevertheless reused {prefix_cached} tokens")
            forced_this_round = False
            if not is_prefill:
                decode_rounds += 1
                if a.graph == "exact" and decode_rounds == 2:
                    # Same real batch and native T=1 target sampler; only the
                    # scheduling choice is forced to cover ordinary decode.
                    for seq in seqs:
                        seq.num_scheduled_tokens = 1
                    forced_this_round = forced_ordinary_decode = True
            before_counts = {s.seq_id: s.num_completion_tokens for s in seqs}
            anchors_before = {s.seq_id: len(s) - 1 for s in seqs}
            active_seqs[0] = seqs
            output = runner.run(seqs, is_prefill)
            if is_prefill and a.fixed_anchor is not None:
                # Only completed prefill rows produce a real token.
                output = [a.fixed_anchor if s.num_cached_tokens + s.num_scheduled_tokens >= len(s) else t
                          for s, t in zip(seqs, output)]
            if len(output) != len(seqs):
                raise RuntimeError("runner output batch length mismatch")
            entries = []
            for s, o in zip(seqs, output):
                proposed = len(s.draft_tokens)
                accepted = max(0, len(o) - 1) if proposed and isinstance(o, list) else 0
                C = anchors_before[s.seq_id]
                entries.append({"seq_id": s.seq_id, "proposed": proposed, "accepted_prefix": accepted,
                                "verified_output_tokens": len(o) if isinstance(o, list) else 1,
                                "remaining_before": s.max_tokens - before_counts[s.seq_id],
                                "anchor_absolute_position": C,
                                "expected_draft_history_start": max(0, C - a.window) if a.window else 0,
                                "expected_draft_history_end": C,
                                "target_cached_tokens_before_forward": s.num_cached_tokens,
                                "target_block_table_before_forward": list(s.block_table)})
            delivered = scheduler.postprocess(seqs, output, is_prefill)
            end = time.perf_counter()
            for s, entry in zip(seqs, entries):
                entry["delivered"] = s.num_completion_tokens - before_counts[s.seq_id]
                if s.is_finished:
                    done[s.seq_id] = list(s.completion_token_ids)
            rounds.append({"start_s": t0 - started, "end_s": end - started,
                           "prefill": bool(is_prefill), "active_b": len(seqs),
                           "forced_ordinary_decode": forced_this_round,
                           "delivered": delivered, "sequences": entries})
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        requests = []
        for sid, h in sorted(hook.requests.items()):
            tokens = done[sid]
            requests.append({"seq_id": sid, "arrival_s": h["arrival"] - started,
                             "finish_s": h["end"] - started,
                             "deliveries": [{"at_s": ts - started, "count": n} for ts, n in h["batches"]],
                             "output_token_ids": tokens, "output_sha256": token_hash(tokens)})
        graph_delta = delta(before, snapshot(runner))
        for category, items in events.items():
            for index, event_start, event_end in items:
                if index is not None and index < len(rounds):
                    key = category + "_gpu_event_ms"
                    rounds[index][key] = rounds[index].get(key, 0.0) + event_start.elapsed_time(event_end)
        run = {**ident, "wall_s": wall, "requests": requests, "rounds": rounds,
               "delivered_tokens": sum(len(x) for x in done.values()),
               "preemptions": len(preemptions), "prefix_cached_tokens": prefix_cached,
               "graph_delta": graph_delta, "conditional_q_checks": checks,
               "target_numeric_checks": numeric, "forced_ordinary_decode": forced_ordinary_decode,
               "diagnostic_eager_fallbacks": ({"target_verify": 1, "target_decode": 1}
                                              if a.graph == "exact" else {}),
               "gpu_event_ms": {k: sum(start.elapsed_time(end) for _, start, end in rows)
                                for k, rows in events.items()},
               "commit_timing_note": ("event span for commit helper" if commit_instrumented
                                      else "not instrumented for W=0 or missing helper"),
               "draft_gpu_event_note": "includes one extra eager q diagnostic forward; not performance data",
               "verify_gpu_event_note": "includes eager/exact target diagnostic comparison; not performance data",
               "memory_after_run": memory(runner), "max_model_len": runner.config.max_model_len,
               "window_stats": dict(getattr(runner, "dspark_window_stats", {})),
               "prefix_lookup_count": scheduler.block_manager.prefix_lookup_count,
               "prefix_hit_blocks": scheduler.block_manager.prefix_hit_blocks,
               "kv_block_size": runner.block_size,
               "target_kv_blocks": runner.config.num_kvcache_blocks}
        write(run)
        if not checks or not math.isfinite(checks[0]["max_conditional_kl"]) or checks[0]["max_conditional_kl"] >= 0.02:
            raise RuntimeError("same-predecessor eager vs actual proposal q comparison failed")
        if a.graph != "eager" and graph_delta.get("graph_hits", 0) <= 0:
            raise RuntimeError("requested draft graph did not replay")
        if a.graph != "eager" and graph_delta.get("verify_hits", 0) <= 0:
            raise RuntimeError("requested verify graph did not replay")
        if a.graph == "exact" and (graph_delta.get("graph_exact_hits", 0) <= 0 or
                                   graph_delta.get("verify_exact_hits", 0) <= 0):
            raise RuntimeError("requested exact draft and verify graphs did not both replay")
        if a.graph == "exact":
            if not forced_ordinary_decode or not all(numeric.values()):
                raise RuntimeError("target verify and forced ordinary decode numeric probes incomplete")
            if any(graph_delta.get(k, 0) for k in
                   ("graph_padded_hits", "verify_padded_hits", "decode_graph_padded_hits")):
                raise RuntimeError("padding occurred during exact-all smoke")
            if graph_delta.get("decode_graph_fallbacks", {}) != {"enforce_eager": 1}:
                raise RuntimeError("unexpected target decode fallback outside the one eager probe")
            for name, probe in numeric.items():
                if (not math.isfinite(probe["target_logit_max_kl"]) or probe["target_logit_max_kl"] >= 0.02
                        or not math.isfinite(probe["target_kv_relative_rms"]) or probe["target_kv_relative_rms"] >= 0.05
                        or not math.isfinite(probe["dspark_kv_or_scratch_relative_rms"])
                        or probe["dspark_kv_or_scratch_relative_rms"] >= 0.05):
                    raise RuntimeError(f"target {name} exact vs eager numeric comparison failed")
        if a.graph == "padded" and graph_delta.get("graph_padded_hits", 0) <= 0:
            raise RuntimeError("requested padded draft graph did not replay")
        if a.graph == "padded" and graph_delta.get("verify_padded_hits", 0) <= 0:
            raise RuntimeError("requested padded verify graph did not replay")
        if scheduler.block_manager.prefix_lookup_count or scheduler.block_manager.prefix_hit_blocks:
            raise RuntimeError("prefix lookup or hits occurred despite disabled policy")
        if graph_delta.get("graph_runtime_captures", 0) or graph_delta.get("head_runtime_captures", 0):
            raise RuntimeError("unexpected measured-phase graph capture; preserve raw record and inspect")
        print(f"真实 DSpark smoke 完成：{a.mode}，B={a.batch}，输出 {run['delivered_tokens']} tokens", flush=True)
    except BaseException as exc:
        write({**{k: v for k, v in ident.items() if k != "type"}, "type": "failure",
               "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        raw.close()
        if llm is not None:
            import atexit
            llm.exit()
            atexit.unregister(llm.exit)


if __name__ == "__main__":
    main()
