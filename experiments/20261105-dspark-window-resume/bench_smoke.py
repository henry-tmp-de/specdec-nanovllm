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


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


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
    p.add_argument("--max-seqs", type=int, help="默认 batch；补桶 B=3/11/15 需设为 4/16/16")
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
    if (a.mode == "w0") != (a.window == 0):
        p.error("w0 必须 window=0，full_cache/ring 必须 window>0")
    if a.window and a.window % 256:
        p.error("window 必须为 256 的整数倍")
    buckets = (1, 2, 4, 8, 16)
    if a.graph == "padded" and a.batch in buckets:
        p.error("padded smoke 必须使用未精确捕获的 batch，例如 3、11、15")
    if a.graph == "exact":
        buckets = tuple(sorted(set(buckets + (a.batch,))))
    if a.graph != "eager":
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
        if torch.cuda.device_count() != 1:
            raise RuntimeError("guard must expose exactly one CUDA device")
        actual_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if actual_uuid is None or str(actual_uuid).lower() != os.environ["DS_GPU_UUID"].lower():
            raise RuntimeError(f"CUDA device UUID mismatch or unavailable: {actual_uuid!r}")
        from nanovllm import LLM, SamplingParams
        from nanovllm.config import Config
        from nanovllm.engine.token_hook import TokenDeliveryHook
        required = {"dspark_window_tokens", "dspark_window_storage", "enable_prefix_cache"}
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
            before_counts = {s.seq_id: s.num_completion_tokens for s in seqs}
            anchors_before = {s.seq_id: len(s) - 1 for s in seqs}
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
               "gpu_event_ms": {k: sum(start.elapsed_time(end) for _, start, end in rows)
                                for k, rows in events.items()},
               "commit_timing_note": ("event span for commit helper" if commit_instrumented
                                      else "not instrumented for W=0 or missing helper"),
               "draft_gpu_event_note": "includes one extra eager q diagnostic forward; not performance data",
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
