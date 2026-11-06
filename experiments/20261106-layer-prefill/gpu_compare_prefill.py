"""Guarded real-engine layer-prefill correctness diagnostic, not a speed test.

Runs fresh full, one-stream same-DAG serial and two-stream pipeline engines on
identical fixed histories. Page padding is zeroed and valid prompt K/V slots
NaN-initialized before each prefill; logits stay in CPU memory only.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_common import (OLD, git_identity, guard_environment, options_for, read_inputs, snapshot_layer,
                          target_provenance, token_hash, validate_layer_execution)
from gpu_identity import verify_visible_gpu

spec = importlib.util.spec_from_file_location("layer_prefill_comparison_previous_bench", OLD / "bench_formal.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

PAIRS = (("full", "serial", 0.02, 0.05), ("full", "pipeline", 0.02, 0.05),
         ("serial", "pipeline", 1e-4, 0.01))
ZERO_REF_MAX_ABS = 1e-6
DECODE_STEPS = 16


def compare_logits(reference, candidate, kl_limit, torch):
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("logit vectors differ in shape")
    if not (torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item()):
        return {"finite": False, "pass": False, "kl": None, "max_abs": None,
                "reference_argmax": None, "candidate_argmax": None,
                "reference_top1_top2_margin": None, "candidate_top1_top2_margin": None}
    a, b = reference.float(), candidate.float()
    loga, logb = torch.log_softmax(a, dim=-1), torch.log_softmax(b, dim=-1)
    kl = (loga.exp() * (loga - logb)).sum().item()
    max_abs = (a - b).abs().max().item()
    topa, topb = torch.topk(a, 2), torch.topk(b, 2)
    idx_a, idx_b = int(topa.indices[0].item()), int(topb.indices[0].item())
    margin_a, margin_b = (topa.values[0] - topa.values[1]).item(), (topb.values[0] - topb.values[1]).item()
    finite = all(math.isfinite(x) for x in (kl, max_abs, margin_a, margin_b))
    return {"finite": finite, "pass": finite and -1e-6 <= kl < kl_limit and idx_a == idx_b,
            "kl": kl if finite else None, "kl_direction": "reference || candidate", "kl_limit": kl_limit,
            "max_abs": max_abs if finite else None, "reference_argmax": idx_a, "candidate_argmax": idx_b,
            "reference_top1_top2_margin": margin_a if finite else None,
            "candidate_top1_top2_margin": margin_b if finite else None}


def compare_kv(reference, candidate, rms_limit, torch):
    if reference.shape != candidate.shape or not reference.numel():
        raise ValueError("K/V samples differ in shape or are empty")
    if not (torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item()):
        return {"finite": False, "pass": False, "reference_rms": None, "relative_rms": None, "max_abs": None}
    a, b = reference.float(), candidate.float()
    diff = a - b
    ref_rms = a.square().mean().sqrt().item()
    err_rms = diff.square().mean().sqrt().item()
    max_abs = diff.abs().max().item()
    finite = all(math.isfinite(x) for x in (ref_rms, err_rms, max_abs))
    rel = err_rms / ref_rms if finite and ref_rms else None
    passed = finite and ((rel is not None and rel < rms_limit) or
                         (ref_rms == 0 and max_abs <= ZERO_REF_MAX_ABS))
    return {"finite": finite, "pass": passed, "reference_rms": ref_rms if finite else None,
            "relative_rms": rel, "relative_rms_limit": rms_limit, "max_abs": max_abs if finite else None,
            "zero_reference_max_abs_limit": ZERO_REF_MAX_ABS if ref_rms == 0 else None}


def sample_positions(length):
    return sorted({p for p in (0, 127, 128, 255, 256, length - 2, length - 1) if 0 <= p < length})


def diagnostic_cases(manifest):
    return {"b2_257_513": read_inputs(manifest, "code", [257, 513])}


def collect_case(llm, inputs, identity, mode, chunk, torch):
    """Instrument only this diagnostic; prefill capture occurs before block release."""
    from nanovllm import SamplingParams
    runner, scheduler = llm.model_runner, llm.scheduler
    prompts = [ids for ids, _ in inputs]
    histories = [ids[:DECODE_STEPS + 1] for ids in prompts]
    if any(len(x) != DECODE_STEPS + 1 for x in histories):
        raise ValueError("prompts too short for fixed diagnostic history")
    if scheduler.waiting or scheduler.running:
        raise RuntimeError("diagnostic cohort requires an idle scheduler")
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=DECODE_STEPS + 1, ignore_eos=True))
    seqs = list(scheduler.waiting)
    if len(seqs) != len(prompts):
        raise RuntimeError("scheduler did not retain all diagnostic requests")
    id_to_input = {seq.seq_id: i for i, seq in enumerate(seqs)}
    state = {"seqs": None, "logits": [[] for _ in prompts], "kv": [None for _ in prompts],
             "prefill_calls": [], "sentinel_blocks": []}
    before, graph_before = snapshot_layer(runner), old.snapshot(runner)
    old_sampler, old_run = runner.sampler, runner.run

    def fixed_sampler(logits, temperatures):
        active = state["seqs"]
        if active is None or logits.ndim != 2 or logits.shape[0] != len(active):
            raise RuntimeError("sampler logits do not match scheduled request rows")
        # This explicit synchronization is diagnostic only and is not used by
        # the performance entry point.
        host = logits.detach().to(device="cpu", dtype=torch.float32)
        chosen = []
        for index, seq in enumerate(active):
            which = id_to_input[seq.seq_id]
            history_index = seq.num_completion_tokens
            if history_index >= len(histories[which]):
                raise RuntimeError("fixed history exhausted")
            state["logits"][which].append(host[index].clone())
            chosen.append(histories[which][history_index])
        return torch.tensor(chosen, dtype=torch.int64, device=logits.device)

    def observed_run(active, is_prefill):
        state["seqs"] = active
        if is_prefill:
            owned = set()
            for seq in active:
                which = id_to_input[seq.seq_id]
                if seq.num_cached_tokens != 0 or seq.num_scheduled_tokens != len(prompts[which]):
                    raise RuntimeError("prefill did not schedule the full uncached prompt")
                needed = (len(prompts[which]) + runner.block_size - 1) // runner.block_size
                blocks = list(seq.block_table[:needed])
                if (len(blocks) != needed or len(set(blocks)) != needed or
                        any(type(block) is not int or not 0 <= block < runner.kv_cache.shape[2] or block in owned
                            for block in blocks)):
                    raise RuntimeError("prefill blocks missing or shared between requests")
                owned.update(blocks)
                for block_index, block in enumerate(blocks):
                    # FlashAttention may legally touch masked page padding:
                    # zero it, but keep all logical prompt slots as NaN until
                    # their own chunk writes. Future chunks must remain hidden.
                    runner.kv_cache[:, :, block].zero_()
                    count = min(runner.block_size, len(prompts[which]) - block_index * runner.block_size)
                    runner.kv_cache[:, :, block, :count].fill_(float("nan"))
                state["sentinel_blocks"].append({"input_index": which, "block_count": len(blocks),
                                                  "physical_blocks": blocks, "block_size": runner.block_size,
                                                  "sample_positions": sample_positions(len(prompts[which])),
                                                  "valid_prefix_slots": len(prompts[which]),
                                                  "preimage": "logical prompt slots NaN; page padding zero"})
            state["prefill_calls"].append({"active_b": len(active),
                                           "scheduled_tokens": [s.num_scheduled_tokens for s in active]})
        output = old_run(active, is_prefill)
        if is_prefill:
            for seq in active:
                which = id_to_input[seq.seq_id]
                positions = sample_positions(len(prompts[which]))
                values = []
                for pos in positions:
                    block = seq.block_table[pos // runner.block_size]
                    values.append(runner.kv_cache[:, :, block, pos % runner.block_size].detach().to("cpu").clone())
                state["kv"][which] = {"positions": positions, "values": values}
        return output

    runner.sampler = fixed_sampler
    runner.run = observed_run
    try:
        while not llm.is_finished():
            if time.time() >= identity["deadline_unix"]:
                raise TimeoutError("GPU guard deadline reached")
            llm.step()
        after, graph_after = snapshot_layer(runner), old.snapshot(runner)
    finally:
        runner.sampler = old_sampler
        runner.run = old_run
    lengths = [len(x) for x in prompts]
    if (len(state["prefill_calls"]) != 1 or state["prefill_calls"][0]["active_b"] != len(prompts)
            or state["prefill_calls"][0]["scheduled_tokens"] != lengths
            or any(len(x) != DECODE_STEPS + 1 for x in state["logits"]) or any(x is None for x in state["kv"])):
        raise RuntimeError("diagnostic did not complete full prefill and 16 decode steps")
    delta = validate_layer_execution(mode, before, after, lengths, runner.config.hf_config.num_hidden_layers, chunk)
    graph_delta = old.delta(graph_before, graph_after)
    if (graph_delta.get("decode_graph_exact_hits", 0) < DECODE_STEPS or
            graph_delta.get("decode_graph_padded_hits", 0) or graph_delta.get("decode_graph_fallbacks", {})):
        raise RuntimeError("ordinary decode exact graph was not observed")
    info = {"input_metadata": [metadata for _, metadata in inputs], "lengths": lengths,
            "fixed_history_sha256": [token_hash(x) for x in histories], "prefill_calls": state["prefill_calls"],
            "sentinel_blocks": state["sentinel_blocks"], "layer_before": before, "layer_after": after,
            "layer_delta": delta, "graph_delta": graph_delta}
    return {"logits": state["logits"], "kv": state["kv"], "info": info}


def run_mode(mode, args, cases, identity, torch):
    from nanovllm import LLM
    options = options_for(mode, 2, 513, DECODE_STEPS + 1, args.chunk, args.budget)
    started = time.perf_counter()
    llm = LLM(args.target, **options)
    try:
        runner = llm.model_runner
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        if any(getattr(runner.config, key) != value for key, value in options.items()):
            raise RuntimeError("engine did not retain requested diagnostic config")
        if (runner.config.quant_weights or runner.config.draft_quant_weights or runner.config.draft_model or
                runner.config.dspark_quant_preset != "none" or runner.config.enable_prefix_cache or
                llm.scheduler.block_manager.enable_prefix_cache or runner.spec_proposer is not None or
                runner.draft_model is not None or runner.dspark_model is not None):
            raise RuntimeError("target-only unquantized prefix-off policy was not retained")
        if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in runner.model.parameters()):
            raise RuntimeError("target parameters must remain BF16")
        if mode == "full" and (hasattr(runner, "layer_prefill_stats") or hasattr(runner, "layer_prefill_executor")):
            raise RuntimeError("disabled path unexpectedly initialized layer-prefill state")
        if list(runner.graph_bs) != [1, 2]:
            raise RuntimeError("decode graph buckets do not cover B1/B2")
        mode_info = {"load_initial_graph_s": load_s, "target_kv_blocks": runner.config.num_kvcache_blocks,
                     "kv_block_size": runner.block_size, "num_layers": runner.config.hf_config.num_hidden_layers,
                     "memory_after_init": old.memory(runner)}
        results = {name: collect_case(llm, inputs, identity, mode, args.chunk, torch)
                   for name, inputs in cases.items()}
        return results, mode_info
    finally:
        llm.exit()
        atexit.unregister(llm.exit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--manifest", required=True, help="existing 1K prompt manifest")
    p.add_argument("--out", required=True, help="new JSONL path outside frozen source")
    p.add_argument("--budget", type=float, choices=(0.70, 0.85), required=True)
    p.add_argument("--chunk", type=int, choices=(128,), default=128)
    args = p.parse_args()
    cases = diagnostic_cases(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    identity, stage = {}, "preflight"
    with out.open("x", encoding="utf-8") as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        try:
            identity.update(guard_environment(os.environ), **git_identity())
            if identity["engine_dirty"]:
                raise RuntimeError("source is dirty; use a committed frozen clone")
            import torch
            properties = verify_visible_gpu(torch, identity["gpu_uuid"])
            identity.update(gpu_name=properties.name, target_provenance=target_provenance(args.target),
                            chunk=args.chunk, budget=args.budget, fixed_decode_steps=DECODE_STEPS,
                            diagnostic_instrumented=True,
                            note="actual logits, fixed diagnostic history; sampler replaced only in this non-performance probe")
            if not identity["target_provenance"]["weight_files"]:
                raise RuntimeError("target checkpoint safetensors not found")
            results = {}
            for mode in ("full", "serial", "pipeline"):
                stage = "mode_" + mode
                if time.time() >= identity["deadline_unix"]:
                    raise TimeoutError("GPU guard deadline reached")
                results[mode], mode_info = run_mode(mode, args, cases, identity, torch)
                for case, data in results[mode].items():
                    write({"type": "mode_complete", **identity, "mode": mode, "case": case,
                           **mode_info, **data["info"]})
                gc.collect()
                torch.cuda.empty_cache()
                print(f"真实模型诊断完成 {mode}", flush=True)
            stage = "numeric_comparison"
            total, failures = 0, 0
            for left, right, kl_limit, rms_limit in PAIRS:
                for case in cases:
                    a, b = results[left][case], results[right][case]
                    if (a["info"]["lengths"] != b["info"]["lengths"] or
                            a["info"]["fixed_history_sha256"] != b["info"]["fixed_history_sha256"] or
                            [x["actual_input_sha256"] for x in a["info"]["input_metadata"]] !=
                            [x["actual_input_sha256"] for x in b["info"]["input_metadata"]]):
                        raise RuntimeError("comparison inputs or fixed histories differ")
                    for request_index in range(len(a["logits"])):
                        for step, (x, y) in enumerate(zip(a["logits"][request_index], b["logits"][request_index])):
                            result = compare_logits(x, y, kl_limit, torch)
                            write({"type": "logit_check", "reference": left, "candidate": right, "case": case,
                                   "request_index": request_index, "step": step,
                                   "step_kind": "prefill" if step == 0 else "decode", **result})
                            total += 1
                            failures += not result["pass"]
                        leftkv, rightkv = a["kv"][request_index], b["kv"][request_index]
                        if leftkv["positions"] != rightkv["positions"]:
                            raise RuntimeError("KV comparison positions differ")
                        for pos, x, y in zip(leftkv["positions"], leftkv["values"], rightkv["values"]):
                            if x.shape != y.shape or x.shape[0] != 2:
                                raise RuntimeError("sampled KV cache shapes differ")
                            for kind in range(2):
                                for layer in range(x.shape[1]):
                                    result = compare_kv(x[kind, layer], y[kind, layer], rms_limit, torch)
                                    write({"type": "kv_check", "reference": left, "candidate": right,
                                           "case": case, "request_index": request_index, "position": pos,
                                           "kv": "K" if kind == 0 else "V", "layer": layer, **result})
                                    total += 1
                                    failures += not result["pass"]
            write({"type": "complete", **identity, "comparisons": total, "failed_comparisons": failures,
                   "pass": failures == 0})
            if failures:
                raise RuntimeError(f"numeric diagnostic failed {failures}/{total} checks")
            print(f"数值诊断完成：{total} 项均通过", flush=True)
        except BaseException as exc:
            write({"type": "failure", **identity, "failure_phase": stage, "error": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
