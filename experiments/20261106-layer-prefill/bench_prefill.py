"""Guarded B4/4K/output128 layer-prefill performance measurement.

One process runs one mode and one panel: one untimed warmup cohort, then one
timed cohort. Output throughput uses actual delivered tokens / request wall.
This is not the separate numerical comparison; one sample is not a CI.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_common import (MODES, OLD, ROOT, git_identity, guard_environment, options_for, read_inputs,
                          snapshot_layer, target_provenance, validate_layer_execution)
from gpu_identity import verify_visible_gpu

spec = importlib.util.spec_from_file_location("layer_prefill_previous_bench", OLD / "bench_formal.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def validate_result(row, lengths):
    """Recompute old per-request metrics, permitting the actual no-decode cap=1."""
    result = old.compute_run(row)
    prefill = [x for x in row["rounds"] if x["prefill"]]
    decode = [x for x in row["rounds"] if not x["prefill"]]
    if row["output_tokens"] == 1 and not decode:
        result["errors"] = [x for x in result["errors"] if x != "first decode did not reach requested B"]
    if (len(prefill) != 1 or prefill[0]["active_b"] != len(lengths) or
            row.get("observed_prefill_lengths") != list(lengths)):
        result["errors"].append("single full prefill did not admit the requested B and token lengths")
    if row["output_tokens"] == 1 and decode:
        result["errors"].append("output cap=1 unexpectedly performed decode")
    if row.get("prefix_lookup_count") != 0 or row.get("prefix_hit_blocks") != 0:
        result["errors"].append("prefix lookup or hit occurred")
    graph = row.get("target_decode_graph")
    if not isinstance(graph, dict):
        result["errors"].append("target decode graph telemetry missing")
    elif (graph.get("padded") != 0 or not isinstance(graph.get("fallbacks"), dict)
          or any(graph["fallbacks"].values()) or (row["output_tokens"] > 1 and graph.get("exact", 0) <= 0)):
        result["errors"].append("decode graph missing, padded, or fell back")
    result["valid"] = not result["errors"]
    result["input_tokens"] = sum(lengths)
    result["prefill_only_input_tps"] = (sum(lengths) / row["wall_s"] if row["output_tokens"] == 1 else None)
    result["full_request_output_tps"] = result["full_request_tps"] if row["output_tokens"] > 1 else None
    result["decode_span_s"] = (decode[-1]["end_s"] - decode[0]["start_s"] if decode else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="new JSONL path outside frozen source")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--batch", type=int, choices=(4,), default=4)
    p.add_argument("--ctx", type=int, choices=(4096,), default=4096)
    p.add_argument("--output-tokens", type=int, choices=(128,), default=128)
    p.add_argument("--chunk", type=int, choices=(128,), default=128)
    p.add_argument("--budget", type=float, choices=(0.70, 0.85), required=True)
    p.add_argument("--panel", choices=("code", "math", "general"), default="code")
    p.add_argument("--repeats", type=int, choices=(1,), default=1)
    p.add_argument("--seed", type=int, default=20261106)
    args = p.parse_args()
    inputs = read_inputs(args.manifest, args.panel, [args.ctx] * args.batch)
    options = options_for(args.mode, args.batch, args.ctx, args.output_tokens, args.chunk, args.budget)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    identity, stage, llm = {}, "preflight", None
    with out.open("x", encoding="utf-8") as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        try:
            identity.update(guard_environment(os.environ), **git_identity())
            if identity["engine_dirty"]:
                raise RuntimeError("source is dirty; use a committed frozen clone")
            import torch
            from nanovllm import LLM
            properties = verify_visible_gpu(torch, identity["gpu_uuid"])
            identity.update(mode=args.mode, batch=args.batch, ctx=args.ctx, output_tokens=args.output_tokens,
                            chunk=args.chunk, max_inflight=2, budget=args.budget, seed=args.seed, temperature=1.0,
                            serial_definition="same diagonal DAG, one worker stream; not chunk-major or scheduler chunking",
                            primary_timing_repeats=1, warmup_cohorts=1,
                            timing_note="single timed cohort; no confidence interval or repeat range",
                            panel=args.panel, input_metadata=[metadata for _, metadata in inputs],
                            target_provenance=target_provenance(args.target), gpu_name=properties.name,
                            torch_version=torch.__version__, graph_policy="exact_all", diagnostic_instrumented=False)
            if not identity["target_provenance"]["weight_files"]:
                raise RuntimeError("target checkpoint safetensors not found")
            stage = "model_load_and_graph_capture"
            start = time.perf_counter()
            llm = LLM(args.target, **options)
            runner = llm.model_runner
            torch.cuda.synchronize()
            load_s = time.perf_counter() - start
            if any(getattr(runner.config, name) != expected for name, expected in options.items()):
                raise RuntimeError("runner did not retain requested config")
            if (runner.config.quant_weights or runner.config.draft_quant_weights or runner.config.draft_model or
                    runner.config.dspark_quant_preset != "none" or runner.config.enable_prefix_cache or
                    llm.scheduler.block_manager.enable_prefix_cache or runner.spec_proposer is not None or
                    runner.draft_model is not None or runner.dspark_model is not None):
                raise RuntimeError("target-only unquantized prefix-off policy was not retained")
            if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in runner.model.parameters()):
                raise RuntimeError("target parameters must remain BF16")
            layers = runner.config.hf_config.num_hidden_layers
            if args.mode == "full" and (hasattr(runner, "layer_prefill_stats") or hasattr(runner, "layer_prefill_executor")):
                raise RuntimeError("disabled path unexpectedly initialized layer-prefill state")
            identity.update(load_initial_graph_s=load_s, target_kv_blocks=runner.config.num_kvcache_blocks,
                            kv_block_size=runner.block_size, num_layers=layers,
                            decode_graph_buckets=list(runner.graph_bs))
            if identity["decode_graph_buckets"] != list(range(1, args.batch + 1)):
                raise RuntimeError("decode graph buckets differ from requested exact policy")
            prefill_calls = []
            original_run = runner.run
            def observed_run(seqs, is_prefill):
                if is_prefill:
                    prefill_calls.append({"seq_ids": [s.seq_id for s in seqs],
                                          "scheduled_tokens": [s.num_scheduled_tokens for s in seqs],
                                          "cached_tokens": [s.num_cached_tokens for s in seqs]})
                return original_run(seqs, is_prefill)
            runner.run = observed_run
            stage = "warmup"
            warm_before = snapshot_layer(runner)
            warm_state = {"rounds": []}
            warm_started = time.perf_counter()
            warm = old.run_once(llm, [x for x, _ in inputs], min(args.output_tokens, 16), args.seed,
                                identity["deadline_unix"], warm_state, False)
            warm_s = time.perf_counter() - warm_started
            warm_after = snapshot_layer(runner)
            if len(prefill_calls) != 1 or prefill_calls[0]["scheduled_tokens"] != [args.ctx] * args.batch:
                raise RuntimeError("warmup did not admit the complete fixed batch")
            validate_layer_execution(args.mode, warm_before, warm_after, [args.ctx] * args.batch, layers, args.chunk)
            write({"type": "warmup", **identity, "warmup_s": warm_s, "warmup_request_wall_s": warm["wall_s"],
                   "layer_before": warm_before, "layer_after": warm_after, "prefill_calls": prefill_calls,
                   "memory_after_warmup": old.memory(runner)})
            print(f"预热完成：{args.mode} B={args.batch} ctx={args.ctx} output={args.output_tokens}", flush=True)
            stage = "measured_requests"
            for repeat in range(args.repeats):
                before = snapshot_layer(runner)
                prefill_calls.clear()
                state = {"rounds": []}
                result = old.run_once(llm, [x for x, _ in inputs], args.output_tokens, args.seed,
                                      identity["deadline_unix"], state, False)
                after = snapshot_layer(runner)
                if len(prefill_calls) != 1 or prefill_calls[0]["cached_tokens"] != [0] * args.batch:
                    raise RuntimeError("prefill cache policy or admitted steps differ")
                delta = validate_layer_execution(args.mode, before, after, [args.ctx] * args.batch, layers, args.chunk)
                record = {"type": "run", **identity, "repeat": repeat,
                          "requested_batch": args.batch, "ignore_eos": True, "arm": "nospec",
                          "observed_prefill_lengths": prefill_calls[0]["scheduled_tokens"],
                          "prefill_calls": list(prefill_calls), "layer_before": before, "layer_after": after,
                          "layer_delta": delta, "memory_after_run": old.memory(runner), **result}
                metrics = validate_result(record, [args.ctx] * args.batch)
                record["metrics"] = metrics
                write(record)
                if not metrics["valid"]:
                    raise RuntimeError("raw measurement contract failed: " + "; ".join(metrics["errors"]))
                print(f"完成 {args.mode} B={args.batch} ctx={args.ctx} {args.panel} repeat={repeat}", flush=True)
            write({"type": "complete", **identity, "panel_count": 1, "repeats_per_panel": args.repeats})
        except BaseException as exc:
            write({"type": "failure", **identity, "failure_phase": stage, "error": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            if llm is not None:
                import atexit
                llm.exit()
                atexit.unregister(llm.exit)


if __name__ == "__main__":
    main()
