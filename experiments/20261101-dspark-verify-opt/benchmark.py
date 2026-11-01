"""Padding ablations with shared workload, bounded graph warmup and raw JSONL."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE_PATH = ROOT / "experiments/20261101-dspark-integration/bench.py"
spec = importlib.util.spec_from_file_location("prior_bench", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def snapshot(runner):
    p = runner.spec_proposer
    result = {"verify_hits": runner.verify_graph_hits,
              "verify_padded_hits": runner.verify_graph_padded,
              "verify_exact_hits": runner.verify_graph_hits - runner.verify_graph_padded,
              "verify_fallbacks": dict(runner.verify_graph_fallbacks)}
    names = ("graph_hits", "graph_exact_hits", "graph_padded_hits", "graph_captures",
             "graph_runtime_captures", "graph_runtime_capture_s", "graph_runtime_shapes",
             "graph_fallbacks", "active_batch_histogram", "head_graph_hits", "head_graph_captures",
             "head_runtime_captures", "head_runtime_capture_s", "head_runtime_batches",
             "n_graphed", "n_eager", "n_padded", "n_rounds", "n_batch_forwards",
             "n_catchup_forwards", "n_catchup_tokens")
    for name in names:
        if p is not None and hasattr(p, name):
            value = getattr(p, name)
            result[name] = value.copy() if isinstance(value, (dict, list)) else value
    return result


def delta(before, after):
    result = {}
    for key, value in after.items():
        old = before.get(key, {} if isinstance(value, dict) else [] if isinstance(value, list) else 0)
        if isinstance(value, dict):
            result[key] = {str(k): v - old.get(k, 0) for k, v in value.items() if v - old.get(k, 0)}
        elif isinstance(value, list):
            result[key] = value[len(old):]
        else:
            result[key] = value - old
    return result


base.snapshot = snapshot
base.delta = delta


def memory():
    import torch
    result = {"allocated_bytes": torch.cuda.memory_allocated(),
              "reserved_bytes": torch.cuda.memory_reserved()}
    stats = torch.cuda.memory_stats()
    result["private_pool_stats"] = {str(k): v for k, v in stats.items()
                                    if "private" in k.lower() or "graph" in k.lower()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("nospec-pad", "ar-pad", "dspark-original", "dspark-ablations"),
                    required=True)
    ap.add_argument("--batch", type=int, choices=(8, 16), required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ar-model", required=True)
    ap.add_argument("--dspark-model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--deadline-unix", type=float, default=float(os.environ.get("DS_RUN_DEADLINE_UNIX", "0")))
    args = ap.parse_args()
    import torch
    from nanovllm import LLM
    buckets = (1, 2, 4, 8, 16)
    options = dict(max_num_seqs=args.batch, max_model_len=1536,
                   max_num_batched_tokens=max(args.batch * args.ctx, 1536),
                   gpu_memory_utilization=0.70, enforce_eager=False,
                   spec_graph_bs=buckets, spec_graph_pad=args.arm == "ar-pad")
    if args.arm == "ar-pad":
        options.update(spec_k=6, spec_method="draft", draft_model=args.ar_model)
    elif args.arm.startswith("dspark"):
        options.update(spec_k=7, spec_method="dspark", draft_model=args.dspark_model,
                       dspark_split_graph=args.arm == "dspark-ablations", dspark_graph_pad=False)
    identity = {"arm": args.arm, "requested_batch": args.batch, "ctx": args.ctx,
                "output_tokens": args.output_tokens, "seed": args.seed,
                "gpu_memory_utilization": 0.70, "device_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "graph_buckets": buckets, "gpu": torch.cuda.get_device_name(),
                "k": 6 if args.arm == "ar-pad" else 7 if args.arm.startswith("dspark") else 0}
    llm = None
    try:
        started = time.perf_counter()
        llm = LLM(args.target, **options)
        torch.cuda.synchronize()
        mr, prop = llm.model_runner, llm.model_runner.spec_proposer
        identity.update(load_and_initial_graph_s=time.perf_counter() - started,
                        kv_blocks=int(mr.config.num_kvcache_blocks),
                        kv_block_size=mr.config.kvcache_block_size,
                        initial_graph_mem=mr.graph_mem, memory_after_init=memory())
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if manifest["context_tokens"] != args.ctx:
            raise ValueError("manifest context mismatch")
        panels = {}
        for panel in ("code", "math", "general"):
            rows = manifest["panels"][panel][:args.batch]
            if len(rows) != args.batch:
                raise ValueError("not enough rows in manifest")
            prompts, meta = [x["token_ids"] for x in rows], [x["metadata"] for x in rows]
            if any(len(p) != args.ctx or base.token_hash(p) != m["input_sha256"]
                   for p, m in zip(prompts, meta)):
                raise ValueError("manifest prompt verification failed")
            panels[panel] = prompts, meta
        events = base.install_events(mr)
        memory_before_prewarm = memory()
        prewarm_shapes = []
        if args.arm == "dspark-ablations":
            # At 1024 input + 256 output, only these two context buckets can be
            # reached.  Only the original maximum B can occur at the first 1024
            # step; the remaining bucket shapes serve the tail at ctx 1280.
            prewarm_shapes = [(args.batch, 1280), (args.batch, 1024)]
            prewarm_shapes += [(b, 1280) for b in reversed(buckets) if b < args.batch]
            prewarm_start = time.perf_counter()
            prop.prewarm(prewarm_shapes, range(args.batch, 0, -1))
            torch.cuda.synchronize()
            prewarm_s = time.perf_counter() - prewarm_start
        else:
            prewarm_s = 0.0
        memory_after_prewarm = memory()
        # Warm target prefill/decode and the original or AR draft. Seed is reset
        # at each one_run after every graph warmup; runtime captures are counted.
        warm_start = time.perf_counter()
        warm = base.one_run(llm, panels["code"][0], min(args.output_tokens, 16), args.seed,
                            args.deadline_unix, events)
        base.write(args.out, {"type": "warmup", **identity, "prewarm_s": prewarm_s,
                              "prewarm_shapes": prewarm_shapes,
                              "prewarm_head_batches": list(range(args.batch, 0, -1))
                                  if args.arm == "dspark-ablations" else [],
                              "memory_before_prewarm": memory_before_prewarm,
                              "memory_after_prewarm": memory_after_prewarm,
                              "warmup_s": time.perf_counter() - warm_start,
                              "graph_delta": warm["graph_delta"]})
        for panel in ("code", "math", "general"):
            prompts, meta = panels[panel]
            for repeat in range(args.repeats):
                modes = ["split-none", "split-target", "split-both"] if args.arm == "dspark-ablations" else [args.arm]
                if len(modes) == 3:
                    shift = repeat % 3
                    modes = modes[shift:] + modes[:shift]
                for mode in modes:
                    if args.arm == "dspark-ablations":
                        mr.graph_pad = mode in ("split-target", "split-both")
                        mr.config.dspark_graph_pad = mode == "split-both"
                    result = base.one_run(llm, prompts, args.output_tokens, args.seed,
                                          args.deadline_unix, events)
                    base.write(args.out, {"type": "run", **identity, "mode": mode,
                                          "panel": panel, "repeat": repeat, "inputs": meta,
                                          "memory_after_run": memory(), **result})
                    print(json.dumps({"arm": args.arm, "mode": mode, "B": args.batch,
                                      "panel": panel, "repeat": repeat,
                                      "full_request_tps": result["full_request_tps"],
                                      "graph_delta": result["graph_delta"]}), flush=True)
    except BaseException as exc:
        base.write(args.out, {"type": "failure", **identity, "error": type(exc).__name__,
                              "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if llm is not None:
            llm.exit()
            import atexit
            atexit.unregister(llm.exit)


if __name__ == "__main__":
    main()
