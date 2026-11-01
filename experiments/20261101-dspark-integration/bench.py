"""One arm and batch size, with raw per-run/round measurements in JSONL."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def load_wiki(tokenizer, wiki_path, ctx, batch):
    text = Path(wiki_path).read_text(encoding="utf-8")
    # Chunking bounds tokenizer work on long natural text. IDs are fixed in a
    # shared manifest, so chunk boundaries are identical for every arm.
    wiki = []
    for start in range(0, len(text), 4096):
        wiki.extend(tokenizer.encode(text[start:start + 4096], add_special_tokens=False, verbose=False))
        if len(wiki) >= (batch + 1) * ctx:
            break
    return wiki


def build_inputs(tokenizer, dataset_dir, wiki_path, ctx, batch, panel, wiki=None):
    files = {"code": "humaneval.jsonl", "math": "gsm8k.jsonl", "general": "alpaca.jsonl"}
    path = Path(dataset_dir) / files[panel]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if wiki is None:
        wiki = load_wiki(tokenizer, wiki_path, ctx, batch)
    if len(rows) < batch or len(wiki) < (batch + 1) * ctx:
        raise ValueError("insufficient distinct public tasks or WikiText-2 filler")
    prompts, metadata = [], []
    for i in range(batch):
        turns = rows[i].get("turns")
        if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
            raise ValueError(f"unsupported DeepSpec dataset schema at {path}:{i+1}")
        task = tokenizer.apply_chat_template([{"role": "user", "content": turns[0]}],
                                             tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False, return_dict=False)
        if isinstance(task, Mapping):
            task = task["input_ids"]
        task = list(task)
        if task and isinstance(task[0], list):
            if len(task) != 1:
                raise ValueError("chat template returned more than one request")
            task = task[0]
        if not task or any(not isinstance(token, Integral) for token in task):
            raise TypeError(f"chat template did not return integer token IDs for {panel} row {i}")
        truncated = len(task) > ctx
        if truncated:
            ids = task[-ctx:]
            fill = 0
        else:
            fill = ctx - len(task)
            ids = wiki[i * ctx:i * ctx + fill] + task
        if len(ids) != ctx:
            raise RuntimeError("prompt length differs from fixed context")
        if any(not isinstance(token, Integral) or token < 0 for token in ids):
            raise ValueError("prompt contains an invalid token ID")
        prompts.append(ids)
        metadata.append({"dataset": str(path.name), "row": i, "input_sha256": token_hash(ids),
                         "task_tokens": len(task), "wiki_tokens": fill,
                         "task_truncated_from_left": truncated})
    return prompts, metadata


def clear_prefix(llm):
    bm = llm.scheduler.block_manager
    if bm.used_block_ids or llm.scheduler.running or llm.scheduler.waiting:
        raise RuntimeError("cannot clear prefix table while requests are active")
    bm.hash_to_block_id.clear()


def snapshot(runner):
    p = runner.spec_proposer
    result = {"verify_hits": runner.verify_graph_hits,
              "verify_fallbacks": dict(runner.verify_graph_fallbacks)}
    for name in ("graph_hits", "graph_captures", "graph_fallbacks", "active_batch_histogram",
                 "n_graphed", "n_eager", "n_padded", "n_rounds", "n_batch_forwards",
                 "n_catchup_forwards", "n_catchup_tokens"):
        if p is not None and hasattr(p, name):
            value = getattr(p, name)
            result[name] = dict(value) if isinstance(value, dict) else value
    return result


def delta(before, after):
    result = {}
    for key, value in after.items():
        old = before.get(key, {} if isinstance(value, dict) else 0)
        if isinstance(value, dict):
            result[key] = {str(k): int(v - old.get(k, 0)) for k, v in value.items()
                           if v - old.get(k, 0)}
        else:
            result[key] = int(value - old)
    return result


def install_events(runner):
    import torch
    events = {"draft": [], "verify": []}
    def wrap(obj, name, bucket):
        original = getattr(obj, name)
        def timed(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            events[bucket].append((start, end))
            return result
        setattr(obj, name, timed)
    if runner.spec_proposer is not None:
        wrap(runner.spec_proposer, "propose_batch", "draft")
        wrap(runner, "run_verify_forward", "verify")
    return events


def one_run(llm, prompts, output_len, seed, deadline, events):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook
    clear_prefix(llm)
    runner, sched = llm.model_runner, llm.scheduler
    before = snapshot(runner)
    for rows in events.values():
        rows.clear()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    hook = TokenDeliveryHook()
    llm.token_hook = hook
    sched.token_hook = hook
    torch.cuda.synchronize()
    started = time.perf_counter()
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True))
    records, done = [], {}
    prefix_cached, preemptions = 0, 0
    saw_first_decode = False
    original_preempt = sched.preempt
    def preempt(seq):
        nonlocal preemptions
        preemptions += 1
        return original_preempt(seq)
    sched.preempt = preempt
    try:
        while not llm.is_finished():
            if deadline and time.time() >= deadline:
                raise TimeoutError("shared GPU budget deadline reached")
            begin = time.perf_counter()
            seqs, is_prefill = sched.schedule()
            if not seqs:
                raise RuntimeError("scheduler could not admit a request")
            if preemptions:
                raise RuntimeError(f"requested batch capacity insufficient: {preemptions} preemptions")
            if not is_prefill and not saw_first_decode:
                saw_first_decode = True
                if len(seqs) != len(prompts):
                    raise RuntimeError(f"requested batch capacity insufficient: first decode active "
                                       f"B={len(seqs)} < requested B={len(prompts)}")
            if is_prefill:
                prefix_cached += sum(s.num_cached_tokens for s in seqs)
            output = runner.run(seqs, is_prefill)
            proposal = sum(len(s.draft_tokens) for s in seqs)
            accepted = (sum(max(0, len(o) - 1) for o in output)
                        if output and isinstance(output[0], list) else 0)
            post_start = time.perf_counter()
            delivered = sched.postprocess(seqs, output, is_prefill)
            post_end = time.perf_counter()
            records.append({"active_b": len(seqs), "prefill": bool(is_prefill),
                            "wall_s": post_end - begin, "postprocess_s": post_end - post_start,
                            "delivered": delivered, "proposed": proposal, "accepted": accepted})
            for seq in seqs:
                if seq.is_finished:
                    done[seq.seq_id] = list(seq.completion_token_ids)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        summaries = hook.summary()
        ttfts = [v["ttft"] for v in summaries.values()]
        tpots = [v["tpot"] for v in summaries.values()]
        decode = [r for r in records if not r["prefill"]]
        full_batch = [r for r in decode if r["active_b"] == len(prompts)]
        delivered = sum(len(ids) for ids in done.values())
        if len(done) != len(prompts) or any(len(ids) != output_len for ids in done.values()):
            raise RuntimeError("completion count or output length mismatch")
        def rate(rows):
            seconds = sum(r["wall_s"] for r in rows)
            return sum(r["delivered"] for r in rows) / seconds if seconds else None
        proposed = sum(r["proposed"] for r in decode)
        accepted = sum(r["accepted"] for r in decode)
        return {"wall_s": wall, "full_request_tps": delivered / wall,
                "prefill_s": sum(r["wall_s"] for r in records if r["prefill"]),
                "decode_s": sum(r["wall_s"] for r in decode),
                "decode_tps": rate(decode), "full_batch_decode_tps": rate(full_batch),
                "full_batch_decode_rounds": len(full_batch),
                "ttft_s": ttfts, "ttft_median_s": statistics.median(ttfts),
                "tpot_s": tpots, "tpot_median_s": statistics.median(tpots),
                "delivered_tokens": delivered, "proposed_tokens": proposed,
                "accepted_prefix_tokens": accepted,
                "acceptance_prefix_fraction": accepted / proposed if proposed else None,
                "mean_delivered_per_decode_round_per_sequence":
                    sum(r["delivered"] for r in decode) / max(1, sum(r["active_b"] for r in decode)),
                "active_b_rounds": dict(Counter(r["active_b"] for r in decode)),
                "prefix_cached_tokens": prefix_cached, "preemptions": preemptions,
                "postprocess_s": sum(r["postprocess_s"] for r in records),
                "gpu_stream_span_ms": {key: sum(a.elapsed_time(b) for a, b in spans)
                                       for key, spans in events.items()},
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "graph_delta": delta(before, snapshot(runner)),
                "output_sha256": [token_hash(ids) for _, ids in sorted(done.items())],
                "rounds": records}
    finally:
        sched.preempt = original_preempt
        sched.token_hook = None
        llm.token_hook = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("nospec", "ar", "dspark"), required=True)
    ap.add_argument("--batch", type=int, choices=(1, 2, 4, 8, 16), required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ar-model", required=True)
    ap.add_argument("--dspark-model", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--manifest", help="precomputed CPU workload with token IDs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--deadline-unix", type=float, default=float(os.environ.get("DS_RUN_DEADLINE_UNIX", "0")))
    args = ap.parse_args()
    import torch
    from nanovllm import LLM
    graph_bs = (1, 2, 4, 8, 16)
    kwargs = dict(max_num_seqs=args.batch, max_model_len=1536,
                  max_num_batched_tokens=max(args.batch * args.ctx, 1536),
                  gpu_memory_utilization=0.85, enforce_eager=False,
                  spec_graph_bs=graph_bs, spec_graph_pad=False)
    if args.arm == "ar":
        kwargs.update(spec_k=6, spec_method="draft", draft_model=args.ar_model)
    elif args.arm == "dspark":
        kwargs.update(spec_k=7, spec_method="dspark", draft_model=args.dspark_model)
    llm = None
    identity = {"arm": args.arm, "requested_batch": args.batch, "ctx": args.ctx,
                "output_tokens": args.output_tokens, "seed": args.seed,
                "k": {"nospec": 0, "ar": 6, "dspark": 7}[args.arm],
                "graph_buckets": graph_bs, "gpu": torch.cuda.get_device_name(),
                "device_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        t0 = time.perf_counter()
        llm = LLM(args.target, **kwargs)
        torch.cuda.synchronize()
        identity["load_and_initial_graph_s"] = time.perf_counter() - t0
        identity["kv_blocks"] = int(llm.model_runner.config.num_kvcache_blocks)
        identity["kv_block_size"] = llm.model_runner.config.kvcache_block_size
        identity["graph_memory_gb"] = llm.model_runner.graph_mem
        events = install_events(llm.model_runner)
        if args.manifest:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            if manifest["context_tokens"] != args.ctx:
                raise ValueError("workload manifest context does not match --ctx")
            prompt_sets = {}
            for panel in ("code", "math", "general"):
                rows = manifest["panels"][panel][:args.batch]
                if len(rows) != args.batch:
                    raise ValueError("workload manifest has too few rows")
                prompts = [r["token_ids"] for r in rows]
                meta = [r["metadata"] for r in rows]
                if any(len(p) != args.ctx or token_hash(p) != m["input_sha256"]
                       for p, m in zip(prompts, meta)):
                    raise ValueError("workload token length or fingerprint mismatch")
                prompt_sets[panel] = (prompts, meta)
        else:
            wiki_ids = load_wiki(llm.tokenizer, args.wiki, args.ctx, args.batch)
            prompt_sets = {panel: build_inputs(llm.tokenizer, args.datasets, args.wiki,
                                               args.ctx, args.batch, panel, wiki_ids)
                           for panel in ("code", "math", "general")}
        # Warm both context buckets (1024 and 1280) without retaining prefix hits.
        warm_prompts, _ = prompt_sets["code"]
        tw = time.perf_counter()
        warm = one_run(llm, warm_prompts, min(args.output_tokens, 16), args.seed,
                       args.deadline_unix, events)
        write(args.out, {"type": "warmup", **identity, "warmup_s": time.perf_counter() - tw,
                         "graph_delta": warm["graph_delta"]})
        for panel in ("code", "math", "general"):
            prompts, meta = prompt_sets[panel]
            for repeat in range(args.repeats):
                result = one_run(llm, prompts, args.output_tokens, args.seed,
                                 args.deadline_unix, events)
                write(args.out, {"type": "run", **identity, "panel": panel,
                                 "repeat": repeat, "inputs": meta, **result})
                print(json.dumps({"arm": args.arm, "B": args.batch, "panel": panel,
                                  "repeat": repeat, "full_request_tps": result["full_request_tps"],
                                  "decode_tps": result["decode_tps"]}), flush=True)
    except BaseException as exc:
        write(args.out, {"type": "failure", **identity, "error": type(exc).__name__,
                         "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if llm is not None:
            llm.exit()
            # LLMEngine also registers exit with atexit; avoid a second call.
            import atexit
            atexit.unregister(llm.exit)


if __name__ == "__main__":
    main()
