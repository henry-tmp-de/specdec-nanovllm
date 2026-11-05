"""B3/11/15 padded verify, ragged fallback and ring-owner GPU contract.

One real model process per B, intentionally forced acceptance.  It probes
identical inputs through eager, exact and padded target verify, restores no
synthetic data: padded is run last, so ring scratch committed by the real
engine is the padded replay's data.  This diagnostic is not a throughput run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bench_smoke import prewarm_shapes, read_manifest, snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", required=True, type=int, choices=(3, 11, 15))
    ap.add_argument("--ctx", type=int, default=252)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--output-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    if not 0 < a.ctx <= a.window or a.window % 256 or a.output_tokens < 16:
        ap.error("要求 0 < ctx <= 对齐的 window，输出至少 16")
    data = read_manifest(a.manifest, "code", a.batch, a.ctx)
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    handle = dst.open("x", encoding="utf-8")
    def write(record):
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
    identity = {"guard_id": os.environ.get("DS_GUARD_ID"), "gpu_uuid": os.environ.get("DS_GPU_UUID"),
                "batch": a.batch, "ctx": a.ctx, "window": a.window, "output_tokens": a.output_tokens,
                "seed": a.seed, "forced_acceptance": True, "diagnostic_instrumented": True,
                "q_and_target_kl_threshold": 0.02, "kv_relative_rms_threshold": 0.05}
    llm = None
    restore_verify = None
    try:
        if not all(os.environ.get(x) for x in ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX")):
            raise RuntimeError("必须通过 gpu_control.py 在授权 GPU 启动")
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("guard must expose exactly one CUDA device")
        actual_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if actual_uuid is None or str(actual_uuid).lower() != os.environ["DS_GPU_UUID"].lower():
            raise RuntimeError(f"CUDA device UUID mismatch or unavailable: {actual_uuid!r}")
        from nanovllm import LLM, SamplingParams
        import nanovllm.spec_decode.verify as verifier
        restore_verify = verifier.verify_batch
        real_b, graph_b = a.batch, (4 if a.batch == 3 else 16)
        production_buckets = (1, 2, 4, 8, 16)
        exact_buckets = tuple(sorted(set(production_buckets + (real_b,))))
        max_len = a.ctx + a.output_tokens + 7
        llm = LLM(a.target, max_num_seqs=graph_b, max_model_len=max_len,
                  max_num_batched_tokens=max(a.ctx * real_b, max_len), gpu_memory_utilization=0.70,
                  tensor_parallel_size=1, enforce_eager=False, spec_cuda_graph=True,
                  spec_method="dspark", spec_k=7, draft_model=a.draft,
                  spec_graph_bs=exact_buckets, spec_graph_pad=True, dspark_split_graph=True,
                  dspark_graph_pad=True, dspark_window_tokens=a.window,
                  dspark_window_storage="ring", enable_prefix_cache=False, spec_overlap_mixed=False)
        runner, scheduler = llm.model_runner, llm.scheduler
        prop, model = runner.spec_proposer, runner.dspark_model
        if runner.config.max_model_len < max_len:
            raise RuntimeError("model max position capacity is insufficient")
        standard_verify = {k: v for k, v in runner.verify_graphs.items() if k[0] in production_buckets}
        exact_verify = runner.verify_graphs[(real_b, 7)]
        runner.verify_graphs = standard_verify
        target_guard_block = runner.config.num_kvcache_blocks - 1
        ring_guard_block = runner.config.num_draft_blocks - 1
        if target_guard_block < 0 or ring_guard_block < 0:
            raise RuntimeError("no guard block available")
        ring_guard_slots = slice(ring_guard_block * runner.block_size, (ring_guard_block + 1) * runner.block_size)
        runner.kv_cache[:, :, target_guard_block].fill_(0.375)
        model.cache[:, :, ring_guard_slots].fill_(0.25)
        target_guard = runner.kv_cache[:, :, target_guard_block].clone()
        ring_guard = model.cache[:, :, ring_guard_slots].clone()
        shapes = prewarm_shapes(real_b, graph_b, a.ctx, a.output_tokens, a.window,
                                runner.block_size, production_buckets)
        shapes.append((real_b, max(runner.block_size, ((a.ctx + runner.block_size - 1) // runner.block_size) * runner.block_size)))
        prop.prewarm(sorted(set(shapes)), range(1, real_b + 1))
        torch.cuda.synchronize()
        before = snapshot(runner)
        runner.config.spec_graph_bs = production_buckets
        row_index = {}
        forced_per_row = [0] * real_b
        observations = {"draft_probe": None, "target_probe": None, "ragged": [],
                        "owner_survival": [], "forced": [], "active_batches": []}
        current = {"snapshot_owners": False, "seqs": [], "prior": {}, "ragged": False}
        original_propose, original_forward = prop.propose_batch, runner.run_verify_forward

        def max_kl(x, y):
            p, q = torch.log_softmax(x.float(), -1), torch.log_softmax(y.float(), -1)
            return float((p.exp() * (p - q)).sum(-1).max())
        def rms(x, y):
            x, y = x.float(), y.float()
            return float((x - y).square().mean().sqrt() / x.square().mean().sqrt().clamp_min(1e-12))
        def fixed_chains(seqs):
            return [[1000 + row_index[s.seq_id] * 8 + j for j in range(7)] for s in seqs]
        def fixed_q(base, anchors, chains):
            previous = anchors
            result = []
            for j in range(7):
                result.append(base[:, j] + model.markov_head(previous))
                previous = torch.tensor([chain[j] for chain in chains], device="cuda", dtype=torch.long)
            return torch.stack(result, dim=1)

        @torch.inference_mode()
        def proposal(seqs):
            anchors, lengths, tables, bucket, temperatures = prop._inputs(seqs)
            if observations["draft_probe"] is None:
                if len(seqs) != real_b:
                    raise AssertionError("first proposal did not reach requested batch")
                state = torch.cuda.get_rng_state()
                runner.config.spec_graph_bs = exact_buckets
                _, q_exact_actual = original_propose(seqs)
                exact_base = prop.graphs[(real_b, bucket)][2][:real_b].detach().clone()
                exact_rng = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(state)
                runner.config.spec_graph_bs = production_buckets
                chains, q_actual = original_propose(seqs)
                padded_base = prop.graphs[(graph_b, bucket)][2][:real_b].detach().clone()
                padded_rng = torch.cuda.get_rng_state()
                eager_base = model.base_logits(anchors, lengths, tables, bucket)
                fixed = fixed_chains(seqs)
                q_exact = fixed_q(exact_base, anchors, fixed)
                q_padded = fixed_q(padded_base, anchors, fixed)
                q_eager = fixed_q(eager_base, anchors, fixed)
                actual_reconstruction = fixed_q(padded_base, anchors, chains)
                observations["draft_probe"] = {
                    "batch": real_b, "padded_bucket": graph_b, "context_bucket": bucket,
                    "q_eager_exact_kl": max_kl(q_eager, q_exact),
                    "q_eager_padded_kl": max_kl(q_eager, q_padded),
                    "q_exact_padded_kl": max_kl(q_exact, q_padded),
                    "actual_own_chain_kl": max_kl(actual_reconstruction, q_actual),
                    "actual_own_chain_max_abs": float((q_actual.float() - actual_reconstruction.float()).abs().max()),
                    "exact_padded_rng_equal": bool(torch.equal(exact_rng, padded_rng))}
                write({"type": "draft_probe", **identity, **observations["draft_probe"]})
                return fixed, q_padded
            chains, actual = original_propose(seqs)
            b = len(seqs)
            bucket_b = next((x for x in production_buckets if b <= x <= graph_b), None)
            if bucket_b is None:
                raise AssertionError("no draft padding bucket")
            graph_base = prop.graphs[(bucket_b, bucket)][2][:b].detach().clone()
            fixed = fixed_chains(seqs)
            return fixed, fixed_q(graph_base, anchors, fixed)

        @torch.inference_mode()
        def verify_forward(ids, positions):
            if observations["target_probe"] is not None:
                before_ragged = runner.verify_graph_fallbacks.get("ragged_draft_len", 0)
                result = original_forward(ids, positions)
                after_ragged = runner.verify_graph_fallbacks.get("ragged_draft_len", 0)
                if after_ragged > before_ragged:
                    expected = sum(1 + len(s.draft_tokens) for s in current["seqs"])
                    observations["ragged"].append({"query_rows": int(ids.numel()), "expected_rows": expected,
                                                    "output_rows": int(result.shape[0]),
                                                    "active_b": len(current["seqs"])})
                return result
            mapping = runner._verify_extra["slot_mapping"].long()
            if mapping.numel() != ids.numel() or bool((mapping < 0).any()):
                raise AssertionError("invalid real verify slots")
            blocks, offsets = torch.div(mapping, runner.block_size, rounding_mode="floor"), mapping % runner.block_size
            prior_graphs = runner.verify_graphs
            try:
                # Run on the same input, order eager -> exact -> padded so the
                # scratch and target KV used by subsequent commit are padded.
                runner.verify_graphs = {}
                eager = original_forward(ids, positions).detach().clone()
                tgt_e = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                scratch_e = model.scratch[:, :, :ids.numel()].detach().clone()
                runner.verify_graphs = {**standard_verify, (real_b, 7): exact_verify}
                exact_before = runner.verify_graph_hits
                exact = original_forward(ids, positions).detach().clone()
                if runner.verify_graph_hits != exact_before + 1:
                    raise AssertionError("exact target verify graph did not replay")
                tgt_x = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                scratch_x = model.scratch[:, :, :ids.numel()].detach().clone()
                runner.verify_graphs = standard_verify
                padded_before = runner.verify_graph_padded
                padded = original_forward(ids, positions).detach().clone()
                if runner.verify_graph_padded != padded_before + 1:
                    raise AssertionError("padded target verify graph did not replay")
                tgt_p = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                scratch_p = model.scratch[:, :, :ids.numel()].detach().clone()
            finally:
                runner.verify_graphs = prior_graphs
            observations["target_probe"] = {"query_rows": int(ids.numel()),
                "eager_exact_kl": max_kl(eager, exact), "eager_padded_kl": max_kl(eager, padded),
                "target_kv_eager_exact_rms": rms(tgt_e, tgt_x), "target_kv_eager_padded_rms": rms(tgt_e, tgt_p),
                "scratch_eager_exact_rms": rms(scratch_e, scratch_x),
                "scratch_eager_padded_rms": rms(scratch_e, scratch_p)}
            write({"type": "target_probe", **identity, **observations["target_probe"]})
            return padded

        def force_verify(q, p, candidates, temperatures=None, draft_is_point_mass=False):
            idx = (int(candidates[0, 0]) - 1000) // 8
            if not 0 <= idx < real_b:
                raise AssertionError("candidate row owner not recognized")
            desired = (0, 3, 7)[(idx + forced_per_row[idx]) % 3]
            forced_per_row[idx] += 1
            n = min(desired, candidates.shape[1])
            observations["forced"].append({"owner": idx, "accepted": int(n), "proposed": int(candidates.shape[1])})
            accepted = torch.tensor([n], device=candidates.device)
            return verifier.VerifyResult(accepted=accepted, bonus=torch.tensor([2000 + idx], device=candidates.device),
                n_proposed=torch.tensor([candidates.shape[1]], device=candidates.device),
                accept_mask=torch.arange(candidates.shape[1], device=candidates.device)[None] < accepted[:, None])
        prop.propose_batch = proposal
        runner.run_verify_forward = verify_forward
        verifier.verify_batch = force_verify
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        for i, (prompt, _) in enumerate(data):
            llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=a.output_tokens, ignore_eos=True))
            row_index[scheduler.waiting[-1].seq_id] = i
        deadline = float(os.environ["DS_RUN_DEADLINE_UNIX"])
        first_decode = True
        while not llm.is_finished():
            if time.time() >= deadline:
                raise TimeoutError("GPU guard deadline reached")
            seqs, is_prefill = scheduler.schedule()
            if not seqs:
                raise RuntimeError("failed to schedule B")
            if any(target_guard_block in s.block_table or ring_guard_block in s.draft_block_table for s in seqs):
                raise RuntimeError("guard block unexpectedly allocated to a real owner")
            current["seqs"] = seqs
            before_fallback = runner.verify_graph_fallbacks.get("ragged_draft_len", 0)
            if not is_prefill:
                observations["active_batches"].append(len(seqs))
                if first_decode and len(seqs) != real_b:
                    raise RuntimeError("requested fixed B not reached")
                first_decode = False
                counts = [min(7, max(0, s.max_tokens - s.num_completion_tokens - 1))
                          if s.num_scheduled_tokens > 1 else 0 for s in seqs]
                current["ragged"] = len(set(counts)) > 1 and any(counts)
                current["snapshot_owners"] = not observations["owner_survival"] or (current["ragged"] and not any(x["ragged"] for x in observations["owner_survival"]))
                current["prior"] = {}
                if current["snapshot_owners"]:
                    _, lengths, tables, _, _ = prop._inputs(seqs)
                    for i, s in enumerate(seqs):
                        slots = [int(x) for x in tables[i].tolist() if x >= 0]
                        idx = torch.tensor(slots, device="cuda", dtype=torch.long)
                        current["prior"][s.seq_id] = (slots, model.cache.index_select(2, idx).clone(), int(lengths[i]))
            output = runner.run(seqs, is_prefill)
            if is_prefill:
                output = [100 + row_index[s.seq_id] if s.num_cached_tokens + s.num_scheduled_tokens >= len(s) else x
                          for s, x in zip(seqs, output)]
            if not is_prefill and current["snapshot_owners"]:
                outcomes = []
                cu = runner._last_cu_seqlens_q
                if not output or not isinstance(output[0], list) or cu is None or len(cu) != len(seqs) + 1:
                    raise AssertionError("owner probe requires a real verify with query boundaries")
                for i, (s, raw) in enumerate(zip(seqs, output)):
                    slots, prior, C = current["prior"][s.seq_id]
                    landed = len(raw) if isinstance(raw, list) else 1
                    drop = max(0, len(slots) + landed - a.window)
                    remaining_slots = torch.tensor(slots[drop:], device="cuda", dtype=torch.long)
                    now = model.cache.index_select(2, remaining_slots)
                    # Independently reconstruct scratch-row and owner-ring mapping.
                    # Finished owners legitimately release their ring without commit.
                    still_active = landed < s.max_tokens - s.num_completion_tokens
                    committed_ok = None
                    if still_active:
                        if cu[i] + landed > cu[i + 1]:
                            raise AssertionError("owner commit exceeds its verify query rows")
                        M = a.window // runner.block_size
                        dst = [s.draft_block_table[(p // runner.block_size) % M] * runner.block_size
                               + p % runner.block_size for p in range(C, C + landed)]
                        dst_tensor = torch.tensor(dst, device="cuda", dtype=torch.long)
                        src_tensor = torch.arange(cu[i], cu[i] + landed, device="cuda", dtype=torch.long)
                        committed_ok = bool(torch.equal(model.cache.index_select(2, dst_tensor),
                                                        model.scratch.index_select(2, src_tensor)))
                    outcomes.append({"owner": row_index[s.seq_id], "anchor": C, "landed": landed,
                                     "surviving_slots": len(slots) - drop,
                                     "unchanged": bool(torch.equal(prior[:, :, drop:], now)),
                                     "continues": still_active, "commit_matches_own_scratch": committed_ok})
                observations["owner_survival"].append({"ragged": current["ragged"], "owners": outcomes,
                    "real_ragged_fallback": runner.verify_graph_fallbacks.get("ragged_draft_len", 0) > before_fallback})
                write({"type": "owner_commit_probe", **identity, **observations["owner_survival"][-1]})
            scheduler.postprocess(seqs, output, is_prefill)
        torch.cuda.synchronize()
        after = snapshot(runner)
        draft, target = observations["draft_probe"], observations["target_probe"]
        graph_counts = {"draft_hits": after["graph_hits"] - before["graph_hits"],
                        "draft_padded": after["graph_padded_hits"] - before["graph_padded_hits"],
                        "target_hits": after["verify_hits"] - before["verify_hits"],
                        "target_padded": after["verify_padded_hits"] - before["verify_padded_hits"],
                        "ragged_fallback": after["verify_fallbacks"].get("ragged_draft_len", 0) - before["verify_fallbacks"].get("ragged_draft_len", 0),
                        "draft_runtime_captures": after["graph_runtime_captures"] - before["graph_runtime_captures"],
                        "head_runtime_captures": after["head_runtime_captures"] - before["head_runtime_captures"]}
        guard_checks = {"target_guard_unchanged": bool(torch.equal(runner.kv_cache[:, :, target_guard_block], target_guard)),
                        "ring_guard_unchanged": bool(torch.equal(model.cache[:, :, ring_guard_slots], ring_guard)),
                        "sentinel_zero": bool(torch.count_nonzero(model.cache[:, :, model.sentinel]) == 0),
                        "prefix_lookup_count": scheduler.block_manager.prefix_lookup_count,
                        "prefix_hit_blocks": scheduler.block_manager.prefix_hit_blocks}
        failures = []
        if not draft or any(draft[k] >= 0.02 for k in ("q_eager_exact_kl", "q_eager_padded_kl", "q_exact_padded_kl")):
            failures.append("draft q KL")
        if not draft or draft["actual_own_chain_kl"] >= 0.02 or not draft["exact_padded_rng_equal"]:
            failures.append("draft real-chain q or padded RNG")
        if not target or any(target[k] >= 0.02 for k in ("eager_exact_kl", "eager_padded_kl")):
            failures.append("target verify KL")
        if not target or any(target[k] >= 0.05 for k in ("target_kv_eager_exact_rms", "target_kv_eager_padded_rms", "scratch_eager_exact_rms", "scratch_eager_padded_rms")):
            failures.append("target or projected scratch KV RMS")
        if not all(graph_counts[k] > 0 for k in ("draft_hits", "draft_padded", "target_hits", "target_padded", "ragged_fallback")):
            failures.append("required graph or ragged path not observed")
        if graph_counts["draft_runtime_captures"] or graph_counts["head_runtime_captures"]:
            failures.append("runtime graph capture")
        if not observations["ragged"] or not all(x["query_rows"] == x["expected_rows"] == x["output_rows"] for x in observations["ragged"]):
            failures.append("ragged true row count")
        if (not any(x["ragged"] and x["real_ragged_fallback"] for x in observations["owner_survival"])
                or not all(item["unchanged"] and (not item["continues"] or item["commit_matches_own_scratch"])
                           for x in observations["owner_survival"] for item in x["owners"])):
            failures.append("padded/ragged owner surviving KV")
        if not all(guard_checks[k] for k in ("target_guard_unchanged", "ring_guard_unchanged", "sentinel_zero")) or guard_checks["prefix_lookup_count"] or guard_checks["prefix_hit_blocks"]:
            failures.append("guards or disabled prefix")
        if not {0, 3, 7}.issubset({x["accepted"] for x in observations["forced"]}):
            failures.append("forced acceptance coverage")
        write({"type": "padding_contract", **identity, "pass": not failures, "failures": failures,
               "observations": observations, "graph_counts": graph_counts, "guard_checks": guard_checks,
               "window_stats": dict(runner.dspark_window_stats), "prewarm_shapes": shapes})
        print(f"补桶与 ragged 合同 B={real_b}：{'通过' if not failures else '失败'}", flush=True)
        if failures:
            raise RuntimeError("contract failures: " + ", ".join(failures))
    except BaseException as exc:
        write({"type": "failure", **identity, "error": type(exc).__name__,
               "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if restore_verify is not None:
            verifier.verify_batch = restore_verify
        handle.close()
        if llm is not None:
            import atexit
            llm.exit()
            atexit.unregister(llm.exit)


if __name__ == "__main__":
    main()
