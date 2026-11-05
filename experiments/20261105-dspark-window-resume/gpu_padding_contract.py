"""B3/11/15 verify, ragged fallback and ring-owner GPU contract.

One real model process per B, intentionally forced acceptance. The historical
bucket_padded mode preserves its original test; exact_all tests same-input
eager/exact with restored query-slot/scratch preimages. This is not throughput.
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
from gpu_identity import verify_visible_gpu


def contract_graph_settings(batch, policy):
    if batch not in (3, 11, 15) or policy not in ("bucket_padded", "exact_all"):
        raise ValueError("invalid contract batch or graph policy")
    if policy == "exact_all":
        # Reserve one additional unused ring owner as a guard block; measured
        # real B and target/draft speculative graph shapes remain 1..batch.
        return {"max_seqs": batch + 1, "graph_batch": batch,
                "production": tuple(range(1, batch + 1)),
                "capture": tuple(range(1, batch + 1)), "padding": False}
    graph_batch = 4 if batch == 3 else 16
    production = (1, 2, 4, 8, 16)
    return {"max_seqs": graph_batch, "graph_batch": graph_batch, "production": production,
            "capture": tuple(sorted(set(production + (batch,)))), "padding": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", required=True, type=int, choices=(3, 11, 15))
    ap.add_argument("--graph-policy", choices=("bucket_padded", "exact_all"), default="bucket_padded")
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
                "seed": a.seed, "graph_policy": a.graph_policy,
                "forced_acceptance": True, "diagnostic_instrumented": True,
                "q_and_target_kl_threshold": 0.02, "kv_relative_rms_threshold": 0.05}
    llm = None
    restore_verify = None
    try:
        if not all(os.environ.get(x) for x in ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX")):
            raise RuntimeError("必须通过 gpu_control.py 在授权 GPU 启动")
        import torch
        verify_visible_gpu(torch, os.environ["DS_GPU_UUID"])
        from nanovllm import LLM, SamplingParams
        from nanovllm.config import Config
        from dataclasses import fields
        import nanovllm.spec_decode.verify as verifier
        restore_verify = verifier.verify_batch
        real_b = a.batch
        settings = contract_graph_settings(real_b, a.graph_policy)
        graph_b, max_seqs = settings["graph_batch"], settings["max_seqs"]
        production_buckets, exact_buckets = settings["production"], settings["capture"]
        options = {}
        if a.graph_policy == "exact_all":
            if "decode_graph_bs" not in {field.name for field in fields(Config)}:
                raise RuntimeError("engine missing decode_graph_bs")
            options["decode_graph_bs"] = tuple(range(1, max_seqs + 1))
        identity.update(max_num_seqs=max_seqs, spec_graph_buckets=production_buckets,
                        guard_extra_sequences=max_seqs - real_b,
                        decode_graph_buckets=options.get("decode_graph_bs"))
        max_len = a.ctx + a.output_tokens + 7
        llm = LLM(a.target, max_num_seqs=max_seqs, max_model_len=max_len,
                  max_num_batched_tokens=max(a.ctx * real_b, max_len), gpu_memory_utilization=0.70,
                  tensor_parallel_size=1, enforce_eager=False, spec_cuda_graph=True,
                  spec_method="dspark", spec_k=7, draft_model=a.draft,
                  spec_graph_bs=exact_buckets, spec_graph_pad=settings["padding"], dspark_split_graph=True,
                  dspark_graph_pad=settings["padding"], dspark_window_tokens=a.window,
                  dspark_window_storage="ring", enable_prefix_cache=False, spec_overlap_mixed=False, **options)
        runner, scheduler = llm.model_runner, llm.scheduler
        prop, model = runner.spec_proposer, runner.dspark_model
        if runner.config.max_model_len < max_len:
            raise RuntimeError("model max position capacity is insufficient")
        if options and runner.config.decode_graph_bs != options["decode_graph_bs"]:
            raise RuntimeError("engine did not retain exact decode graph buckets")
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
        shapes = prewarm_shapes(real_b, max_seqs, a.ctx, a.output_tokens, a.window,
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
                if a.graph_policy == "exact_all":
                    chains, q_actual = original_propose(seqs)
                    q_actual = q_actual.detach().clone()
                    exact_base = prop.graphs[(real_b, bucket)][2][:real_b].detach().clone()
                    eager_base = model.base_logits(anchors, lengths, tables, bucket)
                    fixed = fixed_chains(seqs)
                    q_exact = fixed_q(exact_base, anchors, fixed)
                    q_eager = fixed_q(eager_base, anchors, fixed)
                    actual_reconstruction = fixed_q(exact_base, anchors, chains)
                    observations["draft_probe"] = {
                        "batch": real_b, "exact_bucket": real_b, "context_bucket": bucket,
                        "q_eager_exact_kl": max_kl(q_eager, q_exact),
                        "actual_own_chain_kl": max_kl(actual_reconstruction, q_actual),
                        "actual_own_chain_max_abs": float((q_actual.float() - actual_reconstruction.float()).abs().max())}
                    write({"type": "draft_probe", **identity, **observations["draft_probe"]})
                    return fixed, q_exact
                state = torch.cuda.get_rng_state()
                runner.config.spec_graph_bs = exact_buckets
                _, q_exact_actual = original_propose(seqs)
                exact_base = prop.graphs[(real_b, bucket)][2][:real_b].detach().clone()
                exact_rng = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(state)
                runner.config.spec_graph_bs = production_buckets
                chains, q_actual = original_propose(seqs)
                q_actual = q_actual.detach().clone()
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
            if a.graph_policy == "exact_all":
                expected_pos, expected_slots = [], []
                owned_blocks = [block for seq in current["seqs"] for block in seq.block_table]
                if len(set(owned_blocks)) != len(owned_blocks):
                    raise AssertionError("target owners share or duplicate physical blocks")
                for seq in current["seqs"]:
                    for pos in range(len(seq) - 1, len(seq) + len(seq.draft_tokens)):
                        if pos // runner.block_size >= len(seq.block_table):
                            raise AssertionError("target block table does not cover verify query")
                        expected_pos.append(pos)
                        expected_slots.append(seq.block_table[pos // runner.block_size] * runner.block_size + pos % runner.block_size)
                if positions.tolist() != expected_pos or mapping.tolist() != expected_slots or len(set(expected_slots)) != len(expected_slots):
                    raise AssertionError("target verify query slots do not match real owners")
                target_pre = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                scratch_pre = model.scratch[:, :, :ids.numel()].detach().clone()
                try:
                    runner.verify_graphs = {}
                    eager = original_forward(ids, positions).detach().clone()
                    tgt_e = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                    scratch_e = model.scratch[:, :, :ids.numel()].detach().clone()
                    runner.kv_cache[:, :, blocks, offsets] = target_pre
                    model.scratch[:, :, :ids.numel()].copy_(scratch_pre)
                    restored_target = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                    restored_scratch = model.scratch[:, :, :ids.numel()].detach().clone()
                    for before, after in ((target_pre, restored_target), (scratch_pre, restored_scratch)):
                        if not torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)):
                            raise AssertionError("query/scratch preimage restore was not bitwise identical")
                    runner.verify_graphs = standard_verify
                    exact_before = runner.verify_graph_hits - runner.verify_graph_padded
                    exact = original_forward(ids, positions).detach().clone()
                    if runner.verify_graph_hits - runner.verify_graph_padded != exact_before + 1:
                        raise AssertionError("exact target verify graph did not replay")
                    tgt_x = runner.kv_cache[:, :, blocks, offsets].detach().clone()
                    scratch_x = model.scratch[:, :, :ids.numel()].detach().clone()
                finally:
                    runner.verify_graphs = prior_graphs
                observations["target_probe"] = {"query_rows": int(ids.numel()),
                    "eager_exact_kl": max_kl(eager, exact),
                    "target_kv_eager_exact_rms": rms(tgt_e, tgt_x),
                    "scratch_eager_exact_rms": rms(scratch_e, scratch_x),
                    "same_query_preimage_restored": True}
                write({"type": "target_probe", **identity, **observations["target_probe"]})
                return exact
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
                        "draft_exact": after["graph_exact_hits"] - before["graph_exact_hits"],
                        "draft_padded": after["graph_padded_hits"] - before["graph_padded_hits"],
                        "target_hits": after["verify_hits"] - before["verify_hits"],
                        "target_exact": after["verify_exact_hits"] - before["verify_exact_hits"],
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
        q_keys = ("q_eager_exact_kl",) if a.graph_policy == "exact_all" else ("q_eager_exact_kl", "q_eager_padded_kl", "q_exact_padded_kl")
        target_keys = ("eager_exact_kl",) if a.graph_policy == "exact_all" else ("eager_exact_kl", "eager_padded_kl")
        kv_keys = (("target_kv_eager_exact_rms", "scratch_eager_exact_rms") if a.graph_policy == "exact_all" else
                   ("target_kv_eager_exact_rms", "target_kv_eager_padded_rms", "scratch_eager_exact_rms", "scratch_eager_padded_rms"))
        if not draft or any(not math.isfinite(draft[k]) or draft[k] >= 0.02 for k in q_keys):
            failures.append("draft q KL")
        if (not draft or not math.isfinite(draft["actual_own_chain_kl"]) or draft["actual_own_chain_kl"] >= 0.02
                or (a.graph_policy == "bucket_padded" and not draft["exact_padded_rng_equal"])):
            failures.append("draft real-chain q or padded RNG")
        if not target or any(not math.isfinite(target[k]) or target[k] >= 0.02 for k in target_keys):
            failures.append("target verify KL")
        if not target or any(not math.isfinite(target[k]) or target[k] >= 0.05 for k in kv_keys):
            failures.append("target or projected scratch KV RMS")
        required_paths = (("draft_exact", "target_exact", "ragged_fallback") if a.graph_policy == "exact_all" else
                          ("draft_hits", "draft_padded", "target_hits", "target_padded", "ragged_fallback"))
        if not all(graph_counts[k] > 0 for k in required_paths):
            failures.append("required graph or ragged path not observed")
        if a.graph_policy == "exact_all" and (graph_counts["draft_padded"] or graph_counts["target_padded"]):
            failures.append("padding occurred during exact-all contract")
        if graph_counts["draft_runtime_captures"] or graph_counts["head_runtime_captures"]:
            failures.append("runtime graph capture")
        if not observations["ragged"] or not all(x["query_rows"] == x["expected_rows"] == x["output_rows"] for x in observations["ragged"]):
            failures.append("ragged true row count")
        if (not any(x["ragged"] and x["real_ragged_fallback"] for x in observations["owner_survival"])
                or not all(item["unchanged"] and (not item["continues"] or item["commit_matches_own_scratch"])
                           for x in observations["owner_survival"] for item in x["owners"])):
            failures.append("graph/ragged owner surviving KV")
        if not all(guard_checks[k] for k in ("target_guard_unchanged", "ring_guard_unchanged", "sentinel_zero")) or guard_checks["prefix_lookup_count"] or guard_checks["prefix_hit_blocks"]:
            failures.append("guards or disabled prefix")
        if not {0, 3, 7}.issubset({x["accepted"] for x in observations["forced"]}):
            failures.append("forced acceptance coverage")
        write({"type": "exact_owner_contract" if a.graph_policy == "exact_all" else "padding_contract",
               **identity, "pass": not failures, "failures": failures,
               "observations": observations, "graph_counts": graph_counts, "guard_checks": guard_checks,
               "window_stats": dict(runner.dspark_window_stats), "prewarm_shapes": shapes})
        print(f"{a.graph_policy} / ragged 合同 B={real_b}：{'通过' if not failures else '失败'}", flush=True)
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
