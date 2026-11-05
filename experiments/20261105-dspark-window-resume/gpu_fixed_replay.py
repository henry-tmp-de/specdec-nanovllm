"""Real-model fixed-history DSpark KV/q diagnostic with forced acceptance.

Runs W0, full_cache and ring serially on one guarded GPU.  The target history,
anchor and seven candidate predecessors are fixed across runs.  Rejections are
forced solely to test commit semantics; this is not a sampling/performance test.
The first position compares W0 against W>=history; later positions compare the
same W in full_cache and ring. Run eager and graph as separate guarded stages.
"""
from __future__ import annotations

import argparse
import gc
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
from bench_smoke import prewarm_shapes, read_manifest, snapshot, token_hash
from gpu_identity import verify_visible_gpu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="new JSONL file")
    ap.add_argument("--graph", choices=("eager", "graph"), required=True)
    ap.add_argument("--ctx", type=int, default=252)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--output-tokens", type=int, default=532)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    if a.window <= 0 or a.window % 256 or not 0 < a.ctx <= a.window:
        ap.error("window 必须 256 对齐，且 0 < ctx <= window，保证首轮全历史对照")
    if a.ctx + a.output_tokens < 3 * a.window + 8:
        ap.error("输出预算不足以从首个位置跨越至少两个完整环及其后验收轮")
    prompt, metadata = read_manifest(a.manifest, "code", 1, a.ctx)[0]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("x", encoding="utf-8")
    def write(row):
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    identity = {"guard_id": os.environ.get("DS_GUARD_ID"), "gpu_uuid": os.environ.get("DS_GPU_UUID"),
                "graph": a.graph, "ctx": a.ctx, "window": a.window, "seed": a.seed,
                "output_tokens": a.output_tokens, "input_sha256": token_hash(prompt),
                "input_metadata": metadata, "diagnostic_forced_acceptance": True,
                "q_kl_threshold": 0.02, "kv_relative_rms_threshold": 0.05}
    try:
        if not all(os.environ.get(x) for x in ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX")):
            raise RuntimeError("必须通过本轮 gpu_control.py 启动 GPU 数值对拍")
        import torch
        verify_visible_gpu(torch, os.environ["DS_GPU_UUID"])
        from nanovllm import LLM, SamplingParams
        import nanovllm.spec_decode.verify as verifier
        from nanovllm.engine.token_hook import TokenDeliveryHook
        anchor = prompt[-1]
        fixed_candidates = (prompt[-7:] if len(prompt) >= 7 else [anchor] * 7)
        deadline = float(os.environ["DS_RUN_DEADLINE_UNIX"])

        def one_mode(mode):
            window = 0 if mode == "w0" else a.window
            llm = None
            old_verify = verifier.verify_batch
            snapshots, forced, rejected_checks, ordinary_steps = [], [], [], []
            output_history = []
            current = {"number": 0, "slots": None, "before": None, "C": None,
                       "last_bucket": None, "capture": False, "after_ordinary": False}
            try:
                max_len = a.ctx + a.output_tokens + 7
                llm = LLM(a.target, max_num_seqs=1, max_model_len=max_len,
                          max_num_batched_tokens=max(a.ctx, max_len), gpu_memory_utilization=0.70,
                          tensor_parallel_size=1, enforce_eager=a.graph == "eager",
                          spec_cuda_graph=a.graph == "graph", spec_method="dspark", spec_k=7,
                          draft_model=a.draft, spec_graph_bs=(1,), spec_graph_pad=False,
                          dspark_split_graph=True, dspark_graph_pad=False,
                          dspark_window_tokens=window, dspark_window_storage="ring" if mode == "ring" else "full_cache",
                          enable_prefix_cache=False, spec_overlap_mixed=False)
                runner, scheduler = llm.model_runner, llm.scheduler
                prop, model = runner.spec_proposer, runner.dspark_model
                if runner.config.max_model_len < max_len:
                    raise RuntimeError("model max position capacity is insufficient")
                if a.graph == "graph":
                    shapes = prewarm_shapes(1, 1, a.ctx, a.output_tokens, window, runner.block_size, (1,))
                    prop.prewarm(shapes, [1])
                    torch.cuda.synchronize()
                else:
                    shapes = []
                before_graph = snapshot(runner)
                original_propose = prop.propose_batch

                @torch.inference_mode()
                def fixed_propose(seqs):
                    if len(seqs) != 1:
                        raise RuntimeError("fixed-history replay requires B=1")
                    seq = seqs[0]
                    C = len(seq) - 1
                    anchors, lengths, tables, bucket, _ = prop._inputs(seqs)
                    if int(lengths[0]) != C:
                        raise AssertionError("anchor position does not equal target history")
                    if window:
                        slots = [int(x) for x in tables[0].tolist() if x >= 0]
                        expected_len = min(C, window)
                    else:
                        blocks = tables[0].tolist()
                        slots = [int(blocks[pos // runner.block_size]) * runner.block_size + pos % runner.block_size
                                 for pos in range(C)]
                        expected_len = C
                    if len(slots) != expected_len or min(slots, default=0) < 0 or len(set(slots)) != len(slots):
                        raise AssertionError("context table has missing or duplicate live slots")
                    slot_tensor = torch.tensor(slots, device="cuda", dtype=torch.long)
                    current.update(slots=slots, before=model.cache.index_select(2, slot_tensor).clone(), C=C)
                    sampled, actual_q = original_propose(seqs)
                    if len(sampled[0]) != 7:
                        raise AssertionError("real DSpark proposal did not produce seven candidates")
                    if a.graph == "graph":
                        graph_base = prop.graphs[(1, bucket)][2][:1].detach().clone()
                    else:
                        graph_base = None
                    base = model.base_logits(anchors, lengths, tables, bucket)
                    previous = anchors
                    eager_q, graph_q = [], []
                    for j, token in enumerate(fixed_candidates):
                        correction = model.markov_head(previous)
                        # Production adds in the native dtype before sampling.
                        eager_q.append((base[:, j] + correction).float())
                        if graph_base is not None:
                            graph_q.append((graph_base[:, j] + correction).float())
                        previous = torch.tensor([token], device="cuda", dtype=torch.long)
                    eager_q = torch.stack(eager_q, dim=1)
                    actual_logits = eager_q.to(actual_q.dtype)
                    C_bucket = C // a.window
                    capture = (len(snapshots) < 6 or C_bucket != current["last_bucket"]
                               or current["after_ordinary"])
                    current["last_bucket"], current["after_ordinary"] = C_bucket, False
                    if capture:
                        logical_target_positions = sorted({x for x in (0, C // 2, C - 1, 255, 256, 511, 512, 767, 768)
                                                           if 0 <= x < C})
                        if len(seq.block_table) < math.ceil(C / runner.block_size):
                            raise AssertionError("target cache does not cover complete confirmed history")
                        tb = torch.tensor([seq.block_table[x // runner.block_size] for x in logical_target_positions],
                                          device="cuda", dtype=torch.long)
                        ti = torch.tensor([x % runner.block_size for x in logical_target_positions],
                                          device="cuda", dtype=torch.long)
                        target_kv = runner.kv_cache[:, :, tb, ti].detach().cpu()
                        graph_kl = None
                        if graph_q:
                            gq = torch.stack(graph_q, dim=1)
                            p, q = torch.log_softmax(eager_q, -1), torch.log_softmax(gq, -1)
                            graph_kl = float((p.exp() * (p - q)).sum(-1).max())
                        snapshots.append({"C": C, "history_hash": token_hash(seq.token_ids),
                                          "history_start": C - expected_len, "visible_len": expected_len,
                                          "target_sample_positions": logical_target_positions,
                                          "target_blocks": len(seq.block_table),
                                          "q": eager_q.detach().cpu(),
                                          "kv": current["before"].detach().cpu(),
                                          "target_kv": target_kv,
                                          "graph_same_predecessor_q_kl": graph_kl})
                    return [list(fixed_candidates)], actual_logits

                def force_verify(q, p, candidates, temperatures=None, draft_is_point_mass=False):
                    count = min((0, 3, 7)[len(forced) % 3], candidates.shape[1])
                    forced.append(int(count))
                    chosen_bonus = fixed_candidates[(len(forced) - 1) % 7]
                    n = torch.tensor([count], device=candidates.device)
                    mask = torch.arange(candidates.shape[1], device=candidates.device)[None] < n[:, None]
                    bonus = torch.tensor([chosen_bonus], device=candidates.device)
                    return verifier.VerifyResult(accepted=n, bonus=bonus,
                        n_proposed=torch.tensor([candidates.shape[1]], device=candidates.device), accept_mask=mask)
                prop.propose_batch = fixed_propose
                verifier.verify_batch = force_verify
                torch.manual_seed(a.seed)
                torch.cuda.manual_seed_all(a.seed)
                llm.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=a.output_tokens, ignore_eos=True))
                step = 0
                while not llm.is_finished():
                    if time.time() >= deadline:
                        raise TimeoutError("GPU guard deadline reached")
                    seqs, prefill = scheduler.schedule()
                    if len(seqs) != 1:
                        raise RuntimeError("fixed replay failed to admit B=1")
                    seq = seqs[0]
                    if not prefill:
                        step += 1
                        if step in (5, 25, 50):
                            # Force a real engine ordinary-decode step after scheduler reservation.
                            seq.num_scheduled_tokens = 1
                            ordinary_steps.append({"step": step, "anchor": len(seq) - 1})
                    force_before = len(forced)
                    output = runner.run(seqs, prefill)
                    if prefill:
                        output = [anchor]
                    elif len(forced) == force_before:
                        if not output or isinstance(output[0], list):
                            raise AssertionError("unexpected no-verifier output shape")
                        # Also fix a last-token ordinary fallback (remaining=1).
                        output = [fixed_candidates[step % 7]]
                        current["after_ordinary"] = True
                    if (not prefill and len(forced) > force_before and forced[-1] == 0 and window):
                        prior = current["before"]
                        slots = current["slots"]
                        C = current["C"]
                        # At a full window, only the oldest position may be evicted.
                        skip = 1 if C >= window else 0
                        live = torch.tensor(slots[skip:], device="cuda", dtype=torch.long)
                        after = model.cache.index_select(2, live)
                        unchanged = bool(torch.equal(prior[:, :, skip:], after))
                        rejected_checks.append({"anchor": C, "surviving_slots": len(slots) - skip,
                                                "unchanged_after_reject": unchanged})
                        if not unchanged:
                            raise AssertionError("rejected suffix overwrote surviving history KV")
                    delivered = scheduler.postprocess(seqs, output, prefill)
                    if delivered:
                        output_history = list(seq.completion_token_ids)
                torch.cuda.synchronize()
                graph_delta = {k: v for k, v in snapshot(runner).items() if k in
                               ("graph_hits", "graph_exact_hits", "graph_padded_hits", "graph_fallbacks",
                                "graph_runtime_captures", "head_runtime_captures", "verify_hits",
                                "verify_padded_hits", "verify_fallbacks", "context_bucket_histogram")}
                if a.graph == "graph" and (graph_delta.get("graph_hits", 0) <= before_graph.get("graph_hits", 0)
                                            or graph_delta.get("verify_hits", 0) <= before_graph.get("verify_hits", 0)):
                    raise AssertionError("requested exact draft/verify graph was not actually used")
                if scheduler.block_manager.prefix_lookup_count or scheduler.block_manager.prefix_hit_blocks:
                    raise AssertionError("prefix lookup occurred")
                if len(output_history) != a.output_tokens or not {0, 3, 7}.issubset(set(forced)):
                    raise AssertionError("forced acceptance or delivered output coverage incomplete")
                if len(ordinary_steps) < 2 or not snapshots or max(x["C"] for x in snapshots) < 3 * a.window:
                    raise AssertionError("ordinary decode or multi-wrap coverage incomplete")
                if window and (not rejected_checks or not all(x["unchanged_after_reject"] for x in rejected_checks)):
                    raise AssertionError("no successful reject/commit checks")
                return {"mode": mode, "snapshots": snapshots, "forced": forced,
                        "ordinary_steps": ordinary_steps, "reject_checks": rejected_checks,
                        "output_hash": token_hash(output_history), "prewarm_shapes": shapes,
                        "graph_counters_after": graph_delta,
                        "window_stats": dict(runner.dspark_window_stats)}
            finally:
                verifier.verify_batch = old_verify
                if llm is not None:
                    import atexit
                    llm.exit()
                    atexit.unregister(llm.exit)
                gc.collect()
                torch.cuda.empty_cache()

        modes = {}
        for mode in ("w0", "full_cache", "ring"):
            modes[mode] = one_mode(mode)
            write({"type": "mode_complete", **identity,
                   **{k: v for k, v in modes[mode].items() if k != "snapshots"},
                   "snapshot_anchors": [s["C"] for s in modes[mode]["snapshots"]]})
            # Break proposer/closure cycles only after the mode stack is gone.
            gc.collect()
            torch.cuda.empty_cache()
            print(f"固定历史对拍完成：{mode}，采样快照 {len(modes[mode]['snapshots'])}", flush=True)
        if len({v["output_hash"] for v in modes.values()}) != 1:
            raise AssertionError("controlled history differs across storage modes")

        def compare(left, right, full_history_only=False):
            rows = []
            ls = {s["C"]: s for s in modes[left]["snapshots"]}
            rs = {s["C"]: s for s in modes[right]["snapshots"]}
            if ls.keys() != rs.keys():
                raise AssertionError("sampled anchors differ across replay modes")
            for C in sorted(ls):
                x, y = ls[C], rs[C]
                if full_history_only and C > a.window:
                    continue
                if (x["history_hash"] != y["history_hash"] or x["visible_len"] != y["visible_len"]
                        or x["target_sample_positions"] != y["target_sample_positions"]):
                    raise AssertionError(f"logical histories differ at C={C}")
                p, q = torch.log_softmax(x["q"].float(), -1), torch.log_softmax(y["q"].float(), -1)
                kl = float((p.exp() * (p - q)).sum(-1).max())
                def rms(u, v):
                    u, v = u.float(), v.float()
                    return float((u - v).square().mean().sqrt() / u.square().mean().sqrt().clamp_min(1e-12))
                row = {"C": C, "history_start": x["history_start"], "visible_len": x["visible_len"],
                       "q_max_kl": kl, "q_max_abs": float((x["q"].float() - y["q"].float()).abs().max()),
                       "context_kv_relative_rms": rms(x["kv"], y["kv"]),
                       "target_kv_sample_relative_rms": rms(x["target_kv"], y["target_kv"]),
                       "graph_eager_kl_left": x["graph_same_predecessor_q_kl"],
                       "graph_eager_kl_right": y["graph_same_predecessor_q_kl"]}
                if not (math.isfinite(kl) and kl < 0.02 and row["context_kv_relative_rms"] < 0.05
                        and row["target_kv_sample_relative_rms"] < 0.05
                        and all(v is None or (math.isfinite(v) and v < 0.02) for v in
                                (row["graph_eager_kl_left"], row["graph_eager_kl_right"]))):
                    raise AssertionError(f"fixed-history numerical comparison failed: {row}")
                rows.append(row)
            if not rows:
                raise AssertionError("no eligible numerical snapshots")
            return rows
        full_ring = compare("full_cache", "ring")
        w0_full = compare("w0", "full_cache", full_history_only=True)
        report = {"type": "fixed_replay", **identity, "pass": True,
                  "controls": {"w0_vs_full_when_window_covers_history": w0_full,
                               "full_cache_vs_ring_same_window": full_ring},
                  "modes": {key: {k: v for k, v in data.items() if k != "snapshots"}
                            for key, data in modes.items()},
                  "target_kv_note": "target KV sampled at specified logical positions; target context itself is full",
                  "sampler_note": "forced acceptances and fixed candidate chain are diagnostic, not a sampling test"}
        write(report)
        print(f"固定历史数值对拍通过：同窗 {len(full_ring)} 个位置，W>=历史 {len(w0_full)} 个位置", flush=True)
    except BaseException as exc:
        write({"type": "failure", **identity, "error": type(exc).__name__,
               "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        f.close()


if __name__ == "__main__":
    main()
