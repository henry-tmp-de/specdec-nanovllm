"""Target and DSpark exact/eager/padded comparisons for B=3,11,15."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.spec_decode.verify import verify_batch
from nanovllm.utils.context import reset_context


def max_kl(expected, actual):
    p = torch.log_softmax(expected.float(), dim=-1)
    q = torch.log_softmax(actual.float(), dim=-1)
    return float((p.exp() * (p - q)).sum(-1).max())


def relative_rms(expected, actual):
    d = expected.float() - actual.float()
    return float(d.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-12))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Extra exact B buckets exist only for this numerical reference, not for
    # the performance experiment where buckets stay 1/2/4/8/16.
    exact_test = (1, 2, 3, 4, 8, 11, 15, 16)
    production = (1, 2, 4, 8, 16)
    llm = LLM(args.target, max_num_seqs=16, max_model_len=512, max_num_batched_tokens=512,
              gpu_memory_utilization=0.60, spec_method="dspark", spec_k=7,
              draft_model=args.draft, spec_graph_bs=exact_test, enforce_eager=False,
              dspark_split_graph=True, dspark_graph_pad=True, spec_graph_pad=True)
    mr, prop = llm.model_runner, llm.model_runner.spec_proposer
    standard_verify = {key: value for key, value in mr.verify_graphs.items() if key[0] in production}
    exact_verify = {key: value for key, value in mr.verify_graphs.items() if key[0] in (3, 11, 15)}
    manifest = json.loads(Path(args.manifest).read_text())
    prompts = [row["token_ids"][:32] for row in manifest["panels"]["code"][:15]]
    if mr.config.num_kvcache_blocks <= 16:
        raise RuntimeError("insufficient KV blocks for independent smoke-test rows")
    guard_block = mr.config.num_kvcache_blocks - 1
    mr.kv_cache[:, :, guard_block].fill_(0.375)
    guard_slots = slice(guard_block * mr.block_size, (guard_block + 1) * mr.block_size)
    mr.dspark_model.cache[:, :, guard_slots].fill_(0.25)
    target_guard = mr.kv_cache[:, :, guard_block].clone()
    draft_guard = mr.dspark_model.cache[:, :, guard_slots].clone()
    records = []
    first_q = first_p = None
    for batch in (3, 11, 15):
        seqs = []
        for i in range(batch):
            seq = Sequence(prompts[i], SamplingParams(max_tokens=32, temperature=1.0))
            seq.block_table = [i]
            seq.num_scheduled_tokens = len(seq)
            seqs.append(seq)
        ids, positions = mr.prepare_prefill(seqs, advance_draft=False)
        mr.run_model(ids, positions, True)
        reset_context()
        for i, seq in enumerate(seqs):
            seq.append_token((1234 + i) % mr.draft_hf_config.vocab_size)
            seq.draft_tokens = [(800 + 7 * i + j) % mr.draft_hf_config.vocab_size for j in range(7)]
            seq.num_scheduled_tokens = 8

        def target(mode):
            mr.verify_graphs = {} if mode == "eager" else dict(standard_verify)
            if mode == "exact":
                mr.verify_graphs[(batch, 7)] = exact_verify[(batch, 7)]
            ids, pos = mr.prepare_verify(seqs)
            out = mr.run_verify_forward(ids, pos).clone()
            reset_context()
            # Prefix + speculative candidate slots in every real physical block.
            target_slice = mr.kv_cache[:, :, :batch, :40].clone()
            slot_ids = torch.tensor([i * mr.block_size + j for i in range(batch) for j in range(40)],
                                    device="cuda", dtype=torch.long)
            draft_slice = mr.dspark_model.cache[:, :, slot_ids].clone()
            return out, target_slice, draft_slice

        p_e, t_e, d_e = target("eager")
        p_x, t_x, d_x = target("exact")
        padded_before = mr.verify_graph_padded
        p_p, t_p, d_p = target("padded")
        if mr.verify_graph_padded != padded_before + 1:
            raise AssertionError(f"target padded graph was not replayed for B={batch}")

        # Capture every needed draft graph *before* seeding, including exact B
        # for reference. Production selection excludes those extra shapes.
        mr.config.enforce_eager = False
        mr.config.spec_graph_bs = exact_test
        context_bucket = 256
        prop.prewarm([(batch, context_bucket), (4 if batch == 3 else 16, context_bucket)], [batch])
        def draft(mode):
            mr.config.enforce_eager = mode == "eager"
            mr.config.spec_graph_bs = exact_test if mode == "exact" else production
            mr.config.dspark_graph_pad = mode == "padded"
            torch.manual_seed(72351)
            torch.cuda.manual_seed_all(72351)
            tokens, q = prop.propose_batch(seqs)
            if mode == "eager":
                anchors, lengths, tables, ctx, _ = prop._inputs(seqs)
                base_logits = mr.dspark_model.base_logits(anchors, lengths, tables, ctx)
            else:
                graph_batch = batch if mode == "exact" else (4 if batch == 3 else 16)
                base_logits = prop.graphs[(graph_batch, context_bucket)][2][:batch].clone()
            return tokens, q.clone(), torch.cuda.get_rng_state().clone(), base_logits
        x_e, q_e, rng_e, base_e = draft("eager")
        x_x, q_x, rng_x, base_x = draft("exact")
        before = prop.graph_padded_hits
        x_p, q_p, rng_p, base_p = draft("padded")
        if prop.graph_padded_hits != before + 1:
            raise AssertionError(f"draft padded graph was not replayed for B={batch}")
        def conditional(bases, chain):
            previous = torch.tensor([s.last_token for s in seqs], device="cuda")
            result = []
            for j in range(7):
                result.append(bases[:, j] + mr.dspark_model.markov_head(previous))
                previous = torch.tensor([x[j] for x in chain], device="cuda")
            return torch.stack(result, dim=1)
        # Compare each q under the *same* actual predecessor chain. Different
        # valid random draws across eager and graph do not mean a q error.
        c_e, c_x, c_p = (conditional(v, x_x) for v in (base_e, base_x, base_p))
        own_error = max(float((q.float() - conditional(v, chain).float()).abs().max())
                        for q, v, chain in ((q_e, base_e, x_e), (q_x, base_x, x_x),
                                            (q_p, base_p, x_p)))
        row = {"batch": batch, "target_eager_exact_kl": max_kl(p_e, p_x),
               "target_eager_padded_kl": max_kl(p_e, p_p),
               "target_kv_eager_exact_rms": relative_rms(t_e, t_x),
               "target_kv_eager_padded_rms": relative_rms(t_e, t_p),
               "draft_context_kv_eager_exact_rms": relative_rms(d_e, d_x),
               "draft_context_kv_eager_padded_rms": relative_rms(d_e, d_p),
               "q_eager_exact_kl": max_kl(c_e, c_x), "q_eager_padded_kl": max_kl(c_e, c_p),
               "q_exact_padded_kl": max_kl(c_x, c_p), "actual_own_predecessor_q_max_abs": own_error,
               "exact_padded_rng_equal": torch.equal(rng_x, rng_p),
               "eager_exact_rng_equal": torch.equal(rng_e, rng_x),
               "exact_padded_tokens_equal": x_x == x_p,
               "eager_exact_tokens_equal": x_e == x_x,
               "target_guard_unchanged": torch.equal(mr.kv_cache[:, :, guard_block], target_guard),
               "draft_guard_unchanged": torch.equal(mr.dspark_model.cache[:, :, guard_slots], draft_guard),
               "sentinel_zero": bool(torch.count_nonzero(mr.dspark_model.cache[:, :, mr.dspark_model.sentinel]) == 0)}
        records.append(row)
        if batch == 3:
            first_p, first_q = p_p[0].clone(), q_p[0, 0].clone()

    # The batch-size padding path does not legalize uneven candidate lengths.
    # Exercise the real ragged fallback and its true output row count.
    seqs[-1].draft_tokens = seqs[-1].draft_tokens[:3]
    before_ragged = mr.verify_graph_fallbacks.get("ragged_draft_len", 0)
    ragged_ids, ragged_positions = mr.prepare_verify(seqs)
    ragged_logits = mr.run_verify_forward(ragged_ids, ragged_positions)
    ragged_rows = sum(1 + len(s.draft_tokens) for s in seqs)
    ragged_ok = (mr.verify_graph_fallbacks.get("ragged_draft_len", 0) == before_ragged + 1
                 and ragged_logits.shape[0] == ragged_rows)
    reset_context()

    # Actual model distributions, production verifier; only B real q samples.
    torch.manual_seed(991)
    p_prob, q_prob = torch.softmax(first_p.float(), -1), torch.softmax(first_q.float(), -1)
    candidates = torch.multinomial(q_prob, 8192, replacement=True)
    produced = []
    for chunk in candidates.split(32):
        n = chunk.numel()
        qr = first_q.view(1, 1, -1).expand(n, 1, -1)
        pr = first_p.view(1, 1, -1).expand(n, 2, -1)
        result = verify_batch(qr, pr, chunk.view(n, 1))
        produced.append(torch.where(result.accepted > 0, chunk, result.bonus))
    produced = torch.cat(produced)
    top = torch.topk(p_prob, 16).indices
    observed = torch.stack([(produced == tok).float().mean() for tok in top])
    expected = p_prob[top]
    observed = torch.cat((observed, (1 - observed.sum()).view(1)))
    expected = torch.cat((expected, (1 - expected.sum()).view(1)))
    tv = float((observed - expected).abs().sum() / 2)
    checks = all(row["target_eager_exact_kl"] < 0.02 and row["target_eager_padded_kl"] < 0.02
                 and row["q_eager_exact_kl"] < 0.02 and row["q_eager_padded_kl"] < 0.02
                 and row["q_exact_padded_kl"] < 0.01
                 and row["actual_own_predecessor_q_max_abs"] < 0.1
                 and row["target_kv_eager_padded_rms"] < 0.05
                 and row["draft_context_kv_eager_padded_rms"] < 0.05
                 and row["exact_padded_rng_equal"] and row["target_guard_unchanged"]
                 and row["draft_guard_unchanged"] and row["sentinel_zero"] for row in records)
    report = {"pass": bool(checks and ragged_ok and tv < 0.06), "records": records,
              "ragged_fallback_ok": ragged_ok, "ragged_true_output_rows": ragged_rows,
              "actual_target_proposal_samples": 8192, "target_top16_other_tv": tv,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    llm.exit()
    import atexit
    atexit.unregister(llm.exit)
    if not report["pass"]:
        raise SystemExit("GPU graph padding contract failed")


if __name__ == "__main__":
    main()
