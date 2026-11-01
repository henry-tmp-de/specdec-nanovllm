"""Integrated B=1/2 smoke, including verify-graph feature/KV replay parity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.verify import verify_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--batch", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(772)
    llm = LLM(a.target, max_num_seqs=a.batch, max_model_len=512,
              max_num_batched_tokens=512, gpu_memory_utilization=0.70,
              spec_method="dspark", spec_k=7, draft_model=a.draft,
              spec_graph_bs=(1, 2), enforce_eager=False)
    mr, prop = llm.model_runner, llm.model_runner.spec_proposer
    checks = {"proposal_calls": 0, "candidate_rows": 0, "conditional_logits_max_abs": 0.0,
              "verify_calls": 0, "feature_kv_max_abs": 0.0, "verify_graph_hits": 0}
    first_distribution = {}
    orig_prop, orig_verify = prop.propose_batch, mr.run_verify_forward

    @torch.inference_mode()
    def checked_prop(seqs):
        chains, got = orig_prop(seqs)
        if not first_distribution:
            first_distribution["q"] = got[0, 0].detach().clone()
        anchors, lengths, tables, bucket, _ = prop._inputs(seqs)
        hidden = mr.dspark_model.forward_hidden(anchors, lengths, tables, bucket)
        base = F.linear(hidden, mr.dspark_model.lm_head_weight)
        previous = anchors
        for j in range(7):
            want = base[:, j] + mr.dspark_model.markov_head(previous)
            error = float((want.float() - got[:, j].float()).abs().max())
            checks["conditional_logits_max_abs"] = max(checks["conditional_logits_max_abs"], error)
            previous = torch.tensor([chain[j] for chain in chains], device=anchors.device)
        checks["proposal_calls"] += 1
        checks["candidate_rows"] += len(seqs)
        return chains, got

    @torch.inference_mode()
    def checked_verify(input_ids, positions):
        prior = mr.verify_graph_hits
        logits = orig_verify(input_ids, positions)
        if "p" not in first_distribution:
            first_distribution["p"] = logits[0].detach().clone()
        if mr.verify_graph_hits > prior:
            # Snapshot values written by replay, independently recompute features
            # in eager mode for the same positions and compare projected KV.
            slots = mr._verify_extra["slot_mapping"].long()
            before = mr.dspark_model.cache[:, :, slots].clone()
            _, features = mr.model(input_ids, positions, mr.dspark_target_layers)
            mr.dspark_model.write_context(features, positions, slots)
            after = mr.dspark_model.cache[:, :, slots]
            error = float((before.float() - after.float()).abs().max())
            checks["feature_kv_max_abs"] = max(checks["feature_kv_max_abs"], error)
            checks["verify_graph_hits"] += 1
        checks["verify_calls"] += 1
        return logits

    prop.propose_batch = checked_prop
    mr.run_verify_forward = checked_verify
    prompts = [llm.tokenizer.encode(f"Explain a different method for finding prime numbers, example {i}.\n")
               for i in range(a.batch)]
    outputs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=24, ignore_eos=True),
                           use_tqdm=False)
    if a.batch == 1:
        # Empirical first-token distribution using *actual* target/draft logits
        # and the production rejection sampler. Group top-16 target tokens and
        # all remaining tokens; no large [N,vocab] tensor is kept resident.
        torch.manual_seed(20261101)
        q = torch.softmax(first_distribution["q"].float(), -1)
        p = torch.softmax(first_distribution["p"].float(), -1)
        proposals = torch.multinomial(q, 8192, replacement=True)
        delivered = []
        for chunk in proposals.split(32):
            n = chunk.numel()
            qlog = first_distribution["q"].view(1, 1, -1).expand(n, 1, -1)
            plog = first_distribution["p"].view(1, 1, -1).expand(n, 2, -1)
            result = verify_batch(qlog, plog, chunk.view(n, 1))
            delivered.append(torch.where(result.accepted > 0, chunk, result.bonus))
        samples = torch.cat(delivered)
        top = torch.topk(p, 16).indices
        empirical = torch.stack([(samples == tok).float().mean() for tok in top])
        expected = p[top]
        empirical = torch.cat((empirical, (1 - empirical.sum()).view(1)))
        expected = torch.cat((expected, (1 - expected.sum()).view(1)))
        checks["distribution_samples"] = samples.numel()
        checks["distribution_top16_plus_other_tv"] = float((empirical - expected).abs().sum() / 2)
        checks["distribution_max_bin_error"] = float((empirical - expected).abs().max())
    torch.cuda.synchronize()
    checks.update(batch=a.batch, output_lengths=[len(x["token_ids"]) for x in outputs],
                  draft_graph_hits=prop.graph_hits, draft_graph_captures=prop.graph_captures,
                  draft_graph_fallbacks=prop.graph_fallbacks,
                  verify_graph_fallbacks=mr.verify_graph_fallbacks,
                  kv_blocks=mr.config.num_kvcache_blocks,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    checks["pass"] = (checks["output_lengths"] == [24] * a.batch
                      and checks["proposal_calls"] > 0 and prop.graph_hits > 0
                      and checks["verify_graph_hits"] > 0
                      and checks["conditional_logits_max_abs"] < 0.1
                      and checks["feature_kv_max_abs"] < 0.1
                      and (a.batch != 1 or checks["distribution_top16_plus_other_tv"] < 0.06))
    Path(a.out).write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)
    llm.exit()
    import atexit
    atexit.unregister(llm.exit)
    if not checks["pass"]:
        raise SystemExit("integrated engine smoke failed")


if __name__ == "__main__":
    main()
