"""Forced accept lengths are diagnostic only; production sampling is unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm import LLM, SamplingParams
import nanovllm.spec_decode.verify as verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    llm = LLM(a.target, max_num_seqs=1, max_model_len=512, max_num_batched_tokens=512,
              gpu_memory_utilization=0.70, spec_method="dspark", spec_k=7,
              draft_model=a.draft, spec_graph_bs=(1,), enforce_eager=False)
    mr, prop, model = llm.model_runner, llm.model_runner.spec_proposer, llm.model_runner.dspark_model
    history = {"features": None, "verify_features": None, "forced": [], "checks": []}
    prefill_orig, forward_orig, proposal_orig, verify_orig = (mr._prefill_forward,
        mr.run_verify_forward, prop.propose_batch, verify.verify_batch)

    @torch.inference_mode()
    def prefill(seqs):
        output = prefill_orig(seqs)
        # Context remains installed until run() samples and resets it.
        from nanovllm.utils.context import get_context
        # Recreate the same target inputs because _prefill_forward only returns logits.
        ids = torch.tensor(seqs[0].token_ids, device="cuda", dtype=torch.long)
        positions = torch.arange(ids.numel(), device="cuda")
        _, features = mr.model(ids, positions, mr.dspark_target_layers)
        history["features"] = features.clone()
        return output

    @torch.inference_mode()
    def target_forward(ids, positions):
        result = forward_orig(ids, positions)
        _, features = mr.model(ids, positions, mr.dspark_target_layers)
        history["verify_features"] = features.clone()
        return result

    @torch.inference_mode()
    def proposal(seqs):
        chains, logits = proposal_orig(seqs)
        seq = seqs[0]
        context = history["features"]
        if context.shape[0] != len(seq) - 1:
            raise AssertionError("confirmed target features do not match anchor position")
        actual_cache, actual_sentinel = model.cache, model.sentinel
        nslots = ((len(seq) + mr.block_size - 1) // mr.block_size) * mr.block_size
        c = model.cfg
        model.cache = torch.zeros(2, c.num_hidden_layers, nslots + 1, c.num_key_value_heads,
                                  c.head_dim, device="cuda", dtype=model.fc.weight.dtype)
        model.sentinel = nslots
        try:
            positions = torch.arange(context.shape[0], device="cuda")
            model.write_context(context, positions, positions)
            anchors = torch.tensor([seq.last_token], device="cuda")
            lengths = torch.tensor([len(seq) - 1], device="cuda")
            blocks = torch.arange(nslots // mr.block_size, device="cuda", dtype=torch.int32)[None]
            hidden = model.forward_hidden(anchors, lengths, blocks, nslots)
            base = F.linear(hidden, model.lm_head_weight)
            previous = anchors
            max_kl = 0.0
            for i, token in enumerate(chains[0]):
                want = base[:, i] + model.markov_head(previous)
                p = torch.log_softmax(want.float(), -1)
                q = torch.log_softmax(logits[0:1, i].float(), -1)
                max_kl = max(max_kl, float((p.exp() * (p - q)).sum()))
                previous = torch.tensor([token], device="cuda")
            history["checks"].append({"confirmed_context": int(context.shape[0]),
                                      "max_conditional_kl": max_kl})
        finally:
            model.cache, model.sentinel = actual_cache, actual_sentinel
        return chains, logits

    @torch.inference_mode()
    def forced(q, p, candidates, temperatures=None, draft_is_point_mass=False):
        # Cover all-rejected, partial acceptance and full acceptance in that
        # order; these are NOT samples for measuring output distribution.
        schedule = (0, 3, 7, 0)
        n = min(schedule[len(history["forced"]) % len(schedule)], candidates.shape[1])
        history["forced"].append(n)
        history["features"] = torch.cat((history["features"], history["verify_features"][:n + 1]), dim=0)
        accepted = torch.tensor([n], device=candidates.device)
        mask = torch.arange(candidates.shape[1], device=candidates.device)[None] < n
        bonus = p[:, n].argmax(-1)
        return verify.VerifyResult(accepted=accepted, bonus=bonus,
                                   n_proposed=torch.tensor([candidates.shape[1]], device=candidates.device),
                                   accept_mask=mask)

    mr._prefill_forward, mr.run_verify_forward = prefill, target_forward
    prop.propose_batch, verify.verify_batch = proposal, forced
    tokens = llm.tokenizer.encode(Path(a.wiki).read_text(encoding="utf-8"), add_special_tokens=False)[100:164]
    try:
        output = llm.generate([tokens], SamplingParams(temperature=1.0, max_tokens=24, ignore_eos=True),
                              use_tqdm=False)
    finally:
        verify.verify_batch = verify_orig
    torch.cuda.synchronize()
    report = {"forced_accept_lengths": history["forced"], "checks": history["checks"],
              "output_length": len(output[0]["token_ids"]), "diagnostic_forced_acceptance": True}
    report["pass"] = (history["forced"][:3] == [0, 3, 7] and len(history["checks"]) >= 4
                      and max(x["max_conditional_kl"] for x in history["checks"]) < 0.01
                      and report["output_length"] == 24)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    llm.exit()
    import atexit
    atexit.unregister(llm.exit)
    if not report["pass"]:
        raise SystemExit("rollback smoke failed")


if __name__ == "__main__":
    main()
