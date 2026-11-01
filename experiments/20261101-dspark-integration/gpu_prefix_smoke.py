"""Check shared physical prefix slots and next logits across cache reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm import LLM, SamplingParams


def kl(a, b):
    lp, lq = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    return float((lp.exp() * (lp - lq)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    llm = LLM(a.target, max_num_seqs=1, max_model_len=1024, max_num_batched_tokens=1024,
              gpu_memory_utilization=0.70, spec_method="dspark", spec_k=7,
              draft_model=a.draft, spec_graph_bs=(1,), enforce_eager=False)
    mr, prop = llm.model_runner, llm.model_runner.spec_proposer
    tokens = llm.tokenizer.encode(Path(a.wiki).read_text(encoding="utf-8"), add_special_tokens=False)
    warm, prompt = tokens[:512], tokens[2048:2560]
    if len(prompt) != 512:
        raise ValueError("WikiText-2 too short")
    phase = {"name": "warm"}
    stats = {}
    original_prefill = mr._prefill_forward
    original_propose = prop.propose_batch

    @torch.inference_mode()
    def prefill(seqs):
        label = phase["name"]
        cached = int(seqs[0].num_cached_tokens)
        if label != "warm":
            stats[label] = {"cached_tokens": cached}
            if cached:
                blocks = seqs[0].block_table[:cached // mr.block_size]
                indices = torch.cat([torch.arange(b * mr.block_size, (b + 1) * mr.block_size,
                                                  device="cuda") for b in blocks])
                stats[label]["reused_kv"] = mr.dspark_model.cache[:, :, indices].clone()
        logits, temperature = original_prefill(seqs)
        if label != "warm":
            stats[label]["target_logits"] = logits[0].clone()
            if not cached:
                b = seqs[0].block_table[0]
                idx = torch.arange(b * mr.block_size, (b + 1) * mr.block_size, device="cuda")
                stats[label]["written_kv"] = mr.dspark_model.cache[:, :, idx].clone()
        return logits, temperature

    @torch.inference_mode()
    def propose(seqs):
        chains, logits = original_propose(seqs)
        label = phase["name"]
        if label in ("fresh", "cached") and "draft_logits" not in stats[label]:
            stats[label]["anchor"] = seqs[0].last_token
            stats[label]["draft_logits"] = logits[0, 0].clone()
        return chains, logits

    mr._prefill_forward = prefill
    prop.propose_batch = propose
    sp = SamplingParams(temperature=1.0, max_tokens=16, ignore_eos=True)
    llm.generate([warm], sp, use_tqdm=False)
    # All requests have finished; prevent the warmup affecting the fresh run.
    llm.scheduler.block_manager.hash_to_block_id.clear()
    for label in ("fresh", "cached"):
        phase["name"] = label
        torch.manual_seed(20261101)
        torch.cuda.manual_seed_all(20261101)
        llm.generate([prompt], sp, use_tqdm=False)
    torch.cuda.synchronize()
    fresh, cached = stats["fresh"], stats["cached"]
    same_anchor = fresh["anchor"] == cached["anchor"]
    result = {"fresh_cached_tokens": fresh["cached_tokens"],
              "reused_cached_tokens": cached["cached_tokens"],
              "shared_prefix_kv_bit_equal": torch.equal(fresh["written_kv"], cached["reused_kv"]),
              "prefill_target_logit_kl": kl(fresh["target_logits"], cached["target_logits"]),
              "same_first_anchor": same_anchor,
              "draft_first_step_kl": kl(fresh["draft_logits"], cached["draft_logits"]) if same_anchor else None,
              "draft_graph_hits": prop.graph_hits, "verify_graph_hits": mr.verify_graph_hits}
    result["pass"] = (result["fresh_cached_tokens"] == 0 and result["reused_cached_tokens"] >= 256
                      and result["shared_prefix_kv_bit_equal"] and same_anchor
                      and result["prefill_target_logit_kl"] < 0.02
                      and result["draft_first_step_kl"] < 0.02)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    llm.exit()
    import atexit
    atexit.unregister(llm.exit)
    if not result["pass"]:
        raise SystemExit("prefix cache smoke failed")


if __name__ == "__main__":
    main()
