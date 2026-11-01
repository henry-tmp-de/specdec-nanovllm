"""Compare real-target-feature DSpark proposals against official DeepSpec."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    llm = LLM(a.target, max_num_seqs=2, max_model_len=1536, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.65, spec_method="dspark", spec_k=7,
              draft_model=a.draft, spec_graph_bs=(1, 2), enforce_eager=False)
    mr, proposer = llm.model_runner, llm.model_runner.spec_proposer
    cfg = AutoConfig.from_pretrained(a.draft, local_files_only=True)
    ref = Qwen3DSparkModel.from_pretrained(a.draft, config=cfg, dtype=torch.bfloat16,
                                           attn_implementation="sdpa", local_files_only=True,
                                           device_map={"": 0}).eval()
    features = {}
    observations = []
    orig_prefill, orig_propose = mr._prefill_forward, proposer.propose_batch

    @torch.inference_mode()
    def prefill(seqs):
        result = orig_prefill(seqs)
        if any(seq.num_cached_tokens for seq in seqs):
            raise RuntimeError("reference prefill must not use cached prefix")
        flat = [token for seq in seqs for token in seq.token_ids]
        positions = [i for seq in seqs for i in range(len(seq))]
        ids = torch.tensor(flat, device="cuda", dtype=torch.long)
        pos = torch.tensor(positions, device="cuda", dtype=torch.long)
        _, captured = mr.model(ids, pos, mr.dspark_target_layers)
        offset = 0
        for seq in seqs:
            features[seq.seq_id] = captured[offset:offset + len(seq)].clone()
            offset += len(seq)
        return result

    @torch.inference_mode()
    def propose(seqs):
        chains, logits = orig_propose(seqs)
        if observations:
            return chains, logits
        for row, seq in enumerate(seqs):
            history = features[seq.seq_id]
            if history.shape[0] != len(seq) - 1:
                raise RuntimeError("target features must stop before anchor")
            length = history.shape[0]
            ids = torch.full((1, 7), cfg.mask_token_id, device="cuda", dtype=torch.long)
            ids[0, 0] = seq.last_token
            pos = torch.arange(length + 7, device="cuda").unsqueeze(0)
            hidden = ref._forward_backbone(position_ids=pos, noise_embedding=ref.embed_tokens(ids),
                                            target_hidden_states=history.unsqueeze(0),
                                            attention_mask=None, is_causal=False)
            base = ref.compute_logits(hidden)
            previous = torch.tensor([seq.last_token], device="cuda")
            max_kl, max_abs, top1 = 0.0, 0.0, 0
            for j, sampled in enumerate(chains[row]):
                expected = ref.markov_head.apply_step_logits(base[:, j], token_ids=previous,
                                                             hidden_states=None).float()
                actual = logits[row:row + 1, j].float()
                p, q = torch.log_softmax(expected, -1), torch.log_softmax(actual, -1)
                max_kl = max(max_kl, float((p.exp() * (p - q)).sum()))
                max_abs = max(max_abs, float((expected - actual).abs().max()))
                top1 += int(expected.argmax(-1) == actual.argmax(-1))
                previous = torch.tensor([sampled], device="cuda")
            observations.append({"context": int(length), "max_conditional_kl": max_kl,
                                 "max_logit_abs": max_abs, "top1_matches": top1, "positions": 7})
        return chains, logits

    mr._prefill_forward = prefill
    proposer.propose_batch = propose
    manifest = json.loads(Path(a.manifest).read_text())
    prompts = [r["token_ids"] for r in manifest["panels"]["code"][:2]]
    outputs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True),
                           use_tqdm=False)
    torch.cuda.synchronize()
    report = {"batch": 2, "observations": observations,
              "output_lengths": [len(x["token_ids"]) for x in outputs],
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    report["pass"] = (len(observations) == 2 and all(x["context"] == 1024 and
                      x["max_conditional_kl"] < 0.02 for x in observations)
                      and report["output_lengths"] == [8, 8])
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    llm.exit()
    import atexit
    atexit.unregister(llm.exit)
    if not report["pass"]:
        raise SystemExit("end-to-end reference failed")


if __name__ == "__main__":
    main()
