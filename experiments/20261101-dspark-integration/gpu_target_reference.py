"""Compare native target selected layer outputs with HF Qwen3, on GPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    llm = LLM(a.target, max_num_seqs=2, max_model_len=512, max_num_batched_tokens=512,
              gpu_memory_utilization=0.45, enforce_eager=True)
    mr = llm.model_runner
    ids = [llm.tokenizer.encode("Explain how a hash table handles collisions. " * 2),
           llm.tokenizer.encode("Find the sum of the integers from one to twenty. " * 2)]
    seqs = []
    for i, tokens in enumerate(ids):
        seq = Sequence(tokens, SamplingParams(max_tokens=8))
        seq.num_scheduled_tokens = len(tokens)
        seq.block_table = [i]
        seqs.append(seq)
    inputs, positions = mr.prepare_prefill(seqs, advance_draft=False)
    native_hidden, native_features = mr.model(inputs, positions, (1, 9, 17, 25, 33))
    native_logits = mr.model.compute_logits(native_hidden)
    reset_context()
    ref = AutoModelForCausalLM.from_pretrained(a.target, dtype=torch.bfloat16,
                                               attn_implementation="sdpa", local_files_only=True,
                                               device_map={"": 0}).eval()
    offset = 0
    layer_errors, logit_kls, top1 = [], [], []
    for i, tokens in enumerate(ids):
        target_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        result = ref(target_ids, use_cache=False, output_hidden_states=True)
        expected = torch.cat([result.hidden_states[layer + 1][0] for layer in (1, 9, 17, 25, 33)], dim=-1)
        actual = native_features[offset:offset + len(tokens)]
        offset += len(tokens)
        dif = actual.float() - expected.float()
        layer_errors.append(float(dif.square().mean().sqrt() /
                                  expected.float().square().mean().sqrt().clamp_min(1e-12)))
        p = torch.log_softmax(result.logits[0, -1].float(), -1)
        q = torch.log_softmax(native_logits[i].float(), -1)
        logit_kls.append(float((p.exp() * (p - q)).sum()))
        top1.append(int(result.logits[0, -1].argmax()) == int(native_logits[i].argmax()))
    report = {"batch": 2, "prompt_lengths": [len(x) for x in ids],
              "selected_layers": [1, 9, 17, 25, 33], "relative_feature_rms": layer_errors,
              "target_logit_kl": logit_kls, "target_greedy_top1_matches": top1,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    report["pass"] = max(layer_errors) < 0.05 and max(logit_kls) < 0.02 and all(top1)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    llm.exit()
    import atexit
    atexit.unregister(llm.exit)
    if not report["pass"]:
        raise SystemExit("target feature alignment failed")


if __name__ == "__main__":
    main()
