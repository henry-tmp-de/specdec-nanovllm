"""Real checkpoint DSpark vs official model, on one explicitly selected GPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm.spec_decode.dspark_model import DSparkModel, load_official_weights


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(881)
    c = AutoConfig.from_pretrained(args.draft, local_files_only=True)
    ref = Qwen3DSparkModel.from_pretrained(args.draft, config=c, dtype=torch.bfloat16,
                                           attn_implementation="sdpa", local_files_only=True,
                                           device_map={"": 0}).eval()
    ours = DSparkModel(c, ref.embed_tokens.weight, ref.lm_head.weight).to(device="cuda", dtype=torch.bfloat16)
    # nn.Module.to(dtype=...) also casts buffers; production constructs this
    # RoPE buffer in FP32, so reconstruct it at the production precision.
    theta = (getattr(c, "rope_parameters", None) or {}).get("rope_theta", 1_000_000)
    ours.inv_freq = 1.0 / (theta ** (torch.arange(0, c.head_dim, 2, device="cuda",
                                                  dtype=torch.float32) / c.head_dim))
    load_official_weights(ours, args.draft)
    ours.bind_cache(8, 16)
    lengths = [31, 47]
    feature = torch.randn(2, 47, len(c.target_layer_ids) * c.hidden_size,
                          device="cuda", dtype=torch.bfloat16)
    pos = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    slots = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    slots[lengths[0]:] += 3 * 16
    selected = torch.cat((feature[0, :lengths[0]], feature[1, :lengths[1]]), dim=0)
    ours.write_context(selected, pos, slots)
    anchor = torch.tensor([444, 555], device="cuda")
    lens = torch.tensor(lengths, device="cuda")
    tables = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
    actual = ours.forward_hidden(anchor, lens, tables, 48)
    rows = []
    for i, n in enumerate(lengths):
        ids = torch.full((1, 7), c.mask_token_id, device="cuda", dtype=torch.long)
        ids[0, 0] = anchor[i]
        row = ref._forward_backbone(position_ids=torch.arange(n + 7, device="cuda").unsqueeze(0),
                                    noise_embedding=ref.embed_tokens(ids),
                                    target_hidden_states=feature[i:i+1, :n],
                                    attention_mask=None, is_causal=False)
        rows.append(row)
    expected = torch.cat(rows)
    hidden_diff = (actual.float() - expected.float())
    hidden_relative_rms = float(hidden_diff.square().mean().sqrt() /
                                expected.float().square().mean().sqrt().clamp_min(1e-12))
    sampled, logits = ours.propose(anchor, lens, tables, 48, torch.tensor([0.8, 1.0], device="cuda"))
    base = ref.compute_logits(expected)
    previous = anchor
    max_kl, max_logit_abs, matching_top1 = 0.0, 0.0, 0
    for j in range(7):
        true = ref.markov_head.apply_step_logits(base[:, j], token_ids=previous, hidden_states=None).float()
        got = logits[:, j].float()
        temps = torch.tensor([0.8, 1.0], device="cuda")[:, None]
        lp, lq = torch.log_softmax(true / temps, -1), torch.log_softmax(got / temps, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        max_kl = max(max_kl, float(kl.max()))
        max_logit_abs = max(max_logit_abs, float((true - got).abs().max()))
        matching_top1 += int((true.argmax(-1) == got.argmax(-1)).sum())
        previous = sampled[:, j]
    # Check that CUDA graph replay consumes fresh features rather than the data
    # present during capture. This captures the same cache write operations as
    # the target graph and uses a sentinel for padding writes.
    sf, sp, ss = selected.clone(), pos.clone(), slots.clone()
    ours.write_context(sf, sp, ss)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.write_context(sf, sp, ss)
        graph_hidden = ours.forward_hidden(anchor, lens, tables, 48)
    sf.copy_(sf * 0.75)
    graph.replay()
    graph_value = graph_hidden.clone()
    ours.write_context(sf, sp, ss)
    eager_value = ours.forward_hidden(anchor, lens, tables, 48)
    graph_max_abs = float((graph_value.float() - eager_value.float()).abs().max())
    torch.cuda.synchronize()
    result = {"checkpoint": args.draft, "dtype": "bfloat16", "batch": 2,
              "contexts": lengths, "block": 7, "candidate_count": sampled.shape[1],
              "hidden_max_abs": float(hidden_diff.abs().max()),
              "hidden_relative_rms": hidden_relative_rms,
              "conditional_logits_max_abs": max_logit_abs, "max_kl_p_to_q": max_kl,
              "top1_matching_rows": matching_top1, "top1_total_rows": 14,
              "fresh_feature_graph_max_abs": graph_max_abs,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    result["pass"] = (hidden_relative_rms < 0.03 and max_kl < 0.01 and graph_max_abs == 0.0
                      and torch.isfinite(logits).all().item())
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit("official checkpoint numerical reference failed")


if __name__ == "__main__":
    main()
