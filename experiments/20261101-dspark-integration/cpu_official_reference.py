"""Compare batched inference against the official DeepSpec model on CPU.

Run with PYTHONPATH pointing at a DeepSpec checkout; no checkpoint or GPU is
needed. This deliberately uses synthetic, shared weights, so it establishes
operator/position alignment, not correctness of a released checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen3Config
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel


def ours_module():
    if "_dspark_ours" in sys.modules:
        return sys.modules["_dspark_ours"]
    path = Path(__file__).resolve().parents[2] / "nanovllm/spec_decode/dspark_model.py"
    spec = importlib.util.spec_from_file_location("_dspark_ours", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(dtype):
    torch.manual_seed(55)
    c = Qwen3Config(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                    vocab_size=31, attention_bias=False, max_position_embeddings=64,
                    rms_norm_eps=1e-6, tie_word_embeddings=False)
    c.target_layer_ids = [0, 1]
    c.mask_token_id = 3
    c.block_size = 7
    c.num_anchors = 1
    c.markov_rank = 4
    c.markov_head_type = "vanilla"
    c.enable_confidence_head = True
    c.confidence_head_with_markov = True
    c._attn_implementation = "sdpa"
    ref = Qwen3DSparkModel(c).eval().to(dtype=dtype)
    impl = ours_module().DSparkModel(c, ref.embed_tokens.weight, ref.lm_head.weight).eval().to(dtype=dtype)
    state = {k: v for k, v in ref.state_dict().items()
             if k not in ("embed_tokens.weight", "lm_head.weight")}
    impl.load_state_dict(state, strict=True)
    impl.bind_cache(4, 4)
    feature = torch.randn(2, 5, 32).to(dtype)
    lens = [3, 5]
    pos = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    slots = torch.tensor([0, 1, 2, 8, 9, 10, 11, 12])
    selected = torch.cat((feature[0, :3], feature[1, :5]), dim=0)
    impl.write_context(selected, pos, slots)
    anchors = torch.tensor([4, 5])
    tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    actual = impl.forward_hidden(anchors, torch.tensor(lens), tables, 8)
    expected = []
    for i, length in enumerate(lens):
        noise_ids = torch.full((1, 7), c.mask_token_id, dtype=torch.long)
        noise_ids[0, 0] = anchors[i]
        positions = torch.arange(length + 7).unsqueeze(0)
        with torch.no_grad():
            row = ref._forward_backbone(position_ids=positions,
                                        noise_embedding=ref.embed_tokens(noise_ids),
                                        target_hidden_states=feature[i:i+1, :length],
                                        attention_mask=None, is_causal=False)
        expected.append(row)
    expected = torch.cat(expected)
    tolerance = 5e-5 if dtype == torch.float32 else 0.04
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tolerance, atol=tolerance)
    torch.manual_seed(333)
    sampled, corrected = impl.propose(anchors, torch.tensor(lens), tables, 8, torch.tensor([0.7, 1.0]))
    base = ref.compute_logits(expected)
    previous = anchors
    max_logits = 0.0
    for j in range(7):
        official_logits = ref.markov_head.apply_step_logits(base[:, j], token_ids=previous, hidden_states=None)
        diff = (corrected[:, j].float() - official_logits.float()).abs().max().item()
        max_logits = max(max_logits, diff)
        torch.testing.assert_close(corrected[:, j].float(), official_logits.float(),
                                   rtol=tolerance, atol=tolerance)
        previous = sampled[:, j]
    return {"dtype": str(dtype), "max_hidden_abs": (actual.float() - expected.float()).abs().max().item(),
            "max_conditional_logits_abs": max_logits, "batch": 2, "contexts": lens,
            "block": 7, "candidate_count": int(sampled.shape[1])}


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps([run(torch.float32), run(torch.bfloat16)], ensure_ascii=False, indent=2))
