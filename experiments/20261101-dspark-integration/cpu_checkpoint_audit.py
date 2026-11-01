"""Validate released checkpoint shape and its frozen target weights without CUDA."""
from __future__ import annotations

import argparse
from glob import glob
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm.spec_decode.dspark_model import DSparkModel, validate_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = AutoConfig.from_pretrained(a.draft, local_files_only=True)
    tc = AutoConfig.from_pretrained(a.target, local_files_only=True)
    validate_config(c, tc, 7)
    with torch.device("meta"):
        embedding = torch.empty(c.vocab_size, c.hidden_size)
        lm_head = torch.empty(c.vocab_size, c.hidden_size)
        m = DSparkModel(c, embedding, lm_head)
    expected = {k: list(v.shape) for k, v in m.named_parameters()}
    expected.update({"embed_tokens.weight": list(embedding.shape), "lm_head.weight": list(lm_head.shape)})
    actual = {}
    draft_files = sorted(glob(str(Path(a.draft) / "*.safetensors")))
    if len(draft_files) != 1:
        raise ValueError("expected exactly one released checkpoint file")
    target_index = json.loads((Path(a.target) / "model.safetensors.index.json").read_text())
    target_file = Path(a.target) / target_index["weight_map"]["model.embed_tokens.weight"]
    equal = {"embed_tokens.weight": True, "lm_head.weight": True}
    with safe_open(draft_files[0], framework="pt", device="cpu") as draft, \
            safe_open(str(target_file), framework="pt", device="cpu") as target:
        for k in draft.keys():
            actual[k] = list(draft.get_slice(k).get_shape())
        target_embed = target.get_slice("model.embed_tokens.weight")
        for key in equal:
            value = draft.get_slice(key)
            for start in range(0, c.vocab_size, 4096):
                end = min(c.vocab_size, start + 4096)
                if not torch.equal(value[start:end], target_embed[start:end]):
                    equal[key] = False
                    break
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = {k: [expected[k], actual[k]] for k in set(expected) & set(actual) if expected[k] != actual[k]}
    result = {"draft": a.draft, "target": a.target, "checkpoint_keys": len(actual),
              "missing": missing, "unexpected": extra, "wrong_shapes": wrong,
              "target_ties_embedding": bool(tc.tie_word_embeddings),
              "shared_target_exact": equal}
    result["pass"] = not missing and not extra and not wrong and all(equal.values())
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("checkpoint audit failed")


if __name__ == "__main__":
    main()
