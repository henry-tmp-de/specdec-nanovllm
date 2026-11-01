"""Validate the actual tokenizer and all three panel input contracts on CPU."""
import argparse
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import build_inputs, load_wiki


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True, use_fast=True)
    vocab_size = len(tokenizer)
    report = {}
    manifest = {"context_tokens": 1024, "panels": {}}
    wiki = load_wiki(tokenizer, args.wiki, 1024, 16)
    for panel in ("code", "math", "general"):
        prompts, meta = build_inputs(tokenizer, args.datasets, args.wiki, 1024, 16, panel, wiki)
        assert all(len(x) == 1024 and all(isinstance(y, int) and 0 <= y < vocab_size for y in x)
                   for x in prompts)
        assert len({m["input_sha256"] for m in meta}) == 16
        report[panel] = {"requests": len(prompts), "lengths": sorted(set(map(len, prompts))),
                         "wiki_tokens": [m["wiki_tokens"] for m in meta],
                         "task_truncated": sum(m["task_truncated_from_left"] for m in meta),
                         "first_input_sha256": meta[0]["input_sha256"]}
        manifest["panels"][panel] = [{"token_ids": ids, "metadata": m} for ids, m in zip(prompts, meta)]
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
