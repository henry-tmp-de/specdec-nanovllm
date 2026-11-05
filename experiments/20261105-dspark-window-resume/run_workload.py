"""Construct fixed-token panels from local datasets and WikiText or an old manifest.

Both are synthetic throughput workloads, not natural long questions or quality
benchmarks.  The tokenizer mode reads local files on CPU; no network or GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def build(source, ctx):
    old_ctx = source["context_tokens"]
    if ctx < old_ctx:
        raise ValueError("new context cannot be shorter than the source")
    result = {"context_tokens": ctx, "panels": {},
              "construction": {"method": "prepend concatenated source WikiText token IDs",
                               "source_context_tokens": old_ctx,
                               "added_tokens_per_request": ctx - old_ctx,
                               "interpretation": "synthetic fixed-length workload; not a natural long task"}}
    for panel in ("code", "math", "general"):
        rows = source["panels"][panel]
        if not rows:
            raise ValueError(f"empty panel: {panel}")
        filler = []
        for row in rows:
            ids, meta = row["token_ids"], row["metadata"]
            n = meta["wiki_tokens"]
            if (len(ids) != old_ctx or not isinstance(n, int) or not 0 <= n <= old_ctx
                    or meta.get("task_truncated_from_left") or meta["task_tokens"] != old_ctx - n
                    or any(type(x) is not int or x < 0 for x in ids)
                    or token_hash(ids) != meta["input_sha256"]):
                raise ValueError(f"invalid source row in {panel}")
            filler.extend(ids[:n])
        if not filler:
            raise ValueError(f"source contains no WikiText filler in {panel}")
        new_rows = []
        extra = ctx - old_ctx
        for i, row in enumerate(rows):
            offset = (i * max(1, extra)) % len(filler)
            # Repetition (if any) and rotation are explicit in metadata.
            prefix = [filler[(offset + j) % len(filler)] for j in range(extra)]
            ids = prefix + row["token_ids"]
            meta = dict(row["metadata"])
            meta.update(input_sha256=token_hash(ids), source_input_sha256=meta["input_sha256"],
                        wiki_tokens=meta["wiki_tokens"] + extra,
                        wiki_fraction=(meta["wiki_tokens"] + extra) / ctx,
                        added_wiki_tokens=extra, filler_pool_tokens=len(filler), filler_offset=offset,
                        filler_wrap_count=(offset + extra - 1) // len(filler) if extra else 0,
                        synthetic_context=True)
            new_rows.append({"token_ids": ids, "metadata": meta})
        if len({r["metadata"]["input_sha256"] for r in new_rows}) != len(new_rows):
            raise ValueError(f"duplicated constructed prompts in {panel}")
        result["panels"][panel] = new_rows
    return result


def build_from_local(tokenizer_dir, dataset_dir, wiki_path, ctx, batch):
    """Use the historical task/chat formatting and non-overlapping wiki slices."""
    from transformers import AutoTokenizer
    prior_path = Path(__file__).resolve().parents[1] / "20261101-dspark-integration" / "bench.py"
    spec = importlib.util.spec_from_file_location("historical_bench", prior_path)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    # Some Transformers tokenizers rebuild their vocabulary on __len__;
    # calling it once per token makes the 4K manifest prohibitively slow.
    vocab_size = len(tokenizer)
    wiki = prior.load_wiki(tokenizer, wiki_path, ctx, batch)
    if len(wiki) < (batch + 1) * ctx:
        raise ValueError("local WikiText does not contain enough token IDs for distinct slices")
    result = {"context_tokens": ctx, "panels": {},
              "construction": {"method": "non-overlapping contiguous WikiText token slices plus task suffix",
                               "source": "local DeepSpec datasets and WikiText",
                               "tokenizer_path": str(tokenizer_dir), "wiki_path": str(wiki_path),
                               "datasets_path": str(dataset_dir),
                               "wiki_sha256": hashlib.sha256(Path(wiki_path).read_bytes()).hexdigest(),
                               "interpretation": "synthetic fixed-length workload; not a natural long task"}}
    for panel in ("code", "math", "general"):
        prompts, meta = prior.build_inputs(tokenizer, dataset_dir, wiki_path, ctx, batch, panel, wiki)
        result["panels"][panel] = []
        for i, (ids, item) in enumerate(zip(prompts, meta)):
            if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
                raise ValueError(f"invalid token ID in {panel} row {i}")
            item.update(wiki_fraction=item["wiki_tokens"] / ctx,
                        wiki_slice_start=i * ctx, wiki_slice_end=i * ctx + item["wiki_tokens"],
                        synthetic_context=True)
            result["panels"][panel].append({"token_ids": ids, "metadata": item})
        if len({r["metadata"]["input_sha256"] for r in result["panels"][panel]}) != batch:
            raise ValueError(f"duplicated constructed prompts in {panel}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="原 1024-token workload-manifest.json（仅 smoke / CPU）")
    source.add_argument("--datasets", help="本地 DeepSpec eval_datasets 目录（正式 workload）")
    p.add_argument("--tokenizer", help="本地目标模型 tokenizer 目录，配合 --datasets")
    p.add_argument("--wiki", help="本地 WikiText 文件，配合 --datasets")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--ctx", type=int, required=True, help="目标长度，例如 4096 或容量实验的 16384")
    p.add_argument("--out", required=True, help="新文件，不覆盖")
    a = p.parse_args()
    if a.ctx <= 0 or a.rows <= 0:
        p.error("ctx 和 rows 必须为正")
    if a.datasets:
        if not a.wiki or not a.tokenizer:
            p.error("--datasets 需要 --wiki 和 --tokenizer")
        result = build_from_local(a.tokenizer, a.datasets, a.wiki, a.ctx, a.rows)
    else:
        if a.wiki or a.tokenizer:
            p.error("--source 不接受 --wiki 或 --tokenizer")
        source_manifest = json.loads(Path(a.source).read_text(encoding="utf-8"))
        result = build(source_manifest, a.ctx)
    with Path(a.out).open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    summary = {panel: {"requests": len(rows),
                       "wiki_fraction_min": min(r["metadata"]["wiki_fraction"] for r in rows),
                       "wiki_fraction_max": max(r["metadata"]["wiki_fraction"] for r in rows),
                       "max_filler_wraps": max(r["metadata"].get("filler_wrap_count", 0) for r in rows),
                       "task_truncated": sum(r["metadata"]["task_truncated_from_left"] for r in rows)}
               for panel, rows in result["panels"].items()}
    print("已生成合成上下文：" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
