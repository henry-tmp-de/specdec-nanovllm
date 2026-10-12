#!/usr/bin/env python
"""slide_equiv.py —— 一档配置的 out_md5 / logits_md5 对拍（同脚本指向不同树）。

    CTX=1024 WINDOW=0 B=1 OUTLEN=64 python -u scripts/slide_equiv.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")


def main():
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from acceptance_matrix import build_prompts

    ctx = int(os.environ.get("CTX", "1024"))
    window = int(os.environ.get("WINDOW", "0"))
    head = int(os.environ.get("HEAD", "0"))
    align = os.environ.get("ALIGN", "0") == "1"
    B = int(os.environ.get("B", "1"))
    OUTLEN = int(os.environ.get("OUTLEN", "64"))
    K = int(os.environ.get("K", "6"))
    MAXLEN = int(os.environ.get("MAXLEN", "16640"))
    MAX_SEQS = int(os.environ.get("MAX_SEQS", "1"))
    TAG = os.environ.get("TAG", "eq")

    kw = dict(enforce_eager=False, max_model_len=MAXLEN,
              max_num_batched_tokens=max(MAXLEN, 16384), max_num_seqs=MAX_SEQS,
              spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if window:
        kw["spec_draft_window"] = window
    if head:
        kw["spec_draft_head_blocks"] = head
    if align:
        kw["spec_draft_win_align"] = True
    tok = AutoTokenizer.from_pretrained(TARGET)
    prompts = build_prompts(tok, ctx, B)
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)

    llm = LLM(TARGET, **kw)
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=4,
                                             ignore_eos=True), use_tqdm=False)
    res = llm.generate(prompts, sp, use_tqdm=False)
    h = hashlib.md5()
    for x in res:
        h.update(",".join(map(str, x["token_ids"])).encode())
        h.update(b"|")

    print("@@EQ@@" + json.dumps(dict(tag=TAG, ctx=ctx, B=B, window=window,
                                     head=head, align=align, k=K, outlen=OUTLEN,
                                     out_md5=h.hexdigest()), ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
