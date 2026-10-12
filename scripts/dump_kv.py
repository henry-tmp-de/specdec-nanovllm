#!/usr/bin/env python
"""dump_kv.py —— 把 draft KV cache 在「窗口 prefill 刚做完、decode 还没开始」那一刻存下来。

用来验证：引擎写进环里的 KV，内容是不是就等于「按窗口那段 token 前向算出来的 KV」。
（P=0 纯滑窗的接受率远低于离线模型对同一窗口的预期，几何/块表/ctx_len 都对得上，
 所以只剩「写进去的内容不对」这一种可能。）

用法: CTX=8192 WINDOW=1280 HEAD=0 TAG=p0m5 python -u scripts/dump_kv.py
"""
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
    import nanovllm.spec_decode.draft_proposer as DP

    ctx = int(os.environ.get("CTX", "8192"))
    window = int(os.environ.get("WINDOW", "1280"))
    head = int(os.environ.get("HEAD", "0"))
    TAG = os.environ.get("TAG", "x")
    OUT = os.environ.get("OUT", f"out/eng/kv_{TAG}.pt")

    KVH = {}

    import nanovllm.engine.model_runner as MR
    orig_alloc = MR.ModelRunner.allocate_kv_cache

    def alloc(self):
        r = orig_alloc(self)
        KVH["kv"] = getattr(self, "draft_kv_cache", None)
        print(f"[kv] captured draft_kv_cache: {None if KVH['kv'] is None else tuple(KVH['kv'].shape)}",
              flush=True)
        return r

    MR.ModelRunner.allocate_kv_cache = alloc

    snap = {}
    orig_fg = DP.DraftModelProposer._forward_group

    def fg(self, tokens, positions, block_tables, context_lens, slot_mapping):
        if "v" not in snap and KVH.get("kv") is not None:
            kv = KVH["kv"]
            snap["v"] = kv[1, 0].clone().cpu()          # 层 0 的 v_cache
            snap["k"] = kv[0, 0].clone().cpu()          # 层 0 的 k_cache
            snap["n_layers"] = int(kv.shape[1])
            snap["n_blocks"] = int(kv.shape[2])
            snap["bs"] = int(kv.shape[3])
            snap["positions"] = [int(x) for x in positions]
            snap["ctx_len"] = [int(x) for x in context_lens]
            snap["block_table"] = [list(map(int, b)) for b in block_tables]
            print(f"[kv] 快照于第一次 decode 前向之前: pos={snap['positions']} "
                  f"ctx_len={snap['ctx_len']} bt={snap['block_table']}", flush=True)
        return orig_fg(self, tokens, positions, block_tables, context_lens, slot_mapping)

    DP.DraftModelProposer._forward_group = fg

    kw = dict(enforce_eager=False, max_model_len=16640,
              max_num_batched_tokens=16640, max_num_seqs=4,
              spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if window:
        kw["spec_draft_window"] = window
    if head:
        kw["spec_draft_head_blocks"] = head
    llm = LLM(TARGET, **kw)

    tok = AutoTokenizer.from_pretrained(TARGET)
    prompt = build_prompts(tok, ctx, 1)[0]
    snap["prompt"] = prompt
    snap["window"] = window
    snap["head"] = head
    snap["ctx"] = ctx
    llm.generate([prompt], SamplingParams(temperature=1.0, max_tokens=8,
                                          ignore_eos=True), use_tqdm=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(snap, OUT)
    print(f"[kv] saved {OUT} (blocks={snap['n_blocks']} bs={snap['bs']} "
          f"layers={snap['n_layers']})", flush=True)


if __name__ == "__main__":
    main()
