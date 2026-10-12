#!/usr/bin/env python
"""eng_dump_q.py —— 把引擎【第一次投机轮】里 draft 的真实 logits 倒出来存盘。

配合 cmp_dump_q.py：拿同样的 prompt，在离线 HF 模型上按「全上下文 / 只喂窗口 /
喂 head+窗口」三种 draft 上下文各算一遍，看引擎那一次到底等于哪一种。

用法: CTX=8192 WINDOW=1280 HEAD=0 B=1 TAG=p0m5 python -u scripts/eng_dump_q.py
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
    import nanovllm.spec_decode.verify as V
    import nanovllm.spec_decode.draft_proposer as DP

    ctx = int(os.environ.get("CTX", "8192"))
    window = int(os.environ.get("WINDOW", "1280"))
    head = int(os.environ.get("HEAD", "0"))
    TAG = os.environ.get("TAG", "x")
    OUT = os.environ.get("OUT", f"out/eng/dump_{TAG}.pt")

    fg_log = []
    orig_fg = DP.DraftModelProposer._forward_group

    def fg(self, tokens, positions, block_tables, context_lens, slot_mapping):
        fg_log.append(dict(pos=[int(x) for x in positions],
                           cl=[int(x) for x in context_lens],
                           bt=[list(map(int, b)) for b in block_tables]))
        return orig_fg(self, tokens, positions, block_tables, context_lens, slot_mapping)

    DP.DraftModelProposer._forward_group = fg

    dump = {}
    orig_vb = V.verify_batch

    def vb(draft_probs, target_logits, draft_tokens, temperatures=None,
           draft_is_point_mass=False):
        if not draft_is_point_mass and "q" not in dump:
            dump["q"] = draft_probs[0].detach().float().cpu()
            dump["p"] = target_logits[0].detach().float().cpu()
            dump["tokens"] = [int(x) for x in draft_tokens[0]]
            dump["temp"] = float(temperatures[0]) if temperatures is not None else 1.0
            dump["fg"] = fg_log[:8]
        return orig_vb(draft_probs, target_logits, draft_tokens, temperatures,
                       draft_is_point_mass)

    V.verify_batch = vb

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
    dump["prompt"] = prompt
    dump["window"] = window
    dump["head"] = head
    dump["ctx"] = ctx
    ntok = int(os.environ.get("NTOK", "16"))
    gen = llm.generate([prompt], SamplingParams(temperature=1.0, max_tokens=ntok,
                                                ignore_eos=True), use_tqdm=False)
    dump["gen"] = list(gen[0]["token_ids"])
    print(f"[dump] TAG={TAG} window={window} head={head} "
          f"fg(前{len(dump.get('fg', []))}) = {json.dumps(dump.get('fg', []))}",
          flush=True)
    print(f"[dump] draft_tokens={dump.get('tokens')} temp={dump.get('temp')}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(dump, OUT)
    print(f"[dump] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
