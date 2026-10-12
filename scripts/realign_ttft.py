#!/usr/bin/env python
"""realign_ttft.py —— 修复后重定位：单一配置的 TTFT / prefill 拆解（口径与
`acceptance_prefill_probe.py` 的条件①一致：spec_k=0 + enforce_eager=True，
这样量到的是【prefill 本身】的代价，与台账 §2.7.1 的 101.5/93.1 ms 可直接比）。

env: CTX WINDOW HEAD ALIGN B REPS MAXLEN MAX_SEQS TAG OUT
输出：@@TTFT@@ json —— prefill_ms(中位) / hook_ttft_ms / draft_blocks / target_blocks
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.token_hook import TokenDeliveryHook
    from acceptance_matrix import build_prompts

    ctx = int(os.environ.get("CTX", "8192"))
    window = int(os.environ.get("WINDOW", "0"))
    head = int(os.environ.get("HEAD", "0"))
    align = os.environ.get("ALIGN", "0") == "1"
    B = int(os.environ.get("B", "1"))
    REPS = int(os.environ.get("REPS", "3"))
    MAXLEN = int(os.environ.get("MAXLEN", "16640"))
    MAX_SEQS = int(os.environ.get("MAX_SEQS", "4"))
    TAG = os.environ.get("TAG", "ttft")
    OUT = os.environ.get("OUT", "")

    kw = dict(enforce_eager=False, max_model_len=MAXLEN,
              max_num_batched_tokens=max(MAXLEN, 16384),
              max_num_seqs=MAX_SEQS, spec_k=6, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0)
    if window:
        kw["spec_draft_window"] = window
    if head:
        kw["spec_draft_head_blocks"] = head
    if align:
        kw["spec_draft_win_align"] = True
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner

    llm.scheduler.spec_k = 0          # = 表里的 U（排除解码投机的影响）
    mr.enforce_eager = True

    T = {"prefill": []}
    orig_run = ModelRunner.run

    def run(self, seqs, is_prefill):
        t0 = time.time()
        r = orig_run(self, seqs, is_prefill)
        torch.cuda.synchronize()
        if is_prefill and all(s.block_table for s in seqs):
            T["prefill"].append(time.time() - t0)
        return r
    ModelRunner.run = run

    prompts = build_prompts(llm.tokenizer, ctx, B)
    sp = SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True)

    ttft, wall = [], []
    llm.generate(prompts, sp, use_tqdm=False)      # warmup
    T["prefill"].clear()
    for _ in range(REPS):
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        torch.cuda.synchronize()
        t0 = time.time()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        wall.append(time.time() - t0)
        llm.token_hook = None
        llm.scheduler.token_hook = None
        summ = hook.summary()
        ttft.append(statistics.median(
            [v["ttft"] for v in summ.values() if v["ttft"] == v["ttft"]]))

    p = sorted(T["prefill"])
    out = dict(tag=TAG, ctx=ctx, B=B, window=window, head=head, align=align,
               maxlen=MAXLEN, max_seqs=MAX_SEQS,
               draft_blocks=int(mr.config.num_draft_blocks),
               target_blocks=int(mr.config.num_kvcache_blocks),
               prefill_ms=round(1000 * statistics.median(p), 2),
               prefill_min_ms=round(1000 * p[0], 2),
               prefill_max_ms=round(1000 * p[-1], 2),
               hook_ttft_ms=round(1000 * statistics.median(ttft), 2),
               wall_ms=round(1000 * statistics.median(wall), 2))
    print("@@TTFT@@" + json.dumps(out, ensure_ascii=False), flush=True)
    if OUT:
        json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
