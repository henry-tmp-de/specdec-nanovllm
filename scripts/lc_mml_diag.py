#!/usr/bin/env python
"""lc_mml_diag.py —— 诊断：ctx=4096 下「窗口不截断」却与全上下文不同轨迹，是不是
max_model_len（静态块表宽度）变了造成的？

背景：长上下文组的 4 个进程都用 max_model_len=16640（块表 65 列），
而历史验收表用的是 8192（32 列）。块表宽度变化会让 paged 那条路的浮点归约顺序变，
在 temp=1.0 下足以让采样轨迹发散（见 acceptance_matrix.py 顶部那条口径陷阱）。

本脚本在【同一份负载】上只翻 max_model_len / spec_draft_window，各跑一遍，比：
  · accept_rate  · out_md5  · ms/round
用来判断「lc_w0 在 ctx=4096 的 α=0.6062 偏离 published S2 的 0.6359」是
  (a) 窗口路径的数值差，还是 (b) max_model_len 改变导致的轨迹漂移。

用法:
  CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2391 WINDOW=0    MAXLEN=8192 \
      python -u scripts/lc_mml_diag.py
  CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2391 WINDOW=8448 MAXLEN=8192 \
      python -u scripts/lc_mml_diag.py
"""
import hashlib
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

from acceptance_matrix import TARGET, DRAFT_BF16, build_prompts, md5_ids  # noqa: E402

K, OUTLEN, SEED = 6, 256, 20261011


def main():
    WINDOW = int(os.environ.get("WINDOW", "0"))
    MAXLEN = int(os.environ.get("MAXLEN", "8192"))
    CTX = int(os.environ.get("CTX", "4096"))
    B = int(os.environ.get("B", "1"))
    SEQS = int(os.environ.get("SEQS", "4"))
    RUNS = int(os.environ.get("RUNS", "5"))

    import torch
    from nanovllm import LLM, SamplingParams
    import nanovllm.spec_decode.verify as V

    kw = dict(enforce_eager=False, max_model_len=MAXLEN,
              max_num_batched_tokens=max(MAXLEN, 16384), max_num_seqs=SEQS,
              spec_k=K, spec_method="draft", draft_model=DRAFT_BF16,
              spec_batch_threshold=0)
    if WINDOW:
        kw["spec_draft_window"] = WINDOW
    llm = LLM(TARGET, **kw)
    cfg = llm.model_runner.config

    S = {"proposed": 0, "accepted": 0}
    orig_vb = V.verify_batch

    def tvb(draft_probs, target_logits, draft_tokens, temperatures=None,
            draft_is_point_mass=False):
        res = orig_vb(draft_probs, target_logits, draft_tokens, temperatures,
                      draft_is_point_mass)
        if not draft_is_point_mass:
            S["proposed"] += int(res.n_proposed.sum())
            S["accepted"] += int(res.accept_mask.sum())
        return res
    V.verify_batch = tvb

    prompts = build_prompts(llm.tokenizer, CTX, B)
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)          # warmup

    recs = []
    for _ in range(RUNS):
        S["proposed"] = S["accepted"] = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.synchronize()
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.time() - t0
        recs.append(dict(wall_s=wall, alpha=S["accepted"] / S["proposed"],
                         md5=md5_ids([o["token_ids"] for o in outs]),
                         ntok=sum(len(o["token_ids"]) for o in outs)))

    print("@@DIAG@@" + json.dumps(dict(
        window=WINDOW, maxlen=MAXLEN, ctx=CTX, B=B, seqs=SEQS,
        blocks=[cfg.num_kvcache_blocks, cfg.num_draft_blocks, cfg.draft_window_blocks],
        wall_s=statistics.median([r["wall_s"] for r in recs]),
        alpha=statistics.median([r["alpha"] for r in recs]),
        md5=recs[0]["md5"], md5_same=len({r["md5"] for r in recs}) == 1,
        ntok=recs[0]["ntok"]), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
