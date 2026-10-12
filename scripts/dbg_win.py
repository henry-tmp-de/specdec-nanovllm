#!/usr/bin/env python
"""dbg_win.py —— 把 draft 每次前向【真实用的上下文】打出来（只观测，不改引擎）。

目的：引擎侧确定性仪器显示 P=0（纯滑窗）的 draft 分布远差于「只喂最后 M 块」
      的预期（α_step 0.366 vs 0.626），而 P=2 却和预期一致。
      本探针记录 draft 每次 decode 前向的 (positions, context_lens, block_tables)，
      以及窗口 prefill 的 ring 布局，用来定位差在哪。

用法: WINDOW=1280 HEAD=0 TAG=p0m5 python -u scripts/dbg_win.py
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
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from acceptance_matrix import build_prompts
    import nanovllm.spec_decode.draft_proposer as DP
    import nanovllm.engine.model_runner as MR

    ctx = int(os.environ.get("CTX", "8192"))
    window = int(os.environ.get("WINDOW", "1280"))
    head = int(os.environ.get("HEAD", "0"))
    B = int(os.environ.get("B", "1"))
    TAG = os.environ.get("TAG", "x")

    LOG = []

    orig_fg = DP.DraftModelProposer._forward_group

    def fg(self, tokens, positions, block_tables, context_lens, slot_mapping):
        if len(LOG) < 40:
            LOG.append(dict(pos=[int(x) for x in positions],
                            cl=[int(x) for x in context_lens],
                            bt=[list(map(int, b)) for b in block_tables],
                            slots=[int(x) for x in slot_mapping]))
        return orig_fg(self, tokens, positions, block_tables, context_lens, slot_mapping)

    DP.DraftModelProposer._forward_group = fg

    orig_pref = MR.ModelRunner._run_draft_prefill_window

    def pref(self, seqs):
        for s in seqs:
            print(f"[dbg] PREFILL seq len={len(s)} ring={list(s.draft_block_table)} "
                  f"tokens[:4]={list(s.token_ids[:4])}", flush=True)
        return orig_pref(self, seqs)

    MR.ModelRunner._run_draft_prefill_window = pref

    # 记录窗口 prefill 算出来的 (b0, head_end, w0, end)
    orig_b0 = DP.DraftWindow.b0
    B0LOG = []

    def b0(self, pos, valid_from=0):
        r = orig_b0(self, pos, valid_from)
        if len(B0LOG) < 8:
            B0LOG.append(dict(pos=int(pos), valid_from=int(valid_from), b0=int(r),
                              P=self.P, M=self.M, bs=self.bs))
        return r

    DP.DraftWindow.b0 = b0

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
    prompts = build_prompts(tok, ctx, B)
    cfg = llm.model_runner.config
    print(f"[dbg] TAG={TAG} window={cfg.spec_draft_window} head={cfg.spec_draft_head_blocks} "
          f"M={llm.model_runner.draft_window_blocks} P={llm.model_runner.draft_head_blocks} "
          f"num_draft_blocks={cfg.num_draft_blocks}", flush=True)

    out = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=24,
                                               ignore_eos=True), use_tqdm=False)
    print(f"[dbg] B0LOG(前{len(B0LOG)}): {json.dumps(B0LOG)}", flush=True)
    print(f"[dbg] 前 3 次 draft decode 前向:", flush=True)
    for e in LOG[:6]:
        print("      pos", e["pos"], "ctx_len", e["cl"], "bt", e["bt"], flush=True)
    # 统计 context_len 的分布
    cls = [c for e in LOG for c in e["cl"]]
    if cls:
        print(f"[dbg] context_len: min={min(cls)} max={max(cls)} 前 24 个={cls[:24]}", flush=True)


if __name__ == "__main__":
    main()
