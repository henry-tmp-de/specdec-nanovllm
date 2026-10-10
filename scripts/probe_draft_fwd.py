#!/usr/bin/env python
"""probe_draft_fwd.py —— 「draft 单次前向」的带宽效率（Phase 0 补充）

协调方要的那个因素：草稿一次前向读 1.2 GB，graphed 约 2 ms，
≈ 实测天花板(842 GB/s) 的多少？**如果草稿本来就没跑满带宽，减字节的收益会比按比例推算的更少。**

做法：直接 replay draft 的 CUDA 图 N 次，用 profiler 读纯 device 时间。
同时量 target 的普通 decode 图做对照（它已知在 81%）。

用法: CUDA_VISIBLE_DEVICES=0 python -u scripts/probe_draft_fwd.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
CEIL = 842.0  # GB/s，实测 copy 流式天花板

N = 60


def _dedup_bytes(model):
    """唯一权重字节数（按 storage 去重，tie_word_embeddings 会重复计数）。"""
    seen, tot = set(), 0
    for p in model.parameters():
        key = (p.data_ptr(), p.numel(), p.element_size())
        if key in seen:
            continue
        seen.add(key)
        tot += p.numel() * p.element_size()
    return tot


def dev_us(fn, iters=N):
    prof = profile(activities=[ProfilerActivity.CUDA])
    with prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for ev in prof.events():
        if getattr(ev, "device_type", None) == torch.autograd.DeviceType.CUDA:
            tot += float(getattr(ev, "device_time", 0) or 0)
    del prof
    return tot / iters


def main():
    llm = LLM(TARGET, enforce_eager=False, max_model_len=8192,
              max_num_batched_tokens=16384, spec_k=6, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0)
    mr = llm.model_runner
    llm.generate(["def f(x):\n    return x + 1\n"], SamplingParams(
        temperature=1.0, max_tokens=16, ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()

    dw = sum(p.numel() * p.element_size() for p in mr.draft_model.parameters())
    tw = sum(p.numel() * p.element_size() for p in mr.model.parameters())
    dw_d = _dedup_bytes(mr.draft_model)
    tw_d = _dedup_bytes(mr.model)
    print(f"去重后 draft {dw_d/1e9:.3f} GB  target {tw_d/1e9:.3f} GB")
    res = {"draft_weight_bytes": dw, "target_weight_bytes": tw, "ceiling_GBs": CEIL}
    print(f"draft 权重 {dw/1e9:.3f} GB  target 权重 {tw/1e9:.3f} GB")

    for B in sorted(mr.draft_graphs):
        g = mr.draft_graphs[B][0]
        t = dev_us(lambda: g.replay())
        bw = dw / t / 1e3
        res[f"draft_fwd_us_B{B}"] = t
        res[f"draft_fwd_GBs_B{B}"] = bw
        res[f"draft_fwd_GBs_B{B}_dedup"] = _dedup_bytes(mr.draft_model) / t / 1e3
        res[f"draft_fwd_pct_ceiling_B{B}"] = bw / CEIL * 100
        print(f"draft 前向 B={B}: {t:8.2f} us  {bw:6.1f} GB/s  = 天花板 {bw/CEIL*100:5.1f}%")

    for B in sorted(mr.graphs):
        g = mr.graphs[B]
        t = dev_us(lambda: g.replay())
        bw = tw / t / 1e3
        res[f"target_decode_us_B{B}"] = t
        res[f"target_decode_GBs_B{B}"] = bw
        res[f"target_decode_pct_ceiling_B{B}"] = bw / CEIL * 100
        print(f"target 普通 decode B={B}: {t:8.2f} us  {bw:6.1f} GB/s = 天花板 {bw/CEIL*100:5.1f}%")

    print("@@DF@@" + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
