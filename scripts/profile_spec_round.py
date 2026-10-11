#!/usr/bin/env python
"""profile_spec_round.py —— 真实投机「一轮」的账：propose vs verify vs 残差。

不修改引擎代码，只在进程内 monkeypatch 三处计时（CUDA event，不含 profiler）：
  · DraftModelProposer.propose_batch  -> 一轮里 6 次草稿前向 + 采样 + CPU 打包
  · ModelRunner.run_verify            -> 验证前向 + 拒绝采样
  · ModelRunner.run                   -> 整轮

报告：每轮 ms 的构成，以及 propose 内部「前向 vs 非前向」的拆分
（在 propose_batch 里插桩 _forward_group，把 6 次 replay 单独计时）。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<p1-work> python -u scripts/profile_spec_round.py
env: CTX OUTLEN WIN
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.draft_proposer import DraftModelProposer
from nanovllm.engine.model_runner import ModelRunner

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
CTX = int(os.environ.get("CTX", 4096))
OUTLEN = int(os.environ.get("OUTLEN", 64))
WIN = int(os.environ.get("WIN", 0))
MAX_SEQS = int(os.environ.get("MAX_SEQS", 512))
PINNED = os.environ.get("PINNED", "")


def _ev():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, k, ms):
        self.d[k] = self.d.get(k, 0.0) + ms

    def count(self, k):
        self.d[k + "_n"] = self.d.get(k + "_n", 0) + 1


ACC = Acc()


def patch():
    orig_pb = DraftModelProposer.propose_batch
    orig_fg = DraftModelProposer._forward_group
    orig_rv = ModelRunner.run_verify
    orig_run = ModelRunner.run

    def fg(self, *a, **kw):
        s, e = _ev()
        s.record()
        t0 = time.perf_counter()
        try:
            return orig_fg(self, *a, **kw)
        finally:
            t1 = time.perf_counter()
            e.record()
            torch.cuda.synchronize()
            ACC.add("draft_fwd_ms", s.elapsed_time(e))
            ACC.add("draft_fwd_cpu_ms", (t1 - t0) * 1e3)
            ACC.count("draft_fwd")

    def pb(self, *a, **kw):
        s, e = _ev()
        s.record()
        try:
            return orig_pb(self, *a, **kw)
        finally:
            e.record()
            torch.cuda.synchronize()
            ACC.add("propose_ms", s.elapsed_time(e))
            ACC.count("propose")

    def rv(self, *a, **kw):
        s, e = _ev()
        s.record()
        try:
            return orig_rv(self, *a, **kw)
        finally:
            e.record()
            torch.cuda.synchronize()
            ACC.add("verify_ms", s.elapsed_time(e))
            ACC.count("verify")

    def run(self, seqs, is_prefill):
        s, e = _ev()
        s.record()
        try:
            return orig_run(self, seqs, is_prefill)
        finally:
            e.record()
            torch.cuda.synchronize()
            if is_prefill:
                ACC.add("prefill_ms", s.elapsed_time(e))
                ACC.count("prefill")
            else:
                ACC.add("round_ms", s.elapsed_time(e))
                ACC.count("round")

    DraftModelProposer._forward_group = fg
    DraftModelProposer.propose_batch = pb
    ModelRunner.run_verify = rv
    ModelRunner.run = run


def build_ids(llm, need):
    tok = llm.tokenizer
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "In a distant galaxy, researchers study attention kernels. " * 800)
    assert len(base) >= need
    return base[:need]


def main():
    print("=== profile_spec_round ctx=%d out=%d win=%d gpu=%s ==="
          % (CTX, OUTLEN, WIN, os.environ.get("CUDA_VISIBLE_DEVICES")))
    kw = dict(enforce_eager=False, max_model_len=8192, max_num_batched_tokens=16384,
              spec_k=6, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0,
              spec_graph_bs=(1,), max_num_seqs=MAX_SEQS)
    if WIN:
        kw["spec_draft_window"] = WIN
    if PINNED != "":
        kw["spec_draft_pinned_meta"] = PINNED == "1"
    llm = LLM(TARGET, **kw)
    ids = build_ids(llm, CTX + 16)
    sp = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
    llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()

    ACC.d.clear()
    patch()
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)
    hook_before = (llm.model_runner.spec_proposer.n_rounds,
                   llm.model_runner.spec_proposer.n_batch_forwards,
                   llm.model_runner.spec_proposer.n_catchup_forwards)
    t0 = time.time()
    out = llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
    pr = llm.model_runner.spec_proposer
    rounds = pr.n_rounds - hook_before[0]
    fwds = pr.n_batch_forwards - hook_before[1]
    cat = pr.n_catchup_forwards - hook_before[2]
    ntok = len(out[0]["token_ids"])

    r = ACC.d
    res = dict(ctx=CTX, outlen=OUTLEN, win=WIN, wall_s=round(wall, 3),
               tokens=ntok, rounds=rounds, draft_forwards=fwds, catchup=cat,
               ms_per_round=round(r.get("round_ms", 0) / max(rounds, 1), 3),
               propose_ms_per_round=round(r.get("propose_ms", 0) / max(rounds, 1), 3),
               verify_ms_per_round=round(r.get("verify_ms", 0) / max(rounds, 1), 3),
               draft_fwd_us_per_call=round(r.get("draft_fwd_ms", 0) * 1e3 / max(fwds, 1), 1),
               draft_fwd_cpu_us_per_call=round(r.get("draft_fwd_cpu_ms", 0) * 1e3 / max(fwds, 1), 1),
               propose_us=round(r.get("propose_ms", 0) * 1e3, 1),
               verify_us=round(r.get("verify_ms", 0) * 1e3, 1),
               round_us=round(r.get("round_ms", 0) * 1e3, 1),
               prefill_us=round(r.get("prefill_ms", 0) * 1e3, 1))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if rounds:
        tot = r.get("round_ms", 0)
        print("--- 一轮构成 (ms) ---")
        print("  propose  %.3f  (%.1f%%)  [其中 6 次前向 %.3f = %.1f%%]"
              % (r.get("propose_ms", 0) / rounds,
                 100 * r.get("propose_ms", 0) / tot,
                 r.get("draft_fwd_ms", 0) / rounds,
                 100 * r.get("draft_fwd_ms", 0) / tot))
        print("  verify   %.3f  (%.1f%%)" % (r.get("verify_ms", 0) / rounds,
                                             100 * r.get("verify_ms", 0) / tot))
        print("  residual %.3f  (%.1f%%)"
              % ((tot - r.get("propose_ms", 0) - r.get("verify_ms", 0)) / rounds,
                 100 * (tot - r.get("propose_ms", 0) - r.get("verify_ms", 0)) / tot))
        print("  propose 里非前向部分 = %.3f ms/轮 (CPU 打包 + 采样)"
              % ((r.get("propose_ms", 0) - r.get("draft_fwd_ms", 0)) / rounds))
    print("@@ROUND@" + json.dumps(res, ensure_ascii=False))
    try:
        import atexit
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
