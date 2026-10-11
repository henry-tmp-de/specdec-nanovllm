#!/usr/bin/env python
"""bench_pinned.py —— 草稿「锁页元数据」优化的验收脚本（MODE=equiv | bench）

equiv: 同一 prompt + 同一随机种子，把 spec_draft_pinned_meta 开/关各跑一遍，
       逐字节比较：输出 token ids、每次草稿前向 logits 的 bitwise 校验和、
       Gumbel 采样噪声校验和、接受率。**任何一项不同就是 bug。**

bench: 配对 A/B（交替、≥7 对）+ A/A 空转噪声地板。量的是【每轮耗时】
       （tok/s 被轮数随机性淹没）。运行时翻 self.pinned_meta 即可，不用重建。

用法:
  CUDA_VISIBLE_DEVICES=0 CTX=4096 OUTLEN=64 MODE=equiv python -u scripts/bench_pinned.py
  CUDA_VISIBLE_DEVICES=0 CTX=4096 OUTLEN=64 MODE=bench TRIALS=8 python -u scripts/bench_pinned.py
"""
import os
import sys
import json
import time
import statistics

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
MODE = os.environ.get("MODE", "equiv")
TRIALS = int(os.environ.get("TRIALS", 8))
SEED = int(os.environ.get("SEED", 1234))

ACC = {"round_ms": 0.0, "rounds": 0, "accepted": 0, "proposed": 0,
       "fwd_logits": None, "noise": None}


def build_ids(llm, need):
    tok = llm.tokenizer
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "In a distant galaxy, researchers study attention kernels. " * 800)
    return base[:need]


def install(record_hashes: bool):
    orig_fg = DraftModelProposer._forward_group
    orig_rv = ModelRunner.run_verify
    orig_run = ModelRunner.run

    def fg(self, *a, **kw):
        out = orig_fg(self, *a, **kw)
        if record_hashes:
            torch.cuda.synchronize()
            h = hash(out.detach().float().cpu().numpy().tobytes())
            ACC["fwd_logits"] = ACC["fwd_logits"] or []
            ACC["fwd_logits"].append(h)
        return out

    def rv(self, *a, **kw):
        seqs = a[0]
        out = orig_rv(self, *a, **kw)
        try:
            k = self.config.spec_k
            for sq in seqs:
                ACC["accepted"] += int(getattr(sq, "last_accepted", 0))
                ACC["proposed"] += k
        except Exception:
            pass
        return out

    def run(self, seqs, is_prefill):
        if is_prefill:
            return orig_run(self, seqs, is_prefill)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_run(self, seqs, is_prefill)
        e.record()
        torch.cuda.synchronize()
        ACC["round_ms"] += s.elapsed_time(e)
        ACC["rounds"] += 1
        return out

    DraftModelProposer._forward_group = fg
    ModelRunner.run_verify = rv
    ModelRunner.run = run


def one_trial(llm, ids, sp, pinned, record_hashes):
    pr = llm.model_runner.spec_proposer
    pr.pinned_meta = pinned
    ACC["round_ms"] = 0.0
    ACC["rounds"] = 0
    ACC["accepted"] = 0
    ACC["proposed"] = 0
    ACC["fwd_logits"] = [] if record_hashes else None
    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    out = llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tok = list(out[0]["token_ids"])
    return dict(tokens=tok, wall=wall, ms_per_round=ACC["round_ms"] / max(ACC["rounds"], 1),
                rounds=ACC["rounds"], accepted=ACC["accepted"],
                proposed=ACC["proposed"],
                accept_rate=ACC["accepted"] / max(ACC["proposed"], 1),
                fwd_hash=(hash(tuple(ACC["fwd_logits"])) if record_hashes else None),
                n_fwd=(len(ACC["fwd_logits"]) if record_hashes else 0))


def main():
    print("=== bench_pinned mode=%s ctx=%d out=%d ===" % (MODE, CTX, OUTLEN))
    llm = LLM(TARGET, enforce_eager=False, max_model_len=8192,
              max_num_batched_tokens=16384, spec_k=6, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0, spec_graph_bs=(1,))
    ids = build_ids(llm, CTX + 16)
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)
    llm.generate([ids], SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True),
                 use_tqdm=False)
    torch.cuda.synchronize()
    install(record_hashes=(MODE != "bench"))

    if MODE == "equiv":
        a = one_trial(llm, ids, sp, False, True)
        b = one_trial(llm, ids, sp, True, True)
        same_tok = a["tokens"] == b["tokens"]
        same_fwd = a["fwd_hash"] == b["fwd_hash"]
        res = {"mode": "equiv",
               "tokens_identical": same_tok,
               "n_tokens": len(a["tokens"]),
               "fwd_logits_identical": same_fwd,
               "n_draft_forwards": a["n_fwd"],
               "accept_rate_off": round(a["accept_rate"], 4),
               "accept_rate_on": round(b["accept_rate"], 4),
               "accept_rate_identical": abs(a["accept_rate"] - b["accept_rate"]) < 1e-12,
               "first_diff_index": (next((i for i in range(min(len(a["tokens"]), len(b["tokens"])))
                                          if a["tokens"][i] != b["tokens"][i]), None)
                                    if not same_tok else None)}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("@@PINNED@" + json.dumps(res, ensure_ascii=False))
        return

    # ---------- bench: 交替 A/B + A/A 噪声地板 ----------
    pairs = []
    floor = []
    for i in range(TRIALS):
        order = [(False, True), (True, False)][i % 2]
        r = {}
        for tag, pin in zip(("a", "b"), order):
            r[tag] = one_trial(llm, ids, sp, pin, False)
        # 归一化到 (baseline=off, opt=on)
        off = r["a"] if not order[0] else r["b"]
        on = r["b"] if not order[0] else r["a"]
        pairs.append({"i": i,
                      "off_ms": round(off["ms_per_round"], 3),
                      "on_ms": round(on["ms_per_round"], 3),
                      "ratio": round(on["ms_per_round"] / off["ms_per_round"], 4),
                      "off_tok": len(off["tokens"]), "on_tok": len(on["tokens"]),
                      "off_ar": round(off["accept_rate"], 4),
                      "on_ar": round(on["accept_rate"], 4)})
        # 噪声地板：同一配置连跑两次
        f1 = one_trial(llm, ids, sp, False, False)
        f2 = one_trial(llm, ids, sp, False, False)
        floor.append(round(f2["ms_per_round"] / f1["ms_per_round"], 4))
        print("pair %d: off=%.3f on=%.3f ratio=%.4f | floor=%.4f"
              % (i, pairs[-1]["off_ms"], pairs[-1]["on_ms"], pairs[-1]["ratio"], floor[-1]))

    ratios = [p["ratio"] for p in pairs]
    faster = sum(1 for r in ratios if r < 1.0)
    med = statistics.median(ratios)
    fl = statistics.median(floor)
    res = {"mode": "bench", "ctx": CTX, "outlen": OUTLEN, "trials": TRIALS,
           "pairs": pairs, "noise_floor_median": fl,
           "ratio_median": round(med, 4), "ratio_min": min(ratios), "ratio_max": max(ratios),
           "on_faster": faster, "n": len(ratios),
           "tokens_identical_all": all(p["off_tok"] == p["on_tok"] for p in pairs),
           "accept_identical_all": all(p["off_ar"] == p["on_ar"] for p in pairs),
           "speedup_pct": round((1 - med) * 100, 2)}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("@@PINNED@" + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
