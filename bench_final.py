"""最终 benchmark：修复后的 draft 路线"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
mode = sys.argv[1]; k = int(sys.argv[2]); mt = int(sys.argv[3])

kw = dict(enforce_eager=True, max_num_batched_tokens=16384)
if mode == "draft":
    kw.update(spec_k=k, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

llm = LLM(TARGET, **kw)
S = {"verify": 0, "proposed": 0, "landed": 0}
if mode == "draft":
    ov = llm.model_runner.run_verify
    def tv(seqs):
        S["proposed"] += sum(len(s.draft_tokens) for s in seqs)
        out = ov(seqs); S["verify"] += 1; S["landed"] += sum(len(x) for x in out); return out
    llm.model_runner.run_verify = tv
    steps = {"n": 0}
    op = llm.scheduler.postprocess
    def tp(seqs, t, pf):
        steps["n"] += 1; return op(seqs, t, pf)
    llm.scheduler.postprocess = tp

P = "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\ndef calculate_max(numbers):\n"
sp = SamplingParams(temperature=1.0, max_tokens=mt, ignore_eos=True)
torch.cuda.synchronize(); t0 = time.time()
outs = llm.generate([P], sp, use_tqdm=False)
torch.cuda.synchronize()
dt = time.time() - t0
n = len(outs[0]["token_ids"])
ns = steps["n"] if mode == "draft" else n
print("@@B@@" + json.dumps({
    "mode": mode, "k": k, "gen_s": round(dt, 3),
    "tokens": n, "steps": ns, "tok_per_step": round(n / max(ns, 1), 3),
    "tok_per_s": round(n / dt, 1),
    "verify": S["verify"], "proposed": S["proposed"],
    "accept_rate": round((S["landed"] - S["verify"]) / max(S["proposed"], 1), 3),
    "distinct": len(set(outs[0]["token_ids"])),
}))
