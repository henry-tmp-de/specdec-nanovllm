"""draft model 路线：0.6B draft -> 4B target"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

mode = sys.argv[1]           # "base" | "draft"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 60

kwargs = dict(enforce_eager=True, max_num_batched_tokens=16384)
if mode == "draft":
    kwargs.update(spec_k=k, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

t0 = time.time()
llm = LLM(TARGET, **kwargs)
init_s = time.time() - t0

S = {"verify": 0, "proposed": 0, "landed": 0}
orig_verify = llm.model_runner.run_verify
def traced(seqs):
    S["proposed"] += sum(len(s.draft_tokens) for s in seqs)
    out = orig_verify(seqs)
    S["verify"] += 1
    S["landed"] += sum(len(x) for x in out)
    return out
llm.model_runner.run_verify = traced

steps = {"n": 0}
orig_post = llm.scheduler.postprocess
def traced_post(seqs, t, pf):
    steps["n"] += 1
    return orig_post(seqs, t, pf)
llm.scheduler.postprocess = traced_post

PROMPT = "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\ndef calculate_max(numbers):\n"
sp = SamplingParams(temperature=0.2, max_tokens=max_tokens, ignore_eos=True)

torch.cuda.synchronize(); t1 = time.time()
outs = llm.generate([PROMPT], sp, use_tqdm=False)
torch.cuda.synchronize()
gen_s = time.time() - t1

n_out = len(outs[0]["token_ids"])
print("@@JSON@@" + json.dumps({
    "mode": mode, "k": k,
    "init_s": round(init_s, 2), "gen_s": round(gen_s, 3),
    "n_tokens": n_out, "steps": steps["n"],
    "tok_per_step": round(n_out / max(steps["n"], 1), 3),
    "tok_per_s": round(n_out / gen_s, 1),
    "verify": S["verify"], "proposed": S["proposed"], "landed": S["landed"],
    "accept_rate": round((S["landed"] - S["verify"]) / max(S["proposed"], 1), 3),
    "text": outs[0]["text"][:120].replace("\n", "\n"),
}))
