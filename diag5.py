"""统计投机解码的真实接受率"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)

orig_verify = llm.model_runner.run_verify
S = {"verify": 0, "proposed": 0, "landed": 0}
def traced_verify(seqs):
    S["proposed"] += sum(len(s.draft_tokens) for s in seqs)
    out = orig_verify(seqs)
    S["verify"] += 1
    S["landed"] += sum(len(x) for x in out)
    return out
llm.model_runner.run_verify = traced_verify

steps = {"n": 0}
orig_post = llm.scheduler.postprocess
def traced_post(seqs, t, pf):
    steps["n"] += 1
    return orig_post(seqs, t, pf)
llm.scheduler.postprocess = traced_post

prompt = ("def add(a, b): return a + b\n"
          "def sub(a, b): return a - b\n"
          "def add(a, b): return a + b\n"
          "def sub(a, b): return a - b\n")
sp = SamplingParams(temperature=1.0, max_tokens=60, ignore_eos=True)
t0 = time.time()
outs = llm.generate([prompt], sp, use_tqdm=False)
torch.cuda.synchronize()
dt = time.time() - t0

n_out = len(outs[0]["token_ids"])
print(f"\n耗时 {dt:.2f}s   总 step {steps['n']}   产出 {n_out} token")
print(f"verify 路径: {S['verify']} 次   提议 {S['proposed']}   落地 {S['landed']}")
if S["verify"]:
    print(f"  verify 路径内: 提议/次 = {S['proposed']/S['verify']:.2f}, "
          f"落地/次 = {S['landed']/S['verify']:.2f}")
print(f"  投机占比 = {S['verify']}/{steps['n']} = {S['verify']/max(steps['n'],1)*100:.0f}%")
print("\n文本:", outs[0]["text"][:180].replace("\n", "\n"))
