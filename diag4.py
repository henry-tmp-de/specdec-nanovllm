"""查提议为什么总是空"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)
sp = SamplingParams(temperature=1e-4, max_tokens=15, ignore_eos=True)
llm.add_request("The capital of France is Paris. The capital of France is", sp)

p = llm.model_runner.spec_proposer
step = 0
while not llm.is_finished() and step < 40:
    seqs, is_pf = llm.scheduler.schedule()
    s = seqs[0]
    if not is_pf:
        cand = p.propose(s.token_ids, 3)
        n_actual = 1 + len(s.draft_tokens)
        print(f"[step {step}] len(token_ids)={len(s.token_ids)}  提议={cand}  "
              f"实际送forward={n_actual}  n_sched={s.num_scheduled_tokens}")
    toks = llm.model_runner.call("run", seqs, is_pf)
    llm.scheduler.postprocess(seqs, toks, is_pf)
    step += 1
print("\nproposer stats:", p.stats())
print("生成:", s.completion_token_ids)
