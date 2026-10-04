"""直接看 postprocess 前后 num_cached_tokens 的变化"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
sched = llm.scheduler
mr = llm.model_runner

orig = sched.postprocess
n = [0]
def traced(seqs, toks, pf):
    s = seqs[0]
    before = s.num_cached_tokens
    r = orig(seqs, toks, pf)
    print(f"[{n[0]}{'P' if pf else 'D'}] num_cached {before} -> {s.num_cached_tokens} "
          f"| len(tok)={len(s.token_ids)} | n_sched_was | toks={str(toks)[:30]}")
    n[0] += 1
    return r
sched.postprocess = traced

sp = SamplingParams(temperature=0.01, max_tokens=10, ignore_eos=True)
llm.generate(["def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n"], sp, use_tqdm=False)
