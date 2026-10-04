"""查为什么接受率是 0"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)
orig = llm.model_runner.run_verify
def traced(seqs):
    print(f"\n>>> verify: draft={seqs[0].draft_tokens} "
          f"n_sched={seqs[0].num_scheduled_tokens} len(tok)={len(seqs[0].token_ids)}")
    out = orig(seqs)
    print(f"<<< 返回 {out}")
    return out
llm.model_runner.run_verify = traced

prompt = ("def add(a, b): return a + b\n"
          "def sub(a, b): return a - b\n" * 3)
sp = SamplingParams(temperature=1.0, max_tokens=30, ignore_eos=True)
llm.generate([prompt], sp, use_tqdm=False)
