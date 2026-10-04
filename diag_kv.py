"""检查：target 的 KV cache 里，位置上的内容是否是「真实 KV」还是「草稿残留」"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
mr = llm.model_runner

orig_prepare = mr.prepare_prefill
def prep(seqs):
    out = orig_prepare(seqs)
    s = seqs[0]
    print(f"\n[prefill] len={len(s.token_ids)} block_table={s.block_table} "
          f"num_cached={s.num_cached_tokens} n_sched={s.num_scheduled_tokens}")
    return out
mr.prepare_prefill = prep

sched = llm.scheduler
orig_post = sched.postprocess
step = [0]
def post(seqs, toks, pf):
    s = seqs[0]
    print(f"  [post-{step[0]}{'P' if pf else 'D'}] len={len(s.token_ids)} "
          f"block_table={s.block_table} num_cached={s.num_cached_tokens} "
          f"n_sched={s.num_scheduled_tokens} toks={str(toks)[:40]}")
    step[0] += 1
    return orig_post(seqs, toks, pf)
sched.postprocess = post

orig_prep_v = mr.prepare_verify
def prepv(seqs):
    s = seqs[0]
    print(f"  [verify] len={len(s.token_ids)} block_table={s.block_table} "
          f"num_cached={s.num_cached_tokens} n_sched={s.num_scheduled_tokens} "
          f"draft={s.draft_tokens}")
    return orig_prep_v(seqs)
mr.prepare_verify = prepv

sp = SamplingParams(temperature=0.01, max_tokens=8, ignore_eos=True)
llm.generate(["def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n"], sp, use_tqdm=False)
