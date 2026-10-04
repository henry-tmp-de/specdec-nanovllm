"""逐步追踪提议过程"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)
p = llm.model_runner.spec_proposer
print("scheduler 有 proposer 吗:", llm.scheduler.spec_proposer is not None)

prompt = ("def add(a, b): return a + b\n"
          "def sub(a, b): return a - b\n"
          "def add(a, b): return a + b\n"
          "def sub(a, b): return a - b\n")
sp = SamplingParams(temperature=1.0, max_tokens=40, ignore_eos=True)
llm.add_request(prompt, sp)

for step in range(12):
    seqs, is_pf = llm.scheduler.schedule()
    s = seqs[0]
    if not is_pf:
        print(f"[{step}] len={len(s.token_ids)} n_sched={s.num_scheduled_tokens} "
              f"draft={s.draft_tokens} stats={p.stats()}")
    toks = llm.model_runner.call("run", seqs, is_pf)
    llm.scheduler.postprocess(seqs, toks, is_pf)

print("\n最终索引里的 key (前10):")
for k, v in list(p._index.items())[:10]:
    print(f"  {k} -> {list(v)}")
print("\n当前序列末尾 5 token:", s.token_ids[-5:])
