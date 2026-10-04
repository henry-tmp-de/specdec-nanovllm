"""看候选 vs target 分布的差距"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)
p_ = llm.model_runner.spec_proposer
orig = llm.model_runner.run_verify
def traced(seqs):
    s = seqs[0]
    if not s.draft_tokens:
        return orig(seqs)
    # 手动跑一遍看 p[draft]
    from nanovllm.spec_decode.verify import verify_batch, _softmax_with_temp
    input_ids, positions = llm.model_runner.prepare_verify(seqs)
    logits = llm.model_runner.run_model(input_ids, positions, True)
    from nanovllm.utils.context import reset_context
    reset_context()
    cu = llm.model_runner._last_cu_seqlens_q
    sub = logits[cu[0]:cu[0]+1+len(s.draft_tokens)].unsqueeze(0)
    P = _softmax_with_temp(sub, torch.tensor([1.0], device=sub.device))
    print(f"\n>>> draft={s.draft_tokens}")
    for j, d in enumerate(s.draft_tokens):
        pt = float(P[0, j+1, d])
        top = int(P[0, j+1].argmax())
        print(f"    draft[{j}]={d:>6}: p(draft)={pt:.4f}   argmax={top}  p(argmax)={float(P[0,j+1,top]):.4f}")
    return orig(seqs)
llm.model_runner.run_verify = traced

prompt = "def add(a, b): return a + b\n" * 3
sp = SamplingParams(temperature=1.0, max_tokens=25, ignore_eos=True)
llm.generate([prompt], sp, use_tqdm=False)
