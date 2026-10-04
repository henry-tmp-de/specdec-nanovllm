"""打印每一步的 draft 候选、target 分布、接受判定"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.verify import _softmax_with_temp

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

orig = llm.model_runner.run_verify
shown = [0]
def traced(seqs):
    s = seqs[0]
    if s.draft_tokens and shown[0] < 4:
        shown[0] += 1
        input_ids, positions = llm.model_runner.prepare_verify(seqs)
        logits = llm.model_runner.run_model(input_ids, positions, True)
        from nanovllm.utils.context import reset_context
        reset_context()
        cu = llm.model_runner._last_cu_seqlens_q
        n = 1 + len(s.draft_tokens)
        sub = logits[cu[0]:cu[0]+n].unsqueeze(0)
        t = torch.tensor([s.temperature], device=sub.device)
        P = _softmax_with_temp(sub, t)                # target 分布
        Q = s.draft_probs                              # draft 真实分布
        print(f"\n>>> len(token_ids)={len(s.token_ids)} draft={s.draft_tokens}")
        for j, d in enumerate(s.draft_tokens):
            pt = float(P[0, j+1, d]); qt = float(Q[j, d])
            ratio = min(1.0, pt/max(qt,1e-10))
            top = int(P[0, j+1].argmax())
            print(f"    候选[{j}]={d:>6}  p_target={pt:.4f}  q_draft={qt:.4f}  "
                  f"ratio={ratio:.4f}  argmax={top}({float(P[0,j+1,top]):.3f})")
    return orig(seqs)
llm.model_runner.run_verify = traced

sp = SamplingParams(temperature=0.01, max_tokens=16, ignore_eos=True)
llm.generate(["def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n"], sp, use_tqdm=False)
