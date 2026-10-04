"""代码补全场景：对比 n-gram 提议的候选 vs 模型真实输出"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.verify import _softmax_with_temp

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=4, spec_batch_threshold=0)

orig = llm.model_runner.run_verify
shown = [0]
def traced(seqs):
    s = seqs[0]
    if not s.draft_tokens:
        return orig(seqs)
    if shown[0] < 3:
        shown[0] += 1
        input_ids, positions = llm.model_runner.prepare_verify(seqs)
        logits = llm.model_runner.run_model(input_ids, positions, True)
        from nanovllm.utils.context import reset_context
        reset_context()
        cu = llm.model_runner._last_cu_seqlens_q
        n = 1 + len(s.draft_tokens)
        sub = logits[cu[0]:cu[0]+n].unsqueeze(0)
        P = _softmax_with_temp(sub, torch.tensor([1.0], device=sub.device))
        print(f"\n>>> 位置(倒数3个已生成token) = {s.token_ids[-3:]}")
        print(f"    draft = {s.draft_tokens}")
        for j, d in enumerate(s.draft_tokens):
            pt = float(P[0, j+1, d])
            top = int(P[0, j+1].argmax())
            print(f"      候选[{j}]={d:>6}  p={pt:.4f}  |模型最想要={top:>6}  p={float(P[0,j+1,top]):.4f}")
        # 统计：每个位置候选是否为 argmax
        for j, d in enumerate(s.draft_tokens):
            pt = float(P[0, j+1, d]); top = int(P[0, j+1].argmax())
            print(f"      [{j}] p={pt:.4f} argmax={top} {'HIT' if d==top else 'MISS'}")
    return orig(seqs)
llm.model_runner.run_verify = traced

PROMPT = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total


def calculate_max(numbers):
    maximum = numbers[0]
    for num in numbers:
        if num > maximum:
            maximum = num
    return maximum
"""
sp = SamplingParams(temperature=0.2, max_tokens=60, ignore_eos=True)
llm.generate([PROMPT], sp, use_tqdm=False)
