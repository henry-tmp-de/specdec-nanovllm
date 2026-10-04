"""第一次 verify 就分叉了 —— 打印那一步的完整信息"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
mr = llm.model_runner

# 包住 run 的第一次 verify
orig_rv = mr.run_verify
shown = [0]
def traced(seqs):
    s = seqs[0]
    if shown[0] == 0 and s.draft_tokens:
        shown[0] = 1
        print(f"\n{'='*70}")
        print(f"第一次 verify")
        print(f"  len(token_ids)   = {len(s.token_ids)}")
        print(f"  num_cached_tokens= {s.num_cached_tokens}")
        print(f"  last_token       = {s.last_token}")
        print(f"  block_table      = {s.block_table}")
        print(f"  draft 候选       = {s.draft_tokens}")
        # draft 分布
        q = s.draft_probs
        print(f"  draft q[0] 分布: top3 = {torch.topk(q[0],3)}")
        print(f"  draft q[1] 分布: top3 = {torch.topk(q[1],3)}")
        # target 分布
        input_ids, positions = mr.prepare_verify(seqs)
        print(f"  送 target 的 input_ids  = {input_ids.tolist()}")
        print(f"  送 target 的 positions  = {positions.tolist()}")
        logits = mr.run_model(input_ids, positions, True)
        reset_context()
        cu = mr._last_cu_seqlens_q
        print(f"  cu_seqlens_q = {cu}, logits rows = {logits.shape[0]}")
        P = torch.softmax(logits.float(), dim=-1)
        for j in range(logits.shape[0]):
            top = int(P[j].argmax())
            print(f"    位置{j} (pos={positions[j].item()}): argmax={top} "
                  f"p={float(P[j,top]):.4f}  token={llm.tokenizer.decode([top])!r}")
    return orig_rv(seqs)
mr.run_verify = traced

sp = SamplingParams(temperature=0.01, max_tokens=5, ignore_eos=True)
llm.generate(["def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n"], sp, use_tqdm=False)
