"""无损性端到端检查：draft 模式 vs 基线，同 seed 应完全一致"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
mode = sys.argv[1]

kw = dict(enforce_eager=True, max_num_batched_tokens=16384)
if mode == "draft":
    kw.update(spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

torch.manual_seed(1234)
llm = LLM(TARGET, **kw)
# ★ temperature 不能太低：softmax(logits/T) 在 T=0.01 时会压成 one-hot（p=1.0），
#   那样任何模型都一样，无法对比两条路径的差异。用 T=1.0（正常采样）。
sp = SamplingParams(temperature=1.0, max_tokens=30, ignore_eos=True)
out = llm.generate(["def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n"], sp, use_tqdm=False)[0]
print("@@TOK@@" + json.dumps(out["token_ids"]))
print("@@TXT@@" + json.dumps(out["text"]))
