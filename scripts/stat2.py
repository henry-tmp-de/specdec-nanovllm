"""统计检验：单条长序列，比较 token 分布（避免 draft 模式太慢）"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
mode = sys.argv[1]
kw = dict(enforce_eager=True, max_num_batched_tokens=16384)
if mode == "draft":
    kw.update(spec_k=4, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
torch.manual_seed(7)
llm = LLM(TARGET, **kw)
sp = SamplingParams(temperature=1.0, max_tokens=60, ignore_eos=True)
# 同一批 prompt，每条跑 3 次采样，看分布是否一致
P = "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n"
allt = []
for i in range(2):
    allt.extend(llm.generate([P], sp, use_tqdm=False)[0]["token_ids"])
c = collections.Counter(allt)
print("@@S@@" + json.dumps({"mode": mode, "n": len(allt), "distinct": len(c),
                            "top": c.most_common(10)}))
