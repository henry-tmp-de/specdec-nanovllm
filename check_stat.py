"""统计意义上的无损性检验
   随机采样下，逐 token 对比无意义（随机序列本就不同）。
   正确做法：用【固定随机数】跑大样本，比较 token 频率分布。
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
mode, prompt = sys.argv[1], sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N = 40

kw = dict(enforce_eager=True, max_num_batched_tokens=16384)
if mode == "draft":
    kw.update(spec_k=k, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
torch.manual_seed(42)
llm = LLM(TARGET, **kw)
sp = SamplingParams(temperature=1.0, max_tokens=25, ignore_eos=True)

# N 条独立的短prompt，统计输出 token 分布
prompts = [f"{prompt}{i}" for i in range(N)]
all_tokens = []
for p in prompts:
    o = llm.generate([p], sp, use_tqdm=False)[0]
    all_tokens.extend(o["token_ids"])
cnt = collections.Counter(all_tokens)
top = cnt.most_common(12)
print("@@STAT@@" + json.dumps({"mode": mode, "n": len(all_tokens), "top": top,
                                "distinct": len(cnt)}))
