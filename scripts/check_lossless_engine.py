"""引擎级无损性检验：投机路径的输出分布 == 基线路径的输出分布？

方法学要点（避开已知的两个陷阱）
--------------------------------
① temperature 必须 = 1.0。低温度会把分布压成 one-hot，p 恒为 1.0000，
   什么都测不出来（上一轮在这里白改了 8 轮）。
② 不做逐 token 对比。温度 1.0 是随机采样，两次运行本来就该不同。
   要比的是【token 频率分布】。
③ ★ 关键设计：用「基线 vs 基线(第二次独立运行)」当【噪声地板】。
   单看 TV(draft, base) 没有意义 —— 必须和 TV(base, base2) 同量级，
   才能说明差异只是采样噪声，而不是系统性偏差。

用法: python check_lossless_engine.py base|base2|draft|draft2
  base / base2   : 同一份基线配置跑两次 -> 噪声地板
  draft / draft2 : 同一份投机配置跑两次 -> 投机路径自己的噪声地板
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
mode = sys.argv[1]

kw = dict(enforce_eager=False, max_num_batched_tokens=16384)
if mode.startswith("draft"):   # draft / draft2 都用投机配置
    kw.update(spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

llm = LLM(TARGET, **kw)

PROMPTS = [
    "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\ndef calculate_max(numbers):\n",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, d_in, d_hidden, d_out):\n",
    "在机器学习中，梯度下降是一种常用的优化算法，它的基本思想是",
]
PER = 1200
sp = SamplingParams(temperature=1.0, max_tokens=PER, ignore_eos=True)

# 预热
llm.generate([PROMPTS[0]], SamplingParams(temperature=1.0, max_tokens=32, ignore_eos=True),
             use_tqdm=False)

cnt = collections.Counter()
for p in PROMPTS:
    cnt.update(llm.generate([p], sp, use_tqdm=False)[0]["token_ids"])

tot = sum(cnt.values())
print("@@L@@" + json.dumps({"mode": mode, "n": tot, "distinct": len(cnt),
                            "counts": dict(cnt)}))
