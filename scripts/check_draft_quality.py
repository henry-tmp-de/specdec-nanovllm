"""draft 模型的【固有】质量：0.6B 和 4B 的下一 token 分布到底差多少？

为什么必须单独测
----------------
投机解码的接受率有恒等式：
    α = E_{x~q}[ min(1, p[x]/q[x]) ] = Σ_x min(p[x], q[x]) = 1 − TV(p, q)
所以 α 完全由两个模型分布的 TV 距离决定。
引擎里测到 α≈0.16（也就是 TV≈0.84），这对同家族的两个模型来说高得反常。
必须先排除「引擎算错了 draft 的上下文」，才能下结论。

这里绕开引擎，用 transformers 原生直接跑两个模型，
在【完全相同】的上下文上比下一 token 分布 —— 这是 α 的理论上界。

用法: python check_draft_quality.py
"""
import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

tok = AutoTokenizer.from_pretrained(TARGET)
tgt = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16, device_map="cuda").eval()
drf = AutoModelForCausalLM.from_pretrained(DRAFT,  dtype=torch.bfloat16, device_map="cuda").eval()

TEXT = (
    "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n"
    "        total += num\n    return total\n\n"
    "def calculate_max(numbers):\n    largest = numbers[0]\n"
    "    for num in numbers:\n        if num > largest:\n            largest = num\n"
    "    return largest\n\n"
    "The quick brown fox jumps over the lazy dog. Machine learning models are trained on\n"
)
ids = tok(TEXT, return_tensors="pt").input_ids.to("cuda")
L = ids.shape[1]

with torch.no_grad():
    lt = tgt(ids).logits[0].float()          # (L, V)
    lq = drf(ids).logits[0].float()

pt = torch.softmax(lt, dim=-1)
pq = torch.softmax(lq, dim=-1)

# 只看最后 60 个位置（前面的位置上下文太短，不代表真实解码场景）
sl = slice(max(0, L - 60), L)
tv = 0.5 * (pt[sl] - pq[sl]).abs().sum(-1)          # 逐位置 TV
agree = (pt[sl].argmax(-1) == pq[sl].argmax(-1)).float().mean()

# 直接仿真一次「接受率」：从 q 采样，按 min(1,p/q) 接受
torch.manual_seed(0)
samp = torch.multinomial(pq[sl], 200, replacement=True)              # (n_pos, 200)
p_s = pt[sl].gather(1, samp); q_s = pq[sl].gather(1, samp)
alpha = torch.clamp(p_s / q_s.clamp_min(1e-12), max=1.0).mean()

print("@@D@@" + json.dumps({
    "prompt_tokens": L,
    "positions": int(L - max(0, L - 60)),
    "mean_TV": round(float(tv.mean()), 4),
    "min_TV": round(float(tv.min()), 4),
    "max_TV": round(float(tv.max()), 4),
    "argmax_agreement": round(float(agree), 4),
    "implied_acceptance_1_minus_TV": round(1 - float(tv.mean()), 4),
    "sampled_acceptance_alpha": round(float(alpha), 4),
}))
