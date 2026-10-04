"""精度会不会导致『逐 token 完全一致』失败？
   统计：连续 30 步 decode vs 同样的 token 走 verify 路径，argmax 有多少次不同"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import set_context, reset_context

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384)
mr = llm.model_runner
dev = mr.kv_cache.device

toks = llm.tokenizer.encode(
    "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n")

def fwd(tokens, seqlen, nb=16):
    n = len(tokens)
    ids = torch.tensor(tokens, dtype=torch.int64, device=dev)
    pos = torch.arange(seqlen - n, seqlen, device=dev)
    cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=dev)
    slot = torch.arange(seqlen - n, seqlen, dtype=torch.int32, device=dev)
    bt = torch.zeros((1, nb), dtype=torch.int32, device=dev)
    set_context(True, cu_q, cu_k, seqlen, seqlen, slot, None, bt, is_spec_verify=True)
    with torch.inference_mode():
        out = mr.model(ids, pos)
    lg = mr.model.compute_logits(out.clone())
    reset_context()
    return lg.float()

L = len(toks)
diff = 0
total = 0
for step in range(30):
    a = fwd(toks[-1:], L)[0]                      # 单独算（decode）
    b = fwd(toks[-1:] + [500, 600], L + 2)[0]     # 带2个未来token（verify）
    if a.argmax().item() != b.argmax().item():
        diff += 1
    total += 1
    toks = toks + [a.argmax().item()]
    L += 1

print(f"\n{'='*60}")
print(f"对比 {total} 个位置：")
print(f"  argmax 不同: {diff}/{total} = {diff/total*100:.1f}%")
print(f"\n判读：")
print(f"  0%  -> 精度完全不影响，两路径必然一致")
print(f"  少量 -> 精度会让偶发分叉，实属正常，可解释为「数值噪声」")
print(f"  >30% -> 精度不足以解释你看到的『20 次重复同一token』")
