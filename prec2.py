"""最简精度检验：直接对比两次 forward 的 logits"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import set_context, reset_context

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384)
mr = llm.model_runner
dev = mr.kv_cache.device

def fwd(tokens, seqlen, nb_blocks=8):
    """算给定 tokens 在 [0, seqlen) 位置上的 logits"""
    n = len(tokens)
    ids = torch.tensor(tokens, dtype=torch.int64, device=dev)
    pos = torch.arange(seqlen - n, seqlen, device=dev)
    cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=dev)
    slot = torch.arange(seqlen - n, seqlen, dtype=torch.int32, device=dev)
    bt = torch.zeros((1, nb_blocks), dtype=torch.int32, device=dev)
    set_context(True, cu_q, cu_k, seqlen, seqlen, slot, None, bt, is_spec_verify=True)
    with torch.inference_mode():
        out = mr.model(ids, pos)
    lg = mr.model.compute_logits(out.clone())
    reset_context()
    return lg.float()

toks = llm.tokenizer.encode("def calculate_sum(numbers):\n    total = 0\n")
L = len(toks)
print(f"\nprompt token 数 = {L}")

# ① 单独算最后一个 token（decode 模式：q_len=1, kv_len=L）
a = fwd(toks[-1:], L)[0]
# ② 把它和后2 个 token 一起算（verify 模式：q_len=3, kv_len=L+2）
b = fwd(toks[-1:] + [500, 600], L + 2)[0]      # 第 0 行就是同一位置

d = (a - b).abs()
print(f"\n{'='*60}")
print(f"logits 量级    = {a.abs().mean():.4f}")
print(f"max|Δlogit|    = {d.max():.6f}")
print(f"mean|Δlogit|   = {d.mean():.8f}")
print(f"argmax 单独={a.argmax().item()}  批量={b.argmax().item()}")
p1, p2 = torch.softmax(a,-1), torch.softmax(b,-1)
print(f"max|Δprob|     = {(p1-p2).abs().max():.6f}")
print(f"TV 距离        = {0.5*(p1-p2).abs().sum():.6f}")
print(f"\n判读：Δlogit< 0.01 且 argmax 一致 -> 精度影响可忽略")
print(f"      argmax 不一致 -> 精度足以改变采样结果")
