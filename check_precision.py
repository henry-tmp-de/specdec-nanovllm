"""两件事：
   A. 精度假设检验 —— 同一条 token 单独算 vs 批量算，logits 差多少？
   B. draft 的 KV cache 绑定是否真的独立（有没有和 target 串到同一块显存）
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft",
          draft_model=os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B"),
          spec_batch_threshold=0)

mr = llm.model_runner
print("\n" + "=" * 66)
print("B. draft / target 的 KV cache 绑定检查")
print("=" * 66)
print(f"  target kv_cache: {tuple(mr.kv_cache.shape)}  @ {mr.kv_cache.data_ptr():#x}")
print(f"  draft  kv_cache: {tuple(mr.draft_kv_cache.shape)}  @ {mr.draft_kv_cache.data_ptr():#x}")
print(f"  显存地址是否分离: {mr.kv_cache.data_ptr() != mr.draft_kv_cache.data_ptr()}")

# 逐层检查指针，确认没有任何一层串了
t_ptrs, d_ptrs = [], []
for m in mr.model.modules():
    if hasattr(m, "k_cache") and hasattr(m, "v_cache") and m.k_cache.numel():
        t_ptrs.append(m.k_cache.data_ptr())
for m in mr.draft_model.modules():
    if hasattr(m, "k_cache") and hasattr(m, "v_cache") and m.k_cache.numel():
        d_ptrs.append(m.k_cache.data_ptr())
print(f"  target 层数 {len(t_ptrs)}, draft 层数 {len(d_ptrs)}")
overlap = set(t_ptrs) & set(d_ptrs)
print(f"  ★ 两边指针重叠数 = {len(overlap)}  {'（正常，应为 0）' if len(overlap)==0 else '★★★ 串了！'}")

print()
print("=" * 66)
print("A. 精度假设检验：同一 token，单独算 vs 跟别的 token 一起批量算")
print("=" * 66)

bs = 1
toks = llm.tokenizer.encode("def calculate_sum(numbers):\n    total = 0\n")
ctx_len = len(toks)
device = mr.kv_cache.device

def build(seqs_tokens, seqlens, block_tables):
    n = sum(len(s) for s in seqs_tokens)
    ids, pos, slot = [], [], []
    cu_q, cu_k = [0], [0]
    for s, L, bt in zip(seqs_tokens, seqlens, block_tables):
        ids.extend(s); pos.extend(range(L - len(s), L))
        cu_q.append(cu_q[-1] + len(s)); cu_k.append(cu_k[-1] + L)
        for i in range((L - len(s)) // bs, (L + bs - 1) // bs):
            slot_start = bt[i] * 256 + ((L - len(s)) % 256 if i == (L-1)//bs else 0)
            cnt = min(256, L - i * 256) - ((L - len(s)) % 256 if i == (L-1)//bs else 0)
            slot.extend(range(slot_start, slot_start + cnt))
    bt_pad = [b + [0] for b in block_tables]
    T = lambda x, d: torch.tensor(x, dtype=d, device=device)
    return (T(ids, torch.int64), T(pos, torch.int64), T(cu_q, torch.int32),
            T(cu_k, torch.int32), T(slot, torch.int32), T(bt_pad, torch.int32))

one = [toks[-1:]]                      # 单独一个
bt1 = [[0]]

r = {}
for tag, seqs, Ls in [("单独", one, [ctx_len]),
                      ("批量3", [toks[-1:] + [100, 200]], [ctx_len + 2])]:
    seqs = seqs * bs
    ids, pos, cu_q, cu_k, slot, bt = build(seqs, Ls, bt1 * bs)
    set_context(True, cu_q, cu_k, max(Ls), max(Ls), slot, None, bt, is_spec_verify=True)
    with torch.inference_mode():
        out = mr.model(ids, pos)
    lg = mr.model.compute_logits(out)
    reset_context()
    r[tag] = lg[0].float()             # 位置 0 的 logits

d = (r["单独"] - r["批量3"][:len(r['单独'])][0]).abs()
print(f"  logits 形状: {tuple(r['单独'].shape)}")
print(f"  max|Δlogit| = {d.max():.6f}")
print(f"  mean|Δlogit|= {d.mean():.6f}")
print(f"  logit 量级  = {r['单独'].abs().mean():.3f}")
print(f"  相对误差    = {(d.max()/r['单独'].abs().mean()*100):.4f}%")
a1 = r["单独"].argmax().item(); a2 = r["批量3"][:len(r['单独'])][0].argmax().item()
print(f"  argmax:单独={a1} 批量={a2}  {'一致' if a1==a2 else '不一致'}")
p1 = torch.softmax(r["单独"], -1); p2 = torch.softmax(r["批量3"][:len(r['单独'])][0], -1)
print(f"  max|Δprob| = {(p1-p2).abs().max():.6f}")
print(f"  TV距离     = {0.5*(p1-p2).abs().sum():.6f}")
