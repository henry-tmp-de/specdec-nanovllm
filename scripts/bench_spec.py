"""bench_spec.py — 投机解码加速比基准

用法:
  python bench_spec.py base  <max_tokens> <graph 0/1> [temp]
  python bench_spec.py draft <k> <max_tokens> <graph 0/1> [temp]

graph=0 -> enforce_eager=True（不拍图，等于上一轮的配置）
graph=1 -> enforce_eager=False（普通 decode 和投机两条路径都拍图）
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

mode  = sys.argv[1]
k     = int(sys.argv[2])
mt    = int(sys.argv[3])
graph = int(sys.argv[4]) if len(sys.argv) > 4 else 1
temp  = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

kw = dict(enforce_eager=not graph, max_num_batched_tokens=16384)
if mode == "draft":
    kw.update(spec_k=k, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

llm = LLM(TARGET, **kw)
S = {"proposed": 0, "accepted": 0, "verify": 0, "steps": 0}
if mode == "draft":
    # ★ 接受率必须直接数 accept_mask，不能用 landed - verify：
    #   k 个候选全被接受时不会加 bonus，那个式子在「全接受」时只有 (k-1)/k，
    #   永远到不了 1，会把 0.84 的真实接受率报成 0.4 左右。
    import nanovllm.spec_decode.verify as V
    orig_vb = V.verify_batch
    def tvb(draft_probs, target_logits, draft_tokens, temperatures=None,
            draft_is_point_mass=False):
        res = orig_vb(draft_probs, target_logits, draft_tokens, temperatures,
                      draft_is_point_mass)
        if not draft_is_point_mass:
            S["proposed"] += int(res.n_proposed.sum())
            S["accepted"] += int(res.accept_mask.sum())
            S["verify"] += 1
        return res
    V.verify_batch = tvb
    op = llm.scheduler.postprocess
    def tp(seqs, t, pf):
        S["steps"] += 1
        return op(seqs, t, pf)
    llm.scheduler.postprocess = tp

P = ("def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n"
     "        total += num\n    return total\n\ndef calculate_max(numbers):\n")
sp = SamplingParams(temperature=temp, max_tokens=mt, ignore_eos=True)

llm.generate([P], sp, use_tqdm=False)          # warmup
torch.cuda.synchronize()
for key in S:
    S[key] = 0

t0 = time.time()
outs = llm.generate([P], sp, use_tqdm=False)
torch.cuda.synchronize()
dt = time.time() - t0

n = len(outs[0]["token_ids"])
steps = S["steps"] if mode == "draft" else n
print("@@B@@" + json.dumps({
    "mode": mode, "k": k, "graph": graph, "temp": temp,
    "tokens": n, "wall_s": round(dt, 3),
    "tok_per_s": round(n / dt, 1),
    "steps": steps,
    "tok_per_step": round(n / max(steps, 1), 3),
    "accept_rate": round(S["accepted"] / S["proposed"], 3) if S["proposed"] else None,
    "distinct": len(set(outs[0]["token_ids"])),
}))
