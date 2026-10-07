"""一次加载跑完两组检查：
  A. CUDA graph 路径 vs eager 路径，逐元素对拍 logits / 分布
  B. 拒绝采样内部数值：q 是不是概率被当 logits 又 softmax 了一次
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
import nanovllm.spec_decode.verify as V
from nanovllm.spec_decode.verify import _softmax_with_temp

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

llm = LLM(TARGET, enforce_eager=False, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
mr = llm.model_runner

# ---------------- A1. 验证前向：图 vs eager ----------------
assert mr.verify_graph is not None, "verify graph 没建起来"
assert mr.spec_proposer._graph is not None, "draft graph 没注入"
DIFF = []
orig_rvf = mr.run_verify_forward
def wrapped_rvf(input_ids, positions):
    ex = mr._verify_extra
    use_graph = (ex is not None and ex["n"] == 1 and input_ids.numel() == mr._verify_n)
    if not use_graph:
        return orig_rvf(input_ids, positions)
    g = orig_rvf(input_ids, positions).clone()
    e = mr.run_model(input_ids, positions, True)
    DIFF.append({"max_abs": (g - e).abs().max().item(),
                 "argmax_same": int((g.argmax(-1) == e.argmax(-1)).sum().item()),
                 "rows": int(g.shape[0])})
    return g
mr.run_verify_forward = wrapped_rvf

# ---------------- A2. draft 前向：图 vs eager ----------------
prop = mr.spec_proposer
gcnt = {"n": 0}
orig_propose = prop.propose
DDIFF = []
def wrapped_propose(bt, ctx_len, last_token, temp):
    chain_g, probs_g = orig_propose(bt, ctx_len, last_token, temp)
    if gcnt["n"] < 8:
        gcnt["n"] += 1
        chain_e, probs_e = prop._propose_eager(bt, ctx_len, last_token, temp)
        # ★ 只能比第 0 步：两路都是【随机采样】，第 1 步起输入 token 就分叉了，
        #   分布自然不同 —— 那不是 bug，是采样。
        #   第 0 步的输入（last_token / pos / 缓存状态）完全相同，必须一致。
        DDIFF.append({"step0_max_abs": (probs_g[0] - probs_e[0]).abs().max().item(),
                      "step0_argmax_same": bool(probs_g[0].argmax() == probs_e[0].argmax())})
    return chain_g, probs_g
prop.propose = wrapped_propose

# ---------------- B. 拒绝采样内部数值 ----------------
LOG = []
orig_vb = V.verify_batch
def wrapped_vb(draft_probs, target_logits, draft_tokens, temperatures=None,
               draft_is_point_mass=False):
    res = orig_vb(draft_probs, target_logits, draft_tokens, temperatures, draft_is_point_mass)
    if len(LOG) < 400:
        with torch.no_grad():
            q = draft_probs if draft_is_point_mass else _softmax_with_temp(draft_probs, temperatures)
            p = _softmax_with_temp(target_logits, temperatures)
            k = draft_tokens.shape[1]
            pfd = p[:, 1:k+1, :] if p.shape[1] >= k+1 else p[:, :k, :]
            idx = draft_tokens.unsqueeze(-1)
            qt = q.gather(2, idx).squeeze(-1)
            pt = pfd.gather(2, idx).squeeze(-1)
            LOG.append({
                "q_after_softmax_max": float(q.max()),
                "draft_probs_max": float(draft_probs.max()),
                "qt_mean": float(qt.mean()), "pt_mean": float(pt.mean()),
                "ratio_mean": float(torch.clamp(pt / qt.clamp_min(1e-10), max=1.0).mean()),
                "accept_rate": float(res.accept_mask.float().mean()),
            })
    return res
V.verify_batch = wrapped_vb

P = ("def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n"
     "        total += num\n    return total\n\ndef calculate_max(numbers):\n")
sp = SamplingParams(temperature=1.0, max_tokens=48, ignore_eos=True)
outs = llm.generate([P], sp, use_tqdm=False)

def mean(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 7) if rows else None

out = {
    "A1_verify_calls": len(DIFF),
    "A1_verify_max_abs_logit_diff": round(max((r["max_abs"] for r in DIFF), default=-1), 6),
    "A1_verify_argmax_all_same": all(r["argmax_same"] == r["rows"] for r in DIFF),
    "A2_draft_calls": len(DDIFF),
    "A2_draft_step0_max_abs_diff": round(max((r["step0_max_abs"] for r in DDIFF), default=-1), 8),
    "A2_draft_step0_argmax_all_same": all(r["step0_argmax_same"] for r in DDIFF),
    "B_calls": len(LOG),
    "B_draft_probs_max_mean": mean(LOG, "draft_probs_max"),
    "B_q_after_softmax_max_mean": mean(LOG, "q_after_softmax_max"),
    "B_qt_mean": mean(LOG, "qt_mean"),
    "B_pt_mean": mean(LOG, "pt_mean"),
    "B_ratio_mean": mean(LOG, "ratio_mean"),
    "B_accept_rate": mean(LOG, "accept_rate"),
    "gen_tokens": len(outs[0]["token_ids"]),
}
print("@@G@@" + json.dumps(out))
