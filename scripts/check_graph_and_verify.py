"""一次加载跑完两组检查：
  A. CUDA graph 路径 vs eager 路径，逐元素对拍 logits / 分布
     （P6 后：draft 图按 B 分桶、验证图按 (B,k) 分桶，这里比对 B=1 与批量两条）
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
assert mr.verify_graphs, "verify graph 没建起来"
assert mr.spec_proposer._graphs, "draft graph 没注入"
DIFF = []
orig_rvf = mr.run_verify_forward
def wrapped_rvf(input_ids, positions):
    ex = mr._verify_extra
    key = (ex["n"], ex["k"]) if ex is not None else None
    use_graph = (ex is not None and ex["full_k"] and key in mr.verify_graphs
                 and input_ids.numel() == ex["total_q"])
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
orig_pb = prop.propose_batch
DDIFF = []
def _draft_once(req, force_eager):
    saved = prop._graphs
    if force_eager:
        prop._graphs = {}
    try:
        chains, logits = orig_pb([req])
    finally:
        prop._graphs = saved
    return chains[0], logits[0]

def wrapped_pb(reqs):
    chains, logits = orig_pb(reqs)
    if gcnt["n"] < 8 and len(reqs) == 1:
        gcnt["n"] += 1
        req = dict(reqs[0])
        # ★ 只能比第 0 步：两路都是【随机采样】，第 1 步起输入 token 就分叉了，
        #   分布自然不同 —— 那不是 bug，是采样。
        #   第 0 步的输入（last_token / pos / 缓存状态）完全相同，必须一致。
        chain_e, logits_e = _draft_once(req, force_eager=True)
        DDIFF.append({"step0_max_abs": (logits[0][0] - logits_e[0]).abs().max().item(),
                      "step0_argmax_same": bool(logits[0][0].argmax() == logits_e[0].argmax())})
    return chains, logits
prop.propose_batch = wrapped_pb

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
            # ★ 对齐：候选 j <-> 第 j 行（不是 p[:, 1:k+1] —— 那是整体挪一行的错位写法）
            pfd = p[:, :k, :]
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
    "verify_graph_hits": mr.verify_graph_hits,
    "verify_graph_fallbacks": mr.verify_graph_fallbacks,
    "draft_graphs": sorted(mr.draft_graphs.keys()),
    "verify_graphs": sorted(list(k) for k in mr.verify_graphs.keys()),
    "graph_mem": mr.graph_mem,
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
