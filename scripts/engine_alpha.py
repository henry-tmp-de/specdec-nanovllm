#!/usr/bin/env python
"""engine_alpha.py —— 在【真引擎】上同时测两种接受率口径，一次运行全拿到。

为什么需要它
------------
任务要求把接受率「测硬」。项目里已有的两个口径各有一个坑：

  · **自由生成口径**（`acceptance_matrix.py`：`accept_rate = Σaccepted/Σproposed`）
    —— 在 `temperature=1.0` 下跨批次漂移 ±4~13%，同一棵树重复跑 `out_md5` 都不同。
  · **离线确定性口径**（`probe_draft_accept.py` / `det_alpha.py`：α = 1 − TV(p,q)）
    —— 完全确定，但它是**我们自己按窗口语义重算的 draft 上下文**，绕开了引擎。

本脚本把**恒等式 α = 1 − TV(p, q) 直接搬到引擎自己的前向上**：

    每次 `verify_batch` 拿到的 (draft logits q, target logits p) 就是
    「同一个位置、同一个已确认前缀」下的两个分布 —— TV 一算就是该位置的一步接受概率。
    而且它不依赖采样（分布是前向算出来的，确定），所以**噪声远小于自由生成口径**。

于是同一次运行里同时得到：

  α_free  = Σaccepted / Σproposed     ← 自由生成口径（与 acceptance_matrix 完全一致）
  α_step  = mean_{(round,j)} [1 − TV(p_j, q_j)]   ← 确定性口径
  α_step1 = 只看每个 round 的第 1 个候选（这一位才决定这一轮能推进多少）

★ 只 monkeypatch `nanovllm.spec_decode.verify.verify_batch`（纯观测），
  **引擎代码一个字节没改、target 一个字节没动**。

用法（一个配置一个进程）
------------------------
    CTX=8192 WINDOW=0   HEAD=0 TAG=w0_8k    python -u scripts/engine_alpha.py
    CTX=8192 WINDOW=1280 HEAD=0 TAG=m5_8k   python -u scripts/engine_alpha.py
    CTX=8192 WINDOW=1280 HEAD=2 TAG=p2m5_8k python -u scripts/engine_alpha.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")


def install_patch(cap):
    import torch
    import nanovllm.spec_decode.verify as V

    _orig = V.verify_batch

    def _patched(draft_probs, target_logits, draft_tokens,
                 temperatures=None, draft_is_point_mass=False):
        res = _orig(draft_probs, target_logits, draft_tokens,
                    temperatures, draft_is_point_mass)
        if cap["on"] and not draft_is_point_mass:
            k = draft_probs.shape[1]
            dev = draft_probs.device
            if temperatures is None:
                temperatures = torch.ones(draft_probs.shape[0], device=dev)
            T = temperatures.reshape(-1, 1, 1).float()
            q = torch.softmax(draft_probs.float() / T, dim=-1)
            p = torch.softmax(target_logits[:, :k, :].float() / T, dim=-1)
            tv = (0.5 * (p - q).abs().sum(-1)).detach().cpu()      # (B, k)
            cap["rows"].append(tv)
            cap["rtv"].append(float(tv.mean()))
            cap["acc"] += int(res.accepted.sum())
            cap["prop"] += int(res.n_proposed.sum())
            cap["nacc"].extend([int(x) for x in res.accepted])
            cap["rounds"] += int(draft_probs.shape[0])
            cap["k"].append(k)
            del q, p
        return res

    V.verify_batch = _patched
    return _orig


def main():
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from acceptance_matrix import build_prompts, TARGET as TGT, DRAFT_BF16

    ctx = int(os.environ.get("CTX", "8192"))
    window = int(os.environ.get("WINDOW", "0"))
    head = int(os.environ.get("HEAD", "0"))
    B = int(os.environ.get("B", "4"))
    OUTLEN = int(os.environ.get("OUTLEN", "256"))
    K = int(os.environ.get("K", "6"))
    MAXLEN = int(os.environ.get("MAXLEN", "16640"))
    MAX_SEQS = int(os.environ.get("MAX_SEQS", "4"))
    TAG = os.environ.get("TAG", "run")
    OUT = os.environ.get("OUT", "")

    cap = {"on": False, "rows": [], "acc": 0, "prop": 0, "rounds": 0,
           "nacc": [], "k": [], "rtv": []}
    install_patch(cap)

    kw = dict(enforce_eager=False, max_model_len=MAXLEN,
              max_num_batched_tokens=max(MAXLEN, 16384), max_num_seqs=MAX_SEQS,
              spec_k=K, spec_method="draft", draft_model=DRAFT_BF16,
              spec_batch_threshold=0)
    if window:
        kw["spec_draft_window"] = window
    if head:
        kw["spec_draft_head_blocks"] = head
    if os.environ.get("ALIGN", "0") == "1":
        kw["spec_draft_win_align"] = True
    llm = LLM(TGT, **kw)

    tok = AutoTokenizer.from_pretrained(TGT)
    prompts = build_prompts(tok, ctx, B)
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)

    print(f"[eng] TAG={TAG} ctx={ctx} B={B} window={window} head={head} "
          f"k={K} out={OUTLEN} maxlen={MAXLEN} max_num_seqs={MAX_SEQS}", flush=True)
    # 预热（不进统计）
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=16,
                                              ignore_eos=True), use_tqdm=False)
    cap["on"] = True
    res = llm.generate(prompts, sp, use_tqdm=False)
    cap["on"] = False

    tv = torch.cat(cap["rows"], dim=0)          # (N, maxk)
    nflat = tv.numel()
    flat = tv.reshape(-1)
    step1 = tv[:, 0]
    out = {
        "tag": TAG, "ctx": ctx, "B": B, "window": window, "head": head,
        "k": K, "outlen": OUTLEN, "maxlen": MAXLEN, "max_num_seqs": MAX_SEQS,
        "rounds": cap["rounds"], "positions": int(nflat),
        "tokens_delivered": int(sum(len(r["token_ids"]) for r in res)),
        "alpha_free": round(cap["acc"] / cap["prop"], 5),
        "accepted_per_round": round(cap["acc"] / max(1, cap["rounds"]), 4),
        "tokens_per_round": round(sum(len(r["token_ids"]) for r in res) / max(1, cap["rounds"]), 4),
        "alpha_step": round(1 - float(flat.mean()), 5),
        "alpha_step_se": round(float(flat.std()) / (nflat ** 0.5), 5),
        "tv_mean": round(float(flat.mean()), 5),
        "alpha_step1": round(1 - float(step1.mean()), 5),
        "alpha_step1_se": round(float(step1.std()) / (step1.numel() ** 0.5), 5),
        "tv_mean_by_j": [round(float(tv[:, j].mean()), 5)
                         for j in range(tv.shape[1])],
        "nacc_hist": {str(i): cap["nacc"].count(i) for i in sorted(set(cap["nacc"]))},
        "round_tv_first40": [round(v, 4) for v in cap["rtv"][:40]],
        "round_tv_last40": [round(v, 4) for v in cap["rtv"][-40:]],
        "round_tv_bucket": [round(sum(cap["rtv"][i:i + 50]) / len(cap["rtv"][i:i + 50]), 4)
                            for i in range(0, len(cap["rtv"]), 50)],
    }
    print("@@EA@@" + json.dumps(out, ensure_ascii=False), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
