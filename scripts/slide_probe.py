#!/usr/bin/env python
"""slide_probe.py —— 滑窗路径的「引擎增量状态 vs 理想截断」逐轮对拍探针。

要回答的唯一问题
----------------
引擎自己的 draft 前向（读环形缓冲里【增量写出来】的 KV）与
「把同样的窗口 token 喂给同一个 draft、重新算一遍」得到的分布，
**是不是同一个分布**？

  · 逐位相同（只差 ULP） ⇒ 引擎确实实现了「上下文 = 最后 M 块」这条语义
    ⇒ 剩下的 α 损失是【截断的固有代价】，不是滑窗写入的 bug。
  · 明显不同           ⇒ 环里的 KV 不是「最后 M 块」应有的内容 ⇒ **状态被写坏了**。

顺带量出「截断」对 target 那一侧的影响（p_full vs p_window），
用来解释离线口径为什么在 P=0 上偏乐观。

只 monkeypatch，引擎代码一个字节不改。

用法
----
    CTX=8192 WINDOW=1280 HEAD=0 B=4 TAG=m5 \
        /home/ziru/nano-vllm/venv/bin/python -u scripts/slide_probe.py
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

REC = {"i": -1, "rounds": []}
ST = {}


def _local_forward(model, ids, poss, want_full=False):
    """纯 local-causal 前向（不写任何 cache），返回最后一位的 logits。

    把 model 里所有 k_cache/v_cache 临时摘成空张量 ⇒ attention 里的
    store_kvcache 被跳过，走 flash_attn_varlen 本地 k/v 分支。
    """
    import torch
    from nanovllm.utils.context import set_context, reset_context

    dev = next(model.parameters()).device
    saved = []
    for m in model.modules():
        if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
            saved.append((m, m.k_cache, m.v_cache))
            m.k_cache = torch.tensor([])
            m.v_cache = torch.tensor([])
    n = len(ids)
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    set_context(True, cu, cu, n, n, None, None, None)
    try:
        with torch.inference_mode():
            h = model(torch.tensor(ids, dtype=torch.int64, device=dev),
                      torch.tensor(poss, dtype=torch.int64, device=dev))
            logits = model.compute_logits(h)
        ret = logits[-1].float()
        if want_full:
            ret = (ret, logits.float())
    finally:
        reset_context()
        for m, kc, vc in saved:
            m.k_cache = kc
            m.v_cache = vc
    return ret


def _tv(p, q):
    import torch
    p = torch.softmax(p.float(), dim=-1)
    q = torch.softmax(q.float(), dim=-1)
    return float(0.5 * (p - q).abs().sum())


def install():
    import torch
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.spec_decode.draft_proposer import DraftModelProposer, DraftWindow
    from nanovllm.spec_decode import verify as V

    _orig_req = ModelRunner._draft_request
    _orig_prop = DraftModelProposer.propose_batch
    _orig_ver = V.verify_batch

    def _req(self, seq):
        r = _orig_req(self, seq)
        if ST.get("runner") is None:
            ST["runner"] = self
        r["_probe_tokens"] = list(seq.token_ids)
        return r

    def _prop(self, reqs):
        stride = int(os.environ.get("PROBE_STRIDE", "1"))
        want = (REC["i"] + 1) % stride == 0 and ST.get("on")
        # ★ 参考前向必须在引擎自己的前向【之前】做（引擎前向会改 context /
        #   写 KV 环，之后再算就不是「同一个窗口的干净重算」了）
        ref = None
        _CTRL = {}
        if want:
            r = reqs[0]
            toks = r.get("_probe_tokens")
            if toks is not None:
                M = int(r.get("window_blocks") or 0)
                P = int(r.get("head_blocks") or 0)
                bs = self.block_size
                L = len(toks)
                vf = int(r.get("valid_from", 0))
                if M > 0:
                    w = DraftWindow(r["draft_ring"], bs, M, P)
                    b0 = w.b0(L - 1, vf)
                    clen = w.ctx_len(L - 1, vf)
                else:
                    b0, clen = 0, L
                w0 = b0 * bs
                mode = os.environ.get("REF_MODE", "tailonly")
                if mode == "declared" and P > 0:
                    he = min(P * bs, L)
                    ids = list(toks[0:he]) + list(toks[w0:L])
                    ps = list(range(0, he)) + list(range(w0, L))
                else:
                    ids, ps = list(toks[w0:L]), list(range(w0, L))
                q_win = _local_forward(self.model, ids, ps)
                # ★ 对照组：把「多一个块的前置上下文」这个扰动单独加在同一个窗口上，
                #   量出模型对它的敏感度 TV(q_declared, q_extra_block)
                if os.environ.get("CONTROL", "0") == "1" and w0 >= bs:
                    if mode == "declared" and P > 0:
                        he = min(P * bs, L)
                        ids2 = list(toks[0:he]) + list(toks[w0 - bs:L])
                        ps2 = list(range(0, he)) + list(range(w0 - bs, L))
                    else:
                        ids2 = list(toks[w0 - bs:L])
                        ps2 = list(range(w0 - bs, L))
                    q_x = _local_forward(self.model, ids2, ps2).cpu()
                    _CTRL["v"] = _tv(q_win.cpu(), q_x)
                p_win = _local_forward(ST["runner"].model, list(toks[w0:L]),
                                       list(range(w0, L)))
                ref = dict(L=L, b0=b0, ctx_len=clen, valid_from=vf, w0=w0,
                           nwin=L - w0, q_win=q_win.cpu(), p_win=p_win.cpu())
                if "v" in _CTRL:
                    ref["tv_ctrl_extra_block"] = _CTRL.pop("v")
                if os.environ.get("FULL", "0") == "1":
                    ref["q_full"] = _local_forward(
                        self.model, list(toks), list(range(L))).cpu()

        chains, logits = _orig_prop(self, reqs)
        REC["i"] += 1
        if ref is not None:
            q_eng = logits[0, 0].float().cpu()
            ref["q_engine"] = q_eng
            ref["tv_eng_vs_win"] = _tv(q_eng, ref["q_win"])
            ref["maxabs_eng_vs_win"] = float((q_eng - ref["q_win"]).abs().max())
            ref["max_abs_logit"] = float(q_eng.abs().max())
            ref["tv_eng_vs_full"] = (_tv(q_eng, ref["q_full"])
                                     if "q_full" in ref else None)
            ref.pop("q_engine", None)
            ref.pop("q_full", None)
        REC["rounds"].append(dict(i=REC["i"], rec=ref))
        return chains, logits

    def _ver(draft_probs, target_logits, draft_tokens, temperatures=None,
             draft_is_point_mass=False):
        res = _orig_ver(draft_probs, target_logits, draft_tokens, temperatures,
                        draft_is_point_mass)
        if ST.get("on") and not draft_is_point_mass:
            i = REC["i"]
            if 0 <= i < len(REC["rounds"]):
                k = draft_probs.shape[1]
                dev = draft_probs.device
                if temperatures is None:
                    temperatures = torch.ones(draft_probs.shape[0], device=dev)
                T = temperatures.reshape(-1, 1, 1).float()
                q = torch.softmax(draft_probs.float() / T, dim=-1)
                p = torch.softmax(target_logits[:, :k, :].float() / T, dim=-1)
                tv = (0.5 * (p - q).abs().sum(-1)).detach().cpu()      # (B,k)
                ent = REC["rounds"][i]
                ent["tv_by_j"] = [round(float(x), 5) for x in tv.mean(0)]
                ent["tv_all"] = round(float(tv.mean()), 5)
                ent["acc"] = int(res.accepted.sum())
                ent["npro"] = int(res.n_proposed.sum())
                ref = ent.get("rec")
                if ref is not None:
                    p0 = target_logits[0, 0].float().cpu()
                    ref["tv_p_eng_vs_win"] = _tv(p0, ref["p_win"])
                    ref["tv_pwin_vs_qwin"] = _tv(ref["p_win"], ref["q_win"])
                    # ★ 决定性数字：目标分布与「理想截断草稿分布」的 TV
                    #   = 若引擎的缓存完全等于「重算一遍声明的窗口」时会得到的 TV
                    ref["tv_p_eng_vs_qwin"] = _tv(p0, ref["q_win"])
                    ref.pop("p_win", None)
                    ref.pop("q_win", None)
        return res

    ModelRunner._draft_request = _req
    DraftModelProposer.propose_batch = _prop
    V.verify_batch = _ver


def main():
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from acceptance_matrix import build_prompts

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

    install()

    kw = dict(enforce_eager=False, max_model_len=MAXLEN,
              max_num_batched_tokens=max(MAXLEN, 16384), max_num_seqs=MAX_SEQS,
              spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if window:
        kw["spec_draft_window"] = window
    if head:
        kw["spec_draft_head_blocks"] = head
    if os.environ.get("ALIGN", "0") == "1":
        kw["spec_draft_win_align"] = True
    llm = LLM(TARGET, **kw)

    tok = AutoTokenizer.from_pretrained(TARGET)
    prompts = build_prompts(tok, ctx, B)
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)

    print(f"[probe] TAG={TAG} ctx={ctx} B={B} W={window} P={head} k={K} "
          f"out={OUTLEN}", flush=True)
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=16,
                                             ignore_eos=True), use_tqdm=False)
    ST["on"] = True
    res = llm.generate(prompts, sp, use_tqdm=False)
    ST["on"] = False

    out = {"tag": TAG, "ctx": ctx, "B": B, "window": window, "head": head,
           "k": K, "outlen": OUTLEN,
           "tokens_delivered": [len(r["token_ids"]) for r in res]}
    rounds = []
    for ent in REC["rounds"]:
        e = {kk: vv for kk, vv in ent.items() if kk != "rec"}
        ref = ent.get("rec")
        if ref is not None:
            e["probe"] = {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                          for kk, vv in ref.items()}
        rounds.append(e)
    out["rounds"] = rounds
    tv_all = [r["tv_all"] for r in rounds if "tv_all" in r]
    out["alpha_step"] = round(1 - sum(tv_all) / len(tv_all), 5) if tv_all else None
    out["n_rounds"] = len(tv_all)
    print("@@SLIDE@@")
    print(json.dumps(out, ensure_ascii=False))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("[probe] wrote", OUT)


if __name__ == "__main__":
    main()
