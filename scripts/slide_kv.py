#!/usr/bin/env python
"""slide_kv.py —— 逐位置对拍：引擎环形缓冲里的 KV vs「按声明的窗口重算一遍」的 KV。

对某一轮（PROBE_ROUND），在引擎自己的前向【之前】：
  1. 取该请求已确认的 token_ids（长 L）、按引擎自己声明的 b0/vf 算出窗口 [w0, L)；
  2. 用纯 local-causal 前向把这一段重算一遍，捕获每层 Attention 吃到的 (k, v)；
  3. 按 DraftWindow.slot(p) 把引擎 k_cache/v_cache 里对应槽位读出来；
  4. 逐位置比 max|Δk|、max|Δv|（层默认取 0 与中间层）。

结论读法：若某些位置的 Δ 很大（≫bf16 舍入量级 ~1e-2），
说明环里那一格的 KV ≠ 「最后一次以正确上下文写下去」的值 ⇒ 状态被写坏。

用法：
  CTX=8192 WINDOW=1280 HEAD=0 B=1 OUTLEN=128 K=6 MAX_SEQS=1 PROBE_ROUND=4 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/slide_kv.py
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

ST = {"i": -1, "done": False, "out": None}


def _attn_modules(model):
    return [m for m in model.modules()
            if hasattr(m, "k_cache") and hasattr(m, "v_cache")]


def _ref_forward_capture(model, ids, poss, layers):
    """local-causal 前向，返回 {layer_idx: (k, v)}（RoPE 之后的）。"""
    import torch
    from nanovllm.utils.context import set_context, reset_context

    dev = next(model.parameters()).device
    mods = _attn_modules(model)
    cap = {}
    handles = []
    for li in layers:
        def mk(li=li):
            def hook(module, args):
                cap[li] = (args[1].detach().float().cpu(),
                           args[2].detach().float().cpu())
            return hook
        handles.append(mods[li].register_forward_pre_hook(mk()))
    saved = []
    for m in mods:
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
            logits = model.compute_logits(h)[-1].float().cpu()
    finally:
        reset_context()
        for m, kc, vc in saved:
            m.k_cache = kc
            m.v_cache = vc
        for hd in handles:
            hd.remove()
    return cap, logits


def install():
    import torch
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.spec_decode.draft_proposer import (
        DraftModelProposer, DraftWindow)
    from nanovllm.utils.context import set_context, reset_context

    _orig_req = ModelRunner._draft_request
    _orig_prop = DraftModelProposer.propose_batch

    def _req(self, seq):
        r = _orig_req(self, seq)
        if ST.get("runner") is None:
            ST["runner"] = self
            ST["draft"] = self.spec_proposer.model
        r["_probe_tokens"] = list(seq.token_ids)
        return r

    def _prop(self, reqs):
        pl = os.environ.get("PROBE_L")
        trig = False
        if pl is not None:
            trig = (len(reqs[0].get("_probe_tokens") or []) >= int(pl))
        if not ST["done"] and ST.get("on") and (trig or (pl is None and ST["i"] + 1 == int(
                os.environ.get("PROBE_ROUND", "4")))):
            ST["done"] = True
            ST["out"] = _analyze(ST["draft"], reqs, int(os.environ.get("J", "0")))
        chains, logits = _orig_prop(self, reqs)
        ST["i"] += 1
        return chains, logits

    ModelRunner._draft_request = _req
    DraftModelProposer.propose_batch = _prop


def _analyze(draft, reqs, j):
    """对 reqs[j] 做逐位置 KV 对拍。返回可 JSON 化的 dict。"""
    import torch
    from nanovllm.spec_decode.draft_proposer import DraftWindow
    r = reqs[j]
    toks = r["_probe_tokens"]
    M = int(r.get("window_blocks") or 0)
    P = int(r.get("head_blocks") or 0)
    bs = int(ST["runner"].block_size)
    L = len(toks)
    vf = int(r.get("valid_from", 0))
    w = DraftWindow(r["draft_ring"], bs, M, P)
    b0 = w.b0(L - 1, vf)
    clen = w.ctx_len(L - 1, vf)
    w0 = b0 * bs
    # REF_MODE:
    #   declared -> 头部常驻段 [0,P*bs) + 尾部 [b0*bs, L)（= 引擎声明的可见集）
    #   tailonly -> 只有尾部 [b0*bs, L)
    #   prefill  -> 头部 + 尾部但尾段起点用 REF_START（= 预填充那趟的 chunk）
    mode = os.environ.get("REF_MODE", "declared")
    ref_start = int(os.environ.get("REF_START", "0"))
    head_end = min(P * bs, L) if mode != "tailonly" else 0
    t_start = ref_start if (mode == "prefill" and ref_start) else w0
    segs = []
    if head_end > 0:
        segs.append((0, head_end))
    if t_start < L:
        segs.append((t_start, L))
    ids, poss = [], []
    for (a, b) in segs:
        ids.extend(toks[a:b]); poss.extend(range(a, b))
    ids = list(ids)
    poss_ref = list(poss)   # ★ 参考前向要用【完整】的 positions（不能传过滤后的）
    # 引擎侧只对所声明的窗口 [w0, L) 的槽位负责
    slots = [w.slot(p) if p >= w0 else -1 for p in poss]
    cmp_from = int(os.environ.get("CMP_FROM", str(w0)))
    keep = [i for i, p in enumerate(poss) if p >= cmp_from and slots[i] >= 0]
    poss_all, slots_all = poss, slots
    poss = [poss_all[i] for i in keep]
    slots = [slots_all[i] for i in keep]
    keepidx = keep
    mods = _attn_modules(draft)
    nlayers = len(mods)
    layers = sorted(set([0, nlayers // 2, nlayers - 1]))
    cap, q_ref = _ref_forward_capture(draft, ids, poss_ref, layers)

    dev = next(draft.parameters()).device
    res = {"L": L, "b0": b0, "ctx_len": clen, "w0": w0, "nwin": len(ids),
           "valid_from": vf, "ring": list(r["draft_ring"]), "M": M, "P": P,
           "bs": bs, "nlayers": nlayers, "layers": layers,
           "q_ref_head": [round(float(x), 4) for x in q_ref[:5]],
           "per_layer": {}}
    nblk32 = sum(1 for p in poss if p >= 8192)
    res["n_newblk"] = nblk32
    for li in layers:
        m = mods[li]
        kv_heads = m.k_cache.shape[2]
        hd = m.k_cache.shape[3]
        kE = m.k_cache.reshape(-1, kv_heads, hd).float().cpu()
        vE = m.v_cache.reshape(-1, kv_heads, hd).float().cpu()
        idx = torch.tensor(slots, dtype=torch.long)
        k_eng = kE[idx]
        v_eng = vE[idx]
        k_ref, v_ref = cap[li]
        k_ref = k_ref[keepidx]
        v_ref = v_ref[keepidx]
        dk = (k_eng - k_ref).abs().amax(dim=(1, 2))
        dv = (v_eng - v_ref).abs().amax(dim=(1, 2))
        res["per_layer"][str(li)] = {
            "dk": [round(float(x), 4) for x in dk],
            "dv": [round(float(x), 4) for x in dv],
            "absmax_ref_k": round(float(k_ref.abs().max()), 3),
        }
    import collections
    for li in layers:
        pl2 = res["per_layer"][str(li)]
        agg = collections.OrderedDict()
        for ii, pp in enumerate(poss):
            b = pp // bs
            a = agg.setdefault(b, [0.0, 0.0, 0, 0, 0])
            a[0] = max(a[0], pl2["dk"][ii]); a[1] = max(a[1], pl2["dv"][ii])
            if pl2["dk"][ii] > 0.05: a[2] += 1
            if pl2["dv"][ii] > 0.05: a[3] += 1
            a[4] += 1
        pl2["by_block"] = {str(b): [round(x, 4) if isinstance(x, float) else x
                                    for x in a] for b, a in agg.items()}
    res["poss"] = poss
    res["blocks"] = [p // bs for p in poss]
    res["slots"] = slots
    return res


def main():
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    from acceptance_matrix import build_prompts

    ctx = int(os.environ.get("CTX", "8192"))
    window = int(os.environ.get("WINDOW", "1280"))
    head = int(os.environ.get("HEAD", "0"))
    B = int(os.environ.get("B", "1"))
    OUTLEN = int(os.environ.get("OUTLEN", "128"))
    K = int(os.environ.get("K", "6"))
    MAXLEN = int(os.environ.get("MAXLEN", "16640"))
    MAX_SEQS = int(os.environ.get("MAX_SEQS", "1"))
    TAG = os.environ.get("TAG", "kv")
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
    llm = LLM(TARGET, **kw)

    tok = AutoTokenizer.from_pretrained(TARGET)
    prompts = build_prompts(tok, ctx, B)
    sp = SamplingParams(temperature=1.0, max_tokens=OUTLEN, ignore_eos=True)

    print(f"[kv] TAG={TAG} ctx={ctx} B={B} W={window} P={head} k={K} "
          f"out={OUTLEN} probe_round={os.environ.get('PROBE_ROUND','4')}",
          flush=True)
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=16,
                                             ignore_eos=True), use_tqdm=False)
    ST["on"] = True
    llm.generate(prompts, sp, use_tqdm=False)
    ST["on"] = False

    out = {"tag": TAG, "ctx": ctx, "B": B, "window": window, "head": head,
           "k": K, "probe": ST["out"]}
    print("@@KV@@")
    print(json.dumps(out, ensure_ascii=False))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("[kv] wrote", OUT)


if __name__ == "__main__":
    main()
