#!/usr/bin/env python
"""cmp_dump_q.py —— 引擎倒出来的 draft logits 到底等于哪一种「draft 上下文」？

拿 eng_dump_q.py 存下的 q（引擎第一次投机轮里 draft 对第 1 个候选位置的原始 logits），
在离线 HF 上按几种候选上下文各算一遍下一 token 分布，比三组 TV：

  TV(engine_q, HF_q_X)   —— 引擎那一次到底像哪种上下文
  TV(HF_q_X , HF_p)      —— 该上下文下「理想的」一步拒绝率（= 1 − α）
  TV(engine_q, engine_p) —— 引擎自己那一步的拒绝率（引擎内 TV）

用法: DUMP=out/eng/dump_p0m5.pt python -u scripts/cmp_dump_q.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    import torch
    from transformers import AutoModelForCausalLM

    DUMP = os.environ.get("DUMP", "out/eng/dump_p0m5.pt")
    DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
    TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
    dev = "cuda"
    d = torch.load(DUMP, weights_only=False)
    ids = list(d["prompt"]) + list(d.get("gen") or [])
    q_eng = d["q"][0].float().cuda()
    p_eng = d["p"][0].float().cuda()
    fg = d["fg"][0]
    pos0 = fg["pos"][0]
    M = int(d["window"]) // 256
    P = int(d["head"])
    b = pos0 // 256
    print(f"[cmp] window={d['window']}(M={M}) head={P} ctx={d['ctx']} "
          f"first_fg pos={fg['pos']} cl={fg['cl']} bt={fg['bt']}", flush=True)

    drf = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16,
                                               device_map=dev).eval()
    tgt = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16,
                                               device_map=dev).eval()
    torch.set_grad_enabled(False)

    def hid(model, segs):
        lay_i, lay_p = [], []
        for (a, c) in segs:
            lay_i.extend(ids[a:c])
            lay_p.extend(range(a, c))
        h = model.model(torch.tensor([lay_i], device=dev),
                        position_ids=torch.tensor([lay_p], device=dev),
                        use_cache=False).last_hidden_state[0]
        last_a = segs[-1][0]
        off = sum(c - a for (a, c) in segs[:-1]) + (pos0 - last_a)
        return h[off:off + 1]

    def qof(segs):
        return drf.lm_head(hid(drf, segs))[0].float()

    def clip(segs):
        out = [(a, min(c, pos0 + 1)) for (a, c) in segs]
        return [(a, c) for (a, c) in out if c > a]

    end = pos0 + 1
    cands = {"full": [(0, end)]}
    for k in range(1, M + 3):
        cands[f"tail{k*256}"] = [(max(0, end - k * 256), end)]
    if P:
        cands[f"head{P}+win"] = [(0, P * 256), (max(P * 256, end - M * 256), end)]
    # 引擎那次窗口 prefill 的锚点（b0 = (end-1)//bs - M + 1）:
    cands["eng_chunk_anchor"] = [(max(0, ((fg["pos"][0] - 1) // 256 - M + 1) * 256), end)]

    p_hf = tgt.lm_head(hid(tgt, [(0, end)]))[0].float()
    print(f"[cmp] 引擎自己那一步 TV(draft, target) = "
          f"{float(0.5*(q_eng.softmax(-1)-p_eng.softmax(-1)).abs().sum()):.5f}", flush=True)
    print(f"[cmp] {'上下文':<24}{'token':>7}  TV(eng_q, HF_q)  TV(HF_q, HF_p)", flush=True)
    for name, segs in cands.items():
        segs = clip(segs)
        q = qof(segs)
        tv1 = float(0.5 * (q.softmax(-1) - q_eng.softmax(-1)).abs().sum())
        tv2 = float(0.5 * (q.softmax(-1) - p_hf.softmax(-1)).abs().sum())
        print(f"[cmp] {name:<24}{sum(c-a for a,c in segs):>7}  {tv1:>14.5f}  {tv2:>14.5f}",
              flush=True)
    tvp = float(0.5 * (p_hf.softmax(-1) - p_eng.softmax(-1)).abs().sum())
    print(f"[cmp] target: TV(HF_p, engine_p) = {tvp:.5f}  "
          f"(argmax HF={int(p_hf.argmax())} eng={int(p_eng.argmax())})", flush=True)
    print(f"[cmp] draft argmax: engine={int(q_eng.argmax())}  "
          f"cand(full)={int(qof(clip([(0,end)])).argmax())}", flush=True)


if __name__ == "__main__":
    main()
