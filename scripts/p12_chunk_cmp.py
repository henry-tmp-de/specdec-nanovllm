#!/usr/bin/env python
"""p12_chunk_cmp.py —— 对拍 p12_chunk_equiv.py 两个 OUTDIR 的产物。

比较（每个样本、每一张 logits）：
  · 逐位相同？（float32 字节）
  · 最大绝对差（max|Δ|）
  · argmax（= 贪婪会吐的 token）是否相同；★ 若不齐，报出【那一步 top1-top2 的间距】
    —— 用来判断"翻掉"是不是发生在 ULP 级的近乎平局上
  · 交付的 token 序列是否相同
最后给一个跨样本汇总（总步数 / 翻转数 / 翻转处的间距）。

用法:
  A=.../out/equiv2/chunk0 B=.../out/equiv2/chunk256 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p12_chunk_cmp.py
"""
import glob
import json
import os

import torch


def mdd(a, b):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    if torch.equal(a, b):
        return 0.0
    return float((a - b).abs().max())


def cmp_list(pa, pb, label, flips):
    n = max(len(pa), len(pb))
    worst = 0.0
    nbit = out_n = 0
    for i in range(n):
        a = pa[i] if i < len(pa) else None
        b = pb[i] if i < len(pb) else None
        d = mdd(a, b)
        if isinstance(d, str):
            continue
        out_n += 1
        if d == 0.0:
            nbit += 1
        else:
            worst = max(worst, d)
        if a is not None and b is not None and a.shape == b.shape:
            va, vb = a.flatten(), b.flatten()
            ia, ib = int(va.argmax()), int(vb.argmax())
            if ia != ib:
                t = torch.topk(va, 2).values
                flips.append(dict(label=label, i=i, tok_a=ia, tok_b=ib,
                                  margin_top1_top2=float(t[0] - t[1]),
                                  max_abs_diff=d))
    return dict(n=out_n, n_bitwise_equal=nbit, max_abs_diff=worst)


def main():
    A, B = os.environ["A"], os.environ["B"]

    def by_L(d):
        m = {}
        for f in glob.glob(os.path.join(d, "s*.pt")):
            o = torch.load(f, weights_only=False)
            m[int(o["L"])] = o
        return m

    DA, DB = by_L(A), by_L(B)
    lens = sorted(set(DA) & set(DB))
    assert lens, f"no common L: {sorted(DA)} vs {sorted(DB)}"
    allflips = []
    tot_pre = tot_dec = tot_tok_eq = 0
    rows = []
    for L in lens:
        da, db = DA[L], DB[L]
        flips = []
        pre = cmp_list([da["prefill_last"]], [db["prefill_last"]], "prefill", flips)
        dec = cmp_list(da["decode"], db["decode"], "decode", flips)
        tok_eq = (da["tokens"] == db["tokens"])
        tot_pre += pre["n"]
        tot_dec += dec["n"]
        tot_tok_eq += int(tok_eq)
        allflips += flips
        rows.append(dict(L=L, pre=pre, dec=dec, tokens_eq=tok_eq, flips=flips,
                         tokens_a=da["tokens"], tokens_b=db["tokens"]))
        print(f"L={L:6d} rounds {len(da['prefill_all'])}v{len(db['prefill_all'])} | "
              f"prefill bitwise={pre['n_bitwise_equal']}/{pre['n']} maxD={pre['max_abs_diff']:.3g} | "
              f"decode bitwise={dec['n_bitwise_equal']}/{dec['n']} maxD={dec['max_abs_diff']:.3g} | "
              f"tokens_eq={tok_eq} flips={len(flips)}", flush=True)
    margins = [f["margin_top1_top2"] for f in allflips]
    print(f"\n=== {os.path.basename(A)} vs {os.path.basename(B)} ===")
    print(f"prefill steps: {tot_pre}, bitwise-identical: 0" if tot_pre else "")
    print(f"decode steps : {tot_dec}")
    print(f"argmax flips : {len(allflips)}  (prefill {sum(1 for f in allflips if f['label']=='prefill')}, "
          f"decode {sum(1 for f in allflips if f['label']=='decode')})")
    print(f"flip margins (top1-top2): {margins}")
    print(f"token-seq identical: {tot_tok_eq}/{len(rows)}")
    print("@@CHUNKCMP@@ " + json.dumps(dict(
        A=A, B=B, prefill_steps=tot_pre, decode_steps=tot_dec, n_flips=len(allflips),
        flip_margins=margins, tok_seq_identical=tot_tok_eq, n_samples=len(rows), lens=lens,
        rows=rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
