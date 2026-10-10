#!/usr/bin/env python
"""eval_ppl.py —— INT8 量化敏感度扫描：逐模块困惑度退化（Phase 0.5）

背景：用户直觉「只量化 FFN，因为 FFN 损害较小」。这必须用数据验：
把 q/k/v/o/gate/up/down 每一类**单独**量化（其余保持 bf16），量困惑度退化，
再算「性价比 = 省下的权重字节 / 困惑度退化」。

**完全不碰 nano-vllm 引擎**：用 transformers 直接前向，只把选中的 nn.Linear
换成我们的融合反量化路径（同一个 `nanovllm.layers.quant.int8_linear`）。
这样精度结论与引擎实现解耦，也不会被 paged attention 干扰。

口径：
  · 困惑度在**留出文本**（wikitext-2 raw test）上算，固定窗口 2048、固定分块数。
  · 所有配置用**同一段 token 流**，只有量化范围不同 —— 差异才可比。
  · bf16 权重常驻 GPU（一份克隆），每个配置从它还原后再量化，避免重复加载。

用法：
  python scripts/eval_ppl.py --model <bf16_dir> --configs all
  python scripts/eval_ppl.py --model <bf16_dir> --configs q,k,v,o,gate,up,down,ffn,ffn_no_down,block_all
"""
import os
import re
import sys
import json
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanovllm.layers.quant import quantize_weight, int8_linear, QMAX

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SHORT = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
         "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def load_corpus(n_chunks, seqlen):
    """优先读本地 txt；没有就从 wikitext-2 parquet 抽一段出来。"""
    txt = os.path.join(ROOT, "data", "wikitext2_test.txt")
    if not os.path.isfile(txt):
        pq = os.path.join(ROOT, "data", "wikitext2_test.parquet")
        assert os.path.isfile(pq), f"缺语料：{txt} / {pq}"
        import pyarrow.parquet as pqa
        col = pqa.read_table(pq).column("text").to_pylist()
        os.makedirs(os.path.dirname(txt), exist_ok=True)
        with open(txt, "w", encoding="utf-8") as f:
            f.write("".join(col))
    with open(txt, encoding="utf-8") as f:
        return f.read()


class Int8Wrapper(torch.nn.Module):
    """把 nn.Linear 的前向换成融合反量化的 int8 GEMM（含 bias）。"""

    def __init__(self, orig: torch.nn.Linear, q, s, gran, gs):
        super().__init__()
        self.bias = orig.bias
        self.q = q
        self.s = s
        self.gran = gran
        self.gs = gs

    def forward(self, x):
        return int8_linear(x, self.q, self.s, self.bias, self.gran, self.gs)


def pick(model, cfg):
    """按配置名挑出要量化的模块名列表。"""
    targets = {p: [] for p in PROJ}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        for p in PROJ:
            if name.endswith("." + p) or name == p:
                targets[p].append(name)
    names = []
    if cfg == "bf16":
        return []
    if cfg == "block_all":
        for p in PROJ:
            names += targets[p]
    elif cfg == "ffn":
        for p in ["gate_proj", "up_proj", "down_proj"]:
            names += targets[p]
    elif cfg == "ffn_no_down":
        for p in ["gate_proj", "up_proj"]:
            names += targets[p]
    elif cfg == "attn":
        for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            names += targets[p]
    elif cfg.startswith("first") or cfg.startswith("last") or cfg == "middle":
        n = int(cfg.split(":")[1]) if ":" in cfg else 6
        L = len(targets["q_proj"])
        if cfg.startswith("first"):
            keep = set(range(n))
        elif cfg.startswith("last"):
            keep = set(range(L - n, L))
        else:
            keep = set(range(n, L - n))
        for p in PROJ:
            names += [nm for i, nm in enumerate(targets[p]) if i in keep]
    else:
        for key in cfg.split(","):
            if key in SHORT.values():
                p = [k for k, v in SHORT.items() if v == key][0]
                names += targets[p]
            else:
                raise ValueError(f"未知配置片段: {key}")
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="bf16,block_all")
    ap.add_argument("--granularity", default="per_channel")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=60)
    ap.add_argument("--agree-chunks", type=int, default=20,
                    help="算「与 bf16 的 argmax 一致率」的块数（0=不算）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda"
    text = load_corpus(args.chunks, args.seqlen)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(text, return_tensors="pt").input_ids[0]
    nsamp = args.chunks * args.seqlen
    assert ids.numel() >= nsamp + 1, f"语料太短 {ids.numel()} < {nsamp+1}"
    ids = ids[:nsamp + 1]
    print(f"corpus tokens={ids.numel()} chunks={args.chunks} seqlen={args.seqlen}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev)
    model.eval()
    V = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params/1e9:.3f}B dtype={next(model.parameters()).dtype}")

    # 原始 bf16 权重存一份（GPU），每配置从它还原
    linears = {name: m for name, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)}
    orig_fwd = {name: m.forward for name, m in linears.items()}
    orig_w = {name: m.weight.detach().clone() for name, m in linears.items()}
    total_linear_bytes = sum(w.numel() * 2 for w in orig_w.values())

    def set_all_bf16():
        for name, m in linears.items():
            m.forward = orig_fwd[name]

    def set_quant(nameset):
        for name, m in linears.items():
            m.forward = orig_fwd[name]
        torch.cuda.empty_cache()
        for name in nameset:
            m = linears[name]
            q, s = quantize_weight(orig_w[name], args.granularity, args.group_size)
            m.forward = Int8Wrapper(m, q, s, args.granularity, args.group_size).forward

    @torch.no_grad()
    def run_ppl(names, want_agree):
        nameset = set(names)
        set_quant(nameset)
        nll = 0.0
        ntok = 0
        nll_h = [0.0, 0.0]
        ntok_h = [0, 0]
        half = args.chunks // 2
        agree = 0
        agree_tot = 0
        for i in range(args.chunks):
            a = i * args.seqlen
            chunk = ids[a:a + args.seqlen].unsqueeze(0).to(dev)
            logits = model(chunk).logits[:, :-1].float()
            tgt = chunk[:, 1:]
            l = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                reduction="sum").item()
            nll += l
            ntok += tgt.numel()
            k = 0 if i < half else 1
            nll_h[k] += l
            ntok_h[k] += tgt.numel()
            if want_agree and i < args.agree_chunks:
                set_all_bf16()
                ref = model(chunk).logits[:, :-1].float()
                agree += (ref.argmax(-1) == logits.argmax(-1)).sum().item()
                agree_tot += ref.shape[0] * ref.shape[1]
                set_quant(nameset)
            del logits
        torch.cuda.empty_cache()
        return (math.exp(nll / ntok), ntok,
                (agree / agree_tot if agree_tot else None),
                [math.exp(nll_h[0] / max(ntok_h[0], 1)), math.exp(nll_h[1] / max(ntok_h[1], 1))])

    cfgs = args.configs.split(",")
    results = []
    base_ppl = None
    for cfg in cfgs:
        names = pick(model, cfg)
        quant_bytes = sum(orig_w[n].numel() * 2 for n in names)
        # ★ 反量化后的每参数字节：int8 1B + scale（per_channel: 2B/N；per_group: 2B/G）
        new_bytes = 0
        for n in names:
            N, K = orig_w[n].shape
            new_bytes += N * K + (N * 2 if args.granularity == "per_channel"
                                  else N * (K // args.group_size) * 2)
        ppl, ntok, agree, ppl_h = run_ppl(names, args.agree_chunks > 0 and cfg != "bf16")
        if cfg == "bf16":
            base_ppl = ppl
        d = (ppl / base_ppl - 1.0) * 100 if base_ppl else None
        if cfg == "bf16":
            base_h = ppl_h
        dh = ([ (ppl_h[k] / base_h[k] - 1.0) * 100 for k in (0, 1)]
              if base_ppl else None)
        rec = {
            "config": cfg, "ppl_delta_half_pct": dh, "n_modules": len(names), "ppl": ppl,
            "ppl_delta_pct": d, "tokens": ntok, "argmax_agree": agree,
            "linear_bytes_total": total_linear_bytes,
            "quant_bf16_bytes": quant_bytes,
            "quant_int8_bytes": new_bytes,
            "quant_frac_of_linear": quant_bytes / total_linear_bytes,
            "saved_frac": (quant_bytes - new_bytes) / total_linear_bytes,
            "ppl_half": ppl_h,
        }
        results.append(rec)
        print("@@P@@" + json.dumps(rec, ensure_ascii=False))

    print("\n=== SUMMARY (ppl over %d tokens, %s) ===" % (nsamp, args.granularity))
    print(f"{'config':>14} {'#mod':>5} {'PPL':>9} {'ΔPPL%':>8} {'argmax一致':>10} "
          f"{'占总线性层字节':>14} {'省下字节':>9} {'ΔPPL% 两半(噪声地板)':>20}")
    for r in results:
        ag = "-" if r["argmax_agree"] is None else f"{r['argmax_agree']*100:.2f}%"
        d = "-" if r["ppl_delta_pct"] is None else f"{r['ppl_delta_pct']:+.3f}"
        dh = r.get("ppl_delta_half_pct")
        dhs = "-" if not dh else f"[{dh[0]:+.2f},{dh[1]:+.2f}]"
        print(f"{r['config']:>14} {r['n_modules']:>5} {r['ppl']:>9.4f} {d:>8} {ag:>10} "
              f"{r['quant_frac_of_linear']*100:>13.1f}% {r['saved_frac']*100:>8.1f}% {dhs:>16}")
    print("@@DONE@@")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
