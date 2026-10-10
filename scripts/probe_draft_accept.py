#!/usr/bin/env python
"""probe_draft_accept.py —— 「量化 draft 会不会掉接受率」的大样本离线测量

为什么必须单独测
----------------
引擎里测到的接受率**有采样噪声**：B=1、一轮 6 个候选、64 轮 → 才 384 个样本，
标准误 ~2.2%，比要分辨的效应还大。引擎数字只能说「方向」，说不了「多少」。

但接受率有恒等式（拒绝采样的数学性质）：

        α = E_{x~q}[ min(1, p[x]/q[x]) ] = Σ_x min(p[x], q[x]) = 1 − TV(p, q)

p = target 分布（**一个字节没动**），q = draft 分布（量化后变了）。
在同一批上下文上把 TV 算出来，就能用上万个位置把 Δα 定到 ~0.3% 精度，完全绕开引擎。

用法: CUDA_VISIBLE_DEVICES=0 python -u scripts/probe_draft_accept.py
env: PRESET(ffn/all/down/attn) SEQLEN(2048) NWIN(6) GRAN(per_channel)
"""
import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanovllm.layers.quant import quantize_weight, int8_linear

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
NWIN = int(os.environ.get("NWIN", "6"))
GRAN = os.environ.get("GRAN", "per_channel")
GS = int(os.environ.get("GS", "128"))
PRESET = os.environ.get("PRESET", "ffn")
PATS = {"ffn": r"\.mlp\.(gate_proj|up_proj|down_proj)$",
        "all": r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        "down": r"\.mlp\.down_proj$",
        "attn": r"\.(q_proj|k_proj|v_proj|o_proj)$"}


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    tgt = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16,
                                               device_map="cuda").eval()
    drf = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16,
                                               device_map="cuda").eval()
    with open(WIKI, encoding="utf-8") as f:
        ids_all = tok(f.read()).input_ids
    wins = [ids_all[i * SEQLEN + 137: (i + 1) * SEQLEN + 137] for i in range(NWIN)]
    npos = NWIN * (SEQLEN - 1)
    print(f"{NWIN} 个窗口 × {SEQLEN} token，共 {npos} 个位置参与统计")

    rx = re.compile(PATS[PRESET])
    mods = [(n, m) for n, m in drf.named_modules() if rx.search(n)]
    orig_w = {n: m.weight.detach().clone() for n, m in mods}
    orig_fwd = {n: m.forward for n, m in mods}
    print(f"draft 量化范围 preset={PRESET}: {len(mods)} 个模块")

    def set_quant(on):
        for n, m in mods:
            if on:
                q, s = quantize_weight(orig_w[n], GRAN, GS)
                m.weight = torch.nn.Parameter(q, requires_grad=False)
                m.weight_scale = s
                m.quant_granularity, m.quant_group_size = GRAN, GS

                def fwd(x, _m=m):
                    return int8_linear(x, _m.weight, _m.weight_scale, _m.bias,
                                       _m.quant_granularity, _m.quant_group_size)
                m.forward = fwd
            else:
                m.weight = torch.nn.Parameter(orig_w[n].clone(), requires_grad=False)
                m.forward = orig_fwd[n]

    @torch.no_grad()
    def tvs():
        """逐位置 TV(target, draft) —— 不缓存整份分布，只留每窗口一行。"""
        out = []
        for w in wins:
            x = torch.tensor([w], device="cuda")
            p = torch.softmax(tgt(x).logits[0, :-1].float(), dim=-1)
            q = torch.softmax(drf(x).logits[0, :-1].float(), dim=-1)
            out.append(0.5 * (p - q).abs().sum(-1).cpu())
            del p, q
            torch.cuda.empty_cache()
        return torch.cat(out)

    set_quant(False)
    tv_bf = tvs()
    set_quant(True)
    tv_q = tvs()
    set_quant(False)

    n = tv_bf.numel()
    se_bf = float(tv_bf.std() / n ** 0.5)
    se_q = float(tv_q.std() / n ** 0.5)
    res = {
        "positions": int(n), "preset": PRESET, "granularity": GRAN,
        "tv_target_bf16draft": round(float(tv_bf.mean()), 5), "se_bf": round(se_bf, 5),
        "tv_target_int8draft": round(float(tv_q.mean()), 5), "se_int8": round(se_q, 5),
        "alpha_bf16": round(1 - float(tv_bf.mean()), 5),
        "alpha_int8": round(1 - float(tv_q.mean()), 5),
        "delta_alpha": round(float(tv_bf.mean()) - float(tv_q.mean()), 5),
        "delta_alpha_se": round((se_bf ** 2 + se_q ** 2) ** 0.5, 5),
        "note": "Δα 为负 = 量化后接受率变差；|Δα| < 2*se 时视为不显著",
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("@@DA@@" + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
