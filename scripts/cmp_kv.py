#!/usr/bin/env python
"""cmp_kv.py —— 引擎环里的 draft KV，内容对不对？

引擎在窗口 prefill 时应该把 [b0*bs, end) 这段 token 的 KV 写进环里
（绝对块 b → 物理块 ring[b % M]，槽 = 块*bs + p%bs）。
本脚本用离线 HF 跑同一段 token（绝对位置），拿层 0 的 v_proj 输出对比
引擎快照里对应槽位的内容 —— V 没有 RoPE，可以直接逐元素比。

用法: DUMP=out/eng/kv_p0m5.pt python -u scripts/cmp_kv.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
BS = 256


def main():
    import torch
    from transformers import AutoModelForCausalLM

    DUMP = os.environ.get("DUMP", "out/eng/kv_p0m5.pt")
    dev = "cuda"
    d = torch.load(DUMP, weights_only=False)
    ids = list(d["prompt"])
    v_eng = d["v"]                     # (n_blocks, bs, heads, dim) 层 0
    M = int(d["window"]) // BS
    P = int(d["head"])
    ctx = int(d["ctx"])
    end = ctx
    b0 = max(P, (end - 1) // BS - M + 1)
    segs = []
    if P:
        segs.append((0, min(P * BS, end)))
    if b0 * BS < end:
        segs.append((b0 * BS, end))
    print(f"[kv] window={d['window']} head={P} ctx={ctx} → prefill 段 = {segs}",
          flush=True)

    drf = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16,
                                               device_map=dev).eval()
    torch.set_grad_enabled(False)
    lay_i, lay_p = [], []
    for (a, c) in segs:
        lay_i.extend(ids[a:c])
        lay_p.extend(range(a, c))
    with torch.inference_mode():
        o = drf.model(torch.tensor([lay_i], device=dev),
                      position_ids=torch.tensor([lay_p], device=dev),
                      use_cache=False, output_hidden_states=True)
    # 层 0 注意力的输入 = input_layernorm(hidden_states[0])，V = v_proj(·)（无 RoPE）
    h0 = drf.model.layers[0].input_layernorm(o.hidden_states[0][0])
    nkv = drf.config.num_key_value_heads
    hd = drf.config.head_dim
    v_hf = drf.model.layers[0].self_attn.v_proj(h0).float().cpu().view(
        len(lay_p), nkv, hd)
    del o, h0
    torch.cuda.empty_cache()
    # 映射：布局下标 i ↔ 绝对位置 lay_p[i] ↔ 引擎物理块 (pos//BS)%M、槽 pos%BS
    worst, n, bad = 0.0, 0, 0
    detail = []
    for start in (0, len(lay_p) // 4, len(lay_p) // 2, len(lay_p) - 4):
        for i in range(start, min(start + 4, len(lay_p))):
            p = lay_p[i]
            blk = (p // BS) % M if P == 0 else (
                (p // BS) if p < P * BS else (P + (p // BS - P) % M))
            off = p % BS
            a = v_hf[i]
            b = v_eng[blk, off].float()
            dif = float((a - b).abs().max())
            detail.append((p, blk, off, dif))
            n += 1
            worst = max(worst, dif)
    print(f"[kv] 抽样 {n} 个位置：引擎环里的 V 与离线算出的 V 最大差 = {worst:.5f}",
          flush=True)
    for (p, blk, off, dif) in detail:
        print(f"      pos={p} -> blk={blk} off={off}  max|ΔV| = {dif:.5f}", flush=True)
    # 全量扫描（只看前 N 个位置，避免太慢）
    worst_all, at = 0.0, -1
    for i in range(0, len(lay_p), 7):
        p = lay_p[i]
        blk = (p // BS) % M if P == 0 else (
            (p // BS) if p < P * BS else (P + (p // BS - P) % M))
        dif = float((v_hf[i] - v_eng[blk, p % BS].float()).abs().max())
        if dif > worst_all:
            worst_all, at = dif, p
    print(f"[kv] 全段扫描（每 7 个取 1）：max|ΔV| = {worst_all:.5f} @ pos={at}",
          flush=True)
    # 对照：引擎环里那个槽位的 V 与【别的块】的 V 哪个更像？
    i = len(lay_p) // 2
    p = lay_p[i]
    blk = (p // BS) % M if P == 0 else (
        (p // BS) if p < P * BS else (P + (p // BS - P) % M))
    e = v_eng[blk, p % BS].float()
    print(f"[kv] 对照 pos={p}: 与 HF 同位置 Δ={float((v_hf[i]-e).abs().max()):.5f}；"
          f"与 HF 位置 {p-256} Δ={float((v_hf[i-256]-e).abs().max()):.5f}；"
          f"与 HF 位置 {p+256} Δ={float((v_hf[i+256]-e).abs().max()):.5f}", flush=True)


if __name__ == "__main__":
    main()
