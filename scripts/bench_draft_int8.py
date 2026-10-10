#!/usr/bin/env python
"""bench_draft_int8.py —— draft FFN INT8 的端到端消融（配对 / 交替 / 含空转噪声地板）

范围（用户定）：**只量化 draft（Qwen3-0.6B）的 FFN（gate/up/down）**，target 完全不动。
所以「相对 BF16 target 无损」这句话**仍然严格成立**（target 是参考标准，一个字节没改）。

三个状态在【同一个进程】里交替测量，避免跨进程/跨加载漂移：
    U      普通解码（scheduler.spec_k = 0，draft 不参与）
    S_bf16 投机，draft FFN 保持 bf16
    S_int8 投机，draft FFN int8
★ 空转噪声地板：S_bf16 与 S_int8 交替跑，S_bf16 自身的组间范围就是噪声地板；
  再看 S_int8 相对 S_bf16 的差异有没有超过它。

draft 的权重替换【不重拍图】：两套权重都常驻（int8 那套只多 0.27 GB），
bf16 与 int8 各拍一张 draft 图，切换时只换权重引用 + 重新绑定图字典。

用法:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_draft_int8.py
env: RUNS(9) OUTLEN(256) BS(1,4) CTXS(1024,4096) K(6) GRAN(per_channel)
"""
import os
import sys
import re
import time
import json
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.token_hook import TokenDeliveryHook
from nanovllm.layers.quant import quantize_weight

TARGET = os.path.expanduser(os.environ.get("TARGET", "~/nano-vllm/models/Qwen3-4B"))
DRAFT = os.path.expanduser(os.environ.get("DRAFT", "~/nano-vllm/models/Qwen3-0.6B"))
RUNS = int(os.environ.get("RUNS", "9"))
OUTLEN = int(os.environ.get("OUTLEN", "256"))
BS = [int(x) for x in os.environ.get("BS", "1,4").split(",")]
CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096").split(",")]
K = int(os.environ.get("K", "6"))
GRAN = os.environ.get("GRAN", "per_channel")
GS = int(os.environ.get("GS", "128"))
MAX_MODEL_LEN = 8192
OUT_JSON = os.environ.get("OUT_JSON", "/home/ziru/nano-vllm/draft_int8_ablation.json")
DRAFT_FFN = re.compile(r"\.mlp\.(gate_up_proj|down_proj)$")


def build_ids(tok, need):
    txt = ("The quick brown fox jumps over the lazy dog. "
           "In a distant galaxy, researchers study attention kernels. ")
    return tok.encode(txt * 4000)


def prompts_for(base, ctx, B, stride=64):
    return [base[i * stride: i * stride + ctx] for i in range(B)]


def med(vals):
    vals = sorted(vals)
    m = statistics.median(vals)
    return {"median": m, "min": vals[0], "max": vals[-1]}


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print(f"=== bench_draft_int8 RUNS={RUNS} K={K} BS={BS} CTXS={CTXS} out={OUTLEN} "
          f"gran={GRAN} dev={torch.cuda.get_device_name(0)} ===")
    llm = LLM(TARGET, enforce_eager=False, max_model_len=MAX_MODEL_LEN,
              max_num_batched_tokens=16384, spec_k=K, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0)
    mr = llm.model_runner
    prop = mr.spec_proposer

    # ---------- 找出 draft 的 FFN 模块 ----------
    dmods = [(n, m) for n, m in mr.draft_model.named_modules()
             if DRAFT_FFN.search(n)]
    assert dmods, "没找到 draft 的 FFN 模块"
    orig = {n: m.weight for n, m in dmods}
    nb = sum(w.numel() * 2 for w in orig.values())
    int8_state = {}
    for n, m in dmods:
        q, s = quantize_weight(m.weight.data, GRAN, GS)
        int8_state[n] = (torch.nn.Parameter(q.contiguous(), requires_grad=False),
                         torch.nn.Parameter(s.contiguous(), requires_grad=False))
    na = sum(q.numel() + s.numel() * 2 for q, s in int8_state.values())
    print(f"draft FFN: {len(dmods)} 个模块, {nb/2**20:.1f} -> {na/2**20:.1f} MiB "
          f"({na/nb:.3f}x), 模块={sorted({n.split('.')[-1] for n, _ in dmods})}")

    def install(which):
        for n, m in dmods:
            if which == "int8":
                m.weight, m.weight_scale = int8_state[n]
                m.quant_granularity, m.quant_group_size = GRAN, GS
            else:
                m.weight = orig[n]
                m.weight_scale = None

    # ---------- 两套状态各拍一张 draft 图（之后切换不再重拍）----------
    install("bf16")
    mr.capture_draft_cudagraph()
    G = {"bf16": mr.draft_graphs}
    install("int8")
    mr.capture_draft_cudagraph()
    G["int8"] = mr.draft_graphs

    def set_variant(v):
        if v == "U":
            llm.scheduler.spec_k = 0
            install("bf16")
            prop.bind_cudagraph(G["bf16"])
        else:
            llm.scheduler.spec_k = K
            install(v)
            prop.bind_cudagraph(G[v])
        torch.cuda.synchronize()

    # ---------- 统计钩子：接受率 + 轮数 ----------
    import nanovllm.spec_decode.verify as V
    S = {"proposed": 0, "accepted": 0}
    orig_vb = V.verify_batch

    def tvb(draft_probs, target_logits, draft_tokens, temperatures=None,
            draft_is_point_mass=False):
        res = orig_vb(draft_probs, target_logits, draft_tokens, temperatures,
                      draft_is_point_mass)
        if not draft_is_point_mass:
            S["proposed"] += int(res.n_proposed.sum())
            S["accepted"] += int(res.accept_mask.sum())
        return res
    V.verify_batch = tvb

    def one_run(prompts, outlen):
        for k in S:
            S[k] = 0
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        torch.cuda.synchronize()
        t0 = time.time()
        outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=outlen,
                                                    ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.time() - t0
        llm.token_hook = None
        llm.scheduler.token_hook = None
        ntok = sum(len(o["token_ids"]) for o in outs)
        summ = hook.summary()
        ttft = statistics.median([v["ttft"] for v in summ.values()])
        rounds = S["proposed"] / K if S["proposed"] else ntok   # 每轮提 K 个候选
        return {
            "wall_s": wall, "tokens": ntok, "tok_per_s": ntok / wall,
            "ttft_ms": ttft * 1e3,
            "rounds": rounds, "ms_per_round": 1e3 * wall / max(rounds, 1),
            "accept_rate": (S["accepted"] / S["proposed"]) if S["proposed"] else None,
        }

    base_ids = build_ids(llm.tokenizer, max(CTXS) + 512)
    variants = ["U", "S_bf16", "S_int8"]
    results = []
    for B in BS:
        for ctx in CTXS:
            prompts = prompts_for(base_ids, ctx, B)
            sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)
            recs = {v: [] for v in variants}
            for v in variants:                      # 每个状态先 warmup
                set_variant(v)
                llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            for r in range(RUNS):                   # ★ 交替轮转，配对采样
                for v in variants:
                    set_variant(v)
                    x = one_run(prompts, OUTLEN)
                    x.update(variant=v, B=B, ctx=ctx, run=r)
                    recs[v].append(x)
            row = {"B": B, "ctx": ctx, "outlen": OUTLEN, "runs": RUNS, "k": K}
            for v in variants:
                row[v] = {
                    "tok_per_s": med([x["tok_per_s"] for x in recs[v]]),
                    "ttft_ms": med([x["ttft_ms"] for x in recs[v]]),
                    "ms_per_round": med([x["ms_per_round"] for x in recs[v]]),
                    "rounds": med([x["rounds"] for x in recs[v]]),
                    "wall_s": med([x["wall_s"] for x in recs[v]]),
                    "accept_rate": (med([x["accept_rate"] for x in recs[v]])
                                    if recs[v][0]["accept_rate"] is not None else None),
                    "raw_tok_per_s": [round(x["tok_per_s"], 3) for x in recs[v]],
                    "raw_ms_per_round": [round(x["ms_per_round"], 3) for x in recs[v]],
                }
            # 配对：同一 run 内 S_int8 / S_bf16
            pair = [recs["S_int8"][i]["ms_per_round"] / recs["S_bf16"][i]["ms_per_round"]
                    for i in range(RUNS)]
            pair_u = [recs["S_int8"][i]["tok_per_s"] / recs["U"][i]["tok_per_s"]
                      for i in range(RUNS)]
            idle = [recs["S_bf16"][i]["ms_per_round"] for i in range(RUNS)]
            row["paired_ms_per_round_int8_over_bf16"] = med(pair)
            row["paired_tok_per_s_int8_over_U"] = med(pair_u)
            row["noise_floor_ms_per_round"] = {
                "median": statistics.median(idle),
                "range_pct": 100 * (max(idle) - min(idle)) / statistics.median(idle),
            }
            results.append(row)
            print("@@B@@" + json.dumps(row, ensure_ascii=False))
            with open(OUT_JSON, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'B':>2} {'ctx':>5} | {'U tok/s':>10} {'S_bf16':>10} {'S_int8':>10} | "
          f"{'ms/round bf16':>14} {'int8':>10} {'配对比':>8} | {'接受率 bf16':>12} {'int8':>8} | "
          f"{'噪声地板%':>9}")
    for r in results:
        print(f"{r['B']:>2} {r['ctx']:>5} | {r['U']['tok_per_s']['median']:>10.1f} "
              f"{r['S_bf16']['tok_per_s']['median']:>10.1f} "
              f"{r['S_int8']['tok_per_s']['median']:>10.1f} | "
              f"{r['S_bf16']['ms_per_round']['median']:>14.3f} "
              f"{r['S_int8']['ms_per_round']['median']:>10.3f} "
              f"{r['paired_ms_per_round_int8_over_bf16']['median']:>8.4f} | "
              f"{(r['S_bf16']['accept_rate'] or {'median':float('nan')})['median']:>12.4f} "
              f"{(r['S_int8']['accept_rate'] or {'median':float('nan')})['median']:>8.4f} | "
              f"{r['noise_floor_ms_per_round']['range_pct']:>9.2f}")
    print("@@DONE@@  写盘: " + OUT_JSON)


if __name__ == "__main__":
    main()
