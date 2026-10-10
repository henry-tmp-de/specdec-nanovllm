#!/usr/bin/env python
"""check_plain_ctrl.py —— 「量化 draft 有没有碰到投机路径以外的东西」的对照实验

本轮的量化范围**只在 draft（Qwen3-0.6B）的线性层**上。普通解码（spec_k=0）
从头到尾不碰 draft，所以它**必须一个字节都不变** —— 这是"改动被限制在投机路径内"的
直接证据，也是给"配置开关只加不改"这条约束做验收。

做法：同一个脚本跑两遍（两个独立进程，nano-vllm 只允许一个引擎在跑），
唯一差别是 draft 目录：
    CASE=bf16  → DRAFT = ~/nano-vllm/models/Qwen3-0.6B        （没有 quant_config.json）
    CASE=qall  → DRAFT = ~/nano-vllm/models/Qwen3-0.6B-qall   （有 → 引擎自动量化）

两遍都跑 **普通解码**（spec_k=0），比较 tok/s、ms/token、KV 池块数。
要看的结论是"两列相等"（在噪声里），不是"谁更快"。

用法:
  CUDA_VISIBLE_DEVICES=0 CASE=bf16 python -u scripts/check_plain_ctrl.py
  CUDA_VISIBLE_DEVICES=0 CASE=qall python -u scripts/check_plain_ctrl.py
env: CASE(bf16) REPS(9) OUTLEN(256) BS(1) CTXS(1024,4096)
"""
import os
import sys
import time
import json
import atexit
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT_DIRS = {"bf16": os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B"),
              "qall": os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-qall")}
CASE = os.environ.get("CASE", "bf16")
REPS = int(os.environ.get("REPS", "9"))
OUTLEN = int(os.environ.get("OUTLEN", "256"))
BS = [int(x) for x in os.environ.get("BS", "1").split(",")]
CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096").split(",")]
OUT_JSON = os.environ.get("OUT_JSON", "/home/ziru/nano-vllm/qall-runs/plain_ctrl.json")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")


def build_prompts(tok, ctx, B):
    """★ 与 bench_draft_int8.py 的 PROMPTS=wiki 完全同一段取法，两遍可比。"""
    with open(WIKI, encoding="utf-8") as f:
        ids = tok(f.read()).input_ids
    span = (len(ids) - ctx) // B
    return [ids[i * span: i * span + ctx] for i in range(B)]


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    draft = DRAFT_DIRS[CASE]
    print(f"=== check_plain_ctrl CASE={CASE} draft={draft} REPS={REPS} "
          f"BS={BS} CTXS={CTXS} dev={torch.cuda.get_device_name(0)} ===")
    llm = LLM(TARGET, enforce_eager=False, max_model_len=8192,
              max_num_batched_tokens=16384, spec_k=0,          # ★ 普通解码
              spec_method="draft", draft_model=draft, spec_batch_threshold=0)
    mr = llm.model_runner
    st = getattr(mr, "draft_quant_stats", [])
    nb = sum(s["bytes_before"] for s in st)
    na = sum(s["bytes_after"] for s in st)
    print(f"[draft] 量化模块数={len(st)} 字节 {nb/2**20:.1f} -> {na/2**20:.1f} MiB "
          f"({na/max(nb,1):.3f}x)  KV 池块数={mr.config.num_kvcache_blocks} "
          f"({mr.config.num_kvcache_blocks * mr.block_size} token)")

    results = []
    for B in BS:
        for ctx in CTXS:
            prompts = build_prompts(llm.tokenizer, ctx, B)
            sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)
            llm.generate(prompts, sp, use_tqdm=False)         # warmup
            torch.cuda.synchronize()
            recs = []
            for _ in range(REPS):
                torch.cuda.synchronize()
                t0 = time.time()
                outs = llm.generate(prompts, SamplingParams(temperature=1.0,
                                     max_tokens=OUTLEN, ignore_eos=True),
                                    use_tqdm=False)
                torch.cuda.synchronize()
                wall = time.time() - t0
                ntok = sum(len(o["token_ids"]) for o in outs)
                recs.append({"wall_s": wall, "tokens": ntok,
                             "tok_per_s": ntok / wall, "ms_per_token": 1e3 * wall / ntok})
            row = {"case": CASE, "B": B, "ctx": ctx, "outlen": OUTLEN, "reps": REPS,
                   "draft_quant_modules": len(st), "num_kvcache_blocks":
                   mr.config.num_kvcache_blocks,
                   "draft_quant_bytes_before": nb, "draft_quant_bytes_after": na}
            for k in ("tok_per_s", "ms_per_token"):
                v = sorted(x[k] for x in recs)
                row[k] = {"median": statistics.median(v), "min": v[0], "max": v[-1],
                          "range_pct": 100 * (v[-1] - v[0]) / statistics.median(v)}
            results.append(row)
            print("@@PC@@" + json.dumps(row, ensure_ascii=False))

    old = []
    if os.path.isfile(OUT_JSON):
        with open(OUT_JSON) as f:
            old = json.load(f)
    old = [r for r in old if r.get("case") != CASE] + results
    with open(OUT_JSON, "w") as f:
        json.dump(old, f, indent=2, ensure_ascii=False)

    print(f"\n=== 普通解码对照 (CASE={CASE}) ===")
    print(f"{'B':>2} {'ctx':>5} {'tok/s':>9} {'ms/token':>9} {'组间范围%':>9} "
          f"{'KV块':>6} {'量化模块':>8}")
    for r in results:
        print(f"{r['B']:>2} {r['ctx']:>5} {r['tok_per_s']['median']:>9.2f} "
              f"{r['ms_per_token']['median']:>9.4f} {r['tok_per_s']['range_pct']:>9.2f} "
              f"{r['num_kvcache_blocks']:>6} {r['draft_quant_modules']:>8}")
    print("@@DONE@@  写盘: " + OUT_JSON)
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
