#!/usr/bin/env python
"""profile_draft_breakdown.py —— 草稿模型「单次前向」的逐 kernel 分解。

与 scripts/profile_decode_breakdown.py **同口径**（torch.profiler device time），
但对象是 draft（Qwen3-0.6B）的 decode 图，而不是 target 的普通 decode。

要回答的问题
------------
草稿单次前向 ~2563 us，读 1.192 GB 权重 -> 有效带宽 465 GB/s = 天花板 55%
（target 是 8.045 GB / 10450 us = 770 GB/s = 91.5%）。
那多出来的 ~1.1 ms 到底是什么？分解成：

  A. GEMM / 线性层（含 lm_head）
  B. backbone-only（把 lm_head + logits.copy_ 分离出来）
  C. lm_head 单独
  D. 采样链（softmax + exponential + argmax）—— 注意这条在【图外】eager 跑
  E. eager 整前向（对照：图省了多少）
  F. target decode 图（同一 ctx，同一口径）

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<p1-work> python -u scripts/profile_draft_breakdown.py
env:
  CTX      提示词长度（默认 4096）
  OUTLEN   生成 token 数（默认 64）
  N        replay / microbench 次数（默认 60）
  WIN      draft 滑窗 spec_draft_window（默认 0 = 全上下文，与基线一致）
"""
import os
import sys
import gc
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import set_context, reset_context

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
CEIL = 842.0   # GB/s，实测 copy 流式天花板
CTX = int(os.environ.get("CTX", 4096))
OUTLEN = int(os.environ.get("OUTLEN", 64))
N = int(os.environ.get("N", 60))
WIN = int(os.environ.get("WIN", 0))
SPEC_K = 6

CAT_RULES = [
    ("kv_cache_write", ["store_kvcache"]),
    ("attention",      ["flash", "fmha", "attention", "_attn", "mha"]),
    ("sampling",       ["softmax", "argmax", "exponential", "multinomial", "topk",
                        "sampler", "logprob", "gumbel"]),
    ("gemm_linear",    ["cublas", "cutlass", "xmma", "ampere", "nvjet", "hmma",
                        "gemm", "sgemm", "matmul", "wgrad", "dgrad", "gemv"]),
    ("norm_rope_eltwise", ["rms_norm", "rmsnorm", "silu", "rotary", "rope",
                           "elementwise", "vectorized", "triton_", "embedding",
                           "index", "gather", "concat", "split", "fill",
                           "copy", "add", "mul", "cast", "contiguous", "reshape"]),
]


def categorize(name):
    low = name.lower()
    if "memcpy" in low or "memset" in low:
        return "memcpy"
    for cat, keys in CAT_RULES:
        for k in keys:
            if k in low:
                return cat
    return "other"


def collect(prof):
    cuda_dt = torch.autograd.DeviceType.CUDA
    agg = {}
    total = 0.0
    for ev in prof.events():
        if getattr(ev, "device_type", None) != cuda_dt:
            continue
        d = float(getattr(ev, "device_time", 0) or 0)
        if d <= 0:
            continue
        rec = agg.get(ev.name)
        if rec is None:
            agg[ev.name] = [d, 1]
        else:
            rec[0] += d
            rec[1] += 1
        total += d
    return agg, total


def cat_sum(agg):
    cats = {}
    for name, (tot, cnt) in agg.items():
        c = cats.setdefault(categorize(name), [0.0, 0])
        c[0] += tot
        c[1] += cnt
    return cats


def prof_loop(fn, iters):
    """返回 (agg_raw, total_per_iter_us, agg_per_iter)."""
    prof = profile(activities=[ProfilerActivity.CUDA])
    with prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg, total = collect(prof)
    del prof
    gc.collect()
    iters = max(iters, 1)
    per = {k: [v[0] / iters, v[1] / iters] for k, v in agg.items()}
    return per, total / iters


def wall_loop(fn, iters):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3   # us


def fmt(per, title, top=25):
    rows = sorted(per.items(), key=lambda x: -x[1][0])
    lines = [title, "-" * 96,
             "%-56s %10s %8s %10s" % ("name", "us/iter", "cnt", "us/call")]
    for name, (tot, cnt) in rows[:top]:
        lines.append("%-56s %10.1f %8.1f %10.2f"
                     % (name[:56], tot, cnt, tot / max(cnt, 1e-9)))
    return "\n".join(lines)


def cat_table(per, title):
    total = sum(v[0] for v in per.values())
    cats = cat_sum(per)
    lines = [title, "-" * 96,
             "%-22s %12s %9s %10s" % ("category", "us/iter", "pct", "calls/it")]
    for c, (t, n) in sorted(cats.items(), key=lambda x: -x[1][0]):
        lines.append("%-22s %12.1f %8.2f%% %10.1f"
                     % (c, t, 100 * t / max(total, 1e-9), n))
    return "\n".join(lines), cats, total


def build_prompt_ids(llm, need):
    tok = llm.tokenizer
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "In a distant galaxy, researchers study attention kernels. " * 800)
    assert len(base) >= need, "prompt 太短: %d < %d" % (len(base), need)
    return base[:need]


def dedup_bytes(model):
    seen, tot = set(), 0
    for p in model.parameters():
        k = (p.data_ptr(), p.numel(), p.element_size())
        if k in seen:
            continue
        seen.add(k)
        tot += p.numel() * p.element_size()
    return tot


def main():
    tag = time.strftime("%Y%m%d_%H%M%S")
    print("=== profile_draft_breakdown tag=%s gpu=%s ==="
          % (tag, os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("torch=%s cuda=%s dev=%s" % (torch.__version__, torch.version.cuda,
                                       torch.cuda.get_device_name(0)))
    kw = dict(enforce_eager=False, max_model_len=8192, max_num_batched_tokens=16384,
              spec_k=SPEC_K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_graph_bs=(1,))
    if WIN:
        kw["spec_draft_window"] = WIN
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    d_hf = mr.draft_hf_config
    t_hf = mr.config.hf_config
    res = {"tag": tag, "ctx": CTX, "outlen": OUTLEN, "n_iters": N, "win": WIN,
           "draft_layers": d_hf.num_hidden_layers, "draft_vocab": d_hf.vocab_size,
           "draft_hidden": d_hf.hidden_size,
           "target_layers": t_hf.num_hidden_layers, "target_hidden": t_hf.hidden_size}

    ids = build_prompt_ids(llm, CTX + 16)
    t0 = time.time()
    llm.generate([ids], SamplingParams(temperature=1.0, max_tokens=OUTLEN,
                                       ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()
    print("[warm] generate ctx=%d out=%d wall=%.2fs" % (CTX, OUTLEN, time.time() - t0))

    dw = dedup_bytes(mr.draft_model)
    tw = dedup_bytes(mr.model)
    res["draft_weight_bytes"] = dw
    res["target_weight_bytes"] = tw
    lm_bytes = d_hf.vocab_size * d_hf.hidden_size * 2
    res["draft_lmhead_bytes"] = lm_bytes
    print("[bytes] draft=%.3f GB (lm_head %.3f GB = %.1f%%)  target=%.3f GB"
          % (dw / 1e9, lm_bytes / 1e9, 100 * lm_bytes / dw, tw / 1e9))

    dev = next(mr.draft_model.parameters()).device

    # ================= A. draft 图（完整）=================
    graph, gv = mr.draft_graphs[1]
    cl = gv["context_lens"].tolist()
    bt_width = gv["block_tables"].size(1)
    print("[state] draft graph ctx_lens=%s block_table_width=%d" % (cl, bt_width))
    res["draft_ctx_lens"] = cl
    res["draft_block_table_width"] = bt_width

    perA, tA = prof_loop(lambda: graph.replay(), N)
    wallA = wall_loop(lambda: graph.replay(), N)
    res["draft_full_us_prof"] = tA
    res["draft_full_us_wall"] = wallA
    res["draft_full_GBs"] = dw / wallA / 1e3
    res["draft_full_pct_ceiling"] = dw / wallA / 1e3 / CEIL * 100
    tabA, catsA, _ = cat_table(perA, "A) draft 图（完整）")
    print("\n" + tabA)
    print("\n" + fmt(perA, "A) TOP kernels"))
    res["draft_full_cats"] = {c: {"us": t, "pct": round(100 * t / tA, 2), "calls": n}
                              for c, (t, n) in catsA.items()}
    res["draft_full_top"] = [{"name": k, "us": round(v[0], 2), "calls": v[1]}
                             for k, v in sorted(perA.items(), key=lambda x: -x[1][0])[:25]]
    print("[pct] profiler device-sum=%.1f us/iter, cuda-event wall=%.1f us/iter, "
          "差额=%.1f us (launch gap)" % (tA, wallA, wallA - tA))

    # 从 A 的 kernel 列表里单独拎出 lm_head：整个图里唯一 1 次/iter 的大 GEMV
    bt_width_full = bt_width
    lm_kernels = [(k, v) for k, v in perA.items()
                  if v[1] <= 1.1 and ("gemv" in k.lower() or "gemm" in k.lower()
                                      or "cutlass" in k.lower())]
    res["draft_lmhead_kernel"] = [(k[:80], round(v[0], 1), round(v[1], 2)) for k, v in lm_kernels]
    print("[A/derive] 图内单次大 GEMV（= lm_head）: "
          + ", ".join("%s %.1fus" % (k[:40], v[0]) for k, v in lm_kernels))
    worst = min(bt_width, (int(cl[0]) + mr.block_size - 1) // mr.block_size)

    # ================= B. draft 图 ctx 扫描（拆开「固定 vs 随 ctx」）=================
    print("\n[B] draft 图 ctx 扫描（直接改 context_lens；物理块内容不影响计时）")
    sweep = []
    for cl_v in [1, 256, 512, 1024, 2048, 4096, int(cl[0])]:
        wneed = max(1, (cl_v + mr.block_size - 1) // mr.block_size)
        with torch.inference_mode():
            gv["context_lens"].fill_(cl_v)
            gv["block_tables"].fill_(-1)
            gv["block_tables"][:, :wneed] = torch.arange(
                wneed, dtype=torch.int32, device=dev)
            torch.cuda.synchronize()
        it = max(20, N // 2)
        wv = wall_loop(lambda: graph.replay(), it)
        pv, tv = prof_loop(lambda: graph.replay(), it)
        catv = cat_sum(pv)
        row = {"ctx": cl_v, "wall_us": round(wv, 1), "dev_us": round(tv, 1),
               "gemm_us": round(catv.get("gemm_linear", [0, 0])[0], 1),
               "attn_us": round(catv.get("attention", [0, 0])[0], 1),
               "elt_us": round(catv.get("norm_rope_eltwise", [0, 0])[0], 1)}
        sweep.append(row)
        print("  ctx=%5d  wall=%8.1f  dev=%8.1f  gemm=%7.1f  attn=%7.1f  elt=%7.1f us"
              % (cl_v, wv, tv, row["gemm_us"], row["attn_us"], row["elt_us"]))
    res["draft_ctx_sweep"] = sweep

    # ================= C. lm_head 单独 =================
    with torch.no_grad():
        x = torch.randn(1, d_hf.hidden_size, dtype=torch.bfloat16, device=dev)
        ch = mr.draft_model.compute_logits
        ch(x)
        torch.cuda.synchronize()
    perC, tC = prof_loop(lambda: ch(x), N)
    res["lmhead_solo_us"] = tC
    res["lmhead_GBs"] = lm_bytes / tC / 1e3
    print("\n[C] lm_head 单独: %.1f us  bytes=%.3f GB  %.1f GB/s  (%.1f%% ceiling)"
          % (tC, lm_bytes / 1e9, res["lmhead_GBs"], res["lmhead_GBs"] / CEIL * 100))
    print(fmt(perC, "C) lm_head kernels", 8))

    # ================= D. 采样链（图外）=================
    with torch.no_grad():
        logits = torch.randn(1, d_hf.vocab_size, dtype=torch.bfloat16, device=dev)
        temps = torch.ones(1, 1, dtype=torch.float32, device=dev)
        out_logits = torch.empty(1, SPEC_K, d_hf.vocab_size, dtype=torch.float32, device=dev)

        def sample_chain():
            out_logits[:, 0].copy_(logits.float())
            p = torch.softmax(logits.float() / temps, dim=-1)
            noise = torch.empty_like(p).exponential_(1.0).clamp_min_(1e-10)
            return (p / noise).argmax(dim=-1)

        sample_chain()
        torch.cuda.synchronize()
    perD, tD = prof_loop(sample_chain, N)
    wD = wall_loop(sample_chain, N)
    res["sample_chain_us_dev"] = tD
    res["sample_chain_us_wall"] = wD
    res["sample_chain_x_k_us_wall"] = wD * SPEC_K
    print("\n[D] 采样链（每步 1 次）：device %.1f us / wall %.1f us；一轮 %d 步 = %.1f us(wall)"
          % (tD, wD, SPEC_K, wD * SPEC_K))
    print(fmt(perD, "D) 采样链 kernels", 10))
    res["sample_chain_kernels"] = [{"name": k, "us": round(v[0], 2), "calls": v[1]}
                                   for k, v in sorted(perD.items(), key=lambda x: -x[1][0])[:12]]

    # ================= E. eager 整前向（同 ctx，同口径）=================
    with torch.no_grad():
        inp = torch.zeros(1, dtype=torch.int64, device=dev)
        pos = torch.tensor([CTX - 1], dtype=torch.int64, device=dev)
        wneed = max(1, (CTX + mr.block_size - 1) // mr.block_size)
        slot = torch.full((1,), -1, dtype=torch.int32, device=dev)
        clen = torch.tensor([CTX], dtype=torch.int32, device=dev)
        btwide = torch.full((1, bt_width_full), -1, dtype=torch.int32, device=dev)
        btwide[0, :wneed] = torch.arange(wneed, dtype=torch.int32, device=dev)

        def eager_fwd():
            set_context(False, slot_mapping=slot, context_lens=clen, block_tables=btwide)
            try:
                return ch(mr.draft_model(inp, pos))
            finally:
                reset_context()

        eager_fwd()
        torch.cuda.synchronize()
    itE = max(10, N // 4)
    perE, tE = prof_loop(eager_fwd, itE)
    wE = wall_loop(eager_fwd, itE)
    res["draft_eager_us_wall"] = wE
    res["draft_eager_us_dev"] = tE
    print("\n[E] eager 整前向(ctx=%d): device %.1f / wall %.1f us（图 wall=%.1f us，图省了 %.2fx）"
          % (CTX, tE, wE, wallA, wE / wallA))
    del perE

    # ================= F. target decode 图（同 ctx）=================
    with torch.inference_mode():
        bs = min(mr.graphs)
        gvf = mr.graph_vars
        w = (CTX + mr.block_size - 1) // mr.block_size
        gvf["input_ids"][:bs].fill_(0)
        gvf["positions"][:bs].fill_(CTX - 1)
        gvf["slot_mapping"].fill_(-1)
        gvf["context_lens"].zero_()
        gvf["context_lens"][:bs].fill_(CTX)
        gvf["block_tables"][:bs].fill_(-1)
        gvf["block_tables"][:bs, :w] = torch.arange(w, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        tg = mr.graphs[bs]
        tg.replay()
        torch.cuda.synchronize()
    perF, tF = prof_loop(lambda: tg.replay(), N)
    wallF = wall_loop(lambda: tg.replay(), N)
    res["target_full_us_prof"] = tF
    res["target_full_us_wall"] = wallF
    res["target_full_GBs"] = tw / wallF / 1e3
    res["target_full_pct_ceiling"] = tw / wallF / 1e3 / CEIL * 100
    tabF, catsF, _ = cat_table(perF, "F) target decode 图 B=%d ctx=%d" % (bs, CTX))
    print("\n" + tabF)
    res["target_full_cats"] = {c: {"us": t, "pct": round(100 * t / tF, 2), "calls": n}
                               for c, (t, n) in catsF.items()}
    print("[target] profiler=%.1f us/iter  wall=%.1f us -> %.1f GB/s (%.1f%% ceiling)"
          % (tF, wallF, res["target_full_GBs"], res["target_full_pct_ceiling"]))

    # ================= SUMMARY =================
    print("\n=== SUMMARY ===")
    print("draft  full   prof=%.1f  wall=%.1f us -> %.1f GB/s (%.1f%% ceil)"
          % (tA, wallA, res["draft_full_GBs"], res["draft_full_pct_ceiling"]))
    print("draft  lm_head   %.1f us (图内单次 GEMV)" % sum(v[0] for v in lm_kernels))
    print("draft  eager     wall=%.1f us" % wE)
    print("target full   prof=%.1f  wall=%.1f us -> %.1f GB/s (%.1f%% ceil)"
          % (tF, wallF, res["target_full_GBs"], res["target_full_pct_ceiling"]))
    print("lm_head solo %.1f us = %.1f GB/s | sample/k=%.1f us | sample x6=%.1f us"
          % (tC, res["lmhead_GBs"], wD, wD * SPEC_K))
    print("@@DRAFT@" + json.dumps(res, ensure_ascii=False))
    try:
        import atexit
        atexit.unregister(llm.exit)
    except Exception:
        pass
    try:
        llm.exit()
    except Exception as e:
        print("exit warn:", e)


if __name__ == "__main__":
    main()
