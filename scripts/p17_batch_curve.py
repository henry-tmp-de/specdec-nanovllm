#!/usr/bin/env python
"""p17_batch_curve.py —— P17：把「batch size」这条轴量出来。

要回答的问题
------------
`docs/端到端验收.md` §8 只有 **B∈{1,4}** 四个格子，而那条曲线是**递减**的
（`G → S6`：B=1 是 1.60×/1.11×，B=4 是 0.97×/0.71×）。**B=8/16/32 从来没测过。**
本脚本就是把这条曲线补完：**同一个引擎、同一组负载、同一个口径**，
B 从 1 一路到 32，看「我们的最终配置 `S6`」相对「nano-vllm 默认 `G`」的倍数怎么走。

★ 目的**不是**找好看的数，是**把整条曲线量出来**。B 大了我们比原版慢，就如实写。

四档（都在这一个脚本里，靠运行期/init 期开关翻转，不改 `nanovllm/` 生产代码）
----------------------------------------------------------------------------
    U             target-only 普通解码，**不拍图**        `spec_k=0` + `enforce_eager=True`
    G             target-only 普通解码 + CUDA Graph       `spec_k=0` + `enforce_eager=False`
                  ★ **这才是 nano-vllm 的默认配置**（`config.py:15` 默认 `enforce_eager=False`）
    S6            投机 + 滑窗 + draft int8 直载           `spec_k=6` + `spec_draft_window=1280`
                                                          + `spec_draft_head_blocks=0`
                                                          + `spec_draft_win_align=True`
                                                          + `spec_draft_win_realign=True`
                                                          + draft=`Qwen3-0.6B-qall`
                                                          + `quant_direct_load=True`
    S6_nospec_win 投机但**关掉滑窗**（`spec_draft_window=0`）
                  ★ 用来把「滑窗的贡献」从「投机本身的贡献」里分离出来。
                    大 batch 下滑窗的容量账可能翻负（见下），这一档是判据的一半。

★ 两个必须写明的口径选择（都是为了让这条曲线**可读**，不是为了好看）
--------------------------------------------------------------------
(1) **`max_num_seqs` 逐格按真实并发设 = B**（不是一股脑 32）。
    依据：`docs/优化台账.md` §2.7 那条判据 —— 滑窗的 draft 私有池 = `M × max_num_seqs` 块，
    而 target 与 draft **抢同一份 KV 预算**（`model_runner.py:allocate_kv_cache`）：
        room = budget − draft_pool × draft_block_bytes ;  target_blocks = room // tgt_block_bytes
    所以 `max_num_seqs` 越大，draft 池越大、target 池越小。逐格设 = B 才不会让
    「大 batch」这一格同时混进「配置被撑爆」这件事。

(2) **`spec_graph_bs` 扩到覆盖被测的 B，并打开 `spec_graph_pad`（P16）。**
    `config.py:179` 的默认是 `spec_graph_bs=(1,2,4)` —— 投机路径的两张图
    （draft 单步图、验证图）**只捕获这三个桶**，B≥8 时 `verify_graphs.get((B,k))` 直接
    返回 None ⇒ **整条退回 eager**（`run_verify_forward` 的 `shape_not_captured`）。
    这不是算法差异，是**图桶没配**：eager 的 draft 单步实测 24.8 ms、拍图后 ~2 ms
    （`config.py:34`），k=6 步串行 ⇒ 一格能差出 100+ ms/轮，把整条曲线压成"投机在大 batch 必输"。
    ⇒ 本脚本把 `spec_graph_bs` 设成 `(1,2,4,8,16,32)`（在测量脚本里传参，不动默认值），
      并用 `spec_graph_pad=True` 让收尾尾巴里的 B=31/30/… 也补进最近的桶。
    ⇒ **对照组 `s6stock`** 保留 `(1,2,4)` + `pad=False`，专门量「不配图桶要付多少钱」。

★ 诚实边界（复算前必读）
------------------------
- 主指标 `tok/s(maxB)` = **只在 active B = 最大 batch 的那些轮上**统计（P10 §4.1 的口径，
  剔除收尾尾巴）。同时报 `ms/轮(maxB)`、`tok/轮(maxB)`、`α`、`TTFT`、`active B`、噪声地板。
- **`G` 与 `S6` 不在同一个进程里**（draft 模型、滑窗池布局都是 init 期定的）⇒
  `S6/G` 是**跨进程比值**，带 ~1–2% 的不确定度（P10 §8.3 已量到）。
- **`U` 档是 eager、CPU-bound**，并行/争用下实测 ±4%（P10 §5.3）⇒ 它的数只作参考。
- `S6` 不是无损：纯滑窗相对全上下文有 **1%–6%** 的真实截断代价（台账 §2.7.3）。

用法
----
    # 单格（一个进程）
    CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2411 python -u scripts/p17_batch_curve.py \
        --group s6 --B 8 --runs 3 --out out/p17/s6_B8.json

    # 整条曲线（串行，每格一个进程）
    CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2411 python -u scripts/p17_batch_curve.py \
        --all --runs 3 --outdir out/p17
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from time import perf_counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT_QALL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-qall")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")

# ---- 固定口径（与 P10 逐项相同，保证两个表可比）----
K = 6
CTX = int(os.environ.get("P17_CTX", "1024"))   # 可用 P17_CTX=4096 覆盖（ctx 是 init 期常量，不能靠 CLI 传）
OUTLEN = 256
MAXLEN = 8192
MAX_NUM_BATCHED_TOKENS = 16384
WINDOW = 1280          # M = 5 块 = 1280 token —— 台账 §2.7.3 的新推荐配置
HEAD_BLOCKS = 0
SEED = 20261011
BS_ALL = (1, 2, 4, 8, 16, 32)

# 覆盖被测 B 的图桶（★ 只写在测量脚本里；config.py 的默认仍是 (1,2,4)）
GRAPH_BS_FULL = (1, 2, 4, 8, 16, 32)
GRAPH_BS_STOCK = (1, 2, 4)

# 每一组 = 一个进程（init 期开关只能在这里定）
#   draft=None  ⇒ 完全不加载 draft 模型（= 真正的 nano-vllm 原版）
#   draft="qall"⇒ 加载 Qwen3-0.6B-qall（int8 全线性层）+ 直载
GROUPS = {
    "base":    dict(draft=None,   window=0,      head=0, align=False, realign=False,
                    graph_bs=GRAPH_BS_FULL, graph_pad=False, variants=["U", "G"]),
    # ★ `s6` 组里放两个变体、同进程交替测量 ⇒ **真配对**：
    #   S_nospec 与 S6 的池布局、draft 模型、图桶完全一样，唯一差别是 scheduler.spec_k。
    #   ⇒ `S6/S_nospec` 干净地隔离出「投机本身」这一刀。
    "s6":      dict(draft="qall", window=WINDOW, head=HEAD_BLOCKS, align=True, realign=True,
                    graph_bs=GRAPH_BS_FULL, graph_pad=True,
                    variants=["S_nospec", "S6"]),
    "nowin":   dict(draft="qall", window=0,      head=0, align=False, realign=False,
                    graph_bs=GRAPH_BS_FULL, graph_pad=True, variants=["S6_nospec_win"]),
    # ★ 侧探针：图桶保持出厂默认 (1,2,4) + 不 pad —— 量「不配图桶」的代价
    "s6stock": dict(draft="qall", window=WINDOW, head=HEAD_BLOCKS, align=True, realign=True,
                    graph_bs=GRAPH_BS_STOCK, graph_pad=False, variants=["S6"]),
}
ALL_GROUPS = ["base", "s6", "nowin"]

VARIANT_SPEC = {
    "U":             dict(spec=False, eager=True),
    "G":             dict(spec=False, eager=False),
    # ★ S_nospec：不开投机，但**其余与 S6 完全一致**（draft 仍加载、池布局/图桶一致）
    #   ⇒ 用来把「我们非投机那部分改动的净效果」从「投机本身的净效果」里分出来：
    #        S_nospec / G  = 非投机改动（预期 ≈1.0×，但要实测 —— draft 仍占 KV 会挤 target）
    #        S6 / S_nospec = 投机本身
    "S_nospec":      dict(spec=False, eager=False),
    "S6":            dict(spec=True,  eager=False),
    "S6_nospec_win": dict(spec=True,  eager=False),
}
LADDER = ["U", "G", "S_nospec", "S6", "S6_nospec_win"]


# ----------------------------------------------------------------------
def build_prompts(tokenizer, ctx, B):
    """自然长文（wikitext2），按 B 切成互不重叠的连续片段 —— 与 P10 逐字相同。"""
    with open(WIKI, encoding="utf-8") as f:
        ids = tokenizer(f.read()).input_ids
    assert len(ids) > ctx * B, f"wikitext2 太短：{len(ids)} <= {ctx}*{B}"
    span = (len(ids) - ctx) // B
    return [ids[i * span: i * span + ctx] for i in range(B)]


def med(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    m = statistics.median(vals)
    return {"median": m, "min": vals[0], "max": vals[-1],
            "range_pct": (100.0 * (vals[-1] - vals[0]) / m if m else None)}


def md5_ids(ids):
    h = hashlib.md5()
    for row in ids:
        h.update(",".join(map(str, row)).encode())
        h.update(b"|")
    return h.hexdigest()


# ----------------------------------------------------------------------
def run_group(gname, args):
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook

    g = GROUPS[gname]
    B = int(args.B)

    kw = dict(enforce_eager=False, max_model_len=MAXLEN,
              max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
              max_num_seqs=B)
    if g["draft"]:
        kw.update(spec_k=K, spec_method="draft", draft_model=DRAFT_QALL,
                  spec_batch_threshold=0,
                  spec_graph_bs=tuple(g["graph_bs"]), spec_graph_pad=g["graph_pad"],
                  quant_direct_load=True)
        if g["window"]:
            kw["spec_draft_window"] = g["window"]
            kw["spec_draft_head_blocks"] = g["head"]
            kw["spec_draft_win_align"] = g["align"]
            kw["spec_draft_win_realign"] = g["realign"]

    print(f"=== p17 group={gname} B={B} ctx={CTX} out={OUTLEN} k={K} runs={args.runs} "
          f"window={g['window']} head={g['head']} draft={g['draft']} "
          f"graph_bs={g['graph_bs']} graph_pad={g['graph_pad']} variants={g['variants']} "
          f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"port={os.environ.get('NANOVLLM_PORT', '2333')} "
          f"dev={torch.cuda.get_device_name(0)} ===", flush=True)
    t_load = time.time()
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    prop = mr.spec_proposer
    cfg = mr.config
    print(f"[load] {time.time()-t_load:.1f}s  blocks={cfg.num_kvcache_blocks} "
          f"draft_blocks={cfg.num_draft_blocks} window_blocks={cfg.draft_window_blocks} "
          f"head_blocks={cfg.draft_head_blocks} max_num_seqs={cfg.max_num_seqs} "
          f"draft_quant={cfg.draft_quant_weights} direct={cfg.quant_direct_load} "
          f"graph_mem={getattr(mr, 'graph_mem', None)}", flush=True)

    # ---- 统计钩子：接受率 + 逐轮耗时(按 active B 分档) ----
    import nanovllm.spec_decode.verify as V
    S = {"proposed": 0, "accepted": 0, "steps": [], "preempt": 0}
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

    orig_pp = llm.scheduler.postprocess

    def tpp(seqs, token_ids, is_prefill):
        delivered = orig_pp(seqs, token_ids, is_prefill)
        S["steps"].append((perf_counter(), len(seqs), bool(is_prefill), delivered))
        return delivered
    llm.scheduler.postprocess = tpp

    orig_preempt = llm.scheduler.preempt

    def tpre(seq):
        S["preempt"] += 1
        return orig_preempt(seq)
    llm.scheduler.preempt = tpre

    def set_variant(v):
        c = VARIANT_SPEC[v]
        llm.scheduler.spec_k = K if c["spec"] else 0
        mr.enforce_eager = c["eager"]
        torch.cuda.synchronize()

    def one_run(prompts, outlen):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        for key in list(S):
            S[key] = [] if key == "steps" else 0
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        pc0 = (prop.n_rounds, prop.n_batch_forwards, prop.n_catchup_forwards,
               prop.n_graphed) if prop is not None else (0, 0, 0, 0)
        vh0 = (mr.verify_graph_hits, dict(mr.verify_graph_fallbacks))
        torch.cuda.synchronize()
        t0 = time.time()
        outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=outlen,
                                                    ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.time() - t0
        llm.token_hook = None
        llm.scheduler.token_hook = None
        ids = [o["token_ids"] for o in outs]
        ntok = sum(len(x) for x in ids)
        summ = hook.summary()
        ttfts = [v["ttft"] for v in summ.values() if v["ttft"] == v["ttft"]]

        # ---- 逐轮耗时：从上一个 step 结束到这一步结束；上一跳是 prefill 的那一轮含 prefill，剔除
        steps = S["steps"]
        dec = [s for s in steps if not s[2]]
        durs = []
        for i in range(1, len(steps)):
            if steps[i][2] or steps[i - 1][2]:
                continue
            durs.append((steps[i][0] - steps[i - 1][0], steps[i][1], steps[i][3]))
        by_b = {}
        for dt, b, n in durs:
            by_b.setdefault(b, []).append((dt, n))
        max_b = max(by_b) if by_b else None
        ms_maxB = (1e3 * statistics.median([dt for dt, _ in by_b[max_b]])) if max_b else None
        tok_s_maxB = (sum(n for _, n in by_b[max_b]) / sum(dt for dt, _ in by_b[max_b])
                      if max_b else None)
        # ★ 每 token 成本拆：ms/轮(maxB) ÷ tok/轮(maxB)
        tok_per_round_maxB = ((sum(n for _, n in by_b[max_b]) / len(by_b[max_b]))
                              if max_b else None)
        pc = ((prop.n_rounds - pc0[0], prop.n_batch_forwards - pc0[1],
               prop.n_catchup_forwards - pc0[2], prop.n_graphed - pc0[3])
              if prop is not None else (0, 0, 0, 0))
        fb = {k: v - vh0[1].get(k, 0) for k, v in mr.verify_graph_fallbacks.items()
              if v - vh0[1].get(k, 0) > 0}
        return {
            "wall_s": wall, "tokens": ntok, "tok_per_s": ntok / wall,
            "ttft_ms": 1e3 * statistics.median(ttfts) if ttfts else None,
            "rounds": len(dec),
            "ms_per_round": 1e3 * wall / max(len(dec), 1),
            "ms_per_round_steady": (1e3 * sum(d for d, _, _ in durs) / len(durs)) if durs else None,
            "ms_per_round_maxB": ms_maxB,
            "tok_per_s_maxB": tok_s_maxB,
            "tok_per_round_maxB": tok_per_round_maxB,
            "ms_per_token_maxB": (ms_maxB / tok_per_round_maxB
                                  if (ms_maxB and tok_per_round_maxB) else None),
            "maxB": max_b,
            "rounds_maxB": len(by_b[max_b]) if max_b else 0,
            "rounds_by_B": {str(b): len(v) for b, v in sorted(by_b.items())},
            "tok_per_round": ntok / max(len(dec), 1),
            "accept_rate": (S["accepted"] / S["proposed"]) if S["proposed"] else None,
            "preempt": S["preempt"],
            "proposer_rounds": pc[0], "draft_forwards": pc[1], "catchup_forwards": pc[2],
            "graphed_forwards": pc[3],
            "verify_graph_hits": mr.verify_graph_hits - vh0[0],
            "verify_graph_fallbacks": fb,
            "out_md5": md5_ids(ids),
        }

    prompts = build_prompts(llm.tokenizer, CTX, B)
    variants = list(g["variants"])
    for v in variants:                       # warmup（也把前缀缓存带到稳态）
        set_variant(v)
        llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=64,
                                             ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()
    recs = {v: [] for v in variants}
    for r in range(args.runs):
        for v in variants:                   # ★ 交替轮转 → 配对采样
            set_variant(v)
            x = one_run(prompts, OUTLEN)
            x.update(variant=v, run=r)
            recs[v].append(x)

    cell = dict(group=gname, B=B, ctx=CTX, outlen=OUTLEN, k=K, runs=args.runs,
                prompt="wikitext2", seed=SEED,
                max_num_seqs=cfg.max_num_seqs,
                window=g["window"], head=g["head"], draft=g["draft"],
                graph_bs=list(g["graph_bs"]), graph_pad=g["graph_pad"],
                blocks=dict(num_kvcache_blocks=cfg.num_kvcache_blocks,
                            num_draft_blocks=cfg.num_draft_blocks,
                            draft_window_blocks=cfg.draft_window_blocks,
                            draft_head_blocks=cfg.draft_head_blocks),
                draft_quant_weights=cfg.draft_quant_weights,
                quant_direct_load=cfg.quant_direct_load,
                graph_mem=getattr(mr, "graph_mem", None),
                variants={})
    for v in variants:
        cell["variants"][v] = {
            "tok_per_s": med([x["tok_per_s"] for x in recs[v]]),
            "tok_per_s_maxB": med([x["tok_per_s_maxB"] for x in recs[v]]),
            "ttft_ms": med([x["ttft_ms"] for x in recs[v]]),
            "ms_per_round": med([x["ms_per_round"] for x in recs[v]]),
            "ms_per_round_steady": med([x["ms_per_round_steady"] for x in recs[v]]),
            "ms_per_round_maxB": med([x["ms_per_round_maxB"] for x in recs[v]]),
            "tok_per_round": med([x["tok_per_round"] for x in recs[v]]),
            "tok_per_round_maxB": med([x["tok_per_round_maxB"] for x in recs[v]]),
            "ms_per_token_maxB": med([x["ms_per_token_maxB"] for x in recs[v]]),
            "maxB": recs[v][0]["maxB"],
            "rounds": med([x["rounds"] for x in recs[v]]),
            "rounds_maxB": recs[v][0]["rounds_maxB"],
            "rounds_by_B": recs[v][0]["rounds_by_B"],
            "wall_s": med([x["wall_s"] for x in recs[v]]),
            "accept_rate": med([x["accept_rate"] for x in recs[v]]),
            "preempt": recs[v][0]["preempt"],
            "draft_forwards": med([x["draft_forwards"] for x in recs[v]]),
            "catchup_forwards": med([x["catchup_forwards"] for x in recs[v]]),
            "graphed_forwards": med([x["graphed_forwards"] for x in recs[v]]),
            "verify_graph_hits": recs[v][0]["verify_graph_hits"],
            "verify_graph_fallbacks": recs[v][0]["verify_graph_fallbacks"],
            "raw_tok_per_s_maxB": [round(x["tok_per_s_maxB"], 3) for x in recs[v]],
            "raw_ms_per_round_maxB": [round(x["ms_per_round_maxB"] or float("nan"), 4)
                                      for x in recs[v]],
            "raw_accept_rate": [round(x["accept_rate"], 4) if x["accept_rate"] is not None
                                else None for x in recs[v]],
            "out_md5": recs[v][0]["out_md5"],
            "out_md5_all_same": len({x["out_md5"] for x in recs[v]}) == 1,
            "noise_floor_maxB_pct": med([x["ms_per_round_maxB"] for x in recs[v]])["range_pct"],
            "noise_floor_toks_maxB_pct": med([x["tok_per_s_maxB"] for x in recs[v]])["range_pct"],
        }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(cell, f, indent=2, ensure_ascii=False)
        print("@@WROTE@@" + args.out, flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()
    del llm, mr
    gc.collect()
    torch.cuda.empty_cache()
    return cell


# ----------------------------------------------------------------------
def render(cells):
    """主表：每格四档；再给 S6/G 与 S6_nospec_win/G 的相对提升表。"""
    out = []
    by = {}
    for c in cells:
        by[(c["B"], c["group"])] = c
    bs = sorted({c["B"] for c in cells})

    out.append("### 主表：每一档的绝对数（ctx=%d, out=%d, k=%d）\n" % (CTX, OUTLEN, K))
    out.append("| B | 档 | tok/s(maxB) | active B | TTFT ms | ms/轮(maxB) | tok/轮(maxB) | "
               "ms/token | 接受率 α | 噪声地板 | 轮数@maxB | 总轮数 | 进程 |")
    out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    def fz(d, key, f="%.2f"):
        if d is None:
            return "-"
        x = d.get(key)
        return "-" if x is None else f % x["median"]

    def pick(B, v):
        """在三个组里找持有变体 v 的那一格。"""
        for gname in ("base", "s6", "nowin", "s6stock"):
            c = by.get((B, gname))
            if c and v in c["variants"]:
                return c["variants"][v], c
        return None, None

    for B in bs:
        for v in LADDER:
            d, c = pick(B, v)
            if d is None:
                out.append("| %d | %s | **跑不了** | - | - | - | - | - | - | - | - | - | - |"
                           % (B, v))
                continue
            ar = d["accept_rate"]
            out.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (
                B, v, fz(d, "tok_per_s_maxB", "%.1f"), d["maxB"], fz(d, "ttft_ms", "%.1f"),
                fz(d, "ms_per_round_maxB", "%.3f"), fz(d, "tok_per_round_maxB", "%.3f"),
                fz(d, "ms_per_token_maxB", "%.3f"),
                "-" if ar is None else "%.4f" % ar["median"],
                "%.2f%%" % d["noise_floor_maxB_pct"], d["rounds_maxB"], d["rounds"],
                c["group"]))
        out.append("")

    out.append("\n### 相对提升表（`tok/s(maxB)`；★ `G` = nano-vllm 默认配置）\n")
    out.append("| B | U | **G** | S_nospec | **S6** | S6_nowin | **S6/G** | "
               "**S_nospec/G** | **S6/S_nospec** | **S6_nowin/G** |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def tok(B, v):
        d, _ = pick(B, v)
        if d is None or d["tok_per_s_maxB"] is None:
            return None
        return d["tok_per_s_maxB"]["median"]

    rel = {}
    for B in bs:
        vals = {v: tok(B, v) for v in LADDER}
        G, SN, S6, NW = vals["G"], vals["S_nospec"], vals["S6"], vals["S6_nospec_win"]

        def ratio(a, b):
            return "-" if (a is None or b is None) else "**%.3f×**" % (a / b)

        out.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            B,
            "-" if vals["U"] is None else "%.1f" % vals["U"],
            "-" if G is None else "%.1f" % G,
            "-" if SN is None else "%.1f" % SN,
            "-" if S6 is None else "%.1f" % S6,
            "-" if NW is None else "%.1f" % NW,
            ratio(S6, G), ratio(SN, G), ratio(S6, SN), ratio(NW, G)))
        if G and S6 and SN and NW:
            ds = {v: pick(B, v)[0] for v in ("G", "S_nospec", "S6", "S6_nospec_win")}
            rel[B] = dict(G=G, S_nospec=SN, S6=S6, nowin=NW,
                          S6_over_G=S6 / G, nospec_over_G=SN / G, S6_over_nospec=S6 / SN,
                          nowin_over_G=NW / G,
                          ms_per_token={v: ds[v]["ms_per_token_maxB"]["median"]
                                        for v in ds},
                          tok_per_round={v: ds[v]["tok_per_round_maxB"]["median"]
                                         for v in ds},
                          ms_per_round={v: ds[v]["ms_per_round_maxB"]["median"] for v in ds},
                          alpha={v: (ds[v]["accept_rate"] or {}).get("median") for v in ds},
                          active_B={v: ds[v]["maxB"] for v in ds},
                          noise={v: ds[v]["noise_floor_maxB_pct"] for v in ds})
    out.append("")

    out.append("\n### ★ 每 token 成本拆（`ms/轮(maxB)` ÷ `tok/轮(maxB)`）\n")
    out.append("| B | G ms/tok | G tok/轮 | S6 ms/tok | S6 tok/轮 | S6_nowin ms/tok | "
               "S6_nowin tok/轮 | S6 每轮比 G 贵 | S6 每轮多吐 token |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for B in bs:
        r = rel.get(B)
        if not r:
            continue
        out.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f× | %.2f× |" % (
            B,
            r["ms_per_token"]["G"], r["tok_per_round"]["G"],
            r["ms_per_token"]["S6"], r["tok_per_round"]["S6"],
            r["ms_per_token"]["S6_nospec_win"], r["tok_per_round"]["S6_nospec_win"],
            r["ms_per_round"]["S6"] / r["ms_per_round"]["G"],
            r["tok_per_round"]["S6"] / r["tok_per_round"]["G"]))
    out.append("")

    out.append("\n### 结构证据：active B / α / 噪声地板\n")
    out.append("| B | G active B | S6 active B | nowin active B | G 噪声 | S_nospec 噪声 | "
               "S6 噪声 | nowin 噪声 | S6 α | nowin α |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for B in bs:
        r = rel.get(B)
        if not r:
            continue
        a = r["alpha"]
        out.append("| %d | %d | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %s | %s |" % (
            B, r["active_B"]["G"], r["active_B"]["S6"], r["active_B"]["S6_nospec_win"],
            r["noise"]["G"], r["noise"]["S_nospec"], r["noise"]["S6"],
            r["noise"]["S6_nospec_win"],
            "-" if a["S6"] is None else "%.4f" % a["S6"],
            "-" if a["S6_nospec_win"] is None else "%.4f" % a["S6_nospec_win"]))
    return "\n".join(out), rel


# ----------------------------------------------------------------------
def cmd_all(args):
    os.makedirs(args.outdir, exist_ok=True)
    groups = list(ALL_GROUPS) if args.groups is None else args.groups.split(",")
    bs = [int(x) for x in args.bs.split(",")]
    gpu = args.gpus
    port = args.port
    jobs = [(g, b) for b in bs for g in groups]
    cells = []
    for gname, B in jobs:
        out_path = os.path.join(args.outdir, "%s_B%d.json" % (gname, B))
        log_path = os.path.join(args.outdir, "%s_B%d.log" % (gname, B))
        if args.resume and os.path.exists(out_path):
            print(f"[skip] {out_path} 已存在", flush=True)
            with open(out_path) as f:
                cells.append(json.load(f))
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NANOVLLM_PORT=str(port))
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--group", gname, "--B", str(B), "--runs", str(args.runs),
               "--out", out_path]
        print(f"[launch] group={gname} B={B} gpu={gpu} port={port} -> {log_path}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as logf:
            rc = subprocess.call(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
        print(f"[done] group={gname} B={B} exit={rc} {time.time()-t0:.0f}s", flush=True)
        if rc == 0 and os.path.exists(out_path):
            with open(out_path) as f:
                cells.append(json.load(f))
        else:
            print(f"[warn] group={gname} B={B} 失败（exit={rc}），见 {log_path}", flush=True)
    merged = os.path.join(args.outdir, "merged.json")
    with open(merged, "w") as f:
        json.dump(cells, f, indent=2, ensure_ascii=False)
    table, rel = render(cells)
    with open(os.path.join(args.outdir, "table.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table, flush=True)
    print("\n@@DONE@@ " + merged, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS.keys()))
    ap.add_argument("--B", type=int)
    ap.add_argument("--out")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--groups", default=None, help="逗号分隔；默认 base,s6,nowin")
    ap.add_argument("--bs", default="1,2,4,8,16,32")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--port", default="2411")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "out", "p17"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.all:
        cmd_all(args)
    else:
        assert args.group and args.B, "要么 --all，要么 --group NAME --B N"
        run_group(args.group, args)


if __name__ == "__main__":
    main()
