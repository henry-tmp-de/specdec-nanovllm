#!/usr/bin/env python
"""acceptance_matrix.py —— 端到端统一验收：把「逐级加一项」的消融阶梯跑成一张可复算的表。

为什么需要它（项目最大的缺口）
-----------------------------
各优化的数字散在不同轮次、口径不一（有的 batch=1 有的 B=4；有的 device time 有的墙钟；
接受率有的 0.84 有的 0.99），**互相不可比**，也从未在【同一份代码、同一组负载】上跑成一张表。
本脚本就是那张表：每一行都是**同一份代码上的一个开关翻转**，不是不同代码。

口径声明（★ 违反任何一条这张表就作废）
--------------------------------------
1. 算力口径 = **墙钟**（含 Python dispatch / kernel launch / 同步）。device time 不在本表内。
2. 投机相关的每一行**必须连报接受率** —— 加速比高度依赖接受率（合成素材能把 α 抬到 0.95–0.99）。
3. **主指标是「每轮耗时」**，不是 tok/s（tok/s 被轮数的随机性淹没 ±10–20%）。
   本脚本同时报三个「每轮」口径，防止被 ragged tail 骗：
     · `ms_per_round`      = 墙钟 ÷ decode 轮数（含 prefill 摊销；任务书字面口径）
     · `ms_per_round_steady`= 各 decode 轮耗时的均值（**剔除 prefill**）
     · `ms_per_round_maxB` = 只在 active B = 最大 batch 的那些轮上取中位数（**剔除尾部**）
   ★ 为什么必须报后两个：4 条并发请求落地速度不同，先跑完的会退出，
     尾部会出现 B=3/B=2/B=1。而引擎的 draft 图 / 验证图只按精确 B∈{1,2,4} 捕获，
     **B=3 会整条退回 eager**（`_forward_group` / `run_verify_forward` 的行为，已实测），
     于是尾部每一轮贵好几倍，`墙钟÷轮数` 被它拉高。三个口径一起看才不会被骗。
4. 主结果**成对、交替**测量（同一进程里 A/B/A/B 轮转），报**中位数 + 组间范围**。
   「同一配置多遍的分歧」= **噪声地板**。
5. 不同 batch / 不同输入长度**永不放在同一列比较**。
6. 负载文本 = **自然长文**（wikitext2_test.txt），**不用合成池化素材**。
   ★ 合成素材会把接受率抬到 0.95–0.99，那是本项目最容易误用的坑。
7. 固定 prompt + 固定随机种子（`torch.manual_seed(SEED)` 在每次生成前重置）→ 可复算。

★ 一个必须写明的口径陷阱（本脚本实测发现）
  切换 `spec_batch_draft`（批量 draft）会改变 draft 前向的 GEMM batch 维 →
   浮点归约顺序变 → logits 有 1 ULP 量级差异 → **temp=1.0 采样下轨迹发散**，
   于是各档的接受率不同（B=1 时 bit 相同，B=4 时不同）。
   所以 B=4 的「档间配对比」里混着「不同轨迹」这一项，
   不能当成纯粹的「同一负载下更快的倍数」去读。

负载（固定）
------------
输入 1024 / 4096 token，输出 256 token，temperature=1.0，B=1 / B=4，k=6。

阶梯（每一档只加一项；`S3` 见 docs/端到端验收.md 的说明）
--------------------------------------------------------
    U    target-only 普通解码（enforce_eager=True）        ← 一切对比的零点
    G    U + CUDA Graph（普通 decode 拍图）
    C    旧投机实现（逐请求 draft + 验证图只有单请求形状）
    S1   + 批量 draft
    S2   + 按 (B,k) 捕获验证图（= 现在的默认投机路径）
    S4   + 草稿滑窗 + 解耦分配
    S5   + draft INT8 全线性层就地量化
    S6   + 构造期直载 int8（quant_direct_load=True）

用法
----
    # 单组（一个进程，内部变体交替测量 → 真配对）
    CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2333 python -u scripts/acceptance_matrix.py \
        --group core --out out/core.json --runs 5

    # 一条命令跑整张表（多个组并行跑在不同空卡上，各自一个 rendezvous 端口）
    python -u scripts/acceptance_matrix.py --all --gpus 0,1,2,3 --runs 5 --outdir out/acc

★ 并行覆盖：引擎原来硬编码 tcp://localhost:2333（只能串行）。本脚本用
  `NANOVLLM_PORT`（见 nanovllm/engine/model_runner.py 的 _RENDEZVOUS_PORT，
  默认 "2333" = 原版行为不变）+ `CUDA_VISIBLE_DEVICES` 把多个独立实例铺到多张空卡上。
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
DRAFT_BF16 = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
DRAFT_QALL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-qall")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")

# ---- 固定口径 ----
K = 6
OUTLEN = 256
MAXLEN = 8192
MAX_SEQS = 8          # core 组（无滑窗池；max_num_seqs 对它只影响解码图桶，8 与 12 等价）
WIN_MAX_SEQS = 12     # ★ 滑窗组专用。draft 池 = M*max_num_seqs = 18*12 = 216 块，
                      #   而 target 池只有 ~140 块（预算被 draft 池吃掉一部分）。
                      #   于是「运行期把滑窗压回 0」时 draft 借用 target 的块号
                      #   （≤139）一定落在 draft 池内 —— 这是 in-group `S2` 对照能安全
                      #   存在的前提。用 8 时实测 target 块号到过 151 > 144，
                      #   会被下面的护栏当场拦下（不是静默错误，是有意做成显式报错）。
WINDOW = 4608         # M = 18 块 = 4608 token ≥ 4096+256 → 不截断上下文（见文档的扰动说明）
SEED = 20261011

# 每一组 = 一个进程（init 期的开关只能在这里定）；组内变体靠运行期翻转 → 真配对
# ★ 滑窗组里额外放一个 `S2` 作为【组内对照】：S2 的定义就是"不含滑窗"，
#   所以它把 spec_draft_window 在运行期压回 0（draft 退回共用的全上下文路径），
#   池布局仍是本组 init 的布局 → S4/S2 这一对隔离出"滑窗+解耦分配"这一刀。
GROUPS = {
    "core":            dict(window=0,      draft="bf16", direct=False, variants=["U", "G", "C", "S1", "S2"]),
    "win":             dict(window=WINDOW, draft="bf16", direct=False, variants=["U", "S2", "S4"],
                            max_seqs=WIN_MAX_SEQS),
    "win_int8":        dict(window=WINDOW, draft="qall", direct=False, variants=["U", "S2", "S5"],
                            max_seqs=WIN_MAX_SEQS),
    "win_int8_direct": dict(window=WINDOW, draft="qall", direct=True,  variants=["U", "S2", "S6"],
                            max_seqs=WIN_MAX_SEQS),
}

# 每一档 = 一组运行期开关（U/G/C/S1/S2 全部能在同一个进程里来回翻）
#   window:  "keep" = 保持本组 init 的取值；0 = 运行期压回全上下文（S2 的定义）
VARIANT_SPEC = {
    "U":  dict(spec=False, eager=True,  batch_draft=None, batch_vg=None, window="keep"),
    "G":  dict(spec=False, eager=False, batch_draft=None, batch_vg=None, window="keep"),
    "C":  dict(spec=True,  eager=False, batch_draft=False, batch_vg=False, window=0),
    "S1": dict(spec=True,  eager=False, batch_draft=True,  batch_vg=False, window=0),
    "S2": dict(spec=True,  eager=False, batch_draft=True,  batch_vg=True,  window=0),
    "S4": dict(spec=True,  eager=False, batch_draft=True,  batch_vg=True,  window="keep"),
    "S5": dict(spec=True,  eager=False, batch_draft=True,  batch_vg=True,  window="keep"),
    "S6": dict(spec=True,  eager=False, batch_draft=True,  batch_vg=True,  window="keep"),
}
LADDER = ["U", "G", "C", "S1", "S2", "S4", "S5", "S6"]


# ----------------------------------------------------------------------
def build_prompts(tokenizer, ctx, B):
    """自然长文（wikitext2），按 B 切成互不重叠的连续片段 —— 与 bench_draft_int8 的 wiki 档同一做法。"""
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
    return {"median": statistics.median(vals), "min": vals[0], "max": vals[-1],
            "range_pct": (100.0 * (vals[-1] - vals[0]) / statistics.median(vals)
                          if statistics.median(vals) else None)}


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
    draft_dir = DRAFT_QALL if g["draft"] == "qall" else DRAFT_BF16
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=16384,
              max_num_seqs=g.get("max_seqs", MAX_SEQS), spec_k=K, spec_method="draft",
              draft_model=draft_dir, spec_batch_threshold=0)
    if g["window"]:
        kw["spec_draft_window"] = g["window"]
    if g["direct"]:
        kw["quant_direct_load"] = True

    print(f"=== acceptance_matrix group={gname} window={g['window']} draft={g['draft']} "
          f"direct={g['direct']} runs={args.runs} variants={g['variants']} "
          f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} port={os.environ.get('NANOVLLM_PORT','2333')} "
          f"dev={torch.cuda.get_device_name(0)} ===", flush=True)
    t_load = time.time()
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    prop = mr.spec_proposer
    cfg = mr.config
    print(f"[load] {time.time()-t_load:.1f}s  blocks={cfg.num_kvcache_blocks} "
          f"draft_blocks={cfg.num_draft_blocks} window_blocks={cfg.draft_window_blocks} "
          f"draft_quant={cfg.draft_quant_weights} direct={cfg.quant_direct_load}", flush=True)

    # ---- 统计钩子：接受率 + 逐轮耗时(按 active B 分档) + 结构性计数 ----
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
        # (该步结束时刻, active B, 是否 prefill, 本步真正交付的 token 数)
        S["steps"].append((perf_counter(), len(seqs), bool(is_prefill), delivered))
        return delivered
    llm.scheduler.postprocess = tpp

    # ★ 安全护栏：滑窗组里把 spec_draft_window 压回 0 时，draft 会改用【target 的块号】
    #   去索引 draft 的 KV cache（draft 池只有 num_draft_blocks 块）。这里逐轮挡住
    #   越界——越界会变成 CUDA 非法访存，不是静默错误，但提前报错更好定位。
    if getattr(mr, "draft_kv_cache", None) is not None:
        orig_dr = mr._draft_request

        def dr_guard(seq):
            r = orig_dr(seq)
            n = mr.draft_kv_cache.shape[2]
            ids = list(r["block_table"]) + list(r["draft_ring"])
            if ids and max(ids) >= n:
                raise RuntimeError(
                    f"draft 块号越界：max(block_id)={max(ids)} >= draft 池 {n} 块 "
                    f"(window={mr.config.spec_draft_window})。不允许用这种方式关滑窗。")
            return r
        mr._draft_request = dr_guard

    orig_preempt = llm.scheduler.preempt

    def tpre(seq):
        S["preempt"] += 1
        return orig_preempt(seq)
    llm.scheduler.preempt = tpre

    def set_variant(v):
        c = VARIANT_SPEC[v]
        llm.scheduler.spec_k = K if c["spec"] else 0
        mr.enforce_eager = c["eager"]
        if c["batch_draft"] is not None:
            mr.batch_draft = c["batch_draft"]
            mr.batch_verify_graph = c["batch_vg"]
        # ★ 滑窗开关是【现场读】的（ModelRunner._window_tokens 读 config.spec_draft_window），
        #   所以同一份二进制里能来回翻：0 = 全上下文（S2 的定义），"keep" = 本组 init 值。
        w = g["window"] if c["window"] == "keep" else int(c["window"])
        mr.config.spec_draft_window = w
        # ★★ 光改 config 不够：Scheduler 还会读【BlockManager.draft_window_blocks】决定
        #   两件事 —— ① 前缀缓存命中时让不让 draft 跟着复用（`draft_valid_cached_blocks`），
        #   ② 要不要给序列发滑窗块（`draft_acquire`）。只改 config 不改它，
        #   控制组每轮都会全量补齐（实测 catchup 1044 次 / 62 轮 = 17 次每轮，
        #   而真正的 S2 只有 20 次 —— 那个"2.5×"是补齐造成的假象）。
        bm = llm.scheduler.block_manager
        bm.draft_window_blocks = 0 if w == 0 else mr.draft_window_blocks
        torch.cuda.synchronize()

    def one_run(prompts, outlen):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        for key in list(S):
            S[key] = [] if key == "steps" else 0
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        pc0 = (prop.n_rounds, prop.n_batch_forwards, prop.n_catchup_forwards, prop.n_graphed)
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

        # ---- 逐轮耗时：从上一个 step 结束到这一步结束；★ 上一跳是 prefill 的那一轮含 prefill，剔除 ----
        steps = S["steps"]
        dec = [s for s in steps if not s[2]]
        durs = []                                   # [(秒, 该轮 active B, 该轮交付 token 数)]
        for i in range(1, len(steps)):
            if steps[i][2] or steps[i - 1][2]:
                continue
            durs.append((steps[i][0] - steps[i - 1][0], steps[i][1], steps[i][3]))
        by_b = {}
        for dt, b, n in durs:
            by_b.setdefault(b, []).append((dt, n))
        max_b = max(by_b) if by_b else None
        ms_maxB = (1e3 * statistics.median([dt for dt, _ in by_b[max_b]])) if max_b else None
        # ★ 尾部无关的吞吐：只在 active B = 最大 batch 的那些轮上统计
        tok_s_maxB = (sum(n for _, n in by_b[max_b]) / sum(dt for dt, _ in by_b[max_b])
                      if max_b else None)
        pc = (prop.n_rounds - pc0[0], prop.n_batch_forwards - pc0[1],
              prop.n_catchup_forwards - pc0[2], prop.n_graphed - pc0[3])
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
            "maxB": max_b,
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

    ctxs = [int(x) for x in args.ctxs.split(",")]
    bs = [int(x) for x in args.bs.split(",")]
    variants = list(g["variants"])

    cells = []
    for B in bs:
        for ctx in ctxs:
            prompts = build_prompts(llm.tokenizer, ctx, B)
            for v in variants:                       # 每个变体先 warmup（也把前缀缓存带到稳态）
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
            cell = dict(B=B, ctx=ctx, outlen=OUTLEN, k=K, runs=args.runs,
                        prompt="wikitext2", seed=SEED,
                        block_table=[cfg.num_kvcache_blocks, cfg.num_draft_blocks,
                                     cfg.draft_window_blocks],
                        variants={}, paired={})
            for v in variants:
                cell["variants"][v] = {
                    "tok_per_s": med([x["tok_per_s"] for x in recs[v]]),
                    "ttft_ms": med([x["ttft_ms"] for x in recs[v]]),
                    "ms_per_round": med([x["ms_per_round"] for x in recs[v]]),
                    "ms_per_round_steady": med([x["ms_per_round_steady"] for x in recs[v]]),
                    "ms_per_round_maxB": med([x["ms_per_round_maxB"] for x in recs[v]]),
                    "tok_per_s_maxB": med([x["tok_per_s_maxB"] for x in recs[v]]),
                    "maxB": recs[v][0]["maxB"],
                    "rounds": med([x["rounds"] for x in recs[v]]),
                    "rounds_by_B": recs[v][0]["rounds_by_B"],
                    "tok_per_round": med([x["tok_per_round"] for x in recs[v]]),
                    "wall_s": med([x["wall_s"] for x in recs[v]]),
                    "accept_rate": med([x["accept_rate"] for x in recs[v]]),
                    "preempt": recs[v][0]["preempt"],
                    "draft_forwards": med([x["draft_forwards"] for x in recs[v]]),
                    "catchup_forwards": med([x["catchup_forwards"] for x in recs[v]]),
                    "graphed_forwards": med([x["graphed_forwards"] for x in recs[v]]),
                    "verify_graph_hits": recs[v][0]["verify_graph_hits"],
                    "verify_graph_fallbacks": recs[v][0]["verify_graph_fallbacks"],
                    "raw_ms_per_round": [round(x["ms_per_round"], 4) for x in recs[v]],
                    "raw_ms_per_round_maxB": [round(x["ms_per_round_maxB"] or float("nan"), 4)
                                              for x in recs[v]],
                    "raw_tok_per_s": [round(x["tok_per_s"], 3) for x in recs[v]],
                    "out_md5": recs[v][0]["out_md5"],
                    "out_md5_all_same": len({x["out_md5"] for x in recs[v]}) == 1,
                    "noise_floor_pct": med([x["ms_per_round"] for x in recs[v]])["range_pct"],
                    "noise_floor_maxB_pct": med([x["ms_per_round_maxB"] for x in recs[v]])["range_pct"],
                }
            for i, X in enumerate(variants):         # 配对：同一 run 内 Y/X
                for Y in variants[i + 1:]:
                    ms = [recs[Y][j]["ms_per_round"] / recs[X][j]["ms_per_round"]
                          for j in range(args.runs)]
                    msm = [recs[Y][j]["ms_per_round_maxB"] / recs[X][j]["ms_per_round_maxB"]
                           for j in range(args.runs)]
                    tp = [recs[Y][j]["tok_per_s"] / recs[X][j]["tok_per_s"]
                          for j in range(args.runs)]
                    tpm = [recs[Y][j]["tok_per_s_maxB"] / recs[X][j]["tok_per_s_maxB"]
                           for j in range(args.runs)]
                    cell["paired"][f"{Y}/{X}"] = {"ms_per_round": med(ms),
                                                  "ms_per_round_maxB": med(msm),
                                                  "tok_per_s": med(tp),
                                                  "tok_per_s_maxB": med(tpm)}
            cells.append(cell)
            print("@@" + gname + "@@" + json.dumps(cell, ensure_ascii=False), flush=True)

    out = dict(group=gname, window=g["window"], draft=g["draft"], direct=g["direct"],
               draft_model=draft_dir, variants=variants, k=K, outlen=OUTLEN,
               max_seqs=g.get("max_seqs", MAX_SEQS), max_model_len=MAXLEN, seed=SEED,
               blocks=dict(num_kvcache_blocks=cfg.num_kvcache_blocks,
                           num_draft_blocks=cfg.num_draft_blocks,
                           draft_window_blocks=cfg.draft_window_blocks),
               draft_quant_weights=cfg.draft_quant_weights,
               quant_direct_load=cfg.quant_direct_load,
               cells=cells)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
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
    return out


# ----------------------------------------------------------------------
def render(merged, key="ms_per_round"):
    lines = []
    for gname, out in merged.items():
        lines.append(f"\n### group `{gname}` (window={out['window']}, draft={out['draft']}, "
                     f"direct={out['direct']}, blocks={out['blocks']})")
        for cell in out["cells"]:
            B, ctx = cell["B"], cell["ctx"]
            lines.append(f"\n**B={B} / ctx={ctx} / out={cell['outlen']} / k={cell['k']} "
                         f"/ runs={cell['runs']}**")
            lines.append("| 档 | tok/s | tok/s(maxB) | TTFT ms | ms/轮 | ms/轮(稳态) | ms/轮(maxB) | tok/轮 | 接受率 | 噪声地板 | 配对比 vs 上一档 (ms/轮[maxB] · tok/s[maxB]) |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for v in LADDER:
                if v not in cell["variants"]:
                    continue
                d = cell["variants"][v]

                def fz(x, f="%.2f"):
                    return "-" if x is None else f % x["median"]
                ar = d["accept_rate"]
                ar_txt = "-" if ar is None else "%.4f" % ar["median"]
                nf = d["noise_floor_pct"]
                nf_txt = "-" if nf is None else "%.2f%%" % nf
                idx = LADDER.index(v)
                prev = next((LADDER[j] for j in range(idx - 1, -1, -1)
                             if LADDER[j] in cell["variants"]), None)
                pr = ""
                if prev and f"{v}/{prev}" in cell["paired"]:
                    pp = cell["paired"][f"{v}/{prev}"]
                    m = pp["ms_per_round_maxB"]
                    t = pp["tok_per_s_maxB"]
                    if m and t:
                        pr = (f"{m['median']:.3f} [{m['min']:.3f}~{m['max']:.3f}] · "
                              f"{t['median']:.3f} (vs {prev})")
                lines.append(f"| {v} | {fz(d['tok_per_s'],'%.1f')} | {fz(d['tok_per_s_maxB'],'%.1f')} | "
                             f"{fz(d['ttft_ms'],'%.1f')} | "
                             f"{fz(d['ms_per_round'],'%.3f')} | {fz(d['ms_per_round_steady'],'%.3f')} | "
                             f"{fz(d['ms_per_round_maxB'],'%.3f')} | {fz(d['tok_per_round'],'%.3f')} | "
                             f"{ar_txt} | {nf_txt} | {pr} |")
    return "\n".join(lines)


def cmd_all(args):
    os.makedirs(args.outdir, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",")]
    groups = list(GROUPS.keys())
    pending = list(groups)
    running = []
    gpu_free = list(gpus)
    while pending or running:
        while pending and gpu_free:
            gname = pending.pop(0)
            gpu = gpu_free.pop(0)
            port = args.port_base + 10 * gpu
            out_path = os.path.join(args.outdir, gname + ".json")
            log_path = os.path.join(args.outdir, gname + ".log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NANOVLLM_PORT=str(port))
            logf = open(log_path, "w")
            p = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "--group", gname,
                 "--out", out_path, "--runs", str(args.runs), "--ctxs", args.ctxs, "--bs", args.bs],
                env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"[launch] group={gname} gpu={gpu} port={port} -> {log_path}", flush=True)
            running.append((gname, p, logf, out_path, gpu))
        time.sleep(3)
        still = []
        for gname, p, logf, out_path, gpu in running:
            rc = p.poll()
            if rc is None:
                still.append((gname, p, logf, out_path, gpu))
            else:
                logf.close()
                gpu_free.append(gpu)
                print(f"[done] group={gname} gpu={gpu} exit={rc}", flush=True)
        running = still
    merged = {}
    for gname in groups:
        p = os.path.join(args.outdir, gname + ".json")
        if os.path.exists(p):
            with open(p) as f:
                merged[gname] = json.load(f)
        else:
            print(f"[warn] 缺 {p} —— 该组没跑成功", flush=True)
    with open(os.path.join(args.outdir, "merged.json"), "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    for key in ("ms_per_round", "ms_per_round_maxB"):
        table = render(merged, key)
        fn = os.path.join(args.outdir, "table_%s.md" % key)
        with open(fn, "w") as f:
            f.write(table + "\n")
        print(f"\n=== ACCEPTANCE TABLE (paired key = {key}) ===\n{table}\n")
    print("\n@@DONE@@ " + os.path.join(args.outdir, "merged.json"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS.keys()))
    ap.add_argument("--out")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--ctxs", default="1024,4096")
    ap.add_argument("--bs", default="1,4")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "out", "acc"))
    ap.add_argument("--port-base", type=int, default=2333)
    args = ap.parse_args()
    if args.all:
        cmd_all(args)
    else:
        assert args.group, "要么 --group NAME，要么 --all"
        run_group(args.group, args)


if __name__ == "__main__":
    main()
