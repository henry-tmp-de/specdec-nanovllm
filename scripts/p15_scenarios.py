#!/usr/bin/env python
"""p15_scenarios.py —— P15 的三个基准场景 × 三种对照（★ 同一进程内交替跑）。

对照（用户指定的三档，全部在同一张卡、同一时间段、同一进程里【交替】采样）
    yield      让路式现状（spec_prefill_chunk_tokens>0、overlap 关）
    serial     P15 混合批 + 两个前向【串着跑】   ← 预期收益 ≈ 0
    concurrent P15 混合批 + 两条 CUDA stream     ← 唯一可能为正的一档
  （场景 A 另加一档 exclusive = chunk 0 的排他 prefill，作"为什么要分块"的参照）

★ 为什么三档放在同一个进程里交替：串行 vs 并发这一对的核心变量只有
  "发到几条 stream"，量级又只有几十~几百 ms。分成三个进程、三个时间段跑，
  外部干扰（同机其他进程、SM 时钟、显存温度）会和"重叠"混在一起分不开。
  交替跑能把它们的差压到同一时间窗内。见开发记录「测量方法」一节。

★ 每个条件重复 RUNS 次（默认 3），报【中位数 + 范围】；范围本身即噪声地板。

用法:
  SCEN=A CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2417 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p15_scenarios.py
  SCEN=B ... ; SCEN=C ...
"""
import json
import os
import statistics
import sys
from time import perf_counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")

SCEN = os.environ.get("SCEN", "A").upper()
CTX = int(os.environ.get("CTX", "16384"))
CHUNK = int(os.environ.get("CHUNK", "1024"))
OUT = int(os.environ.get("OUT", "256"))
K = int(os.environ.get("K", "6"))
RUNS = int(os.environ.get("RUNS", "3"))
MAX_SEQS = 4


def pct(xs, q):
    """线性插值分位数（xs 非空）。"""
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook

    MAXLEN = CTX + 256
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=MAX_SEQS, spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if CHUNK:
        kw["spec_prefill_chunk_tokens"] = CHUNK
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids
    print("@@SHEAD@@" + json.dumps(dict(
        scen=SCEN, ctx=CTX, chunk=CHUNK, out=OUT, k=K, runs=RUNS, max_seqs=MAX_SEQS,
        target_blocks=mr.config.num_kvcache_blocks, block_size=mr.block_size,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT"))), flush=True)

    # ★ 每个 run 用一段【起点互不相同】的连续 token：块 hash 从 block0 起链式，
    #   起点不同 ⇒ 首块 hash 不同 ⇒ 一定不命中前缀缓存，也保证三档口径一致。
    cursor = {"v": 4000}

    def fresh_ids(n):
        s = cursor["v"]
        if s + n > len(ids):
            cursor["v"] = 4000
            s = cursor["v"]
        cursor["v"] = s + n
        return ids[s:s + n]

    def set_mode(mode):
        if mode in ("yield", "exclusive"):
            llm.scheduler.overlap_mixed = False
        else:
            llm.scheduler.overlap_mixed = True
            llm.scheduler.overlap_mode = mode
        llm.scheduler.prefill_chunk_tokens = CHUNK if mode != "exclusive" else 0

    def drain(hook):
        """跑到引擎清空为止，返回每步的 (时刻, 本步 prefill token 数, 本步交付数)。"""
        steps = []
        while not llm.is_finished():
            t0 = perf_counter()
            _o, nt = llm.step()
            torch.cuda.synchronize()
            steps.append((perf_counter() - t0, nt))
        return steps

    # ==================================================================
    #  场景 A：decode 饥饿（3 条短请求解码中，突来 1 条 16K 长 prompt）
    # ==================================================================
    def scen_a(mode):
        set_mode(mode)
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        mixed = mr._mixed
        b_stats = dict(mixed.stats) if mixed is not None else None

        sp_dec = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
        dec = []
        for _ in range(3):
            llm.add_request(fresh_ids(64), sp_dec)
            dec.append(llm.scheduler.waiting[-1])
        for _ in range(4):                      # 稳态
            llm.step()
        torch.cuda.synchronize()

        sp_long = SamplingParams(temperature=1.0, max_tokens=4, ignore_eos=True)
        t_inject = perf_counter()
        llm.add_request(fresh_ids(CTX), sp_long)
        long_seq = llm.scheduler.waiting[-1]
        t0 = perf_counter()
        drain(hook)
        wall = perf_counter() - t0
        t_pref_done = hook.requests[long_seq.seq_id]["first_token"]

        # 解码请求的 ITL：全窗口 + 只取"结束在长 prompt prefill 窗口内"的那些
        itl_all, itl_win = [], []
        for sq in dec:
            ts = [t for t, _ in hook.requests[sq.seq_id]["batches"]]
            for a, b in zip(ts, ts[1:]):
                itl_all.append(1000 * (b - a))
                if t_inject <= b <= t_pref_done:
                    itl_win.append(1000 * (b - a))
        llm.token_hook = None
        llm.scheduler.token_hook = None
        a_stats = dict(mixed.stats) if mixed is not None else None
        return dict(
            mode=mode, wall_s=round(wall, 3),
            ttft_long_ms=round(1000 * (t_pref_done - t_inject), 2),
            n_dec_batches=[len(hook.requests[sq.seq_id]["batches"]) for sq in dec],
            delivered=[len(sq.completion_token_ids) for sq in dec],
            itl_p50_ms=round(pct(itl_all, .5), 2), itl_p99_ms=round(pct(itl_all, .99), 2),
            itl_max_ms=round(max(itl_all), 2) if itl_all else None,
            itl_n=len(itl_all),
            itl_win_p50_ms=round(pct(itl_win, .5), 2) if itl_win else None,
            itl_win_max_ms=round(max(itl_win), 2) if itl_win else None,
            itl_win_n=len(itl_win),
            mixed_steps=(None if b_stats is None or a_stats is None else
                         {k: a_stats[k] - b_stats[k] for k in a_stats}),
            verify_graph_hits=mr.verify_graph_hits)

    # ==================================================================
    #  场景 B：prefill 公平（1 条解码 + 3 条 16K 同时到达）
    # ==================================================================
    def scen_b(mode):
        set_mode(mode)
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        sp_dec = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
        llm.add_request(fresh_ids(64), sp_dec)
        dec = llm.scheduler.waiting[-1]
        for _ in range(4):
            llm.step()
        torch.cuda.synchronize()
        sp_long = SamplingParams(temperature=1.0, max_tokens=4, ignore_eos=True)
        longs = []
        t0 = perf_counter()
        for _ in range(3):
            llm.add_request(fresh_ids(CTX), sp_long)
            longs.append(llm.scheduler.waiting[-1])
        drain(hook)
        wall = perf_counter() - t0
        ttfts = [1000 * hook.ttft(sq.seq_id) for sq in longs]
        e2e = [1000 * hook.e2e(sq.seq_id) for sq in longs]
        llm.token_hook = None
        llm.scheduler.token_hook = None
        return dict(mode=mode, wall_s=round(wall, 3),
                    ttft_ms=[round(x, 1) for x in ttfts],
                    ttft_p50_ms=round(pct(ttfts, .5), 1), ttft_p99_ms=round(pct(ttfts, .99), 1),
                    e2e_ms=[round(x, 1) for x in e2e],
                    dec_itl_p50_ms=round(1000 * pct(hook.itl(dec.seq_id), .5), 2)
                    if hook.itl(dec.seq_id) else None,
                    dec_distinct_tokens=len(set(dec.completion_token_ids)))

    # ==================================================================
    #  场景 C：稳态吞吐（连续到达、无空闲）
    # ==================================================================
    def scen_c(mode):
        set_mode(mode)
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)
        backlog = {}                     # 还没入队的
        plan_len = [1024, 8192, 1024, 8192, 4096, 8192, 1024, 4096, 8192, 1024, 4096, 8192]
        order = list(range(len(plan_len)))
        seqs = []
        n_inf = 0
        for i in range(MAX_SEQS):        # 先填满在飞
            llm.add_request(fresh_ids(plan_len[order[i]]), sp)
            seqs.append(llm.scheduler.waiting[-1])
            n_inf += 1
        t0 = perf_counter()
        while not llm.is_finished() or n_inf < len(order):
            llm.step()
            # 无空闲：一有空位立刻补一条（这就是"连续到达"的构造）
            if n_inf < len(order) and len(llm.scheduler.running) < MAX_SEQS:
                llm.add_request(fresh_ids(plan_len[order[n_inf]]), sp)
                seqs.append(llm.scheduler.waiting[-1])
                n_inf += 1
        torch.cuda.synchronize()
        wall = perf_counter() - t0
        total = sum(len(sq.completion_token_ids) for sq in seqs)
        llm.token_hook = None
        llm.scheduler.token_hook = None
        return dict(mode=mode, wall_s=round(wall, 3), n_requests=len(seqs),
                    delivered_tokens=total, tok_per_s=round(total / wall, 2))

    fn = {"A": scen_a, "B": scen_b, "C": scen_c}[SCEN]
    modes = ["exclusive", "yield", "serial", "concurrent"] if SCEN == "A" else \
            ["yield", "serial", "concurrent"]

    # 预热：第一遍会把 torch.compile / 图 / 首次分配的钱都付掉，不能进统计
    for m in modes:
        fn(m)
    print("@@SWARM@@" + json.dumps({"modes": modes}), flush=True)

    rows = {m: [] for m in modes}
    for r in range(RUNS):
        for m in modes:                  # ★ 轮内交替：三档贴在同一时间窗里
            row = fn(m)
            row["run"] = r
            rows[m].append(row)
            print("@@SROW@@ " + json.dumps(row, ensure_ascii=False), flush=True)

    # 汇总：中位数 + 范围
    def agg(m, key):
        vals = [x[key] for x in rows[m]
                if isinstance(x.get(key), (int, float)) and not isinstance(x[key], bool)]
        if not vals:
            return None
        return dict(med=round(statistics.median(vals), 3), lo=round(min(vals), 3),
                    hi=round(max(vals), 3), n=len(vals))

    keys = [k for k in rows[modes[0]][0] if k not in ("mode", "run", "mixed_steps")]
    summ = {m: {k: agg(m, k) for k in keys} for m in modes}
    # 逐 run 的图命中增量（结构验收：混合 step 里 decode 有没有照常吃图）
    # ★ 只有场景 A 记了 verify_graph_hits —— 别的场景没这个键，别让报表把一个
    #   统计口径上的缺失变成一个 KeyError（第一版就是这么崩的）。
    summ["_graph_hits_per_run"] = {
        m: [{"run": x["run"], "delta": x["verify_graph_hits"]
             - (rows[m][x["run"] - 1]["verify_graph_hits"] if x["run"] > 0 else 0)}
            for x in rows[m]] for m in modes if "verify_graph_hits" in rows[m][0]}
    print("@@SSUMM@@ " + json.dumps(dict(scen=SCEN, summary=summ), ensure_ascii=False),
          flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
