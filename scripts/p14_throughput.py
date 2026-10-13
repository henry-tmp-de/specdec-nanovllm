#!/usr/bin/env python
"""p14_throughput.py —— 场景 C：稳态吞吐（连续到达、无空闲，聚合 tok/s）。

★ 口径与另一个 agent（P15）共用：
    ctx=16384 / out=256 / k=6 / max_num_seqs=4 / 自然长文（wikitext2）/
    runs≥3 报中位数 + 范围。

构造：一次性投入 NREQ 条请求（**队列恒非空 ⇒ 引擎无空闲**，真正的并发上限由
      `max_num_seqs=4` 决定），长短交替（16K / 2K），跑到底。
      比的是**聚合吞吐**，不是延迟 —— 轮转在这里没有"减少等待"的直接收益，
      它要证明的是「别把吞吐搞坏」（总工作量不变 ⇒ 不该有损失）。

两档**同进程交替**（off/on 顺序每轮翻转），runs 次，报中位数 + 组间范围。
噪声地板 = 「差值的绝对值 ÷ 中位数」，同一档多遍的分歧即地板（见 docs/优化过程复盘 §1.5）。

用法:
  NREQ=8 RUNS=3 ROTS=off,on CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2540 \\
      /home/ziru/nano-vllm/venv/bin/python -u scripts/p14_throughput.py
env: NREQ RUNS ROTS CHUNK LONG_CTX SHORT_CTX OUT LONG_N
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


def r2(x, nd=2):
    return None if x is None else round(x, nd)


def main():
    NREQ = int(os.environ.get("NREQ", "8"))
    LONG_N = int(os.environ.get("LONG_N", "4"))
    LONG_CTX = int(os.environ.get("LONG_CTX", "16384"))
    SHORT_CTX = int(os.environ.get("SHORT_CTX", "2048"))
    OUT = int(os.environ.get("OUT", "256"))
    CHUNK = int(os.environ.get("CHUNK", "2048"))
    RUNS = int(os.environ.get("RUNS", "3"))
    ROTS = [x.strip() for x in os.environ.get("ROTS", "off,on").split(",")]

    import torch
    from nanovllm import LLM, SamplingParams

    kw = dict(enforce_eager=False, max_model_len=LONG_CTX + 256,
              max_num_batched_tokens=16384, max_num_seqs=4, spec_k=6,
              spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0,
              spec_prefill_chunk_tokens=CHUNK)
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    sch = llm.scheduler
    bm = sch.block_manager

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    print("@@P14THRHEAD@@" + json.dumps(dict(
        nreq=NREQ, long_n=LONG_N, long_ctx=LONG_CTX, short_ctx=SHORT_CTX, out=OUT,
        chunk=CHUNK, runs=RUNS, rots=ROTS, k=6, max_num_seqs=4,
        target_blocks=mr.config.num_kvcache_blocks, block_size=mr.block_size,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT"))), flush=True)

    # 每条请求一段互不重叠的自然长文
    cursor = {"v": 0}

    def take(n):
        s = cursor["v"]
        if s + n > len(ids):
            cursor["v"] = 0
            s = 0
        cursor["v"] = s + n
        return ids[s:s + n]

    prompts = []
    for i in range(NREQ):
        prompts.append(take(LONG_CTX if i < LONG_N else SHORT_CTX))
    prompt_tokens = sum(len(p) for p in prompts)

    # ---- 逐轮打点（吞吐 + KV 峰值，顺带）----
    ROUNDS = []
    orig_schedule = type(sch).schedule

    def sched_wrap(self):
        seqs, is_prefill = orig_schedule(self)
        ROUNDS.append(dict(pref=bool(is_prefill),
                           nseq=len(seqs),
                           ntok=sum(int(s.num_scheduled_tokens) for s in seqs),
                           used=len(self.block_manager.used_block_ids)))
        return seqs, is_prefill

    type(sch).schedule = sched_wrap

    orig_preempt = type(sch).preempt
    N_PREEMPT = [0]

    def preempt_wrap(self, seq):
        N_PREEMPT[0] += 1
        return orig_preempt(self, seq)

    type(sch).preempt = preempt_wrap

    def run_once(rot):
        sch.prefill_chunk_tokens = CHUNK
        sch.prefill_rotate = (rot == "on")
        sch._prefill_chunk_pending = False
        ROUNDS.clear()
        N_PREEMPT[0] = 0
        sp = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
        t0 = perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        wall = perf_counter() - t0

        out_tokens = NREQ * OUT          # ignore_eos + 固定 max_tokens ⇒ 每条正好 OUT 个
        n_pref = sum(1 for r in ROUNDS if r["pref"])
        n_dec = len(ROUNDS) - n_pref
        # ★ 归因用：两档的"批有多宽"直接决定 GEMM/带宽效率 —— 吞吐差必须能从这儿看出来。
        dec_rounds = [r for r in ROUNDS if not r["pref"]]
        pref_rounds = [r for r in ROUNDS if r["pref"]]
        mean_dec_b = (statistics.mean([r["nseq"] for r in dec_rounds]) if dec_rounds else None)
        mean_pref_b = (statistics.mean([r["nseq"] for r in pref_rounds]) if pref_rounds else None)
        mean_pref_tok = (statistics.mean([r["ntok"] for r in pref_rounds]) if pref_rounds else None)
        return dict(
            rot=rot, chunk=CHUNK, wall_s=r2(wall, 3), n_rounds=len(ROUNDS),
            n_prefill_rounds=n_pref, n_decode_rounds=n_dec,
            prompt_tokens=prompt_tokens, out_tokens=out_tokens,
            # 口径①：端到端聚合吞吐（prompt + output 全算）—— 引擎级"总吞吐"
            tok_s_total=r2((prompt_tokens + out_tokens) / wall, 1),
            # 口径②：只算输出 token（纯解码吞吐）
            tok_s_decode=r2(out_tokens / wall, 1),
            # 口径③：prompt 吞吐（prefill 侧）
            tok_s_prefill=r2(prompt_tokens / wall, 1),
            kv_used_peak=max(r["used"] for r in ROUNDS) if ROUNDS else None,
            n_preempt=N_PREEMPT[0],
            mean_decode_batch=r2(mean_dec_b),
            mean_prefill_batch_seqs=r2(mean_pref_b),
            mean_prefill_batch_tokens=r2(mean_pref_tok, 1))

    run_once(ROTS[0])                    # 预热

    out = []
    for run in range(RUNS):
        order = ROTS if run % 2 == 0 else list(reversed(ROTS))
        for rot in order:
            rec = run_once(rot)
            rec["run"] = run
            out.append(rec)
            print("@@P14THR@@" + json.dumps(rec, ensure_ascii=False), flush=True)

    KEY = ("wall_s", "tok_s_total", "tok_s_decode", "tok_s_prefill",
           "mean_decode_batch", "mean_prefill_batch_seqs", "mean_prefill_batch_tokens",
           "n_prefill_rounds", "n_decode_rounds", "kv_used_peak", "n_preempt")
    summ = {}
    for rot in ROTS:
        rows = [r for r in out if r["rot"] == rot]
        agg = dict(n_runs=len(rows))
        for key in KEY:
            xs = [r[key] for r in rows if r.get(key) is not None]
            if xs:
                med = statistics.median(xs)
                spread = (max(xs) - min(xs)) / med if med else 0.0
                agg[key] = dict(median=r2(med), min=r2(min(xs)), max=r2(max(xs)),
                                spread_pct=r2(100 * spread, 2),
                                all=[r2(x) for x in xs])
        summ[rot] = agg
    # 两档之间的差（用中位数），与噪声地板的比值
    if len(ROTS) == 2 and all(r in summ for r in ROTS):
        a, b = summ[ROTS[0]], summ[ROTS[1]]
        summ["delta_on_vs_off_pct"] = {
            k: r2(100 * (b[k]["median"] - a[k]["median"]) / a[k]["median"], 2)
            for k in ("tok_s_total", "tok_s_decode", "tok_s_prefill")
            if k in a and k in b and a[k]["median"]}
        summ["noise_floor_pct"] = {
            k: max(a[k]["spread_pct"], b[k]["spread_pct"])
            for k in ("tok_s_total", "tok_s_decode")
            if k in a and k in b}
    print("@@P14THRSUMM@@" + json.dumps(dict(nreq=NREQ, long_ctx=LONG_CTX, out=OUT,
                                             chunk=CHUNK, runs=RUNS, summary=summ),
                                        ensure_ascii=False), flush=True)

    type(sch).schedule = orig_schedule
    type(sch).preempt = orig_preempt
    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
