#!/usr/bin/env python
"""p12_sched_probe.py —— P12 主探针：分块 prefill 让路的【ITL 尾延迟 / 吞吐】账。

两种模式（env MODE）：
  · mixed    3 条解码稳态解码 + 突来一条长 prompt（ctx）。
             量：稳态解码轮、ITL 的 p50/p99/max、尖刺倍数、长 prompt 的 TTFT、
                 解码请求在 prefill 期间拿到几个 token、prefill 总墙钟 / 轮数。
  · prefill  只跑一条长 prompt（不给解码请求）。
             量：prefill 总墙钟 / 轮数 / tok/s（分块的吞吐代价）。

设计（同进程多条件，把跨进程噪声变成闭合归因）：
  · 一份 engine，运行期翻 `scheduler.prefill_chunk_tokens`（schedule() 每轮现场读）。
  · 每个条件前清空 `hash_to_block_id`（真冷）+ 每条长 prompt 用互不重叠的起点
    ⇒ 一定不命中前缀缓存。
  · 条件【交替】跑 REPEAT 遍：rep0: c0,c1,...,cN; rep1: c0,c1,...;
    取每条件的中位数 + 范围 ⇒ 噪声地板 = 同条件各遍的极差。

用法:
  W=$PWD; CHUNKS=0,256,512,1024 N_DEC=3 CTX=16384 MODE=mixed REPEAT=2 \
  CUDA_VISIBLE_DEVICES=5 NANOVLLM_PORT=2520 \
    /home/ziru/nano-vllm/venv/bin/python -u $W/scripts/p12_sched_probe.py
env: CHUNKS("0,512") CTX(16384) MODE(mixed|prefill) REPEAT(2) N_DEC(3) OUT(64)
     SKIP_OFF(0) SPEC(0) WIN(0) WARM(6)
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


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook

    CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "0,512").split(",")]
    CTX = int(os.environ.get("CTX", "16384"))
    MODE = os.environ.get("MODE", "mixed")
    REPEAT = int(os.environ.get("REPEAT", "2"))
    N_DEC = int(os.environ.get("N_DEC", "3"))
    OUT = int(os.environ.get("OUT", "64"))
    WARM = int(os.environ.get("WARM", "6"))
    SKIP_OFF = os.environ.get("SKIP_OFF", "0") == "1"
    SPEC = int(os.environ.get("SPEC", "0") or 0)
    WIN = int(os.environ.get("WIN", "0") or 0)

    MAXLEN = CTX + 256
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=8, spec_k=(6 if SPEC else 0), spec_method="draft",
              draft_model=(DRAFT if SPEC else None), spec_batch_threshold=0)
    if WIN:
        kw["spec_draft_window"] = WIN
    if SKIP_OFF:
        kw["spec_skip_draft_prefill_when_off"] = True
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    bm = llm.scheduler.block_manager
    llm.scheduler.spec_k = 0            # 主探针走不投机的路径（纯调度问题）

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    print("@@SCHEDHEAD@@" + json.dumps(dict(
        chunks=CHUNKS, ctx=CTX, mode=MODE, repeat=REPEAT, n_dec=N_DEC, out=OUT,
        skip_off=SKIP_OFF, spec=SPEC, win=WIN,
        target_blocks=mr.config.num_kvcache_blocks, block_size=mr.block_size,
        max_num_batched_tokens=MAXLEN,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))), flush=True)

    dec_prompt = ids[:64]
    cur = {"v": 2000}

    def fresh_long():
        s = cur["v"]
        if s + CTX > len(ids):
            raise SystemExit("wikitext 不够")
        cur["v"] = s + CTX
        return ids[s:s + CTX]

    def run_cond(chunk, rep):
        bm.hash_to_block_id.clear()          # 真冷：清前缀缓存索引
        llm.scheduler.prefill_chunk_tokens = chunk
        llm.scheduler._prefill_chunk_pending = False
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        dec_ids = []
        steady_ms = None
        if MODE == "mixed":
            sp_dec = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
            for _ in range(N_DEC):
                llm.add_request(dec_prompt, sp_dec)
                dec_ids.append(llm.scheduler.waiting[-1].seq_id)
            for _ in range(WARM):
                llm.step()
            torch.cuda.synchronize()
            steady = []
            for _ in range(WARM):
                t0 = perf_counter()
                llm.step()
                torch.cuda.synchronize()
                steady.append((perf_counter() - t0) * 1000)
            steady_ms = statistics.median(steady)

        long_ids = fresh_long()
        llm.add_request(long_ids, SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True))
        long_sid = llm.scheduler.waiting[-1].seq_id

        rounds = []
        wall0 = perf_counter()
        while not llm.is_finished():
            before = {sid: hook.requests[sid]["num_tokens"] for sid in dec_ids}
            t0 = perf_counter()
            _out, nt = llm.step()
            torch.cuda.synchronize()
            dt = (perf_counter() - t0) * 1000
            got = sum(hook.requests[sid]["num_tokens"] - before[sid] for sid in dec_ids)
            rounds.append((bool(nt > 0), dt, got))
        wall = (perf_counter() - wall0) * 1000
        llm.token_hook = None
        llm.scheduler.token_hook = None

        pre = [r for r in rounds if r[0]]
        itls_ms = [1000 * x for sid in dec_ids for x in hook.itl(sid)]
        pf_ms = sum(r[1] for r in pre)
        out = dict(
            chunk=chunk, rep=rep, ctx=CTX, mode=MODE,
            steady_round_ms=round(steady_ms, 2) if steady_ms else None,
            n_rounds=len(rounds), n_prefill_rounds=len(pre),
            prefill_total_ms=round(pf_ms, 1),
            prefill_max_ms=round(max((r[1] for r in pre), default=0.0), 1),
            wall_ms=round(wall, 1),
            itl_p50_ms=round(pct(itls_ms, 0.50), 2) if itls_ms else None,
            itl_p99_ms=round(pct(itls_ms, 0.99), 2) if itls_ms else None,
            itl_max_ms=round(max(itls_ms), 2) if itls_ms else None,
            itl_multiple=round(max(itls_ms) / steady_ms, 2) if (itls_ms and steady_ms) else None,
            dec_tokens_during_prefill=sum(r[2] for r in pre),
            ttft_long_ms=round(1000 * hook.ttft(long_sid), 2),
            prefill_tok_per_s=round(CTX / (pf_ms / 1000), 1) if pf_ms else None,
        )
        return out

    # ★ 预热：用最小 chunk 把分块形状 / 图 / 编译都盖掉
    warm_chunk = min([c for c in CHUNKS if c > 0] or [0])
    run_cond(warm_chunk, -1)

    results = []
    for rep in range(REPEAT):
        for c in CHUNKS:
            r = run_cond(c, rep)
            results.append(r)
            print("@@SCHED@@ " + json.dumps(r, ensure_ascii=False), flush=True)

    summ = {}
    for c in CHUNKS:
        rs = [r for r in results if r["chunk"] == c and r["rep"] >= 0]
        if not rs:
            continue
        agg = {"n_repeats": len(rs)}
        for k in ("steady_round_ms", "prefill_total_ms", "prefill_max_ms", "wall_ms",
                  "itl_p50_ms", "itl_p99_ms", "itl_max_ms", "ttft_long_ms",
                  "prefill_tok_per_s", "n_prefill_rounds", "dec_tokens_during_prefill"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                agg[k] = dict(med=round(statistics.median(vals), 2),
                              lo=round(min(vals), 2), hi=round(max(vals), 2))
        summ[str(c)] = agg
    print("@@SCHEDSUMM@@ " + json.dumps(dict(ctx=CTX, mode=MODE, summ=summ), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
