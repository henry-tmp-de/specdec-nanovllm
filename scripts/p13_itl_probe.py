#!/usr/bin/env python
"""p13_itl_probe.py —— 账② 上界：一条长 prompt 到场，正在解码的请求这一轮拿不到 token。

场景（同进程内，逐条件重跑）：
  先让 3 条请求进入稳态解码（把它们的 prefill 那几轮丢掉），
  记下稳态的"每轮耗时"（= 每个解码请求的 ITL）；
  然后【突然】塞进一条长 prompt（ctx = 1K/4K/8K/16K），继续 step 到底，
  把"这一轮"的墙钟单独量出来。

  ⇒ 尖刺倍数 = 尖刺轮墙钟 / 稳态解码轮墙钟（中位数）。
  · 论证 ② 是账① 的下游：prefill 轮里 draft prefill 也要跑 ⇒ 尖刺里含这一笔。
    所以【先修①，② 的尖刺会跟着变小】—— 本脚本同时测 with_draft / no_draft 两条件。

★ 若尖刺只有 2–3 倍，按方案 §6.3 的停止条件收手，别做混合调度（P1 当年就这么被否的）。

用法:
  CTXS=1024,4096,8192,16384 CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2414 \\
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p13_itl_probe.py
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

N_DEC = 3
DEC_LEN = 64
DEC_OUT = 100
WARM_ROUNDS = 6


def main():
    CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096,8192,16384").split(",")]
    maxctx = max(CTXS)
    MAXLEN = maxctx + 256

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook

    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=8, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    # P13 两个开关（默认关 = 老行为）
    if os.environ.get("SKIP_OFF") == "1":
        kw["spec_skip_draft_prefill_when_off"] = True
    CHUNK = int(os.environ.get("CHUNK", "0") or 0)
    if CHUNK:
        kw["spec_prefill_chunk_tokens"] = CHUNK
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    llm.scheduler.spec_k = 0        # U 条件：spec off（draft 已加载 —— 这正是账① 的前提）

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    print("@@ITLHEAD@@" + json.dumps(dict(
        skip_off=(os.environ.get("SKIP_OFF") == "1"), chunk=CHUNK,
        n_tokens=len(ids), n_dec=N_DEC, dec_len=DEC_LEN, dec_out=DEC_OUT,
        target_blocks=mr.config.num_kvcache_blocks, block_size=mr.block_size,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT"))), flush=True)

    dec_prompt = ids[:DEC_LEN]
    # ★ 每个长 prompt 拿一段【起点互不相同】的连续 token：块 hash 从 block0 起带
    #   prefix=-1 链式（compute_hash），起点不同 ⇒ 首块 hash 不同 ⇒ 一定不命中前缀缓存。
    #   第一版按 ctx 各自从 base=2000 起切，短 ctx 的 prompt 成了长 ctx 的前缀，
    #   于是长 ctx 的 with_draft 反而比 no_draft 快（缓存命中）—— 已修。
    long_cursor = {"v": 2000}

    def long_prompt(ctx):
        s = long_cursor["v"]
        if s + ctx > len(ids):
            raise SystemExit(f"wikitext 不够：ctx={ctx} @ {s}")
        long_cursor["v"] = s + ctx
        return ids[s:s + ctx]

    def run_scenario(ctx, with_draft):
        saved = mr.draft_model
        if not with_draft:
            mr.draft_model = None
        try:
            hook = TokenDeliveryHook()
            llm.token_hook = hook
            llm.scheduler.token_hook = hook
            dec_ids = []
            sp_dec = SamplingParams(temperature=1.0, max_tokens=DEC_OUT, ignore_eos=True)
            for _ in range(N_DEC):
                llm.add_request(dec_prompt, sp_dec)
                dec_ids.append(llm.scheduler.waiting[-1].seq_id)
            # ---- 稳态解码 ----
            warm = []
            for r in range(WARM_ROUNDS):
                t0 = perf_counter()
                llm.step()
                torch.cuda.synchronize()
                warm.append(perf_counter() - t0)
            steady = statistics.median(warm[2:])    # 丢掉 setup 那两轮
            # ---- 突来一条长 prompt ----
            sp_pre = SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True)
            llm.add_request(long_prompt(ctx), sp_pre)
            spike_ms = None
            spike_delivered_dec = None
            n_prefill_rounds = 0
            while not llm.is_finished():
                n_dec_before = {sid: hook.requests[sid]["num_tokens"] for sid in dec_ids}
                t0 = perf_counter()
                _out, nt = llm.step()
                torch.cuda.synchronize()
                dt = perf_counter() - t0
                if nt > 0:                       # prefill 轮
                    n_prefill_rounds += 1
                    if spike_ms is None or dt > spike_ms:
                        spike_ms = dt
                        spike_delivered_dec = sum(
                            hook.requests[sid]["num_tokens"] - n_dec_before[sid] for sid in dec_ids)
            llm.token_hook = None
            llm.scheduler.token_hook = None
            # 每个解码请求的 ITL：尖刺那一跳
            itl_max = 0.0
            itl_med = []
            for sid in dec_ids:
                it = hook.itl(sid)
                if it:
                    itl_max = max(itl_max, max(it))
                    itl_med.append(statistics.median(it))
            return dict(
                ctx=ctx, with_draft=with_draft,
                steady_round_ms=round(1000 * steady, 2),
                spike_round_ms=round(1000 * spike_ms, 2) if spike_ms else None,
                spike_multiple=round(spike_ms / steady, 2) if spike_ms else None,
                spike_rounds=n_prefill_rounds,
                spike_delivered_to_decoders=spike_delivered_dec,
                itl_max_ms=round(1000 * itl_max, 2),
                itl_median_ms=round(1000 * statistics.median(itl_med), 2) if itl_med else None,
                itl_multiple=round(itl_max / statistics.median(itl_med), 2) if itl_med else None)
        finally:
            mr.draft_model = saved

    # ★ 预热：第一个 scenario 会被 torch.compile / 图重建 / 首次 CUDA 分配抬高。
    #   实测两处：① 同一条件第一遍 203ms vs 后续 130ms；
    #   ② 分块 prefill 的【第一个块】要付一次 chunk 形状的编译钱
    #      （chunk=2048 时 2055ms vs 同尺寸第二块 376ms）——
    #      所以预热必须用【最大 ctx】把分块形状也编译掉，用 1024 预热盖不住。
    run_scenario(maxctx, True)

    out = []
    for ctx in CTXS:
        for with_draft in (True, False):
            row = run_scenario(ctx, with_draft)
            out.append(row)
            print("@@ITL@@ " + json.dumps(row, ensure_ascii=False), flush=True)

    print("@@ITLSUMM@@ " + json.dumps(dict(results=out), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
