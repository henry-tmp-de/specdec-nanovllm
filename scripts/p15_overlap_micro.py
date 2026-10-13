#!/usr/bin/env python
"""p15_overlap_micro.py —— P15 的决定性微基准：两个前向到底有没有【真重叠】。

不测调度、不测端到端，只测一件事：
    t(只跑 decode 子批) 、t(只跑 prefill 块) 、t(串行跑两个) 、t(并发跑两个)

判据（这是整轮第一优先级的数）
    · t(串行) ≈ t(decode) + t(prefill)   ← 它就是把两个前向排在同一条流上
    · t(并发) ≈ max(...)                 ← 重叠成立
    · 收益 = t(串行) − t(并发)           ← ★ 应当 ≈ min(t_decode, t_prefill)，
                                            也就是"被藏起来的那一个前向"

★★ 为什么要在【多个 prefill 深度】上各测一遍：
   prefill 块的成本随 `seqlen_k` 长（第 i 块的 avg kv 长度 ≈ (i+1)*CHUNK/2），
   而**收益上界 = min(解码轮, prefill 块)**。只测第 1 块（最便宜的那块）会把
   上界**系统性低估**。四个深度一起看，才能回答"上界到底是多少"。

★ 为什么直接调 run_mixed 而不是走 step()：
  这样可以把调度、postprocess、token 采样全部排除在外，只留下"发射两个前向"
  这一件事，避免把调度开销混进重叠的归因里。
  （代价：这一步不改 KV/计数状态，所以只能当【计时】用，不能当正确性用 ——
    正确性由 scripts/p15_smoke.py 的"串行 vs 并发逐 token 相同"负责。）

用法:
  CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2421 CTX=16384 CHUNK=1024 REPS=15 \
    DEPTHS=0,5,10,14 /home/ziru/nano-vllm/venv/bin/python -u scripts/p15_overlap_micro.py
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

N_DEC = int(os.environ.get("N_DEC", "3"))
CTX = int(os.environ.get("CTX", "16384"))
CHUNK = int(os.environ.get("CHUNK", "1024"))
REPS = int(os.environ.get("REPS", "15"))
WARM = int(os.environ.get("WARM", "3"))
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "0,5,10,14").split(",")]


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.mixed_batch import MixedBatchPlan

    MAXLEN = CTX + 256
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=4, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_prefill_chunk_tokens=CHUNK,
              spec_overlap_mixed=True, spec_overlap_mode="concurrent")
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids
    print("@@MICROHEAD@@" + json.dumps(dict(
        ctx=CTX, chunk=CHUNK, reps=REPS, warm=WARM, n_dec=N_DEC, spec_k=6, depths=DEPTHS,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT"))), flush=True)

    # ---- 造场景：3 条解码 + 1 条长 prompt ----
    sp_dec = SamplingParams(temperature=1.0, max_tokens=4096, ignore_eos=True)
    for _ in range(N_DEC):
        llm.add_request(ids[0:64], sp_dec)
    for _ in range(6):                                   # 稳态解码
        llm.step()
    llm.add_request(ids[2000:2000 + CTX], SamplingParams(temperature=1.0, max_tokens=1024,
                                                         ignore_eos=True))
    torch.cuda.synchronize()

    def timed(fn, reps):
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(1000 * (perf_counter() - t0))
        return ts

    def measure(plan, depth):
        dec_seqs, pre_seqs = plan.decode_seqs, plan.prefill_seqs
        for _ in range(WARM):                       # 该深度的形状先编译/热身
            mr.run_mixed(dec_seqs, pre_seqs, "serial")
            mr.run_mixed(dec_seqs, pre_seqs, "concurrent")
            mr.run(dec_seqs, False)
            mr.run(pre_seqs, True)
        buckets = {"decode_only": [], "prefill_only": [], "serial": [], "concurrent": [],
                   "decode_only_nospec": []}
        for _ in range(REPS):
            # ★ 四个条件【逐个交替】采样，把外部干扰压到同一时间窗内。
            buckets["decode_only"] += timed(lambda: mr.run(dec_seqs, False), 1)
            buckets["prefill_only"] += timed(lambda: mr.run(pre_seqs, True), 1)
            buckets["serial"] += timed(lambda: mr.run_mixed(dec_seqs, pre_seqs, "serial"), 1)
            buckets["concurrent"] += timed(
                lambda: mr.run_mixed(dec_seqs, pre_seqs, "concurrent"), 1)
        saved = [s.num_scheduled_tokens for s in dec_seqs]
        for s in dec_seqs:
            s.num_scheduled_tokens = 1
        for _ in range(REPS):
            buckets["decode_only_nospec"] += timed(lambda: mr.run(dec_seqs, False), 1)
        for s, v in zip(dec_seqs, saved):
            s.num_scheduled_tokens = v

        med = {k: statistics.median(v) for k, v in buckets.items()}
        rng = {k: (min(v), max(v)) for k, v in buckets.items()}
        d, p, s, c = (med["decode_only"], med["prefill_only"], med["serial"],
                      med["concurrent"])
        return dict(
            depth=depth,
            prefill_seq_len_k=int(max(sq.num_tokens for sq in pre_seqs)),
            chunk_tokens=int(sum(sq.num_scheduled_tokens for sq in pre_seqs)),
            decode_rows=int(sum(sq.num_scheduled_tokens for sq in dec_seqs)),
            t_decode_ms=round(d, 3), t_prefill_ms=round(p, 3),
            t_decode_nospec_ms=round(med["decode_only_nospec"], 3),
            t_serial_ms=round(s, 3), t_concurrent_ms=round(c, 3),
            serial_minus_sum_ms=round(s - (d + p), 3),
            overlap_gain_ms=round(s - c, 3),
            overlap_upper_ms=round(min(d, p), 3),
            overlap_ratio=round((s - c) / min(d, p), 3),
            decode_slowdown_in_concurrent_ms=round(c - d, 3),
            ranges_ms={k: [round(v[0], 2), round(v[1], 2)] for k, v in rng.items()},
        )

    # ---- 沿 prefill 深度推进，每个目标深度测一组 ----
    targets = sorted(set(DEPTHS))
    want = set(targets)
    rows = []
    idx = -1
    guard = 0
    while want and guard < 400:
        guard += 1
        plan = llm.scheduler.schedule() if hasattr(llm.scheduler, "schedule") else None
        if isinstance(plan, MixedBatchPlan):
            idx += 1
            if idx in want:
                rows.append(measure(plan, idx))
                want.discard(idx)
                print("@@MICROROW@@ " + json.dumps(rows[-1], ensure_ascii=False), flush=True)
            dec, pf = mr.call("run_mixed", plan.decode_seqs, plan.prefill_seqs, plan.mode)
            llm.scheduler.postprocess(plan.prefill_seqs, pf, True)
            llm.scheduler.postprocess(plan.decode_seqs, dec, False)
        else:
            tok = mr.call("run", plan[0], plan[1])
            llm.scheduler.postprocess(plan[0], tok, plan[1])
    print("@@MICROSUMM@@" + json.dumps(dict(rows=rows), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
