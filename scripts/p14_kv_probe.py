#!/usr/bin/env python
"""p14_kv_probe.py —— P14 的代价侧：同时推进 N 条 partial prefill 的 KV 占用。

为什么必须量化
--------------
轮转把"一次只有一条 partial 序列在场"变成"N 条 partial 序列同时在环上"。
而本引擎的块分配是**首次调度时按整条序列一次性分配**的
（`BlockManager.allocate(seq, num_cached_blocks)` 里 `for i in range(num_cached_blocks, seq.num_blocks)`）——
所以老逻辑下第 2、3 条长 prompt 的块【等它自己排到队首才分配】，
轮转之后它们在**第一轮就全部分配好了**。KV 峰值因此可能抬高，甚至触发抢占。

本探针就直接测这个：N 条 16K 同时到达时
  · 池子占用峰值 / 剩余最小值（`used_block_ids` / `free_block_ids` —— 现成的池子打点）
  · 是否触发抢占（`Scheduler.preempt` 调用次数）
  · 是否出现"分配不到块"（`can_allocate` 返回 -1 的次数 = 准入被拒）
  · 完成轮次（确认轮转真的把它们并行推进了）
并给出**推荐 N 与它的边界**（= 池子能容纳几条 16K 的整条分配）。

用法:
  NS=1,2,3,4 RUNS=1 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2550 \\
      /home/ziru/nano-vllm/venv/bin/python -u scripts/p14_kv_probe.py
env: NS RUNS ROTS CHUNK CTX MAX_SEQS
"""
import json
import os
import statistics
import sys

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
    NS = [int(x) for x in os.environ.get("NS", "1,2,3,4").split(",")]
    ROTS = [x.strip() for x in os.environ.get("ROTS", "off,on").split(",")]
    RUNS = int(os.environ.get("RUNS", "1"))
    CHUNK = int(os.environ.get("CHUNK", "2048"))
    CTX = int(os.environ.get("CTX", "16384"))
    MAX_SEQS = int(os.environ.get("MAX_SEQS", "4"))

    import torch
    from nanovllm import LLM, SamplingParams

    kw = dict(enforce_eager=False, max_model_len=CTX + 256,
              max_num_batched_tokens=16384, max_num_seqs=MAX_SEQS, spec_k=6,
              spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0,
              spec_prefill_chunk_tokens=CHUNK)
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    sch = llm.scheduler
    bm = sch.block_manager

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    POOL = mr.config.num_kvcache_blocks
    BS = mr.block_size
    print("@@P14KVHEAD@@" + json.dumps(dict(
        ns=NS, rots=ROTS, chunk=CHUNK, ctx=CTX, max_num_seqs=MAX_SEQS,
        target_blocks=POOL, block_size=BS,
        blocks_per_seq=-(-CTX // BS),          # ceil(ctx / block_size)
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT"))), flush=True)

    # ---------- 打点 ----------
    ROUNDS = []
    orig_schedule = type(sch).schedule

    def sched_wrap(self):
        seqs, is_prefill = orig_schedule(self)
        ROUNDS.append(dict(pref=bool(is_prefill),
                           seqs=[(s.seq_id, int(s.num_scheduled_tokens)) for s in seqs],
                           used=len(self.block_manager.used_block_ids),
                           free=len(self.block_manager.free_block_ids)))
        return seqs, is_prefill

    type(sch).schedule = sched_wrap

    orig_preempt = type(sch).preempt
    CNT = {"preempt": 0, "alloc_fail": 0}

    def preempt_wrap(self, seq):
        CNT["preempt"] += 1
        return orig_preempt(self, seq)

    type(sch).preempt = preempt_wrap

    orig_can_alloc = bm.can_allocate

    def can_alloc_wrap(seq):
        r = orig_can_alloc(seq)
        if r == -1:
            CNT["alloc_fail"] += 1
        return r

    bm.can_allocate = can_alloc_wrap

    cursor = {"v": 60000}

    def take(n):
        s = cursor["v"]
        if s + n > len(ids):
            cursor["v"] = 0
            s = 0
        cursor["v"] = s + n
        return ids[s:s + n]

    def run_once(n_long, rot):
        """1 条解码请求在场 + n_long 条 16K 同时到达（与场景 B 同构，只把 out 压到 4）。"""
        sch.prefill_chunk_tokens = CHUNK
        sch.prefill_rotate = (rot == "on")
        sch._prefill_chunk_pending = False
        ROUNDS.clear()
        CNT["preempt"] = 0
        CNT["alloc_fail"] = 0
        sp_dec = SamplingParams(temperature=1.0, max_tokens=256, ignore_eos=True)
        sp = SamplingParams(temperature=1.0, max_tokens=4, ignore_eos=True)  # 只测 prefill
        # ★ 必须有解码请求在场：`chunk and not self.running → chunk = 0`
        #   （P12 门控）—— 否则分块和轮转都没开启，这条探针就白测了。
        llm.add_request(ids[:64], sp_dec)
        dec_id = sch.waiting[-1].seq_id
        for _ in range(3):
            llm.step()
        torch.cuda.synchronize()
        ROUNDS.clear()          # 预热轮不算
        CNT["preempt"] = 0
        CNT["alloc_fail"] = 0
        long_ids = []
        for _ in range(n_long):
            llm.add_request(take(CTX), sp)
            long_ids.append(sch.waiting[-1].seq_id)
        while not llm.is_finished():
            llm.step()
        torch.cuda.synchronize()
        _ = dec_id

        pref = 0
        cum = {sid: 0 for sid in long_ids}
        done = {}
        for r in ROUNDS:
            if not r["pref"]:
                continue
            pref += 1
            for sid, n in r["seqs"]:
                if sid in cum:
                    cum[sid] += n
                    if cum[sid] >= CTX and sid not in done:
                        done[sid] = pref
        return dict(
            n_long=n_long, rot=rot, chunk=CHUNK,
            n_prefill_rounds=pref, n_rounds_total=len(ROUNDS),
            done_round_list=[done.get(i) for i in long_ids],
            kv_used_peak=max(r["used"] for r in ROUNDS) if ROUNDS else None,
            kv_free_min=min(r["free"] for r in ROUNDS) if ROUNDS else None,
            kv_used_peak_frac=r2((max(r["used"] for r in ROUNDS) / POOL) if ROUNDS else None, 4),
            n_preempt=CNT["preempt"], n_alloc_fail=CNT["alloc_fail"],
            # 每轮平均同时在场的 partial 序列条数（= 轮转的"并发宽度"）
            mean_batch=r2(statistics.mean([len(r["seqs"]) for r in ROUNDS if r["pref"]])
                          if any(r["pref"] for r in ROUNDS) else None))

    # 预热：先跑一条 16K 把分块形状 / 图编译掉
    run_once(1, "off")

    out = []
    for rot in ROTS:
        for n in NS:
            for _ in range(RUNS):
                rec = run_once(n, rot)
                out.append(rec)
                print("@@P14KV@@" + json.dumps(rec, ensure_ascii=False), flush=True)

    # 汇总：rotate=off 时 N=1 作为基线
    base = next((r for r in out if r["rot"] == ROTS[0] and r["n_long"] == 1), None)
    for r in out:
        if base and base["kv_used_peak"]:
            r["kv_peak_vs_N1"] = r2(r["kv_used_peak"] / base["kv_used_peak"], 3)

    print("@@P14KVSUMM@@" + json.dumps(dict(
        pool_blocks=POOL, block_size=BS, ctx=CTX,
        ceil_blocks_per_seq=-(-CTX // BS),
        pool_capacity_seqs_at_ctx=POOL // (-(-CTX // BS)),
        results=out), ensure_ascii=False), flush=True)

    type(sch).schedule = orig_schedule
    type(sch).preempt = orig_preempt
    bm.can_allocate = orig_can_alloc
    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
