#!/usr/bin/env python
"""p14_ab_probe.py —— P14 主指标：场景 A（decode 饥饿）与场景 B（prefill 公平）。

★ 本脚本的口径与另一个 agent（P15）**共用**，两边必须一致，否则没法横向比：

| 场景 | 构造 | 主指标 |
|---|---|---|
| A：decode 饥饿 | 3 条短请求在解码，中途突来 1 条 16K 长 prompt | 解码请求的 ITL p50 / p99 / max |
| B：prefill 公平 | 1 条在解码，**3 条 16K 同时到达** | 三条各自的 TTFT（p50/p99）+ 完成时刻（第几轮） |
| 公共 | ctx=16384、out=256、k=6、max_num_seqs=4、自然长文（wikitext2）、runs≥3、中位数+范围 | |

对照的两档（**同一进程内交替跑**，把跨进程噪声压掉）：
    off = 老调度（`spec_prefill_rotate=False`；分块照常，让路照常）
    on  = P14   （`spec_prefill_rotate=True`）
两档都开 `spec_prefill_chunk_tokens=CHUNK`：★ 轮转只在分块开启时有意义
（没有"每条每轮 ≤ chunk"的上限就无所谓轮转）。

★ 真冷：每次 scenario 前把 `BlockManager.hash_blocks` 打成 no-op **并清空哈希表**，
  让每一档都付【完整 prefill】的钱。否则两档里后跑的那一档会命中先跑那档留下的
  前缀缓存（长 prompt 只剩尾巴要拼），TTFT 直接被缓存偷走 —— 这正是
  docs/端到端验收.md §6 记过的坑（"冷"必须是真的冷）。

用法:
  SCEN=B CHUNK=2048 RUNS=3 ROTS=off,on CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2520 \\
      /home/ziru/nano-vllm/venv/bin/python -u scripts/p14_ab_probe.py
env: SCEN(A|B) CHUNK RUNS ROTS CTX DEC_LEN DEC_OUT OUT
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

CTX = int(os.environ.get("CTX", "16384"))       # 长 prompt 的 token 数
OUT = int(os.environ.get("OUT", "256"))         # 长 prompt 的输出上限
DEC_LEN = int(os.environ.get("DEC_LEN", "64"))  # 解码请求的 prompt 长度
DEC_OUT = int(os.environ.get("DEC_OUT", "256"))
MAX_SEQS = 4
K = 6


def pct(xs, q):
    """线性插值分位数（与 numpy 默认一致，避免下标法在小样本上跳）。"""
    ys = sorted(xs)
    if not ys:
        return float("nan")
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def r2(x, nd=2):
    return None if x is None else round(x, nd)


def main():
    SCEN = os.environ.get("SCEN", "B")
    CHUNK = int(os.environ.get("CHUNK", "2048"))
    RUNS = int(os.environ.get("RUNS", "3"))
    ROTS = [x.strip() for x in os.environ.get("ROTS", "off,on").split(",")]

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook

    MAXLEN = CTX + 256
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=16384,
              max_num_seqs=MAX_SEQS, spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_prefill_chunk_tokens=CHUNK)
    # ★ WIN 只用于"边界线索"那一次测量（开草稿滑窗，看 target 池子能涨到几块）——
    #   它不属于 A/B 的口径，默认不设（= 老行为，全上下文 draft）。
    WIN = int(os.environ.get("WIN", "0") or 0)
    if WIN:
        kw["spec_draft_window"] = WIN
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    sch = llm.scheduler
    bm = sch.block_manager

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    print("@@P14ABHEAD@@" + json.dumps(dict(
        scen=SCEN, chunk=CHUNK, runs=RUNS, rots=ROTS, ctx=CTX, out=OUT,
        dec_len=DEC_LEN, dec_out=DEC_OUT, k=K, max_num_seqs=MAX_SEQS,
        draft_window=int(getattr(mr.config, "spec_draft_window", 0) or 0),
        draft_blocks=mr.config.num_draft_blocks,
        target_blocks=mr.config.num_kvcache_blocks, block_size=mr.block_size,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT"))), flush=True)

    # ---------- 逐轮打点 ----------
    ROUNDS = []
    orig_schedule = type(sch).schedule

    def sched_wrap(self):
        seqs, is_prefill = orig_schedule(self)
        ROUNDS.append(dict(pref=bool(is_prefill),
                           seqs=[(s.seq_id, int(s.num_scheduled_tokens)) for s in seqs],
                           used=len(self.block_manager.used_block_ids),
                           free=len(self.block_manager.free_block_ids),
                           t=perf_counter()))
        return seqs, is_prefill

    type(sch).schedule = sched_wrap

    # ★ 抢占 / 准入被拒：轮转把多条 partial 序列同时摆上台面，最该盯的就是这两个计数
    #   （池子装不下时，老逻辑是"排队等"，轮转是"分配被拒/触发抢占"）。
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

    # ★ 真冷：前缀缓存打成 no-op + 清表（见文件头说明）
    orig_hash = bm.hash_blocks
    NOOP = {"v": False}

    def hash_wrap(*a, **k):
        if NOOP["v"]:
            return None
        return orig_hash(*a, **k)

    bm.hash_blocks = hash_wrap

    # 每个长 prompt 取一段互不重叠的连续 token（起点不同 ⇒ 首块 hash 不同）
    cursor = {"v": 30000}

    def take(n):
        # ★ 取完就回绕。回绕会复用同一段文本 —— 无所谓：整场都把 hash_blocks 打成
        #   no-op（真冷），前缀缓存根本不生效，同一段文本不会带来任何缓存命中。
        s = cursor["v"]
        if s + n > len(ids):
            cursor["v"] = s = 0
        cursor["v"] = s + n
        return ids[s:s + n]

    def run_once(rot):
        """跑一次场景，返回一条结果行。"""
        sch.prefill_chunk_tokens = CHUNK
        sch.prefill_rotate = (rot == "on")
        sch._prefill_chunk_pending = False
        bm.hash_to_block_id.clear()
        NOOP["v"] = True                     # ★ 整场都冷：谁也别想靠缓存偷时间
        ROUNDS.clear()
        CNT["preempt"] = 0
        CNT["alloc_fail"] = 0
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        sch.token_hook = hook

        sp_dec = SamplingParams(temperature=1.0, max_tokens=DEC_OUT, ignore_eos=True)
        sp_long = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)

        # ---- 解码请求先入场并进入稳态（A：3 条；B：1 条）----
        n_dec = 3 if SCEN == "A" else 1
        dec_ids = []
        for _ in range(n_dec):
            llm.add_request(ids[:DEC_LEN], sp_dec)
            dec_ids.append(sch.waiting[-1].seq_id)
        for _ in range(4):                    # 预热几轮，避开首次前向 / 建图
            llm.step()
        torch.cuda.synchronize()

        # ---- 突来长 prompt（A：1 条；B：3 条，同一时刻到）----
        r0 = len(ROUNDS)                      # ★ 轮号从"长 prompt 到场"这一刻起算
        t_arrive = perf_counter()
        long_ids = []
        for _ in range(1 if SCEN == "A" else 3):
            llm.add_request(take(CTX), sp_long)
            long_ids.append(sch.waiting[-1].seq_id)

        t0 = perf_counter()
        while not llm.is_finished():
            llm.step()
        torch.cuda.synchronize()
        wall = perf_counter() - t0

        llm.token_hook = None
        sch.token_hook = None
        NOOP["v"] = False

        rec = dict(rot=rot, chunk=CHUNK, n_dec=n_dec, wall_s=r2(wall, 3))

        # ---- 长 prompt：TTFT / 端到端 ----
        rec["ttft_ms"] = [r2(1000 * hook.ttft(i), 1) for i in long_ids]
        rec["e2e_ms"] = [r2(1000 * hook.e2e(i), 1) for i in long_ids]

        # ---- 解码请求的 ITL：分成"到场前（稳态）"与"到场后"两段 ----
        #  ★ 分界用【区间末端时刻】：一条 ITL 归哪一段，看它结束在到场之前还是之后。
        itl_steady, itl_after = [], []
        for i in dec_ids:
            ts = [t for t, _ in hook.requests[i]["batches"]]
            for a, b in zip(ts, ts[1:]):
                (itl_steady if b <= t_arrive else itl_after).append(b - a)
        rec["itl_steady_med_ms"] = r2(1000 * statistics.median(itl_steady), 2) if itl_steady else None
        rec["itl_n_steady"] = len(itl_steady)
        rec["itl_n_after"] = len(itl_after)
        rec["itl_p50_ms"] = r2(1000 * pct(itl_after, 0.50), 2)
        rec["itl_p99_ms"] = r2(1000 * pct(itl_after, 0.99), 2)
        rec["itl_max_ms"] = r2(1000 * max(itl_after), 2) if itl_after else None
        if itl_after and rec["itl_steady_med_ms"]:
            rec["itl_max_multiple"] = r2(rec["itl_max_ms"] / rec["itl_steady_med_ms"], 1)

        # ---- 逐轮（从长 prompt 到场起算）----
        tail = ROUNDS[r0:]
        pref_round = 0
        cum = {sid: 0 for sid in long_ids}
        done_round = {}
        total_pref_tokens = 0
        for r in tail:
            if not r["pref"]:
                continue
            pref_round += 1
            total_pref_tokens += sum(n for _, n in r["seqs"])
            for sid, n in r["seqs"]:
                if sid in cum:
                    cum[sid] += n
                    if cum[sid] >= CTX and sid not in done_round:
                        done_round[sid] = pref_round
        round_ms = [(b["t"] - a["t"]) * 1000.0 for a, b in zip(ROUNDS, ROUNDS[1:])]
        pf_ms = [round_ms[i] for i in range(len(ROUNDS) - 1)
                 if i >= r0 and ROUNDS[i]["pref"]]
        rec["n_prefill_rounds"] = pref_round
        rec["n_rounds_total"] = len(tail)
        rec["done_round_list"] = [done_round.get(i) for i in long_ids]
        rec["prefill_round_ms_med"] = r2(statistics.median(pf_ms), 1) if pf_ms else None
        rec["prefill_round_ms_max"] = r2(max(pf_ms), 1) if pf_ms else None
        rec["prefill_tokens_total"] = total_pref_tokens
        rec["kv_used_peak"] = max(r["used"] for r in tail) if tail else None
        rec["kv_free_min"] = min(r["free"] for r in tail) if tail else None
        rec["n_preempt"] = CNT["preempt"]
        rec["n_alloc_fail"] = CNT["alloc_fail"]
        # 逐轮轨迹（每轮：是否 prefill / 进几条序列 / 进几个 token / 池子占用）——
        # ★ 诊断用：抢占会把解码序列打回 waiting，`running` 变空 ⇒ chunk 被 P12 门控
        #   压回 0 ⇒ 下一轮变成"整条重拼"，墙钟上是一个大尖刺。没有轨迹看不出来。
        rec["round_trace"] = [[int(r["pref"]), len(r["seqs"]),
                               sum(n for _, n in r["seqs"]), r["used"]] for r in tail]
        return rec

    # ---- 预热（把分块形状 / 图 / 首次分配都编译掉）----
    run_once(ROTS[0])

    out = []
    for run in range(RUNS):
        order = ROTS if run % 2 == 0 else list(reversed(ROTS))
        for rot in order:
            rec = run_once(rot)
            rec["run"] = run
            out.append(rec)
            print("@@P14AB@@" + json.dumps(rec, ensure_ascii=False), flush=True)

    # ---- 汇总：中位数 + 组间范围 ----
    KEY = ("itl_p50_ms", "itl_p99_ms", "itl_max_ms", "itl_steady_med_ms",
           "wall_s", "n_prefill_rounds", "n_rounds_total", "prefill_round_ms_med",
           "prefill_round_ms_max", "kv_used_peak", "prefill_tokens_total")
    summ = {}
    for rot in ROTS:
        rows = [r for r in out if r["rot"] == rot]
        agg = dict(n_runs=len(rows))
        for key in KEY:
            xs = [r[key] for r in rows if r.get(key) is not None]
            if xs:
                agg[key] = dict(median=r2(statistics.median(xs)),
                                min=r2(min(xs)), max=r2(max(xs)),
                                all=[r2(x) for x in xs])
        nl = len(rows[0]["ttft_ms"]) if rows else 0
        agg["ttft_ms"] = [dict(median=r2(statistics.median([r["ttft_ms"][j] for r in rows]), 1),
                               min=r2(min(r["ttft_ms"][j] for r in rows), 1),
                               max=r2(max(r["ttft_ms"][j] for r in rows), 1),
                               all=[r["ttft_ms"][j] for r in rows])
                          for j in range(nl)]
        agg["done_round_list_all"] = [r["done_round_list"] for r in rows]
        summ[rot] = agg
    print("@@P14ABSUMM@@" + json.dumps(dict(scen=SCEN, chunk=CHUNK, ctx=CTX, out=OUT,
                                            runs=RUNS, summary=summ),
                                       ensure_ascii=False), flush=True)

    type(sch).schedule = orig_schedule
    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
