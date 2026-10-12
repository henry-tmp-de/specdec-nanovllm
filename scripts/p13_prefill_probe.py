#!/usr/bin/env python
"""p13_prefill_probe.py —— 账① 上界：draft prefill 的真实代价 + 它到底在算什么。

同进程多条件探针（这个项目验证过的手法，见 docs/端到端验收.md §6）：
  ① with_draft   —— 引擎默认行为：只要进程加载了 draft，prefill 就无条件多跑一遍 0.6B
  ② no_draft     —— 把 mr.draft_model 临时摘掉（run() 里那一步被跳过）
  ③ window0      —— 只在 GROUP=win 时有意义：把 spec_draft_window 压回 0（走全上下文）

★ 冷 / 热分开测，且【冷是真的冷】：
  `docs/端到端验收.md` §6 的 77.8 → 45.7 ms 是在【热前缀缓存】下测的
  （同一批 prompt 反复跑 ⇒ 只剩未命中的尾巴 ≈ 1 个块），量的是"尾巴 + 固定开销"。
  本脚本冷档把 `BlockManager.hash_blocks` 打成 no-op ⇒ 前缀缓存【完全不起作用】，
  每个 prompt 都是完整 prefill；于是 with_draft / no_draft 能跑【同一个 prompt】，
  是真正的 A/B（第一版探针用不同 offset 的"冷"被证明会互相命中，已修）。
  热档恢复 hash_blocks，连跑同一 prompt，第 2 次起是热缓存。

同时把 target / draft 各自 prefill 的【输入 token 数】打出来 —— 这决定修法：
  · draft 跑的是"整条 prompt"还是"未命中的尾巴"？
  · 滑窗路径是不是按 token_ids 重算整个窗口（不认前缀缓存）？

用法:
  GROUP=core CTXS=1024,4096,8192,16384 REPS=3 \\
    CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2413 \\
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p13_prefill_probe.py
env: GROUP(core|win) CTXS REPS WINDOW
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

# 每个 ctx 一个固定的、互不重叠的起点（保证热档也不会互相命中）
BASE = {1024: 0, 4096: 3000, 8192: 20000, 16384: 60000}


def main():
    GROUP = os.environ.get("GROUP", "core")
    CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096,8192,16384").split(",")]
    REPS = int(os.environ.get("REPS", "3"))
    W = int(os.environ.get("WINDOW", "4608"))
    maxctx = max(CTXS)
    MAXLEN = maxctx + 256

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    # ★ win 组用 max_num_seqs=12：draft 池 = M*max_num_seqs = 18*12 = 216 块。
    #   用 8 时（池 144 块）"把 window 压回 0"会让 draft 拿 target 的块号去索引
    #   draft 池，实测 target 块号能到 151 > 144 → CUDA 非法访存。见 acceptance_matrix 的注释。
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=(12 if GROUP == "win" else 8), spec_k=6, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0)
    if GROUP == "win":
        kw["spec_draft_window"] = W
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    bm = llm.scheduler.block_manager
    llm.scheduler.spec_k = 0        # U 条件：spec off（但 draft 已加载 —— 这正是账①的前提）
    mr.enforce_eager = True         # 与表里 U 一致（prefill 本来就强制走 eager，无影响）

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    print("@@P13HEAD@@" + json.dumps(dict(
        group=GROUP, window=(W if GROUP == "win" else 0), n_tokens=len(ids),
        target_blocks=mr.config.num_kvcache_blocks, draft_blocks=mr.config.num_draft_blocks,
        block_size=mr.block_size, dev=torch.cuda.get_device_name(0),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), port=os.environ.get("NANOVLLM_PORT"))),
        flush=True)

    CUR = {"phase": None, "t0": 0.0, "prefill_ms": [], "target_tokens": [], "draft_tokens": []}

    # ---------------- 插桩 ----------------
    orig_run = ModelRunner.run

    def run_wrap(self, seqs, is_prefill):
        hot = bool(is_prefill and seqs and all(s.block_table for s in seqs))
        if hot:
            CUR["phase"] = "prefill"
            CUR["t0"] = perf_counter()
        r = orig_run(self, seqs, is_prefill)
        if CUR["phase"] == "prefill":
            torch.cuda.synchronize()
            CUR["prefill_ms"].append((perf_counter() - CUR["t0"]) * 1000.0)
            CUR["phase"] = None
        return r
    ModelRunner.run = run_wrap

    t_fwd = mr.model.forward

    def t_wrap(input_ids, positions=None):
        if CUR["phase"] == "prefill":
            CUR["target_tokens"].append(int(input_ids.numel()))
        return t_fwd(input_ids, positions)
    mr.model.forward = t_wrap

    d_fwd = mr.draft_model.forward

    def d_wrap(input_ids, positions=None):
        if CUR["phase"] == "prefill":
            CUR["draft_tokens"].append(int(input_ids.numel()))
        return d_fwd(input_ids, positions)
    mr.draft_model.forward = d_wrap

    orig_hash = bm.hash_blocks

    def noop_hash(*a, **k):
        return None

    sp = SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True)

    def run_once(prompt):
        for k in ("prefill_ms", "target_tokens", "draft_tokens"):
            CUR[k].clear()
        llm.generate([prompt], sp, use_tqdm=False)
        return dict(ms=CUR["prefill_ms"][:],
                    target_tokens=CUR["target_tokens"][:],
                    draft_tokens=CUR["draft_tokens"][:])

    # 先跑一次最大 ctx 吸收 torch.compile / 图重建的开销（日志里能看到 dynamo recompile）
    llm.generate([ids[BASE.get(maxctx, 0):BASE.get(maxctx, 0) + maxctx]], sp, use_tqdm=False)

    def measure(cond, ctx):
        saved_draft = mr.draft_model
        saved_win = mr.config.spec_draft_window
        if cond == "no_draft":
            mr.draft_model = None
        elif cond == "window0":
            mr.config.spec_draft_window = 0
        try:
            prompt = ids[BASE[ctx]:BASE[ctx] + ctx]
            # ---- 真冷：前缀缓存打成 no-op，并把【已有的】哈希表清空 ----
            #   ★ 只改 hash_blocks 不够：_deallocate_block 不会删 hash_to_block_id 里的
            #     旧条目（块释放后仍可被命中），上一个条件的热档刚把同一个 prompt 缓存过，
            #     下一个条件的"冷"档就会命中它。第一版探针正是栽在这里
            #     （no_draft 的 cold 与 warm 逐行相等）。清表才是真冷。
            bm.hash_blocks = noop_hash
            bm.hash_to_block_id.clear()
            cold, cold_tok = [], None
            for _ in range(REPS):
                r = run_once(prompt)
                cold.append(statistics.median(r["ms"]) if r["ms"] else float("nan"))
                cold_tok = (r["target_tokens"], r["draft_tokens"])
            # ---- 热：恢复缓存，先跑一次把它缓存上，再连跑 REPS 次 ----
            bm.hash_blocks = orig_hash
            run_once(prompt)
            warm, warm_tok = [], None
            for _ in range(REPS):
                r = run_once(prompt)
                warm.append(statistics.median(r["ms"]) if r["ms"] else float("nan"))
                warm_tok = (r["target_tokens"], r["draft_tokens"])
        finally:
            bm.hash_blocks = orig_hash
            mr.draft_model = saved_draft
            mr.config.spec_draft_window = saved_win
        return dict(
            cond=cond, ctx=ctx,
            cold_ms=round(statistics.median(cold), 2),
            cold_all=[round(x, 2) for x in cold],
            cold_target_tokens=cold_tok[0], cold_draft_tokens=cold_tok[1],
            warm_ms=round(statistics.median(warm), 2),
            warm_all=[round(x, 2) for x in warm],
            warm_target_tokens=warm_tok[0], warm_draft_tokens=warm_tok[1])

    out = []
    conds = ["with_draft", "no_draft"] + (["window0"] if GROUP == "win" else [])
    for ctx in CTXS:
        for cond in conds:
            row = measure(cond, ctx)
            out.append(row)
            print("@@P13@@ " + json.dumps(row, ensure_ascii=False), flush=True)

    print("@@P13SUMM@@ " + json.dumps(dict(group=GROUP, window=(W if GROUP == "win" else 0),
                                          results=out), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
