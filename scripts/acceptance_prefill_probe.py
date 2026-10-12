#!/usr/bin/env python
"""acceptance_prefill_probe.py —— 把 `U`（target-only 普通解码）的 **TTFT / prefill**
拆开，回答「为什么同一个 U 配置在不同组里 TTFT 差 2 倍以上」。

背景（引擎行为，与 spec_k 无关）
--------------------------------
`ModelRunner.run(seqs, is_prefill=True)` 里：

    if self.draft_model is not None:
        self._run_draft_prefill(seqs, input_ids, positions)     # ← 无条件跑 draft prefill！
    ...
    logits = self.run_model(input_ids, positions, is_prefill)   # ← target prefill

**这一句不看 spec_k**。所以只要进程里加载了 draft 模型，
即使是 `spec_k = 0`（我们表里的 `U`），这一步 prefill 也要多跑一遍 0.6B 的前向。
而这条 draft prefill 有两条实现：

  · 全上下文（spec_draft_window == 0）：`self.draft_model(ids, positions)`，
    复用 target 的 context（本地因果 varlen）。
  · 滑窗（> 0）：`_run_draft_prefill_window`，自己 set_context，
    对每条序列跑「最近 M 个块」，走**本地因果**（不带 block_tables）那条路。

本脚本在【同一个进程里】把三种条件各测一遍，先把 prefill 拆开再说别的话：

  ① 有 draft（引擎默认行为，= 表里 U 的实际情况）
  ② 无 draft（把 mr.draft_model 临时摘掉 → run() 里那一步被跳过）
  ③ 有 draft 但 window 强制 0（只在滑窗组里有意义）

用法:
  GROUP=core CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2399 \
      python -u scripts/acceptance_prefill_probe.py
env: GROUP(core|win|win_int8|win_int8_direct) CTX(4096) B(1)
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acceptance_matrix import GROUPS, build_prompts, TARGET, DRAFT_BF16, DRAFT_QALL  # noqa: E402


def main():
    GROUP = os.environ.get("GROUP", "core")
    CTX = int(os.environ.get("CTX", "4096"))
    B = int(os.environ.get("B", "1"))
    REPS = int(os.environ.get("REPS", "3"))

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.token_hook import TokenDeliveryHook

    g = GROUPS[GROUP]
    draft_dir = DRAFT_QALL if g["draft"] == "qall" else DRAFT_BF16
    # ★ 组内可选键（长上下文组带 maxlen/maxntok/max_seqs=4）；老组行为不变
    kw = dict(enforce_eager=False, max_model_len=g.get("maxlen", 8192),
              max_num_batched_tokens=g.get("maxntok", 16384),
              max_num_seqs=g.get("max_seqs", 8), spec_k=6, spec_method="draft",
              draft_model=draft_dir, spec_batch_threshold=0)
    if g["window"]:
        kw["spec_draft_window"] = g["window"]
    if g.get("head"):
        kw["spec_draft_head_blocks"] = g["head"]
    if g["direct"]:
        kw["quant_direct_load"] = True
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner

    llm.scheduler.spec_k = 0        # = 表里的 U
    mr.enforce_eager = True         # = 表里的 U

    T = {"prefill": []}
    orig_run = ModelRunner.run

    def run(self, seqs, is_prefill):
        t0 = time.time()
        r = orig_run(self, seqs, is_prefill)
        torch.cuda.synchronize()
        if is_prefill and all(s.block_table for s in seqs):
            T["prefill"].append(time.time() - t0)
        return r
    ModelRunner.run = run

    prompts = build_prompts(llm.tokenizer, CTX, B)
    sp = SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True)

    TT = {"ttft": [], "wall": [], "nsteps": []}

    def measure(tag, window):
        mr.config.spec_draft_window = window
        T["prefill"].clear()
        llm.generate(prompts, sp, use_tqdm=False)          # warmup
        T["prefill"].clear()
        for k in TT:
            TT[k].clear()
        for _ in range(REPS):
            hook = TokenDeliveryHook()
            llm.token_hook = hook
            llm.scheduler.token_hook = hook
            torch.cuda.synchronize()
            t0 = time.time()
            llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            TT["wall"].append(time.time() - t0)
            llm.token_hook = None
            llm.scheduler.token_hook = None
            summ = hook.summary()
            TT["ttft"].append(statistics.median(
                [v["ttft"] for v in summ.values() if v["ttft"] == v["ttft"]]))
            TT["nsteps"].append(statistics.median([v["n_batches"] for v in summ.values()]))
        p = sorted(T["prefill"])
        return dict(tag=tag, window=window, n=len(p),
                    prefill_ms=round(1000 * statistics.median(p), 2),
                    min_ms=round(1000 * p[0], 2), max_ms=round(1000 * p[-1], 2),
                    hook_ttft_ms=round(1000 * statistics.median(TT["ttft"]), 2),
                    wall_ms=round(1000 * statistics.median(TT["wall"]), 2),
                    n_batches=statistics.median(TT["nsteps"]))

    out = [measure("with_draft_prefill", g["window"])]
    saved = mr.draft_model
    mr.draft_model = None
    out.append(measure("no_draft_prefill", g["window"]))
    mr.draft_model = saved
    # ★ 条件③「有 draft 但运行期把窗口压回 0」只在【draft 池 ≥ target 池】时才安全：
    #   窗口压回 0 后 draft 会改用 target 的块号去索引它自己的 KV cache。
    #   长上下文组按真实并发设 max_num_seqs（=4）⇒ draft 池 < target 池 ⇒ 这一条会越界，
    #   所以它【只在老组里跑】。长上下文的"全上下文 draft prefill"由 lc_w0 组的条件①给。
    if g["window"] and mr.config.num_draft_blocks >= mr.config.num_kvcache_blocks:
        out.append(measure("with_draft_prefill_window0", 0))

    print("@@PREFILL@@" + json.dumps(dict(
        group=GROUP, ctx=CTX, B=B, window=g["window"], head=g.get("head", 0), draft=g["draft"],
        draft_blocks=mr.config.num_draft_blocks, target_blocks=mr.config.num_kvcache_blocks,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), port=os.environ.get("NANOVLLM_PORT"),
        results=out), ensure_ascii=False))

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
