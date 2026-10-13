#!/usr/bin/env python
"""p15_graphprobe.py —— P15 只回答一个问题：**图命中率有没有掉？**

口径
----
· 纯 decode step：P15 不经过 `run_mixed`（CPU 回归测试 §4 已锁），
  `ModelRunner.run()` 一个字节没改 ⇒ 命中率必然与改动前相同。本脚本复核这一点。
· 混合 step：decode 子批同样走 `run(seqs, False)` ⇒ 照常命中它该命中的形状。
  ★ 但"该命中的形状"由 config.spec_graph_bs=(1,2,4) 决定 ——
    B=3 本来就没被捕获（**这是既有行为，不是 P15 引入的**）。
  本脚本把 hits 与逐形状的 fallback 原因打出来，把这件事说清楚。

用法:
  CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2425 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p15_graphprobe.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")
CTX = int(os.environ.get("CTX", "16384"))
CHUNK = int(os.environ.get("CHUNK", "1024"))
N_DEC = int(os.environ.get("N_DEC", "3"))
# ★ 用来证明"B=3 退回 eager"是【既有配置】而不是 P15 引入的：
#   设 GBS=1,2,3,4 把 B=3 也捕获进去，退回计数应当归零。
GBS = os.environ.get("GBS", "")


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.mixed_batch import MixedBatchPlan

    MAXLEN = CTX + 256
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=4, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_prefill_chunk_tokens=CHUNK,
              spec_overlap_mixed=True, spec_overlap_mode="concurrent")
    if GBS:
        kw["spec_graph_bs"] = tuple(int(x) for x in GBS.split(","))
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids
    print("@@GRAPHHEAD@@" + json.dumps(dict(
        n_dec=N_DEC, chunk=CHUNK, ctx=CTX, spec_graph_bs=list(mr.config.spec_graph_bs),
        captured_verify_shapes=sorted(mr.verify_graphs.keys()),
        dev=torch.cuda.get_device_name(0))), flush=True)

    sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)
    for _ in range(N_DEC):
        llm.add_request(ids[0:64], sp)
    for _ in range(4):
        llm.step()
    llm.add_request(ids[2000:2000 + CTX], SamplingParams(temperature=1.0, max_tokens=4,
                                                         ignore_eos=True))
    mixed = pure = 0
    while not llm.is_finished():
        plan = llm.scheduler.schedule()
        if isinstance(plan, MixedBatchPlan):
            mixed += 1
        else:
            pure += 1
        tok = mr.call("run", plan[0], plan[1]) if not isinstance(plan, MixedBatchPlan) else None
        if tok is None:
            dec, pf = mr.call("run_mixed", plan.decode_seqs, plan.prefill_seqs, plan.mode)
            llm.scheduler.postprocess(plan.prefill_seqs, pf, True)
            llm.scheduler.postprocess(plan.decode_seqs, dec, False)
        else:
            llm.scheduler.postprocess(plan[0], tok, plan[1])
    torch.cuda.synchronize()
    print("@@GRAPH@@" + json.dumps(dict(
        mixed_steps=mixed, pure_steps=pure,
        verify_graph_hits=mr.verify_graph_hits,
        verify_graph_fallbacks=mr.verify_graph_fallbacks,
        draft_graphs_bs=sorted(mr.draft_graphs.keys()) if mr.draft_graphs else None),
        ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
