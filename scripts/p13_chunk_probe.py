#!/usr/bin/env python
"""p13_chunk_probe.py —— 逐轮打点：分块 prefill 到底把时间花在哪。

CHUNK=2048 的 ITL 尖刺反而变大（3×）。这个脚本把每一轮的
(is_prefill, 本步 num_scheduled_tokens 之和, 墙钟 ms) 全打出来，
看清楚是"块没切小"还是"每块在重复做全量工作"。

用法:
  CTX=4096 CHUNK=2048 CUDA_VISIBLE_DEVICES=7 NANOVLLM_PORT=2451 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p13_chunk_probe.py
env: CTX(4096) CHUNK(2048) DRAFT(1=带 draft, 0=摘掉)
"""
import json
import os
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


def main():
    CTX = int(os.environ.get("CTX", "4096"))
    CHUNK = int(os.environ.get("CHUNK", "0") or 0)
    WITH_DRAFT = os.environ.get("DRAFT", "1") == "1"
    MAXLEN = CTX + 256
    # ★ 对照用：ITL 探针把 max_model_len/max_num_batched_tokens 设成 maxctx+256（16640）。
    #   这里允许单独放大，用来判断"每块 6× 慢"是不是这个配置差异造成的。
    MAXBT = int(os.environ.get("MAXBT", "0") or 0) or MAXLEN
    MAXML = int(os.environ.get("MAXML", "0") or 0) or MAXLEN

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    kw = dict(enforce_eager=False, max_model_len=MAXML, max_num_batched_tokens=MAXBT,
              max_num_seqs=8, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if CHUNK:
        kw["spec_prefill_chunk_tokens"] = CHUNK
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    llm.scheduler.spec_k = 0
    if not WITH_DRAFT:
        mr.draft_model = None

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    # 每轮记录：本步进了几个 token + 前向里各模型的输入 token 数
    #（用 run_model 的入参代表 target，用 _run_draft_prefill 的入参代表 draft）
    CUR = {"t_tok": 0, "d_tok": []}
    orig_rm = ModelRunner.run_model

    def rm_wrap(self, input_ids, positions, is_prefill):
        CUR["t_tok"] = int(input_ids.numel())
        return orig_rm(self, input_ids, positions, is_prefill)
    ModelRunner.run_model = rm_wrap

    if mr.draft_model is not None:
        d_fwd = mr.draft_model.forward

        def d_wrap(input_ids, positions=None):
            CUR["d_tok"].append(int(input_ids.numel()))
            return d_fwd(input_ids, positions)
        mr.draft_model.forward = d_wrap

    # 先让 3 条请求进入稳态解码，再塞长 prompt
    sp_dec = SamplingParams(temperature=1.0, max_tokens=100, ignore_eos=True)
    for _ in range(3):
        llm.add_request(ids[:64], sp_dec)
    for _ in range(6):
        llm.step()
    torch.cuda.synchronize()

    llm.add_request(ids[2000:2000 + CTX], SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True))

    rows = []
    t_all = perf_counter()
    r = 0
    while not llm.is_finished():
        CUR["t_tok"] = 0
        CUR["d_tok"] = []
        t0 = perf_counter()
        _o, nt = llm.step()
        torch.cuda.synchronize()
        dt = (perf_counter() - t0) * 1000
        rows.append(dict(round=r, is_prefill=bool(nt > 0), step_tokens=int(nt),
                         target_fwd_tokens=CUR["t_tok"],
                         draft_fwd_tokens=list(CUR["d_tok"]),
                         ms=round(dt, 1)))
        r += 1
        if r > 80:
            break
    wall = (perf_counter() - t_all) * 1000

    for row in rows:
        print("@@R@@ " + json.dumps(row), flush=True)
    pf = [x for x in rows if x["is_prefill"]]
    print("@@CHUNKSUMM@@ " + json.dumps(dict(
        ctx=CTX, chunk=CHUNK, with_draft=WITH_DRAFT,
        n_prefill_rounds=len(pf), n_rounds=len(rows),
        prefill_total_ms=round(sum(x["ms"] for x in pf), 1),
        prefill_max_ms=round(max((x["ms"] for x in pf), default=0), 1),
        wall_ms=round(wall, 1)), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
