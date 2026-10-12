#!/usr/bin/env python
"""p13_equiv.py —— P13 改动的【逐字节等价性】对拍（默认路径必须一行不变）。

做法（沿用项目既有手法 §2.9）：同一份脚本 + 同一组固定负载 + 同一随机种子，
分别指向【基线代码】与【改动后的代码】各跑一遍，比较：

  · `n_forwards` / `logits_md5` —— 每一次 `ModelRunner.run_model` 的
    (input_ids 字节 + positions 字节 + is_prefill + logits 字节) 的 bitwise md5
  · `out_md5`        —— 每条请求最终产出的 token_ids
  · `graph_hits` / `fallbacks` —— 图命中 / 退回的结构性计数

要看的结论是【两遍逐字节相同】，不是"谁更快"。

用法（每个 CASE 一个进程 —— 一个进程只能有一个引擎）:
  CASE=core_w0  CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2431 \
      /home/ziru/nano-vllm/venv/bin/python -u scripts/p13_equiv.py
env: CASE(core_w0|core_w0_b4|win4608) ROOTDIR 出 json 到 stdout 的 @@EQUIV@@ 行
"""
import hashlib
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
SEED = 20261011

CASES = {
    # 默认路径（两个新开关都关）：spec 开 + draft 开 + 拍图
    "core_w0":    dict(window=0,    B=1, ctx=1024, out=48),
    "core_w0_b4": dict(window=0,    B=4, ctx=1024, out=48),
    # 滑窗路径（既有功能，也不能被改坏）
    "win4608":    dict(window=4608, B=1, ctx=4096, out=48),
}


def main():
    CASE = os.environ.get("CASE", "core_w0")
    c = CASES[CASE]

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    kw = dict(enforce_eager=False, max_model_len=8192, max_num_batched_tokens=16384,
              max_num_seqs=8, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if c["window"]:
        kw["spec_draft_window"] = c["window"]
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner

    # ---- 逐次前向的 bitwise 指纹 ----
    H = hashlib.md5()
    N = [0]
    orig_rm = ModelRunner.run_model

    def rm_wrap(self, input_ids, positions, is_prefill):
        out = orig_rm(self, input_ids, positions, is_prefill)
        N[0] += 1
        H.update(b"RM")
        H.update(input_ids.detach().to("cpu", torch.int64).numpy().tobytes())
        H.update(positions.detach().to("cpu", torch.int64).numpy().tobytes())
        H.update(b"P" if is_prefill else b"D")
        H.update(out.detach().to("cpu", torch.float32).numpy().tobytes())
        return out
    ModelRunner.run_model = rm_wrap

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids
    B, ctx = c["B"], c["ctx"]
    span = (len(ids) - ctx) // B
    prompts = [ids[i * span: i * span + ctx] for i in range(B)]

    torch.manual_seed(SEED)
    sp = SamplingParams(temperature=1.0, max_tokens=c["out"], ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)

    oh = hashlib.md5()
    for o in outs:
        oh.update(",".join(map(str, o["token_ids"])).encode())
        oh.update(b"|")

    rec = dict(case=CASE, window=c["window"], B=B, ctx=ctx, out=c["out"],
               n_forwards=N[0], logits_md5=H.hexdigest(), out_md5=oh.hexdigest(),
               graph_hits=getattr(mr, "verify_graph_hits", None),
               fallbacks=dict(getattr(mr, "verify_graph_fallbacks", {}) or {}),
               blocks=llm.model_runner.config.num_kvcache_blocks,
               gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), port=os.environ.get("NANOVLLM_PORT"))
    print("@@EQUIV@@" + json.dumps(rec, sort_keys=True), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
