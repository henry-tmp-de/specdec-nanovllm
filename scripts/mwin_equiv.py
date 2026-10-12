#!/usr/bin/env python
"""mwin_equiv.py —— P7-D（开头常驻段 + 滑窗）的【逐字节等价性】对拍。

沿用项目手法（台账 §2.9 / scripts/p13_equiv.py）：**同一份脚本**分别指向
【基线代码 ddce5b8】与【改动后的代码】各跑一遍，比：

  · `logits_md5` —— 每一次 `ModelRunner.run_model` 的
    (input_ids + positions + is_prefill + logits) 的 bitwise md5
  · `out_md5`    —— 最终产出的 token_ids
  · `graph_hits` / `fallbacks` —— 图命中 / 退回的结构计数

三类 CASE：
  A. 默认路径（window=0, head=0）—— **必须与基线逐字节相同**（老路径一行没改）。
  B. 纯滑窗（window>0, head=0）  —— **必须与基线逐字节相同**（P=0 去掉了 head 段）。
  C. P7-D（head>0）              —— 新功能，基线没有；跑出来当【自洽】证据：
       head+window 盖住整条序列时，它看到的上下文 = 全上下文，
       所以 md5 应与 A/B 里同样看到全上下文的那一格相同。

用法（一个 CASE 一个进程）：
  CASE=core_w0 CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2501 \
      /home/ziru/nano-vllm/venv/bin/python -u scripts/mwin_equiv.py
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

# maxlen 16640 的两格：ctx=8192 时 M=33 或 P=1,M=33 都盖住整条序列 → 上下文 = 全上下文。
CASES = {
    # ---- A. 默认路径（老路径，必须与基线逐字节相同）----
    "core_w0":    dict(window=0,    head=0, B=1, ctx=1024, out=48, maxlen=8192,  mseq=8),
    "core_w0_b4": dict(window=0,    head=0, B=4, ctx=1024, out=48, maxlen=8192,  mseq=8),
    "w0_full":    dict(window=0,    head=0, B=1, ctx=8192, out=96, maxlen=16640, mseq=4),
    # ---- B. 纯滑窗（P=0，必须与基线逐字节相同）----
    "win4608":    dict(window=4608, head=0, B=1, ctx=4096, out=48, maxlen=8192,  mseq=8),
    "win8448":    dict(window=8448, head=0, B=1, ctx=8192, out=96, maxlen=16640, mseq=4),
    # ---- C. P7-D（新功能）----
    # head 1 + window 33 = 8704 token >= 8192+96 → 盖住整条序列
    "p1m34":      dict(window=8448, head=1, B=1, ctx=8192, out=96, maxlen=16640, mseq=4),
    "p2m18":      dict(window=4608, head=2, B=1, ctx=4096, out=48, maxlen=8192,  mseq=8),
}


def main():
    CASE = os.environ.get("CASE", "core_w0")
    TEMP = float(os.environ.get("TEMP", "1.0"))     # 低温度 = 抗 ULP 的"代码路径"对照
    c = CASES[CASE]

    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    kw = dict(enforce_eager=False, max_model_len=c["maxlen"],
              max_num_batched_tokens=c["maxlen"], max_num_seqs=c["mseq"],
              spec_k=6, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
    if c["window"]:
        kw["spec_draft_window"] = c["window"]
    if c["head"]:
        kw["spec_draft_head_blocks"] = c["head"]
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    cfg = mr.config

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
    sp = SamplingParams(temperature=TEMP, max_tokens=c["out"], ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)

    oh = hashlib.md5()
    for o in outs:
        oh.update(",".join(map(str, o["token_ids"])).encode())
        oh.update(b"|")

    rec = dict(case=CASE, temp=TEMP, window=c["window"], head=c["head"], B=B, ctx=ctx, out=c["out"],
               n_forwards=N[0], logits_md5=H.hexdigest(), out_md5=oh.hexdigest(),
               graph_hits=getattr(mr, "verify_graph_hits", None),
               fallbacks=dict(getattr(mr, "verify_graph_fallbacks", {}) or {}),
               blocks=dict(target=cfg.num_kvcache_blocks, draft=cfg.num_draft_blocks,
                           M=cfg.draft_window_blocks,
                           P=getattr(cfg, "draft_head_blocks", 0)),
               gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
               port=os.environ.get("NANOVLLM_PORT"))
    print("@@EQUIV@@" + json.dumps(rec, sort_keys=True), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
