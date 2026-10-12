#!/usr/bin/env python
"""p12_chunk_equiv.py —— P12 任务 1：**分块 prefill 的分布级无损性**。

问题：分块会改变 attention 的 query 分组——
  · 不分块：整条 prompt 一次前向，`block_tables is None` ⇒ 本地 varlen 因果注意力；
  · 分块：第 1 块走本地路径，第 2..N 块 `start>0` ⇒ `prepare_prefill` 置了
    `block_tables` ⇒ 走【paged varlen】注意力（从 KV cache 里读 K/V）。
两条路的浮点归约顺序不同 ⇒ **极可能有 ULP 级差异**。P13 只验证了「分块关」的
逐字节等价，分块打开时的无损性【没测】。这个脚本补上。

做法（两个进程各跑一遍，CHUNK=0 与 CHUNK=C，离线对拍）：
  · 每个样本：一条 prompt（长度 L），`max_tokens=OUT`、近贪婪温度。
  · 记录每个 `run_model` 调用的输出 logits（prefill 的最后一个 chunk / 每个 decode 步）
    与最终生成的 token 序列，存成 .pt。
  · 对拍比较：逐张 logits 是否【逐位相同】、最大绝对差、生成 token 是否相同。

为什么"近贪婪"：`SamplingParams` 断言 temperature>1e-10（禁止纯贪婪）。
用 TEMP=1e-6 让 logits/1e-6 支配指数赛跑 ⇒ 采样结果≈argmax，可用于比较。

用法:
  OUTDIR=/home/ziru/nano-vllm/sched-work/out/equiv/chunk512 \
  CHUNK=512 CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2511 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p12_chunk_equiv.py
env: CHUNK(0) LENS("1024,3000,4096,9216") OUT(8) TEMP(1e-6) SPEC(0) WIN(0) OUTDIR
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


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.layers.sampler import Sampler

    import hashlib

    def md5_t(t):
        return hashlib.md5(
            t.detach().contiguous().to("cpu", torch.float32).numpy().tobytes()).hexdigest()

    CHUNK = int(os.environ.get("CHUNK", "0") or 0)
    LENS = [int(x) for x in os.environ.get("LENS", "1024,3000,4096,9216").split(",")]
    OUT = int(os.environ.get("OUT", "8"))
    TEMP = float(os.environ.get("TEMP", "1e-6"))
    SPEC = int(os.environ.get("SPEC", "0") or 0)
    WIN = int(os.environ.get("WIN", "0") or 0)
    OUTDIR = os.environ.get("OUTDIR", os.path.join(ROOT, "out", "equiv", f"chunk{CHUNK}"))
    os.makedirs(OUTDIR, exist_ok=True)

    maxlen = max(LENS) + 64
    kw = dict(enforce_eager=False, max_model_len=maxlen, max_num_batched_tokens=maxlen,
              max_num_seqs=4, spec_k=(6 if SPEC else 0), spec_method="draft",
              draft_model=(DRAFT if SPEC else None), spec_batch_threshold=0)
    if WIN:
        kw["spec_draft_window"] = WIN
    llm = LLM(TARGET, **kw)
    if CHUNK:
        llm.scheduler.prefill_chunk_tokens = CHUNK

    # ---------- 机制对照：FORCE_PAGED=1 让【不分块】的 prefill 也走 paged 路径 ----------
    # 不分块时 prepare_prefill 不置 block_tables（本地 varlen）；分块的 chunk>=2 会置
    # （paged varlen）。所以「分块 vs 不分块」的差 = 「paged vs 本地核」的差。
    # 这个开关把不分块也强制成 paged：若结果与普通不分块【不同】、且与分块【同量级】，
    # 就证明差异来自注意力核，而不是分块本身别的副作用。
    if os.environ.get("FORCE_PAGED") == "1":
        from nanovllm.utils.context import get_context, set_context
        orig_pp = ModelRunner.prepare_prefill

        def pp_force(self, seqs):
            ids, pos = orig_pp(self, seqs)
            ctx = get_context()
            if ctx.is_prefill and ctx.block_tables is None:
                bt = self.prepare_block_tables(seqs)
                set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                            ctx.max_seqlen_q, ctx.max_seqlen_k,
                            ctx.slot_mapping, None, bt)
            return ids, pos
        ModelRunner.prepare_prefill = pp_force

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    # ---------- 记录每一次 run_model 的 logits ----------
    calls = []
    orig_rm = ModelRunner.run_model

    def rm_wrap(self, input_ids, positions, is_prefill):
        out = orig_rm(self, input_ids, positions, is_prefill)
        calls.append((bool(is_prefill), out.detach().to("cpu", torch.float32).clone()))
        return out
    ModelRunner.run_model = rm_wrap

    # ---------- 数 sampler 调用次数（分块会多消费 RNG，见文档）----------
    nsamp = [0]
    orig_s = Sampler.forward

    def s_wrap(self, logits, temperatures):
        nsamp[0] += 1
        return orig_s(self, logits, temperatures)
    Sampler.forward = s_wrap

    # ★ 每条 prompt 用互不重叠的起点 ⇒ 首块 hash 不同 ⇒ 一定不命中前缀缓存（真冷）。
    cursor = {"v": 2000}
    summary = []
    for si, L in enumerate(LENS):
        s = cursor["v"]
        if s + L > len(ids):
            raise SystemExit(f"wikitext 不够: L={L} @ {s}")
        cursor["v"] = s + L
        prompt = ids[s:s + L]
        calls.clear()
        n0 = nsamp[0]
        llm.add_request(prompt, SamplingParams(temperature=TEMP, max_tokens=OUT, ignore_eos=True))
        toks = None
        while not llm.is_finished():
            out, _nt = llm.step()
            for _sid, completion in out:
                toks = list(completion)
        pre = [c for (ip, c) in calls if ip]
        dec = [c for (ip, c) in calls if not ip]
        torch.save(dict(L=L, prompt=prompt, tokens=toks,
                        prefill_last=(pre[-1] if pre else None),
                        prefill_all=pre, decode=dec),
                   os.path.join(OUTDIR, f"s{si}_L{L}.pt"))
        summary.append(dict(idx=si, L=L, n_chunk_rounds=len(pre),
                            n_decode_rounds=len(dec),
                            n_sampler_calls=nsamp[0] - n0,
                            prefill_last_md5=md5_t(pre[-1]) if pre else None,
                            tokens=toks))
        print("@@ROW@@ " + json.dumps(summary[-1], ensure_ascii=False), flush=True)

    print("@@CHUNKEQUIV@@ " + json.dumps(dict(
        chunk=CHUNK, lens=LENS, out=OUT, temp=TEMP, spec=SPEC, win=WIN,
        outdir=OUTDIR, rows=summary), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
