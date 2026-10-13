#!/usr/bin/env python
"""p15_smoke.py —— P15 混合批的最小冒烟：先确认"能跑、账对、算得对"，再上基准。

三件事（同一进程内，靠翻开关对照）：
  ① 让路式（chunk>0、overlap 关）跑一段
  ② P15-串行（overlap 开、mode=serial）
  ③ P15-并发（overlap 开、mode=concurrent）
  ④ ★★ 正确性自查：**串行 vs 并发必须吐出逐 token 相同的序列**。
     两次都在 run_once 开头 torch.manual_seed(同一个种子)，而两条路径的
     host 侧 RNG 消费顺序完全一样（decode 的采样都在 prefill 采样之前），
     所以只要重叠没有改变任何计算，输出就该逐 token 相同。
     这是"重叠只改时间、不改结果"的最强证据 —— 比对比 token 数强得多。

用法:
  CUDA_VISIBLE_DEVICES=4 NANOVLLM_PORT=2415 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p15_smoke.py
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

CTX = 4096          # 冒烟用小一点的 ctx，快
CHUNK = 1024
OUT = 32
SEED = 20261013


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.token_hook import TokenDeliveryHook

    kw = dict(enforce_eager=False, max_model_len=CTX + 256, max_num_batched_tokens=CTX + 256,
              max_num_seqs=4, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_prefill_chunk_tokens=CHUNK)
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids
    print("@@SMOKEHEAD@@" + json.dumps(dict(
        ctx=CTX, chunk=CHUNK, out=OUT, seed=SEED,
        target_blocks=mr.config.num_kvcache_blocks, block_size=mr.block_size,
        dev=torch.cuda.get_device_name(0),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), port=os.environ.get("NANOVLLM_PORT"))),
        flush=True)

    def set_mode(mode):
        if mode == "yield":
            llm.scheduler.overlap_mixed = False
        else:
            llm.scheduler.overlap_mixed = True
            llm.scheduler.overlap_mode = mode

    def run_once(mode, tag, cursor):
        s, e = cursor["v"], cursor["v"] + CTX
        cursor["v"] = e
        set_mode(mode)
        torch.manual_seed(SEED)
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        mixed = llm.model_runner._mixed
        before = dict(mixed.stats) if mixed is not None else None
        before_hits = llm.model_runner.verify_graph_hits
        dec_seqs = []
        sp_dec = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
        for _ in range(3):
            llm.add_request(ids[0:64], sp_dec)
            dec_seqs.append(llm.scheduler.waiting[-1])
        for _ in range(4):                       # 稳态
            llm.step()
        torch.cuda.synchronize()
        llm.add_request(ids[s:e], SamplingParams(temperature=1.0, max_tokens=2,
                                                 ignore_eos=True))
        n_pref = n_dec = 0
        t0 = perf_counter()
        while not llm.is_finished():
            _o, nt = llm.step()
            if nt > 0:
                n_pref += 1
            else:
                n_dec += 1
        torch.cuda.synchronize()
        wall = perf_counter() - t0
        after = dict(mixed.stats) if mixed is not None else None
        after_hits = llm.model_runner.verify_graph_hits
        llm.token_hook = None
        llm.scheduler.token_hook = None
        itl_all = []
        for sq in dec_seqs:
            itl_all.extend(hook.itl(sq.seq_id))
        toks = {sq.seq_id: list(sq.completion_token_ids) for sq in dec_seqs}
        row = dict(tag=tag, mode=mode, wall_s=round(wall, 2),
                   n_prefill_steps=n_pref, n_decode_steps=n_dec,
                   itl_med_ms=round(1000 * statistics.median(itl_all), 2) if itl_all else None,
                   itl_max_ms=round(1000 * max(itl_all), 2) if itl_all else None,
                   delivered={k: len(v) for k, v in toks.items()},
                   mixed_delta=(None if before is None else
                                {k: after[k] - before[k] for k in after}),
                   verify_graph_hits_delta=after_hits - before_hits)
        return row, toks

    out = []
    streams = {}
    cursor = {"v": 2000 * 5}
    for mode in ("yield", "serial", "concurrent"):
        row, toks = run_once(mode, f"smoke-{mode}", cursor)
        streams[mode] = toks
        out.append(row)
        print("@@SMOKE@@ " + json.dumps(row, ensure_ascii=False), flush=True)

    # ★★ 正确性：串行 vs 并发逐 token 相同。
    #   ★ 不能按 seq_id 对齐 —— 每个 run 的请求是新建的，seq_id 每轮递增。
    #     按【交付顺序】对齐（三个 run 都是"先入队的先结束"）。
    s_t = [streams["serial"][k] for k in sorted(streams["serial"])]
    c_t = [streams["concurrent"][k] for k in sorted(streams["concurrent"])]
    eq = len(s_t) == len(c_t) and all(a == b for a, b in zip(s_t, c_t))
    print(f"  [{'PASS' if eq else 'FAIL'}] 串行 vs 并发：交付 token 序列逐 token 相同"
          f"（{len(s_t)} 条请求，各 {[len(x) for x in s_t]} 个 token）", flush=True)
    if not eq:
        for i, (a, b) in enumerate(zip(s_t, c_t)):
            first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
            print(f"     #{i}: len {len(a)} vs {len(b)}, 首个不同 @ {first}", flush=True)
    # 附：让路式只保证"能跑通"，不保证与并发逐 token 相同（两个 run 的
    #     投机轮次划分不同 ⇒ host 侧 RNG 消费顺序不同 ⇒ 采样结果本就不同）
    y_t = [streams["yield"][k] for k in sorted(streams["yield"])]
    print(f"  (参考) 让路式的交付 token 数 {[len(x) for x in y_t]}，"
          f"与并发相同: {[len(x) for x in y_t] == [len(x) for x in c_t]}", flush=True)
    print("@@SMOKESUMM@@ " + json.dumps(dict(lossless_serial_vs_concurrent=eq,
                                             results=out), ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
