#!/usr/bin/env python
"""p16_pad_variant_probe.py —— 空行「零长度」到底安不安全？（实测，不靠猜）

背景
----
vLLM 的 cudagraph padding 把空行的 `seq_lens` / `query_start_loc` 都归零
（`slot_mapping=-1`、`block_table=0`）。本项目原来的 capture 注释写着
「`cu_seqlens=[0,0]` 会让 varlen 内核退化成长度 0」，所以任务书要求
「尽量归零，会崩/不对就给小的合法值，并如实记录」。

本脚本把三种空行形状都在**真实图**上跑一遍，每种都与 eager 对拍：

  A. q=nq, k=1   ← 当前实现（PAD_Q_LEN=None, PAD_CTX_LEN=1）
  B. q=nq, k=0   ← key 侧归零（vLLM 的 seq_lens=0）
  C. q=0,  k=0   ← 两侧都归零（vLLM 的 query_start_loc 台阶）

另外：
  D. draft 图的 context_lens 归零（= vLLM 的 seq_lens=0 in decode attention）
  E. **eager 自身是否逐位可复现** —— 如果 eager 连调两次都不一样，那
     「图 vs eager」的 max|Δ| 就包含噪声地板，必须如实标出来。

用法（服务器）:
  CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2603 \
    /home/ziru/nano-vllm/venv/bin/python -u scripts/p16_pad_variant_probe.py
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
CTX = int(os.environ.get("CTX", "4096"))
CHUNK = int(os.environ.get("CHUNK", "1024"))
K = int(os.environ.get("K", "6"))
N_DEC = int(os.environ.get("N_DEC", "3"))


def emit(tag, obj):
    print(f"@@{tag}@@{json.dumps(obj, ensure_ascii=False)}", flush=True)


def main():
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.spec_decode import graph_pad as gp
    from nanovllm.engine.mixed_batch import MixedBatchPlan

    with open(WIKI, encoding="utf-8") as f:
        text = f.read()
    maxlen = CTX + 256
    llm = LLM(TARGET, enforce_eager=False, max_model_len=maxlen,
              max_num_batched_tokens=maxlen, max_num_seqs=4,
              spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_prefill_chunk_tokens=CHUNK,
              spec_overlap_mixed=True, spec_overlap_mode="concurrent",
              spec_graph_bs=(1, 2, 4), spec_graph_pad=False)
    mr = llm.model_runner
    ids = llm.tokenizer(text).input_ids

    # 先把 3 条请求跑起来（B=3 的投机 step 才是我们要的形状）
    sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)
    for _ in range(N_DEC):
        llm.add_request(ids[0:64], sp)
    for _ in range(3):
        llm.step()

    # ==================================================================
    # E. eager 自身的可复现性（噪声地板）
    # ==================================================================
    cap = {}

    orig_v = mr.run_verify_forward
    orig_p = mr.spec_proposer.propose_batch

    # 在真实 step 里截一份 (input_ids, positions, attention context) ——
    # 不手工拼批，保证形状就是引擎自己会走的那个。
    from nanovllm.utils.context import get_context, set_context

    def hooked_verify(input_ids, positions):
        ex = mr._verify_extra
        if "v_ids" not in cap and ex is not None and ex["full_k"] and ex["n"] == 3:
            cap["v_ids"] = input_ids.clone()
            cap["v_pos"] = positions.clone()
            cap["v_ex"] = dict(ex)
            c = get_context()
            # ★ 直接把 prepare_verify 设好的 context 存下来：step 结束后
            #   run_verify 会 reset_context()，而 seqs 也会被 postprocess 改掉，
            #   所以不能靠「再 prepare 一次」来复现这次前向。
            cap["ctx"] = dict(
                is_prefill=c.is_prefill, cu_seqlens_q=c.cu_seqlens_q,
                cu_seqlens_k=c.cu_seqlens_k, max_seqlen_q=c.max_seqlen_q,
                max_seqlen_k=c.max_seqlen_k, slot_mapping=c.slot_mapping,
                context_lens=c.context_lens, block_tables=c.block_tables,
                is_spec_verify=c.is_spec_verify)
        return orig_v(input_ids, positions)

    def hooked_propose(reqs):
        if "d_reqs" not in cap and len(reqs) == 3:
            cap["d_reqs"] = [dict(r) for r in reqs]
        return orig_p(reqs)

    mr.run_verify_forward = hooked_verify
    mr.spec_proposer.propose_batch = hooked_propose
    for _ in range(8):
        llm.step()
        if "v_ids" in cap and "d_reqs" in cap:
            break
    mr.run_verify_forward = orig_v
    mr.spec_proposer.propose_batch = orig_p

    if "v_ids" not in cap or "d_reqs" not in cap:
        emit("FATAL", dict(msg="没截到 B=3 的 verify/draft 形状", cap=list(cap)))
        return

    res = {}

    def verify_raw():
        """用当前常量跑一次 run_verify_forward（会写静态缓冲区），返回 clone。"""
        set_context(**cap["ctx"])          # 还原 prepare_verify 设好的 context
        out = orig_v(cap["v_ids"], cap["v_pos"])
        return out.clone()

    def eager_verify():
        saved = mr.verify_graphs
        mr.verify_graphs = {}
        set_context(**cap["ctx"])
        out = orig_v(cap["v_ids"], cap["v_pos"]).clone()
        mr.verify_graphs = saved
        return out

    # ---- E1: eager 连跑两次，逐位相同？----
    mr.graph_pad = False
    e1 = eager_verify()
    e2 = eager_verify()
    res["eager_verify_deterministic"] = dict(
        bitwise=bool(torch.equal(e1, e2)),
        max_abs=float((e1 - e2).abs().max()))

    l1 = orig_p(cap["d_reqs"])[1][:, 0].clone()
    l2 = orig_p(cap["d_reqs"])[1][:, 0].clone()
    res["eager_draft_deterministic"] = dict(
        bitwise=bool(torch.equal(l1, l2)),
        max_abs=float((l1 - l2).abs().max()))

    # ---- E2: 同一 RNG 状态下再跑，排除 RNG 的影响 ----
    st = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(st)
    a = orig_p(cap["d_reqs"])[1][:, 0].clone()
    torch.cuda.set_rng_state(st)
    b = orig_p(cap["d_reqs"])[1][:, 0].clone()
    res["eager_draft_same_rng"] = dict(
        bitwise=bool(torch.equal(a, b)),
        max_abs=float((a - b).abs().max()))

    # ==================================================================
    # A/B/C：验证图的三种空行形状
    # ==================================================================
    mr.graph_pad = True
    variants = [
        ("A_q_nq_k_1", None, 1),
        ("B_q_nq_k_0", None, 0),
        ("C_q_0_k_0", 0, 0),
    ]
    outs = {}
    for name, q_len, k_len in variants:
        gp.PAD_Q_LEN = q_len
        gp.PAD_CTX_LEN = k_len
        try:
            mr.verify_graph_hits = 0
            mr.verify_graph_padded = 0
            mr.verify_graph_fallbacks = {}
            out = verify_raw()
            torch.cuda.synchronize()
            outs[name] = out
            r = dict(ok=True, shape=list(out.shape),
                     max_abs_vs_eager=float((out - e1).abs().max()),
                     bitwise_vs_eager=bool(torch.equal(out, e1)),
                     argmax_equal=float(
                         (out.argmax(-1) == e1.argmax(-1)).float().mean()),
                     padded=mr.verify_graph_padded,
                     hits=mr.verify_graph_hits,
                     fallbacks=dict(mr.verify_graph_fallbacks))
        except Exception as exc:                       # noqa: BLE001
            r = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        res["verify_" + name] = r
        emit("VARIANT", dict(name=name, q_len=q_len, k_len=k_len, **r))

    # ★ 关键：三种空行形状彼此之间是否逐位相同？
    #   是 ⇒ 空行的 key/query 长度对真实行**零影响**（block-diagonal attention）。
    base_out = outs.get("A_q_nq_k_1")
    for name, o in outs.items():
        if name == "A_q_nq_k_1":
            continue
        res["cross_" + name + "_vs_A"] = dict(
            bitwise=bool(torch.equal(o, base_out)),
            max_abs=float((o - base_out).abs().max()))
        emit("CROSS", dict(name=name, vs="A_q_nq_k_1",
                           **res["cross_" + name + "_vs_A"]))

    gp.PAD_Q_LEN = None
    gp.PAD_CTX_LEN = 1

    # ==================================================================
    # D：draft 图 context_lens 归零
    # ★ 必须显式打开 proposer 的开关（mr.graph_pad 是验证侧的那个）
    # ==================================================================
    mr.spec_proposer.graph_pad = True
    mr.spec_proposer.n_padded = 0
    base = orig_p(cap["d_reqs"])[1][:, 0].clone()
    base_padded = mr.spec_proposer.n_padded
    for name, ctx_len in [("D_ctx_1", 1), ("D_ctx_0", 0)]:
        gp.PAD_CTX_LEN = ctx_len
        try:
            mr.spec_proposer.n_padded = 0
            o = orig_p(cap["d_reqs"])[1][:, 0].clone()
            torch.cuda.synchronize()
            r = dict(ok=True, max_abs_vs_ctx1=float((o - base).abs().max()),
                     bitwise_vs_ctx1=bool(torch.equal(o, base)),
                     argmax_equal=float((o.argmax(-1) == base.argmax(-1)).float().mean()),
                     padded=mr.spec_proposer.n_padded)
        except Exception as exc:                       # noqa: BLE001
            r = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        res["draft_" + name] = r
        emit("VARIANT", dict(name=name, base_padded=base_padded, **r))
    gp.PAD_CTX_LEN = 1
    mr.spec_proposer.graph_pad = False

    emit("VARIANT_RESULT", res)

    try:
        import atexit
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
