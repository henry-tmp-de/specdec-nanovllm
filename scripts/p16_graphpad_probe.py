#!/usr/bin/env python
"""p16_graphpad_probe.py —— P16 的验收探针：把「未精确捕获的 batch size」补进图。

回答三个问题
------------
① **退回次数**：`verify_graph_fallbacks` 里 `shape_not_captured(B=3,k=6)` 是不是归零？
   （台账里既有的边界：3 条请求的混合 step 实测退回 eager 22 次。）
② **默认关 = 老行为**：pad 关着时，逐前向的输出与改动前的图 / eager 路径**逐位相同**。
③ **开着 = 真实行不受空行影响**：
   · B 恰好被捕获时（B=1/2/4），pad 开/关走的是**同一段精确匹配代码** ⇒ 必须逐位相同；
   · B=3 向上取整到 B=4 时，真实行的 logits 与 eager 的差异必须落在
     **图 vs eager 的已知量级**内（HANDOVER §7.3：max|Δlogit| ≈ 0.30，
     是 flash-attn 分块累加顺序的差，不是逻辑错误）。

★ 怎么拿到「同一个形状的 pad 开 / 关两份输出」：**钩在真实调用点上**，不手工拼
  verify 批 —— 手工拼要复刻 scheduler 的 may_append / 草稿填充，容易造出一个
  引擎里根本不存在的形状。钩子在真实 step 内部触发，`_verify_extra` 由
  `prepare_verify` 正常填好，比对的确实是引擎自己会走的那个形状。

★ 单进程串行跑（端口 2333 被 init_process_group 占用，并发会 EADDRINUSE）。

用法（服务器）:
  CUDA_VISIBLE_DEVICES=0 /home/ziru/nano-vllm/venv/bin/python -u scripts/p16_graphpad_probe.py
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
K = int(os.environ.get("K", "6"))
GBS = os.environ.get("GBS", "1,2,4")


def emit(tag, obj):
    print(f"@@{tag}@@" + json.dumps(obj, ensure_ascii=False), flush=True)


def build(ids_unused=None):
    from nanovllm import LLM
    maxlen = CTX + 256
    return LLM(TARGET, enforce_eager=False, max_model_len=maxlen,
               max_num_batched_tokens=maxlen, max_num_seqs=4,
               spec_k=K, spec_method="draft", draft_model=DRAFT,
               spec_batch_threshold=0, spec_prefill_chunk_tokens=CHUNK,
               spec_overlap_mixed=True, spec_overlap_mode="concurrent",
               spec_graph_bs=tuple(int(x) for x in GBS.split(",")),
               spec_graph_pad=False)


def run_scenario(llm, ids, pad, tag, n_dec=None):
    """跑一遍混合批工作负载，返回图命中 / 退回统计 + 生成的 token。"""
    from nanovllm import SamplingParams
    mr = llm.model_runner
    n_dec = N_DEC if n_dec is None else n_dec
    # ★ 运行期翻开关：同一份二进制、同一批图，只换这一个布尔
    mr.graph_pad = bool(pad)
    mr.spec_proposer.graph_pad = bool(pad)
    mr.verify_graph_hits = 0
    mr.verify_graph_fallbacks = {}
    mr.verify_graph_padded = 0
    mr.spec_proposer.n_graphed = 0
    mr.spec_proposer.n_eager = 0
    mr.spec_proposer.n_padded = 0

    sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)
    for _ in range(n_dec):
        llm.add_request(ids[0:64], sp)
    for _ in range(4):
        llm.step()
    llm.add_request(ids[2000:2000 + CTX],
                    SamplingParams(temperature=1.0, max_tokens=4, ignore_eos=True))
    mixed = pure = 0
    delivered = []
    while not llm.is_finished():
        plan = llm.scheduler.schedule()
        if type(plan).__name__ == "MixedBatchPlan":
            mixed += 1
        else:
            pure += 1
        outs, _ = llm.step()
        delivered.extend((int(sid), [int(t) for t in toks]) for sid, toks in outs)
    out = dict(
        pad=bool(pad), n_dec=n_dec, mixed_steps=mixed, pure_steps=pure,
        verify_graph_hits=mr.verify_graph_hits,
        verify_graph_fallbacks=mr.verify_graph_fallbacks,
        verify_graph_padded=mr.verify_graph_padded,
        draft_graphed=mr.spec_proposer.n_graphed,
        draft_eager=mr.spec_proposer.n_eager,
        draft_padded=mr.spec_proposer.n_padded,
        delivered=delivered,
    )
    emit(f"SCEN_{tag}", out)
    return out


# ======================================================================
#  逐前向对拍：钩在真实调用点上，按 B 分类统计
# ======================================================================
STATS = {}
MAX_REC = int(os.environ.get("MAX_REC", "3"))   # 每个 (kind,B) 最多对拍几次（控制耗时）
SELF = []                                       # draft 自洽性（噪声地板）样本


def _pair(a, b):
    """(max|Δ|, argmax 一致率, 是否逐位相同)。"""
    if a.shape != b.shape:
        return None
    return (float((a - b).abs().max()),
            float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            bool(torch.equal(a, b)))


def _rec(kind, B, off, on, eager):
    """记录三份输出的两两对比。

    · off   = graph_pad=False（B 未捕获 ⇒ eager；B 已捕获 ⇒ 精确匹配的图）
    · on    = graph_pad=True （B 未捕获 ⇒ 向上取整 + 空行；B 已捕获 ⇒ 同一段精确匹配）
    · eager = 强制清空图集合后的纯 eager 基线

    ★ draft 的 on/off/eager 三次调用之间**随机数状态不同**，所以第 2..k 个候选位置
      的输入 token 不同（链式采样）——那些位置本来就不可比。因此 draft 只比
      **第 0 个候选位置**；verify 是一次前向，整块可比。
    """
    s = STATS.setdefault(f"{kind}_B{B}", dict(
        kind=kind, B=B, n=0, rows=0, mode="step0" if kind == "draft" else "full",
        off_vs_on=dict(bitwise=0, max_abs=0.0, argmax=[]),
        on_vs_eager=dict(max_abs=0.0, argmax=[]),
        off_vs_eager=dict(max_abs=0.0, argmax=[])))
    s["n"] += 1
    if kind == "draft":
        off, on, eager = off[:, 0], on[:, 0], eager[:, 0]
    s["rows"] = int(off.shape[0])
    for key, (a, b) in dict(off_vs_on=(off, on), on_vs_eager=(on, eager),
                            off_vs_eager=(off, eager)).items():
        r = _pair(a, b)
        if r is None:
            continue
        mx, ag, bw = r
        if key == "off_vs_on":
            s[key]["bitwise"] += int(bw)
            s[key]["max_abs"] = max(s[key]["max_abs"], mx)
        else:
            s[key]["max_abs"] = max(s[key]["max_abs"], mx)
        s[key].setdefault("argmax", []).append(ag)


def _finalize():
    for s in STATS.values():
        for key in ("off_vs_on", "on_vs_eager", "off_vs_eager"):
            ag = s[key].pop("argmax")
            s[key]["argmax_equal"] = round(sum(ag) / len(ag), 4) if ag else None
            s[key]["max_abs"] = round(s[key]["max_abs"], 5)
    return sorted(STATS.values(), key=lambda s: (s["kind"], s["B"]))


def _should_probe(kind, B):
    s = STATS.get(f"{kind}_B{B}")
    return s is None or s["n"] < MAX_REC


CAP = {}


def install_hooks(llm):
    mr = llm.model_runner
    orig_v = mr.run_verify_forward
    orig_p = mr.spec_proposer.propose_batch

    def hooked_verify(input_ids, positions):
        ex = mr._verify_extra
        if ex is not None and mr.verify_graphs and ex["full_k"] \
                and _should_probe("verify", ex["n"]):
            B = ex["n"]
            if B not in CAP:
                from nanovllm.utils.context import get_context
                c = get_context()
                CAP[B] = dict(ids=input_ids.clone(), pos=positions.clone(),
                              ctx=dict(is_prefill=c.is_prefill,
                                       cu_seqlens_q=c.cu_seqlens_q,
                                       cu_seqlens_k=c.cu_seqlens_k,
                                       max_seqlen_q=c.max_seqlen_q,
                                       max_seqlen_k=c.max_seqlen_k,
                                       slot_mapping=c.slot_mapping,
                                       context_lens=c.context_lens,
                                       block_tables=c.block_tables,
                                       is_spec_verify=c.is_spec_verify))
            saved = mr.verify_graphs
            mr.graph_pad = False
            off = orig_v(input_ids, positions).clone()      # 未捕获 ⇒ eager；捕获 ⇒ 图
            mr.verify_graphs = {}                           # 强制纯 eager 基线
            eager = orig_v(input_ids, positions).clone()
            mr.verify_graphs = saved
            mr.graph_pad = True
            on = orig_v(input_ids, positions).clone()       # 向上取整 + 空行
            _rec("verify", B, off, on, eager)
            mr.graph_pad = True
        return orig_v(input_ids, positions)

    def hooked_propose(reqs):
        B = len(reqs)
        if B >= 1 and _should_probe("draft", B):
            # ★ 三次调用之间必须**复位随机数状态**：propose_batch 里每步都要
            #   采样候选（`exponential_`），候选又会被写进 draft KV 供下一步读
            #   ——不复位的话第 2..k 步的输入 token 天然不同，比出来的差异全是
            #   噪声。复位后第 0 步（我们唯一比较的位置）口径才干净。
            rng = torch.cuda.get_rng_state()
            saved = mr.spec_proposer._graphs
            mr.spec_proposer.graph_pad = False
            # ★ 自洽性：同一条路径连调两次（都复位 RNG），差多少就是本次测量的
            #   噪声地板。它不为 0 的话，下面所有 max|Δ| 都要减去这一项。
            torch.cuda.set_rng_state(rng)
            _, l_a = orig_p(reqs)
            torch.cuda.set_rng_state(rng)
            _, l_b = orig_p(reqs)
            SELF.append(float((l_a[:, 0] - l_b[:, 0]).abs().max()))
            torch.cuda.set_rng_state(rng)
            _, l_off = orig_p(reqs)
            mr.spec_proposer._graphs = {}                   # 强制纯 eager 基线
            torch.cuda.set_rng_state(rng)
            _, l_eager = orig_p(reqs)
            mr.spec_proposer._graphs = saved
            mr.spec_proposer.graph_pad = True
            torch.cuda.set_rng_state(rng)
            _, l_on = orig_p(reqs)
            _rec("draft", B, l_off, l_on, l_eager)
            mr.spec_proposer.graph_pad = True
        return orig_p(reqs)

    mr.run_verify_forward = hooked_verify
    mr.spec_proposer.propose_batch = hooked_propose
    return orig_v, orig_p


def main():
    global torch
    import torch as _t
    torch = _t

    with open(WIKI, encoding="utf-8") as f:
        text = f.read()

    llm = build()
    mr = llm.model_runner
    ids = llm.tokenizer(text).input_ids
    emit("HEAD", dict(
        captured_verify_shapes=sorted(mr.verify_graphs.keys()),
        captured_draft_bs=sorted(mr.draft_graphs.keys()),
        spec_graph_bs=list(mr.config.spec_graph_bs),
        graph_pad_default=mr.graph_pad, k=K,
        dev=torch.cuda.get_device_name(0)))

    # ---------- ① 退回次数（无钩子，干净口径）----------
    a = run_scenario(llm, ids, False, "OFF")
    b = run_scenario(llm, ids, True, "ON")
    key = f"shape_not_captured(B=3,k={K})"
    emit("CMP_SCEN", dict(
        key=key,
        off_shape_not_captured=a["verify_graph_fallbacks"].get(key, 0),
        on_shape_not_captured=b["verify_graph_fallbacks"].get(key, 0),
        off_all_fallbacks=a["verify_graph_fallbacks"],
        on_all_fallbacks=b["verify_graph_fallbacks"],
        off_padded=a["verify_graph_padded"], on_padded=b["verify_graph_padded"],
        off_draft_eager=a["draft_eager"], on_draft_eager=b["draft_eager"],
        off_draft_graphed=a["draft_graphed"], on_draft_graphed=b["draft_graphed"],
        on_draft_padded=b["draft_padded"],
        tokens_equal=a["delivered"] == b["delivered"]))

    # ---------- ② 逐前向对拍（带钩子；4 条请求 ⇒ 覆盖 B=4 的精确桶）----------
    true_v, true_p = install_hooks(llm)
    run_scenario(llm, ids, True, "HOOKED", n_dec=4)
    emit("CMP_FWD", dict(stats=_finalize(),
                         draft_selfcheck=[round(x, 5) for x in SELF]))

    # ---------- ③ 单次验证前向的耗时：eager vs 向上取整的图 ----------
    # ★ 这才是这项优化的收益：B=3 以前整个退回 eager，现在能进 B=4 的图。
    #   ★ 必须先把钩子摘掉（钩子一次调用要跑 3 次前向），并用 hits 计数确认
    #     这次测量**真的** replay 了图，而不是悄悄退回 eager。
    if CAP:
        from nanovllm.utils.context import set_context
        from nanovllm.spec_decode.graph_pad import pick_bucket
        mr.run_verify_forward = true_v            # 摘钩子
        buckets = sorted({b for (b, kk) in mr.verify_graphs if kk == K})

        def timed(B, graph_pad, n=20):
            saved = mr.verify_graphs
            mr.verify_graphs = saved if graph_pad else {}
            mr.graph_pad = graph_pad
            mr.verify_graph_hits = 0
            mr.verify_graph_padded = 0
            mr.verify_graph_fallbacks = {}
            cap = CAP[B]
            # ★ run_verify_forward 的形状判定读的是 self._verify_extra ——
            #   必须按这个 B 重设一遍，否则它用的还是 scenario 最后一步留下的那个 B
            #   （那会把每次测量都判成 static_shape_mismatch，测出来全是 eager）。
            mr._verify_extra = dict(
                n=B, k=K, full_k=True, total_q=B * (K + 1),
                cu_seqlens_q=cap["ctx"]["cu_seqlens_q"],
                cu_seqlens_k=cap["ctx"]["cu_seqlens_k"],
                slot_mapping=cap["ctx"]["slot_mapping"],
                block_tables=cap["ctx"]["block_tables"])
            for _ in range(3):                            # 预热
                set_context(**cap["ctx"])
                true_v(cap["ids"], cap["pos"])
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(n):
                set_context(**cap["ctx"])
                true_v(cap["ids"], cap["pos"])
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1) / n
            info = dict(ms=round(ms, 3), hits=mr.verify_graph_hits,
                        padded=mr.verify_graph_padded,
                        fallbacks=dict(mr.verify_graph_fallbacks))
            mr.verify_graphs = saved
            return info

        timings = {}
        for B in sorted(CAP):
            timings[str(B)] = dict(
                bucket=pick_bucket(B, buckets),
                exact_hit=B in buckets,
                eager=timed(B, False),
                pad_graph=timed(B, True))
        emit("TIMING", dict(buckets=buckets, n=20, ctx_len=CTX, timings=timings))

    try:
        import atexit
        atexit.unregister(llm.exit)
    except Exception:
        pass
    try:
        llm.exit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
