#!/usr/bin/env python
"""det_alpha.py —— 确定性接受率仪器：α = 1 − TV(p, q)，在【固定上下文】上算。

为什么必须换仪器（不要再用自由生成）
------------------------------------
旧的接受率口径是「自由生成 256 token，统计 accepted / proposed」。它在
`temperature=1.0` 下 **会发散**：普通投机自己的接受率跨批次漂移 **±4~13%**
（0.7814 vs 0.6911，两轮串行都复现 0.6911），同一棵树重复跑 3 遍 `out_md5`
都不一样（见 `docs/长上下文-滑窗容量.md` §7.2、`docs/P7-D开头常驻段-开发记录.md` §3.2）。
**这个噪声压过了要测的效应。**

但是接受率有恒等式（拒绝采样的数学性质）：

        α = E_{x~q}[ min(1, p[x]/q[x]) ] = Σ_x min(p[x], q[x]) = 1 − TV(p, q)

p = target 的分布，q = draft 的分布。**两者都只依赖上下文**，在固定上下文上是
确定的 ⇒ 不发散、可复现、可以堆样本量把标准误压到 ~0.3%（沿用台账 §2.8 /
Q1-Q2 的做法：在 wikitext 上取上万个位置，逐位置比 draft 与 target 的下一 token 分布）。

★ 口径声明：这里测的是**逐位置的一步接受概率**（teacher forcing：上下文用
  wikitext 的真实续写，不采样）。引擎里报的「接受率」是整轮 accepted/proposed
  的均值，两者不是同一个数（整轮对候选 2..k 的条件分布不同）——
  **但三配置之间的比较是同口径的**，这正是本任务要的。

draft 的上下文 = 引擎里那个（逐字对应，不是自己发明的口径）
----------------------------------------------------------
引擎出处：
  · `nanovllm/engine/model_runner.py::_run_draft_prefill_window` —— 把
    `[0, head_end)` 与 `[b0*bs, end)` 两段按绝对位置递增拼成一个 chunk，
    走**本地 k/v + 因果**（不传 block_table），RoPE 用**绝对位置**；
  · `nanovllm/spec_decode/draft_proposer.py::DraftWindow` ——
    `b0(p) = max(P, valid_from//bs, p//bs − M + 1)`、`ctx_len = P*bs + (p+1 − b0*bs)`。

所以本脚本对「块 b 里的 query」构造的可见集是：

        visible(b) = [0, min(P*bs, end)) ∪ [b0*bs, end),  b0 = max(P, b − M + 1)

（`end = (b+1)*bs`；两段相接/重叠时退化成 `[0, end)` 一整段 —— 与引擎的
 `head_end = min(P*bs, end)` + 空窗口段一致）。
`M = 0, P = 0` ⇒ 全上下文 ⇒ **普通投机（w0）**。

★ 因果掩码用**layout 顺序**（拼接后的下标）——因为两段按绝对位置递增拼接，
  layout 下标序 == 绝对位置序，所以 HF 的默认因果掩码给出的可见集
  **恰好**是上面的 `visible(b)`，一行自定义 mask 都不用写。

用法（一个配置一个进程；`L` 与 `CFG` 是必填）
---------------------------------------------
    CUDA_VISIBLE_DEVICES=0 L=8192 CFG=p2m5 \\
      /home/ziru/nano-vllm/venv/bin/python -u scripts/det_alpha.py

env：
  L      上下文长度（prompt 的 token 数），例 8192 / 16384
  CFG    w0 | m5 | p2m5 | p2m12 | ...（`p<P>m<M>`；`m<M>` 等价 `p0m<M>`）
  NWIN   窗口数（默认 16，按语料长度自动夹）
  BACK   「生成位」之前多测多少位置（默认 1792 → 加 256 个生成位 = 2048/窗口）
  FWD    生成位个数（默认 256，与 out=256 对齐）
  OUT    结果 json 路径（可选）
  SELFTEST=1  额外做「窗口盖住整条序列 ⇒ q 必须与全上下文逐位相同」的自检
  T      softmax 温度（默认 1.0，与自由生成口径一致）

★ 本脚本【碰不到引擎】：它是离线仪器，只读模型权重算分布。
  target 一个字节没动，引擎代码一个字节没改。
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BS = 256                      # kvcache_block_size（引擎与本项目一贯口径）
TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")


# ======================================================================
#  纯几何（无 torch，可 CPU 单测）—— 必须与引擎逐值一致
# ======================================================================
def visible_segments(P, M, b, bs=BS):
    """draft 在【块 b】里的 query 能看到哪些 token（绝对位置，递增、不重叠）。

    对应 `ModelRunner._run_draft_prefill_window` 的 segs，以及 `DraftWindow.b0`。
    `M <= 0` ⇒ 全上下文（普通投机）。
    """
    end = (b + 1) * bs
    if M <= 0:
        return [(0, end)]
    head_end = min(P * bs, end)
    b0 = max(P, b - M + 1)
    w0 = b0 * bs
    if w0 <= head_end:            # 两段首尾相接 / 重叠 ⇒ 合并成一整段
        return [(0, end)]
    # ★ 逐字对照 `ModelRunner._run_draft_prefill_window`：两段各自只有非空才加。
    #   短序列（b < P）时 head_end 已经 = end，而 w0 > end —— 那时窗口段是【空的】，
    #   引擎的 `if w0 < end` 会把它丢掉；这里也必须丢，否则会返回一个 (a>c) 的假段。
    segs = []
    if head_end > 0:
        segs.append((0, head_end))
    if w0 < end:
        segs.append((w0, end))
    return segs if segs else [(0, end)]


def parse_cfg(cfg):
    cfg = cfg.strip().lower()
    if cfg in ("w0", "full", "base"):
        return 0, 0
    m = re.fullmatch(r"p(\d+)m(\d+)", cfg)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"m(\d+)", cfg)
    if m:
        return 0, int(m.group(1))
    raise ValueError(f"看不懂的 CFG={cfg!r}（用 w0 / m5 / p2m5 这种）")


# ======================================================================
#  GPU 部分
# ======================================================================
def hidden_at(base, ids, want, dev):
    """一次前向跑完 ids，返回 want 位置各自的 hidden（want 必须升序）。

    ★ 用【一次前向】而不是分块 + KV cache：分块会让归约顺序变，
      而自检要求「窗口盖住整条序列」时两条路逐位相同 —— 分块会破坏它。
      16K 的 4B 前向在本机 3090 上单次跑得下（SDPA 走 flash，显存 O(N)）。
    """
    import torch

    inp = torch.tensor([ids], device=dev)
    pos = torch.tensor([list(range(len(ids)))], device=dev)
    with torch.inference_mode():
        h = base(inp, position_ids=pos, use_cache=False).last_hidden_state[0]
    rows = h[torch.tensor(want, device=dev)]
    del h
    torch.cuda.empty_cache()
    return rows


def window_q(drf, ids, P, M, b, dev):
    """draft 在「块 b」的窗口上下文下的下一 token logits。

    返回 (BS, V) —— 第 i 行 = 绝对位置 b*BS+i 之后的 token 分布。
    """
    import torch

    segs = visible_segments(P, M, b)
    lay_ids, lay_pos = [], []
    for (a, c) in segs:
        lay_ids.extend(ids[a:c])
        lay_pos.extend(range(a, c))
    inp = torch.tensor([lay_ids], device=dev)
    pos = torch.tensor([lay_pos], device=dev)
    with torch.inference_mode():
        h = drf.model(inp, position_ids=pos, use_cache=False).last_hidden_state[0]
    last_a = segs[-1][0]
    off = sum(c - a for (a, c) in segs[:-1]) + (b * BS - last_a)
    rows = h[off:off + BS]
    logits = drf.lm_head(rows)
    del h, inp, pos
    torch.cuda.empty_cache()
    return logits


def run_cfg(tgt, drf, ids, P, M, qs, dev, T=1.0):
    """返回逐位置 TV（顺序同 qs）。"""
    import torch

    h_t = hidden_at(tgt.model, ids, qs, dev)
    h_d = hidden_at(drf.model, ids, qs, dev) if M <= 0 else None

    byb = {}
    for i, t in enumerate(qs):
        byb.setdefault(t // BS, []).append(i)

    tv = [0.0] * len(qs)
    for b, ii in sorted(byb.items()):
        p = torch.softmax(tgt.lm_head(h_t[ii]).float() / T, dim=-1)
        if M <= 0:
            q = torch.softmax(drf.lm_head(h_d[ii]).float() / T, dim=-1)
        else:
            qb = window_q(drf, ids, P, M, b, dev)
            q = torch.softmax(
                qb[torch.tensor([t - b * BS for t in (qs[i] for i in ii)],
                                device=dev)].float() / T, dim=-1)
            del qb
        d = 0.5 * (p - q).abs().sum(-1)
        for k, i in enumerate(ii):
            tv[i] = float(d[k])
        del p, q, d
    del h_t, h_d
    torch.cuda.empty_cache()
    return tv


def _stats(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(1, n - 1))
    return dict(n=n, tv_mean=round(mu, 5), tv_se=round(sd / math.sqrt(n), 5),
                alpha=round(1 - mu, 5))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    L = int(os.environ["L"])
    cfg = os.environ.get("CFG", "w0")
    P, M = parse_cfg(cfg)
    NWIN = int(os.environ.get("NWIN", "16"))
    BACK = int(os.environ.get("BACK", "1792"))
    FWD = int(os.environ.get("FWD", "256"))
    T = float(os.environ.get("T", "1.0"))
    OUT = os.environ.get("OUT", "")
    SELFTEST = os.environ.get("SELFTEST", "0") == "1"

    dev = "cuda"
    torch.manual_seed(20261012)
    torch.set_grad_enabled(False)      # 纯测量：不要 autograd 残留

    tok = AutoTokenizer.from_pretrained(TARGET)
    tgt = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16,
                                               device_map=dev).eval()
    drf = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16,
                                               device_map=dev).eval()

    with open(WIKI, encoding="utf-8") as f:
        ids_all = tok(f.read()).input_ids
    W = L + FWD
    NWIN = max(1, min(NWIN, (len(ids_all) - 137) // W))
    wins = [ids_all[137 + i * W: 137 + i * W + W] for i in range(NWIN)]

    qs = list(range(L - 1 - BACK, L - 1 + FWD))       # 窗口内相对下标
    gen_lo = L - 1                                    # 第 1 个生成 token 的 query 位
    assert max(qs) <= W - 1, (max(qs), W)
    gen_j = [j for j, t in enumerate(qs) if t >= gen_lo]

    print(f"[det] L={L} CFG={cfg} -> P={P} M={M} window_tok="
          f"{(P + M) * BS if M else 'full'} | NWIN={NWIN} | "
          f"query/窗口={len(qs)}（其中生成位 {len(gen_j)}）| T={T}", flush=True)
    print(f"[det] 首/末 query 块 {qs[0]//BS}/{qs[-1]//BS} 的可见集："
          f"{visible_segments(P, M, qs[0]//BS)} | "
          f"{visible_segments(P, M, qs[-1]//BS)}", flush=True)

    per_tv, per_win_all, per_win_gen = [], [], []
    for wi, ids in enumerate(wins):
        tv = run_cfg(tgt, drf, ids, P, M, qs, dev, T)
        per_tv.extend(tv)
        per_win_all.append(sum(tv) / len(tv))
        per_win_gen.append(sum(tv[j] for j in gen_j) / len(gen_j))
        print(f"[det]  win{wi:2d}: α_all={1-per_win_all[-1]:.4f} "
              f"α_gen={1-per_win_gen[-1]:.4f}", flush=True)

    gen_tv = [per_tv[wi * len(qs) + j] for wi in range(NWIN) for j in gen_j]
    res = dict(L=L, cfg=cfg, P=P, M=M, T=T, NWIN=NWIN,
               window_tokens=((P + M) * BS if M else None),
               query_per_win=len(qs), back=BACK, fwd=FWD,
               alpha_gen=_stats(gen_tv),
               alpha_all=_stats(per_tv),
               alpha_gen_per_window=[round(1 - v, 5) for v in per_win_gen],
               alpha_gen_window_se=round(_stats(per_win_gen)["tv_se"], 5),
               note="alpha = 1 − mean TV(target, draft)；逐位置一步接受概率；"
                    "alpha_gen 只数 t>=L-1 的生成位（与自由生成 out=256 对齐）")
    print("@@DA@@" + json.dumps(res, ensure_ascii=False), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------------- 自检：窗口盖住整条序列 ⇒ 必须与全上下文逐位相同 ----------
    # 探针刻意取【最后一个 query 块】：M 够大时 b0 = 0，layout 恰好 = [0, W)
    # —— 与 w0 那条全上下文前向【长度都一样】，所以必须逐位相同（不是"接近"）。
    # 更早的块 layout 更短（kernel 分块不同 → bf16 归约顺序不同），
    # 那时只能比分布：TV 应远小于真效应（截断造成的 TV ~0.3）。
    if SELFTEST:
        cov = (W + BS - 1) // BS + 1
        ids = wins[0]
        h_full = hidden_at(drf.model, ids, qs, dev)
        last_b = qs[-1] // BS
        exact = [t for t in qs if t // BS == last_b]
        qf = drf.lm_head(h_full[[qs.index(t) for t in exact]])[..., :8]
        qw = window_q(drf, ids, P, cov, last_b, dev)
        qw = qw[torch.tensor([t - last_b * BS for t in exact], device=dev)][..., :8]
        dmax = float((qf.float() - qw.float()).abs().max())
        print(f"[det] SELFTEST(A) 块 {last_b}（layout=[0,{W})，与全上下文同长）"
              f" M={cov}: max|Δlogits[:8]| = {dmax:.6g} "
              f"({'PASS(逐位相同)' if dmax == 0.0 else 'FAIL'})", flush=True)

        mid_b = qs[len(qs) // 2] // BS
        t0 = qs[len(qs) // 2]
        qf2 = drf.lm_head(h_full[[qs.index(t0)]])
        qw2 = window_q(drf, ids, P, cov, mid_b, dev)[t0 - mid_b * BS:t0 - mid_b * BS + 1]
        p2 = torch.softmax(qf2.float(), dim=-1)
        q2 = torch.softmax(qw2.float(), dim=-1)
        tv2 = float(0.5 * (p2 - q2).abs().sum())
        print(f"[det] SELFTEST(B) 块 {mid_b}（layout 更短 → kernel 分块不同）"
              f" 的 TV = {tv2:.5f}", flush=True)
        res["selftest_exact_max_abs"] = dmax
        res["selftest_mid_tv"] = tv2
        res["selftest_pass"] = bool(dmax == 0.0 and tv2 < 0.02)
        if OUT:
            with open(OUT, "w") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
