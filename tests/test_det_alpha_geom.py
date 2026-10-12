"""纯 CPU 回归：确定性接受率仪器（scripts/det_alpha.py）用的【可见集几何】是否与引擎逐值一致。

为什么需要这个测试
------------------
仪器要回答「把 draft 的上下文截到窗口，接受率掉多少」，它必须**逐值**复刻引擎里
draft 看到的那一段上下文，否则测出来的就不是引擎的行为。引擎那边的权威定义有两处：

  · `ModelRunner._run_draft_prefill_window` —— 把 `[0, head_end)` 与 `[b0*bs, end)`
    两段按绝对位置递增拼成一个 chunk（RoPE 用绝对位置、chunk 内本地因果）；
  · `DraftWindow` —— `b0(p) = max(P, valid_from//bs, p//bs − M + 1)`、
    `ctx_len = P*bs + (p + 1 − b0*bs)`、`block_table = ring[:P] + [...]`。

`det_alpha.visible_segments(P, M, b)` 必须与这两处**逐值**对上：

  §1 两段长度之和 == DraftWindow.ctx_len(b*bs)（覆盖「空档 / 相接 / 短序列」三种边界）
  §2 每段内部连续、两段递增且不重叠；并集 == [0, min(P*bs,end)) ∪ [b0*bs, end)
  §3 可见的每一个 token 都落在 DraftWindow.block_table 列出的物理块里（不乱读）
  §4 M=0 ⇒ 全上下文（普通投机 w0 的口径）
  §5 parse_cfg 的解析

跑：python tests/test_det_alpha_geom.py
"""
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)

_spec = importlib.util.spec_from_file_location(
    "dp_geom", os.path.join(ROOT, "nanovllm/spec_decode/draft_proposer.py"))
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
DraftWindow = _dp.DraftWindow

_spec2 = importlib.util.spec_from_file_location(
    "det_alpha", os.path.join(ROOT, "scripts/det_alpha.py"))
_da = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_da)
visible_segments = _da.visible_segments
parse_cfg = _da.parse_cfg

BS = 256
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# 覆盖：M=0（全上下文）、各种 P/M、以及 b 落在「窗口够到开头」「空档」「正常滑窗」三区
CASES = []
for P in (0, 1, 2, 4, 8):
    for M in (1, 2, 3, 5, 8, 12, 18, 33):
        ring = list(range(P + M))
        w = DraftWindow(ring, BS, M, P)
        for b in range(0, 40):
            CASES.append((P, M, b, w))

# ----------------------------------------------------------------------
# §1 段长之和 == ctx_len（含 valid_from=0 的常态）
bad = []
for (P, M, b, w) in CASES:
    segs = visible_segments(P, M, b)
    got = sum(c - a for (a, c) in segs)
    want = w.ctx_len(b * BS + BS - 1) if M > 0 else (b + 1) * BS
    # M=0 时引擎走的是全上下文（没有 DraftWindow 分支），期望值就是 end
    if got != want:
        bad.append((P, M, b, got, want))
check(f"§1 段长之和 == 引擎 ctx_len（{len(CASES)} 组）", not bad,
      f"bad={bad[:3]}" if bad else "")

# ----------------------------------------------------------------------
# §2 每段内部连续、两段递增不重叠；并集 == 预期两段
bad = []
for (P, M, b, w) in CASES:
    segs = visible_segments(P, M, b)
    end = (b + 1) * BS
    prev_c = -1
    for (a, c) in segs:
        if not (0 <= a < c <= end) or a < prev_c:
            bad.append((P, M, b, "seg", segs))
            break
        prev_c = c
    else:
        if M <= 0:
            want = [(0, end)]
        else:
            head_end = min(P * BS, end)
            b0 = max(P, b - M + 1)
            w0 = b0 * BS
            want = [(0, end)] if w0 <= head_end else [(0, head_end), (w0, end)]
        got = [(a, c) for (a, c) in segs if c > a]
        got = [s for s in got if s[1] > s[0]]
        want = [s for s in want if s[1] > s[0]]
        if got != want:
            bad.append((P, M, b, got, want))
check(f"§2 段结构 == 引擎的 segs（{len(CASES)} 组）", not bad,
      f"bad={bad[:3]}" if bad else "")

# ----------------------------------------------------------------------
# §3 可见 token 都落在 block_table 列出的物理块里
bad = []
for (P, M, b, w) in CASES:
    if M <= 0:
        continue
    bt = set(w.block_table(b * BS + BS - 1))
    for (a, c) in visible_segments(P, M, b):
        for p in range(a, c):
            if w.slot(p) // BS not in bt:
                bad.append((P, M, b, p))
                break
check(f"§3 可见集的每个 token 都在 block_table 里（{len(CASES)} 组）", not bad,
      f"bad={bad[:3]}" if bad else "")

# ----------------------------------------------------------------------
# §4 M=0 ⇒ 全上下文
bad = [(P, M, b) for P in (0, 1, 3) for M in (0, -1) for b in (0, 1, 7, 31)
       if visible_segments(P, M, b) != [(0, (b + 1) * BS)]]
check("§4 M<=0 ⇒ 全上下文 [0, end)", not bad, f"bad={bad[:3]}" if bad else "")

# ----------------------------------------------------------------------
# §5 覆盖率：P+M 块盖住 [0, end) 时，可见集必须恰好是完整前缀
bad = []
for P in (0, 1, 2, 4):
    for b in (0, 1, 3, 10, 33):
        M = b + 1                      # P 块常驻 + M 块窗口 ≥ end（当 P=0 时）
        segs = [(a, c) for (a, c) in visible_segments(P, M, b) if c > a]
        want = [(0, min(P * BS + M * BS, (b + 1) * BS))]
        # 期望：起点 0、连续到 end（没有空档）
        ok = (len(segs) == 1 and segs[0][0] == 0 and segs[0][1] == (b + 1) * BS)
        if not ok:
            bad.append((P, M, b, segs))
check("§5 窗口盖住前缀时不出现空档（连续 [0, end)）", not bad,
      f"bad={bad[:3]}" if bad else "")

# ----------------------------------------------------------------------
# §6 parse_cfg
bad = []
for s, w in [("w0", (0, 0)), ("full", (0, 0)), ("m5", (0, 5)),
             ("p2m5", (2, 5)), ("P2M5", (2, 5)), ("p0m33", (0, 33))]:
    if parse_cfg(s) != w:
        bad.append((s, parse_cfg(s), w))
check("§6 parse_cfg", not bad, f"bad={bad}" if bad else "")

# ----------------------------------------------------------------------
print()
if FAILED:
    print(f"FAILED {len(FAILED)}:")
    for n in FAILED:
        print("  -", n)
    sys.exit(1)
print("ALL PASS")
