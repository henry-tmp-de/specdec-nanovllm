"""纯 CPU 回归：P7-D「开头常驻段 + 最近滑窗」的算术。

约定（见 nanovllm/spec_decode/draft_proposer.DraftWindow）
--------------------------------------------------------
draft 的上下文 = 开头 P 个块（常驻，永不轮转） + 最近 M 个块（环）。
ring 的前 P 个是常驻段、后 M 个是环：

    slot(p) = ring[b]                        * bs + p%bs    当 b = p//bs <  P
            = ring[P + (b-P)%M]              * bs + p%bs    当 b        >= P
    b0(p)   = max(P, valid_from//bs, p//bs - M + 1)         窗口最老的块（>= P）
    clen(p) = P*bs + (p + 1 - b0(p)*bs)
    bt(p)   = ring[0:P] + [ ring[P + (b0+t-P)%M] for t in range(M) ]

硬要求（本文件就是这些）：
  §1 P=0 时逐值等价于改动前那份内联公式（默认行为没变的第一道证据）。
  §2 常驻段与环段【永不重叠】、物理块互不相交。
  §3 块表里 key index j ↔ 绝对位置 的映射正确，且恒按绝对块号递增。
  §4 clen 永不越界（<= (P+M)*bs），且读到的 key 集合恰好 = 开头 ∪ 尾部窗口。
  §5 时间线模拟：任一查询位置读到的每一格都是【它该是的位置】，绝不读到未来、
     也绝不读到空档里那些（根本不写的）陈旧位置。

跑：python tests/test_draft_window_head.py
"""
import os
import sys
import types
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)

_spec = importlib.util.spec_from_file_location(
    "dp_head", os.path.join(ROOT, "nanovllm/spec_decode/draft_proposer.py"))
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
DraftWindow = _dp.DraftWindow
window_valid_from = _dp.window_valid_from
clip_gap_to_window = _dp.clip_gap_to_window
window_request_geom = _dp.window_request_geom

BS = 4
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ======================================================================
print("§1 P=0 逐值等价于改动前的内联公式")
# ----------------------------------------------------------------------
for M in (1, 2, 3, 5, 8):
    ring = [100 + i for i in range(M)]

    def old_slot(p):
        return ring[(p // BS) % M] * BS + p % BS

    def old_b0(p, vf=0):
        return max(vf // BS, p // BS - M + 1, 0)

    def old_clen(p, vf=0):
        return p - old_b0(p, vf) * BS + 1

    def old_bt(p, vf=0):
        b0 = old_b0(p, vf)
        return [ring[(b0 + t) % M] for t in range(M)]

    w = DraftWindow(ring, BS, M, 0)
    bad_s = bad_b = bad_c = bad_t = 0
    for p in range(0, 200):
        for vf in (0, 20, 36, 88):
            if w.slot(p) != old_slot(p):
                bad_s += 1
            if w.b0(p, vf) != old_b0(p, vf):
                bad_b += 1
            if w.ctx_len(p, vf) != old_clen(p, vf):
                bad_c += 1
            if w.block_table(p, vf) != old_bt(p, vf):
                bad_t += 1
    check(f"§1 M={M}: slot 逐值相等", bad_s == 0, f"bad={bad_s}")
    check(f"§1 M={M}: b0 逐值相等", bad_b == 0, f"bad={bad_b}")
    check(f"§1 M={M}: ctx_len 逐值相等", bad_c == 0, f"bad={bad_c}")
    check(f"§1 M={M}: block_table 逐值相等（含长度）", bad_t == 0, f"bad={bad_t}")

# P=0 时 window_valid_from / clip_gap 也逐值不变
for M in (1, 3, 8):
    ok = all(window_valid_from(dvl, BS, M, 0) ==
             (0 if dvl <= 0 else max(0, (dvl - 1) // BS - M + 1) * BS)
             for dvl in (-1, 0, 1, 30, 40, 1000))
    check(f"§1 M={M}: window_valid_from(P=0) 逐值相等", ok)
tok = list(range(500))
ok = all(clip_gap_to_window(st, tok[st:st + ln], BS, 3, tok, 0) ==
         clip_gap_to_window(st, tok[st:st + ln], BS, 3, tok)
         for st in range(0, 200, 7) for ln in (0, 1, 5, 40, 200))
check("§1 clip_gap_to_window(P=0) 与不传 head 逐值相等", ok)

# ======================================================================
print("§2 常驻段与环段互不相交（物理块 / 绝对块号 两个层面）")
# ----------------------------------------------------------------------
for (P, M) in [(1, 1), (1, 5), (2, 5), (4, 5), (1, 12), (8, 3)]:
    ring = list(range(1000, 1000 + P + M))          # 物理 id 用大数，抓排序错误
    w = DraftWindow(ring, BS, M, P)
    head_phys, tail_phys = set(), set()
    for p in range(0, (P + M + 4) * BS):
        b = p // BS
        if b < P:
            head_phys.add(w.slot(p) // BS)
        else:
            tail_phys.add(w.slot(p) // BS)
    check(f"§2 (P={P},M={M}): 常驻段物理块 ∩ 环段物理块 = ∅",
          not (head_phys & tail_phys), f"{head_phys & tail_phys}")
    check(f"§2 (P={P},M={M}): 常驻段恰好用 ring[0:P]",
          head_phys == set(ring[:P]), f"{sorted(head_phys)} vs {ring[:P]}")
    check(f"§2 (P={P},M={M}): b0(p) >= P 恒成立（两段不重叠）",
          all(w.b0(p) >= P for p in range(0, 2000)))

# ======================================================================
print("§3 块表：key index j ↔ 绝对位置 的映射正确 + 按绝对块号递增")
# ----------------------------------------------------------------------
for (P, M) in [(1, 5), (2, 5), (4, 5), (1, 12), (8, 3), (2, 8)]:
    ring = list(range(1000, 1000 + P + M))
    w = DraftWindow(ring, BS, M, P)
    map_bad = mono_bad = len_bad = 0
    for p in range(0, 4000, 7):
        for vf in (0, 100, 333):
            b0 = w.b0(p, vf)
            bt = w.block_table(p, vf)
            clen = w.ctx_len(p, vf)
            if len(bt) != P + M:
                len_bad += 1
            for j in range(clen):
                # key j 应该落在哪个物理块？
                if j < P * BS:
                    want_blk = ring[j // BS]
                else:
                    want_blk = ring[P + ((b0 + (j - P * BS) // BS) - P) % M]
                if bt[j // BS] != want_blk:
                    map_bad += 1
            # 块表按【绝对块号】递增（前 P 项恒为 ring[0:P] = 绝对块 0..P-1）
            abs_blks = list(range(0, P)) + [b0 + t for t in range(M)]
            if abs_blks != sorted(abs_blks):
                mono_bad += 1
            if len(set(abs_blks)) != len(abs_blks):
                mono_bad += 1
    check(f"§3 (P={P},M={M}): key↔位置 映射逐格正确", map_bad == 0, f"bad={map_bad}")
    check(f"§3 (P={P},M={M}): 块表长度恒 = P+M", len_bad == 0, f"bad={len_bad}")
    check(f"§3 (P={P},M={M}): 绝对块号严格递增且无重复", mono_bad == 0, f"bad={mono_bad}")

# ======================================================================
print("§4 可见长度：不越界，且读到的集合 = 开头 ∪ 尾部窗口")
# ----------------------------------------------------------------------
for (P, M) in [(1, 5), (2, 5), (4, 5), (1, 12), (8, 3)]:
    ring = list(range(1000, 1000 + P + M))
    w = DraftWindow(ring, BS, M, P)
    ok = True
    for p in range(0, 4000, 3):
        clen = w.ctx_len(p)
        if not (0 < clen <= (P + M) * BS):
            ok = False
        # 读到的绝对位置集合
        b0 = w.b0(p)
        got = set(range(0, min(P * BS, p + 1))) | set(range(b0 * BS, p + 1))
        exp = set(range(0, min(P * BS, p + 1))) | set(range(b0 * BS, p + 1))
        if got != exp or len(got) != clen:
            ok = False
    check(f"§4 (P={P},M={M}): 0 < clen <= (P+M)*bs 且 |读集合| = clen", ok)

# ======================================================================
print("§5 时间线模拟：只写「常驻段 + 尾部窗口」，读回来必须一格不差")
# ----------------------------------------------------------------------
for (P, M, L) in [(1, 5, 400), (2, 5, 400), (4, 5, 400), (1, 12, 400), (8, 3, 400),
                  (2, 8, 3000)]:
    ring = list(range(1000, 1000 + P + M))
    w = DraftWindow(ring, BS, M, P)
    slots = {}                      # 物理槽 -> 最后写入它的绝对位置
    ok_exact = True
    ok_future = True
    n_read = 0
    for p in range(L):
        # ---- 这一步真正会写的位置：只有 常驻段(p<P*bs) 与 当前尾部窗口 ----
        if p < P * BS:
            slots[w.slot(p)] = p
        b0 = w.b0(p)
        if b0 * BS <= p < (b0 + M) * BS:
            slots[w.slot(p)] = p
        # ---- 读 ----
        bt = w.block_table(p)
        clen = w.ctx_len(p)
        b0r = w.b0(p)
        for j in range(clen):
            slot = bt[j // BS] * BS + (j % BS)
            want = (j if j < P * BS else b0r * BS + (j - P * BS))
            got = slots.get(slot)
            n_read += 1
            if got != want:
                ok_exact = False
            if got is not None and got > p:
                ok_future = False
    check(f"§5 (P={P},M={M},L={L}): 读到的每一格都是它该是的位置（{n_read} 次读）",
          ok_exact)
    check(f"§5 (P={P},M={M},L={L}): 绝不读到未来位置", ok_future)

# ======================================================================
print("§6 常驻段真的常驻：解码很久之后，开头那几个块还在环里")
# ----------------------------------------------------------------------
P, M = 2, 5
ring = list(range(1000, 1000 + P + M))
w = DraftWindow(ring, BS, M, P)
slots = {}
for p in range(0, 200 * BS):            # 模拟「开头 P 块在 prefill 时写完」
    if p < P * BS:
        slots[w.slot(p)] = p
# 之后只写尾部窗口（滚到很远）
for p in range(200 * BS, 400 * BS):
    b0 = w.b0(p)
    if b0 * BS <= p < (b0 + M) * BS:
        slots[w.slot(p)] = p
ok = all(slots.get(ring[b] * BS + off) == b * BS + off
         for b in range(P) for off in range(BS))
check("§6 滚了 200 个块之后，常驻段的内容【一格没被覆盖】", ok)

# ======================================================================
print("§7 window_request_geom / clip 带上 head 之后仍自洽")
# ----------------------------------------------------------------------
tk = list(range(5000))
bad = 0
for P in (0, 1, 2, 4):
    for M in (1, 3, 5, 12):
        for dvl in (0, 1, 500, 1000, 4900, 4999):
            st, gp, vf = window_request_geom(dvl, tk, BS, M, P)
            end = len(tk) - 1
            if not (len(gp) == end - st and (st + s == i for s, i in ())):
                bad += 1
            if len(gp) != end - st:
                bad += 1
            if any(gp[s] != tk[st + s] for s in range(len(gp))):
                bad += 1
            if P and st < P * BS and st != 0:
                bad += 1            # 补齐起点不得落进常驻段里
check("§7 window_request_geom 契约（配对 / 不越界 / 不补常驻段）", bad == 0, f"bad={bad}")

# ======================================================================
print()
if FAILED:
    print(f"FAILED {len(FAILED)}:")
    for n in FAILED:
        print("  -", n)
    sys.exit(1)
print("ALL PASS")
