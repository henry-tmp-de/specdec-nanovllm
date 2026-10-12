"""纯 CPU 回归：`spec_draft_win_realign`（第二次及以后的「对齐漂移」修复）的算术。

背景（见 docs/滑动路径排查-开发记录.md §6 与 docs/修复后重定位-开发记录.md）
----------------------------------------------------------------------------
`spec_draft_win_align` 只修了【第一次】漂移：prefill 把环里那些块对齐到
`b0(end)`（第一个生成位置所在块），并把 `draft_win_start = b0*bs` 当作解码期
窗口下界的地板。但序列再往前生成、跨进下一个块时，`DraftWindow.b0` 的自然项
`p//bs − M + 1` 会超过地板 ⇒ 环里保留的块仍编码着「更早一个块」的上下文。

`window_realign_target(floor, end, bs, M, P)` 就是那个判断（纯算术）：
  自然起点 b0_nat = max(P, (end−1)//bs − M + 1)
  b0_nat*bs > floor → (True, b0_nat*bs)   ← 调用方按新起点重算这扇窗口
  否则              → (False, floor)

硬要求（本文件就是这些）：
  §1 与 DraftWindow.b0 的自然项逐值一致（P ∈ {0,1,2,4} × M ∈ {1,3,5,12}）。
  §2 地板内（第一个块）恒不触发；跨进下一个块恰好触发一次。
  §3 触发后地板单调前进到自然起点，且【同块内不再触发】。
  §4 关闭档（M<=0 / end<=0）恒不触发、地板原样返回 —— 默认行为不变的第一道证据。
  §5 时间线模拟：ALIGN 给定地板后，逐位置调用本函数 ⇒ 读端声明的 b0
     恒等于地板块（自然项永不越位）。这就是「环里的内容 = 声明的窗口」。
  §6 每次重算重写的 token 数 ≤ M*bs（成本上界）。
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
    "dp_realign", os.path.join(ROOT, "nanovllm/spec_decode/draft_proposer.py"))
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
DraftWindow = _dp.DraftWindow
window_realign_target = _dp.window_realign_target

BS = 256
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ======================================================================
print("§1 与 DraftWindow.b0 的自然项逐值一致")
# ----------------------------------------------------------------------
bad = 0
for P in (0, 1, 2, 4):
    for M in (1, 3, 5, 12):
        ring = list(range(1000, 1000 + P + M))
        w = DraftWindow(ring, BS, M, P)
        for end in (1, 5, 256, 300, 1024, 8191, 8192, 8193, 8448, 20000):
            b0_nat = w.b0(end - 1)
            need, ns = window_realign_target(0, end, BS, M, P)
            # floor=0 时 b0_nat>=P>=0 ⇒ new_start = b0_nat*BS，需要触发（除非全为 0）
            if ns != b0_nat * BS:
                bad += 1
check("§1 b0_nat*BS 与 DraftWindow.b0(end-1)*BS 逐值一致（floor=0）", bad == 0, f"bad={bad}")

# ======================================================================
print("§2 地板内不触发；跨进下一个块恰好触发一次")
# ----------------------------------------------------------------------
bad = 0
for M in (1, 3, 5, 12):
    for P in (0, 2):
        floor_end = 8200                     # prompt 长度（落在块 32 内）
        b0p = max(P, floor_end // BS - M + 1)
        floor = b0p * BS
        # 同一个块内（end 到 floor_end 所在块的末尾）都不该触发
        blk_end = (floor_end // BS + 1) * BS
        for end in range(floor_end, blk_end + 1):
            need, _ = window_realign_target(floor, end, BS, M, P)
            if need:
                bad += 1
        # 跨进下一个块（end 的第一位落在块 floor_end//BS + 1）之后应当触发
        need, ns = window_realign_target(floor, blk_end + 1, BS, M, P)
        if not need:
            bad += 1
        if ns != max(P, (blk_end) // BS - M + 1) * BS:
            bad += 1
check("§2 同块不触发 / 跨块触发（M∈{1,3,5,12}, P∈{0,2}）", bad == 0, f"bad={bad}")

# ======================================================================
print("§3 地板单调前进 + 同块内不再触发")
# ----------------------------------------------------------------------
bad = 0
for M in (1, 3, 5, 12):
    floor_end = 8200
    floor = max(0, floor_end // BS - M + 1) * BS
    crossings = 0
    for end in range(floor_end, floor_end + 4 * BS + 7):
        need, ns = window_realign_target(floor, end, BS, M, 0)
        if need:
            if ns <= floor:
                bad += 1                     # 必须严格前进
            floor = ns
            crossings += 1
            # 触发后立刻再问一次同一个 end：不该再触发
            need2, ns2 = window_realign_target(floor, end, BS, M, 0)
            if need2 or ns2 != floor:
                bad += 1
    if crossings != 4:                        # 4 个块边界 = 4 次
        bad += 1
check("§3 地板严格单调 + 每次跨块恰好一次", bad == 0, f"bad={bad}")

# ======================================================================
print("§4 关闭档（M<=0 / end<=0）恒不触发、地板原样")
# ----------------------------------------------------------------------
bad = 0
for (M, end, f) in ((0, 9000, 7168), (-1, 9000, 7168), (5, 0, 7168), (5, -3, 7168)):
    need, ns = window_realign_target(f, end, BS, M, 0)
    if need or ns != max(0, f):
        bad += 1
check("§4 M<=0 / end<=0 ⇒ (False, 地板)", bad == 0, f"bad={bad}")

# ======================================================================
print("§5 时间线模拟：ALIGN 后逐位置调用 ⇒ 读端 b0 恒等于地板块")
# ----------------------------------------------------------------------
bad = 0
for M in (1, 3, 5, 12):
    for P in (0, 2):
        prompt_end = 8192
        floor = max(P, prompt_end // BS - M + 1) * BS
        rewrites = 0
        for end in range(prompt_end, prompt_end + 3 * BS + 5):
            need, ns = window_realign_target(floor, end, BS, M, P)
            if need:
                floor = ns
                rewrites += 1
            # 读端声明的 b0（DraftWindow.b0 带地板）
            declared = max(P, floor // BS, (end - 1) // BS - M + 1)
            if declared * BS != floor:
                bad += 1                      # 自然项越位了 ⇒ 声明与写入不一致
        if rewrites != 3:                     # 走完 3 个块边界
            bad += 1
check("§5 任一位置 declared_b0 == 写入对齐（不变量成立）", bad == 0, f"bad={bad}")

# ======================================================================
print("§6 每次重算重写的 token 数 ≤ M*bs")
# ----------------------------------------------------------------------
bad = 0
for M in (1, 3, 5, 12):
    for P in (0, 2):
        prompt_end = 8192
        floor = max(P, prompt_end // BS - M + 1) * BS
        for end in range(prompt_end, prompt_end + 6 * BS + 5):
            need, ns = window_realign_target(floor, end, BS, M, P)
            if need:
                floor = ns
                # 重算窗口 = [floor, end) 以及常驻段 [0, min(P*BS,end))；尾部 ≤ M*bs
                tail = end - floor
                if tail > M * BS:
                    bad += 1
check("§6 尾部重写长度 ≤ M*bs（成本上界）", bad == 0, f"bad={bad}")

# ======================================================================
print()
if FAILED:
    print(f"FAILED {len(FAILED)}:")
    for n in FAILED:
        print("  -", n)
    sys.exit(1)
print("ALL PASS")
