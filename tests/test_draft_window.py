"""纯 CPU 回归（B 步）：draft 滑窗的环形算术 + 「全上下文路径没被改写」的对拍。

滑窗的约定（见 draft_proposer.DraftWindow）
------------------------------------------
draft 只保留【最近 M 个块】（M = spec_draft_window / block_size），物理布局是
块级环形缓冲：绝对块号 b 永远落在 ring 的第 (b % M) 个块。于是

    slot(p) = ring[(p//bs) % M] * bs + p % bs
    b0(p)   = max(valid_from//bs, p//bs - M + 1, 0)     窗口最老的块
    clen(p) = p - b0(p)*bs + 1                          ≤ M*bs
    bt(p)   = [ ring[(b0+t) % M] for t in range(M) ]     从 b0 起递增

本文件里 §4 是硬要求：**spec_draft_window == 0（默认）时，新的几何计算必须与
改动前那份内联表达式在数值上逐值相等** —— 这是「两条路径都在、默认行为没变」
的证据。§6 则用「槽 → 绝对位置」的模拟证明窗口读到的 key 恰好是最新那些位置
（没有任何陈旧读出）。

跑：python tests/test_draft_window.py
"""
import os
import sys
import types
import importlib.util

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)

_spec = importlib.util.spec_from_file_location(
    "dp_win", os.path.join(ROOT, "nanovllm/spec_decode/draft_proposer.py"))
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
DraftWindow = _dp.DraftWindow
window_valid_from = _dp.window_valid_from
clip_gap_to_window = _dp.clip_gap_to_window
catchup_gap = _dp.catchup_gap
window_request_geom = _dp.window_request_geom

BS = 4
M = 3                      # 窗口 = 最近 3 个块 = 12 个槽
RING = [7, 11, 13]         # 三个物理块 id（故意取非连续值，抓排序/取模错误）
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def mk(M_=M, ring=None):
    return DraftWindow(ring or RING, BS, M_)


# ======================================================================
print("§1 环形算术：slots / 窗口起点 / 可见长度 / 旋转块表")
# ----------------------------------------------------------------------
w = mk()
# --- 早期（还没绕圈）：槽就是自然序，块表是自然序 ---
check("① p<W 时 slot(p) 就是自然序（7*4+p）",
      [w.slot(p) for p in range(0, 5)] == [7 * 4 + 0, 7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3, 11 * 4 + 0],
      f"{[w.slot(p) for p in range(0, 5)]}")
check("① p<W 时 clen = p+1（只有已写过的位置可见）",
      [w.ctx_len(p) for p in (0, 1, 3, 4, 11)] == [1, 2, 4, 5, 12],
      f"{[w.ctx_len(p) for p in (0, 1, 3, 4, 11)]}")
check("① p<W 时块表是自然序（从块 0 起）",
      w.block_table(5) == [7, 11, 13], f"{w.block_table(5)}")
# --- 稳定期：满窗 ---
# 窗口是【块粒度】的（最近 M 个块），所以有效长度在 (M-1)*bs+1 ~ M*bs 之间：
# 查询落在块内越靠后，能看到的历史越多。到块末尾就正好是满窗 M*bs。
check("① 稳定期 clen ∈ [(M-1)*bs+1, M*bs]，且块末尾取满窗",
      all((M - 1) * BS + 1 <= w.ctx_len(p) <= M * BS for p in (12, 13, 40, 99))
      and w.ctx_len(4 * 4 - 1) == M * BS,
      f"{[w.ctx_len(p) for p in (12, 13, 40, 99)]} 末尾={w.ctx_len(4 * 4 - 1)}")
check("① 稳定期 slot 按 p%W 绕圈",
      w.slot(12) == 7 * 4 + 0 and w.slot(13) == 7 * 4 + 1 and w.slot(24) == 7 * 4 + 0,
      f"{w.slot(12)} {w.slot(13)} {w.slot(24)}")
check("① 块表从 b0 起递增（旋转正确）",
      w.block_table(24) == [RING[(24 // BS - M + 1 + t) % M] for t in range(M)],
      f"{w.block_table(24)}  b0={w.b0(24)}")
check("① clen 永远不会超过 M*bs（块表只有 M 项，超了就非法读）",
      all(w.ctx_len(p) <= M * BS for p in range(0, 200)), "")

# ======================================================================
print("§2 valid_from 下界：环里更老的块不可信时，绝不许读")
# ----------------------------------------------------------------------
w = mk()
vf = 5 * BS                       # 从第 5 块起才可信
check("② b0 不低于 valid_from 对齐后的块",
      all(w.b0(p, vf) * BS >= vf for p in range(20, 60)), "")
check("② clen 相应变小（≤ 满窗，且 > 0）",
      all(0 < w.ctx_len(p, vf) <= M * BS for p in range(20, 60)),
      f"{w.ctx_len(24, vf)}")
check("② 传入 valid_from 后块表第一项 = 该块",
      w.block_table(24, vf)[0] == RING[(vf // BS) % M], f"{w.block_table(24, vf)[0]}")

# ======================================================================
print("§3 window_valid_from / clip_gap_to_window")
# ----------------------------------------------------------------------
check("③ 水位为 0 时窗口起点是 0", window_valid_from(0, BS, M) == 0)
check("③ 水位远大于窗口时，起点 = 最近 M 块的块边界",
      window_valid_from(40, BS, M) == ((40 - 1) // BS - M + 1) * BS,
      f"{window_valid_from(40, BS, M)}")

# 大缺口：夹到最近 M 块，且起点对齐块边界
start, gap = clip_gap_to_window(0, list(range(100)), BS, M)
end = 0 + 100
eb = end // BS
check("③ 大缺口被夹到最近 M 个块", start == max(0, eb - M + 1) * BS, f"start={start}")
check("③ 起点对齐块边界", start % BS == 0, f"{start}")
check("③ 夹完的缺口跨越的块数 ≤ M",
      (start + len(gap) - 1) // BS - start // BS + 1 <= M,
      f"blocks={ (start + len(gap) - 1)//BS - start//BS + 1 }")

# 小缺口：原样保留 —— 【不许】为了对齐而移动 start
# ★ 这里原来是 `len(g2) >= 4 and s2 % BS == 0 and s2 <= 10`，恰好把
#   「把 start 向下取整到块边界、gap 却不动」这个 bug 当成正确行为钉住了。
#   那个写法让 gap[s] 落到 start+s 之外（调用方 propose_batch 就是按
#   `positions = catchup_start + s` 配对写入的），把环里正确的 KV 覆盖成错的。
_tok = list(range(1000, 1100))
s2, g2 = clip_gap_to_window(10, _tok[10:14], BS, M, _tok)      # end = 14
check("③ 小缺口原样保留（start 不动、gap 不动）",
      s2 == 10 and g2 == _tok[10:14], f"start={s2} gap={g2}")
check("③ 小缺口配对：gap[s] 就是位置 start+s 上的 token",
      g2 == [_tok[s2 + s] for s in range(len(g2))], f"{g2} vs {_tok[10:14]}")
_pair_bad = 0
for _st in range(0, 40):
    for _ln in range(1, 40):
        _s, _g = clip_gap_to_window(_st, _tok[_st:_st + _ln], BS, M, _tok)
        if _g != _tok[_s:_st + _ln]:
            _pair_bad += 1
check("③ 契约：1540 组 (start,len) 上 gap 恒等于 token_ids[start:end]", _pair_bad == 0,
      f"mismatch={_pair_bad}")

# 缺口伸到窗口之外：起点前移，且 gap 同步裁掉等长前缀（仍然配对）
_s, _g = clip_gap_to_window(0, _tok[0:100], BS, M, _tok)   # end = 100, s_min = 92
check("③ 超出窗口的缺口：起点前移到窗口最老块、gap 同步裁前缀",
      _s == 92 and _g == _tok[92:100], f"start={_s} len={len(_g)}")
check("③ 前移后仍然配对", _g == _tok[_s:100], "")

# 空缺口原样返回
s3, g3 = clip_gap_to_window(12, [], BS, M)
check("③ 空缺口原样返回", g3 == [] and s3 == 12, "")

# ----------------------------------------------------------------------
# ③' ★ 负对照（必须能失败）：把对齐方向反过来（向下取整到块边界）就会
#    破坏配对 —— 这一条证明上面那些断言不是空转。
def _old_align_down(start, gap, block_size, window_blocks):
    """改动前的写法（把 start 向下取整到块边界，gap 不动）。"""
    if window_blocks <= 0 or not gap:
        return start, gap
    end = int(start) + len(gap)
    s_min = max(0, (end // block_size) - window_blocks + 1) * block_size
    if start < s_min:
        drop = min(s_min - start, len(gap))
        start, gap = start + drop, gap[drop:]
    start = (start // block_size) * block_size
    start = max(start, s_min)
    if start >= end:
        return start, []
    return start, list(gap)


_neg_bad = 0
_neg_shifted = 0
for _st in range(0, 40):
    _g0 = _tok[_st:_st + 10]
    _s_old, _g_old = _old_align_down(_st, _g0, BS, M)
    # 旧写法：返回的 gap 声称它们落在 [_s_old, _s_old+len) 上
    if _g_old != _tok[_s_old:_st + len(_g0)]:
        _neg_bad += 1
    if _s_old != _st:
        _neg_shifted += 1
check("③' 负对照：旧的对齐写法确实破坏配对（≥1 组）", _neg_bad > 0,
      f"破坏 {_neg_bad}/40 组，其中 {_neg_shifted} 组起点被挪动")
# 而新写法在同一批输入上全对
_new_bad = 0
for _st in range(0, 40):
    _g0 = _tok[_st:_st + 10]
    _s1, _g1 = clip_gap_to_window(_st, _g0, BS, M, _tok)
    if _g1 != _tok[_s1:_st + len(_g0)]:
        _new_bad += 1
check("③' 新写法在同一批输入上配对全对", _new_bad == 0, f"mismatch={_new_bad}")

# ★ 负对照（真实场景，最锋利的一条）：bs=256、上一轮全接受 → 缺口是位置 1030
#   上的 1 个 token。旧写法把它挪到位置 1024 去写 —— 于是位置 1030 保持陈旧、
#   位置 1024 被写成别人的 KV。两种写法必须给出不同的落点。
_rt = list(range(20000, 20000 + 1032))
_s_old, _g_old = _old_align_down(1030, _rt[1030:1031], 256, 8)
_s_new, _g_new = clip_gap_to_window(1030, _rt[1030:1031], 256, 8, _rt)
check("③' 真实场景负对照：旧写法把位置 1030 的 token 写到 1024（错位 6 格）",
      _s_old == 1024 and _g_old == _rt[1030:1031], f"start={_s_old} {_g_old}")
check("③' 真实场景：新写法原地不动，落点 = 1030（配对正确）",
      _s_new == 1030 and _g_new == _rt[1030:1031], f"start={_s_new} {_g_new}")
check("③' 两种写法的落点不同（所以这条负对照不是空转）", _s_old != _s_new)

# ======================================================================
print("§4 ★ 硬要求：默认档（滑窗关闭）必须与改动前的写法逐值相等")
# ----------------------------------------------------------------------
from nanovllm.engine.block_manager import BlockManager      # noqa: E402
from nanovllm.spec_decode.draft_proposer import DraftModelProposer   # noqa: E402


class _Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 4
        self._b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *a, **k):
        raise RuntimeError("本测试不跑前向")

    def compute_logits(self, h):
        raise RuntimeError("本测试不跑前向")


prop = DraftModelProposer(_Stub(), k=2, block_size=BS)


def old_geom(block_table, pos):
    """改动前 propose_batch 里的【内联表达式】原样抄一份（作为对照真值）。"""
    bi, off = divmod(int(pos), prop.block_size)
    slot = int(block_table[bi]) * prop.block_size + off if 0 <= bi < len(block_table) else -1
    return slot, list(block_table), int(pos) + 1


bad_slot = bad_bt = bad_cl = 0
checked = 0
for bt in ([9, 3, 5], [2], [4, 4, 6], []):
    for pos in range(0, 20):
        r = dict(block_table=list(bt), window_blocks=0, draft_ring=[])
        got = prop._geom(r, pos)
        exp = old_geom(bt, pos)
        checked += 1
        bad_slot += got[0] != exp[0]
        bad_bt += got[1] != exp[1]
        bad_cl += got[2] != exp[2]
check(f"④ {checked} 组 (块表×位置) 上 slot 逐值相等", bad_slot == 0, f"mismatch={bad_slot}")
check("④ block_table 原样传递（不复制语义、不改内容）", bad_bt == 0, f"mismatch={bad_bt}")
check("④ context_len 恒等于 pos+1", bad_cl == 0, f"mismatch={bad_cl}")
check("④ 块表不够时 slot = -1（与旧实现一致的越界保护）",
      prop._geom(dict(block_table=[2], window_blocks=0, draft_ring=[]), 9)[0] == -1)

# 默认档的配置默认值 = 关闭（config.py 要 transformers 才能 import，这里读源码文本）
with open(os.path.join(ROOT, "nanovllm/config.py"), encoding="utf-8") as _f:
    _src = _f.read()
check("④ Config.spec_draft_window 默认 = 0（默认保留全上下文行为）",
      "spec_draft_window: int = 0" in _src, "")
check("④ 默认档不需要 draft 池（BlockManager 默认参数就是 0）",
      "num_draft_blocks: int = 0, draft_window_blocks: int = 0" in
      open(os.path.join(ROOT, "nanovllm/engine/block_manager.py"), encoding="utf-8").read())
# ★ Config 是 @dataclass(slots=True) —— 运行时【不能】凭空加属性。
#   （实机踩过：ModelRunner 里 config.num_draft_blocks = ... 直接 AttributeError，
#    而纯 CPU 测试因为给 config 塞了占位模块，一点都测不出来。）
check("④ num_draft_blocks / draft_window_blocks 已在 Config 里声明",
      "num_draft_blocks: int = 0" in _src and "draft_window_blocks: int = 0" in _src, "")

# 全上下文档下，请求字典里【不许】带出任何窗口量
from nanovllm.engine.sequence import Sequence                          # noqa: E402
from nanovllm.sampling_params import SamplingParams                    # noqa: E402
Sequence.block_size = BS
seq = Sequence([1, 2, 3, 4, 5], SamplingParams())
check("④ 新建 Sequence 的 draft_block_table 为空（默认全上下文）",
      seq.draft_block_table == [])
import pickle                                                          # noqa: E402
seq.draft_block_table = [3, 4, 5]
check("④ draft_block_table 过进程边界（TP>1 每条 rank 用的块表必须一致）",
      pickle.loads(pickle.dumps(seq)).draft_block_table == [3, 4, 5])

# ======================================================================
print("§5 滑窗下的位置/RoPE 契约：丢的是 KV，不是位置编号")
# ----------------------------------------------------------------------
w = mk()
# 位置 p 的写槽只与 p 有关；查询位置 p 的可见区间【终点】永远是 p
check("⑤ 可见区间终点 = 查询自己的位置（不是窗口起点）",
      all(w.b0(p) * BS + w.ctx_len(p) - 1 == p for p in range(0, 60)), "")
check("⑤ 滑窗不改变绝对位置：同一 p 的 slot 在窗口前后一致（%W 取模）",
      all(w.slot(p) == w.slot(p + M * BS) for p in range(0, 20)), "")

# ======================================================================
print("§6 时间线模拟：任意时刻窗口里读到的 key 恰好是最新那些位置（无陈旧读出）")
# ----------------------------------------------------------------------
for Mv in (1, 2, 3, 5):
    ring = [100 + i for i in range(Mv)]
    w = mk(Mv, ring)
    slots = {}                       # slot -> 最后写入它的绝对位置
    ok_exact = True
    ok_newest = True
    L = 60
    for p in range(L):
        slots[w.slot(p)] = p
        b0 = w.b0(p)
        clen = w.ctx_len(p)
        bt = w.block_table(p)
        for j in range(clen):
            slot = bt[j // BS] * BS + (j % BS)
            # 读到的这一格，当前内容是哪个绝对位置？
            got = slots.get(slot)
            want = b0 * BS + j
            if got != want:
                ok_exact = False
            # 且必须 ≤ 查询位置（不能看到未来）
            if got is not None and got > p:
                ok_newest = False
    check(f"⑥ M={Mv}: 窗口读到的每一格都是它该是的位置（无陈旧/错位）", ok_exact)
    check(f"⑥ M={Mv}: 绝不读到未来位置", ok_newest)

# 窗口的「内容」就是最近 M 个块
w = mk(3, [100, 101, 102])
for p in (9, 12, 17, 30):
    b0 = w.b0(p)
    live = {p - k for k in range(0, p + 1)}
    check(f"⑥ p={p}: 窗口起点块 = {b0}（= p//bs-M+1 或 0）",
          b0 == max(0, p // BS - 3 + 1), f"b0={b0}")

# ======================================================================
print("§7 ★ 块表必须从「查询位置算出来的 b0」起（实机踩过的静默错误）")
# ----------------------------------------------------------------------
# 症状：不报错，只是接受率悄悄掉。M=2 时 accept 0.476 -> 0.256，比窗口更小的
# M=1（0.476）还差，才露出马脚 —— 根因是块表用「窗口起点 w0」再算一遍 b0，
# 等于又多减了 (M-1)，key index 映射整体错位，attention 读到别人（甚至未来）的 KV。
# ★ 正确做法是始终用【查询位置】算 b0（block_table(pos) 内部就是这么做的）；
#   调用方如果已经有 end/pos，就用 end-1 而不是 w0。
# ★ 后续进一步发现：滑窗 prefill 干脆不该走带 block_table 的 paged 路径
#   （那一趟就是「最近 M 个块」的本地因果注意力），见 model_runner
#   _run_draft_prefill_window 里的说明 —— 这条断言在 decode 侧依然成立、依然必要。
grid_bad = 0
for _M in (1, 2, 3, 4, 8):
    ring = [200 + i for i in range(_M)]
    w = mk(_M, ring)
    for end in (1, 5, 4 * 4, 4 * 4 + 1, 100, 1024, 1025):
        b0_pre = max(0, (end - 1) // BS - _M + 1)
        want = [ring[(b0_pre + t) % _M] for t in range(_M)]
        got = w.block_table(end - 1)
        w0 = b0_pre * BS
        wrong = w.block_table(w0) if w0 else None
        if got != want:
            grid_bad += 1
        # 记录「用 w0 会错」这件事本身（错的位置必须真的存在，否则这条测试没意义）
        if wrong is not None and wrong == want and b0_pre > 0:
            pass
check("⑦ 用 end-1 算 b0 → 块表起点恒等于 b0_pre（45 组全等）",
      grid_bad == 0, f"mismatch={grid_bad}")
# 反证：至少有一组「用 w0 算会错」，说明这个坑真实存在（测试不是空转）
_found = False
for _M in (2, 3, 4):
    ring = [200 + i for i in range(_M)]
    w = mk(_M, ring)
    end = 40
    b0_pre = max(0, (end - 1) // BS - _M + 1)
    if w.block_table(b0_pre * BS) != w.block_table(end - 1):
        _found = True
check("⑦ 反证：M≥2 时用 w0 算确实会错位（所以这条回归是必要的）", _found)

# ======================================================================
print("§8 ★ 独立验证：「滑窗真的只看到最近 M 个 token」+ 缺口轮次不许砍小窗口")
# ----------------------------------------------------------------------
# 这条是上一轮自己承认没做的地基验证。做法不是"看几何公式对不对"，而是
# 把整条时间线跑一遍并给每一格环内容【打上它自己的绝对位置标签】：
#   写  slot(p) -> 标签 p
#   读  key j   -> slot = bt[j//bs]*bs + j%bs，必须 content[slot] == b0*bs + j
# 任何"读到陈旧/别人的内容 / 读到未来 / 越过窗口"都会被抓出来。
#
# 同时验证「只看到最近 M 个 token」这件事的两半：
#   ① 可见的位置集合 = 最近 M 个块里 ≤ pos 的那些（够不到的更老位置不可见）；
#   ② 这些位置读到的都是自己的 KV（不是别人的）。


def _simulate(M_, L, k, accepts, bs=BS):
    """跑一遍 prefill + 若干轮 decode，返回 (failures, stats)。"""
    ring = [100 + i for i in range(M_)]
    w = mk(M_, ring)
    content = {}                       # slot -> 环里这一格装的是哪个绝对位置的 KV
    bad = []

    def _check_read(pos, vf, where):
        b0 = w.b0(pos, vf)
        b0p = w.b0(pos)                # 自然窗口起点（不含 valid_from）
        clen = w.ctx_len(pos, vf)
        bt = w.block_table(pos, vf)
        # ① 窗口不越界：最老可见位置不早于 pos-M*bs+1；可见长度 ≤ M*bs
        if b0 * bs < pos - M_ * bs + 1:
            bad.append((where, pos, "window_too_old", b0 * bs))
        if clen > M_ * bs:
            bad.append((where, pos, "clen>M*bs", clen))
        # ② 每一格读到的都必须是它自己的位置
        for j in range(clen):
            slot = bt[j // bs] * bs + (j % bs)
            want = b0 * bs + j
            if content.get(slot) != want:
                bad.append((where, pos, "stale_read", want, content.get(slot)))
            if want > pos:
                bad.append((where, pos, "future_read", want))
        # ③ 无缺口时候选位置必须拿到完整窗口（b0 就是自然起点）
        return b0, b0p, clen

    # ---- 窗口 prefill：只写最近 M 个块 ----
    b0p = max(0, (L - 1) // bs - M_ + 1)
    for p in range(b0p * bs, L):
        content[w.slot(p)] = p
    num_tokens, dvl, gap_len = L, L, 0
    stats = []
    for r in range(len(accepts)):
        token_ids = list(range(10000, 10000 + num_tokens))
        dvl = num_tokens - 1 - gap_len
        start, gapt, vf = window_request_geom(dvl, token_ids, bs, M_)
        # 补齐：token[s] 必须落到 start+s
        if gapt != token_ids[start:len(token_ids) - 1]:
            bad.append(("catchup_pair", r, start, len(gapt)))
        for s, _t in enumerate(gapt):
            content[w.slot(start + s)] = start + s
        # k 个候选位置（propose 每步：先 store 再 attend）
        clens = []
        for s in range(k):
            pos = num_tokens - 1 + s
            content[w.slot(pos)] = pos
            b0, b0p_nat, clen = _check_read(pos, vf, f"r{r}s{s}")
            clens.append(clen)
            if b0 != b0p_nat:
                bad.append((f"r{r}s{s}", pos, "collapsed_window", b0, b0p_nat))
        stats.append(dict(r=r, num_tokens=num_tokens, gap=len(gapt),
                          vf=vf, clen=clens))
        a = accepts[r]
        num_tokens += a + 1
        gap_len = 1 if a == k else 0
    return bad, stats


for Mv in (1, 2, 3, 5, 8):
    bad, stats = _simulate(Mv, 40, 4, [4, 2, 4, 0, 4, 3, 4, 1])
    check(f"⑧ M={Mv}: 全时间线无陈旧读/无未来读/不越界/缺口轮次不塌缩", not bad,
          f"{bad[:3]}")
    # 「只看到最近 M 个 token」的硬边界：M 越小可见长度越短，且 ≤ M*bs
    maxc = max(max(s["clen"]) for s in stats)
    check(f"⑧ M={Mv}: 可见长度 ≤ M*bs = {Mv * BS}", maxc <= Mv * BS, f"max clen={maxc}")

# 边界结论：M=2 看得见的位置【严格】多于 M=1（块粒度），且 M=1 看不见上一个块
w1, w2 = mk(1, [0]), mk(2, [0, 1])
_s1 = {w1.b0(p) * BS + j for p in (4 * BS - 1,) for j in range(w1.ctx_len(p))}
_s2 = {w2.b0(p) * BS + j for p in (4 * BS - 1,) for j in range(w2.ctx_len(p))}
check("⑧ 同一查询位置：M=2 的可见位置集合真包含 M=1（滑窗确实随 M 变宽）",
      _s1 < _s2, f"M=1:{min(_s1)}..{max(_s1)}  M=2:{min(_s2)}..{max(_s2)}")
check("⑧ M=1 时上一个块完全不可见（只看到最近 1 个块）",
      min(_s1) == (4 * BS - 1) // BS * BS, f"min={min(_s1)}")

# ----------------------------------------------------------------------
# ⑧' ★ 负对照：把 valid_from 退回「max(wvf, 补齐起点)」就会把窗口砍成 ≈1 块
#     （这正是 W=512 接受率最低那类现象的机制）。必须能失败。
_tk = list(range(10000, 10000 + 1032))
Mv, BSv = 8, 256
_w = DraftWindow([0, 1, 2, 3, 4, 5, 6, 7], BSv, Mv)
dvl, pos = 1030, 1031                       # 上一轮全接受 → 缺口 = 位置 1030 一个 token
_start, _gap, _vf = window_request_geom(dvl, _tk, BSv, Mv)
_clen_new = _w.ctx_len(pos, _vf)
# 旧规则
_wvf = window_valid_from(dvl, BSv, Mv)
_vf_old = max(_wvf, _start)
_clen_old = _w.ctx_len(pos, _vf_old)
check("⑧' 修复后：1-token 缺口下窗口 = 完整上下文（clen = pos+1）",
      _clen_new == pos + 1, f"clen={_clen_new}  vf={_vf}  start={_start}")
check("⑧' 负对照：旧规则 max(wvf, 补齐起点) 把同一窗口砍到 ≈1 个块（能失败）",
      _clen_old < 2 * BSv and _clen_old == pos - (_start // BSv) * BSv + 1,
      f"旧 clen={_clen_old} vs 新 clen={_clen_new}（vf_old={_vf_old}）")
check("⑧' 负对照确实把窗口砍小了（新 > 旧 * 4）",
      _clen_new > 4 * _clen_old, f"{_clen_new} vs {_clen_old}")

# ----------------------------------------------------------------------
# ⑧'' ★ 补齐那一趟前向的上下文：必须与提议阶段用同一个下界。
#     propose_batch 原来硬传 valid_from=base（= 补齐起点），等于把补齐的上下文
#    砍到「补齐起点所在的块」—— 实测 ctx 从 1031 掉到 7，缺口那个位置的 KV
#     就是在极短上下文下算出来的（而它正是 draft 下一步最需要的那个）。
_tk2 = list(range(30000, 30000 + 1032))
_s3, _g3, _vf3 = window_request_geom(1030, _tk2, 256, 8)
_w8 = DraftWindow(list(range(8)), 256, 8)
_pos_c = _s3                                   # 补齐的第一个（也是唯一一个）位置
_clen_catch = _w8.ctx_len(_pos_c, _vf3)
_clen_catch_old = _w8.ctx_len(_pos_c, _s3)     # 旧写法：硬传 valid_from = base
check("⑧'' 补齐前向的 ctx = 完整上下文（与提议阶段同一下界）",
      _clen_catch == _pos_c + 1, f"clen={_clen_catch} pos={_pos_c}")
check("⑧'' 负对照：硬传 valid_from=base 会把补齐的 ctx 砍到 ≈1 个块（能失败）",
      _clen_catch_old <= 2 * 256 and _clen_catch_old < _clen_catch,
      f"旧 clen={_clen_catch_old} vs 新 clen={_clen_catch}")

# 缺口伸到窗口之外（drop）时：补齐必须收紧到 start —— 更老的位置根本没写过
for _M in (1, 2, 4, 8):
    _w = DraftWindow(list(range(_M)), 256, _M)
    _L, _dvl = 6000, 300                        # 缺口 5699 个 token，远超窗口
    _tk = list(range(40000, 40000 + _L))
    _st, _gp, _vfd = window_request_geom(_dvl, _tk, 256, _M)
    assert _st > min(_dvl, _L - 1), (_M, _st, _dvl)      # 确实发生了前移
    _bad = [p for p in range(_st, _L)
            if _w.b0(p, _vfd) * 256 < _st]
    check(f"⑧'' M={_M}: drop 后补齐前向 b0 恒 ≥ 补齐起点（不读没写过更老位置）",
          not _bad, f"{_bad[:3]}")


# 缺口被夹到窗口外（drop 分支）时：起点前移到窗口最老块，且自然下界 ≥ start//bs
for Mv in (1, 2, 4, 8):
    _w = DraftWindow([0] * Mv, 256, Mv)
    for _L, _dvl in ((3000, 500), (3000, 0), (5000, 100)):
        _tk = list(range(10000, 10000 + _L))
        _st, _gp, _vfn = window_request_geom(_dvl, _tk, 256, Mv)
        _smin = max(0, ((_L - 1) // 256 - Mv + 1)) * 256
        assert _st >= _smin, (_st, _smin)
        for _pos in range(_L - 1, _L - 1 + 6):
            _b0 = _w.b0(_pos, _vfn)
            if _b0 * 256 < _st:
                FAILED.append(f"⑧ drop 后仍能读到补齐起点之前的位置 M={Mv} dvl={_dvl}")
check("⑧ 缺口被夹到窗口外时：b0 恒 ≥ 补齐起点所在块（drop 后不读更老的位置）",
      not any("drop 后" in f for f in FAILED), "")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
