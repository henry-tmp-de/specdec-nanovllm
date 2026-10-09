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

# 小缺口：整体保留（不丢有效 token）
s2, g2 = clip_gap_to_window(10, [1, 2, 3, 4], BS, M)      # end = 14
check("③ 小缺口只做块对齐，不丢 token（长度不减少）",
      len(g2) >= 4 and s2 % BS == 0 and s2 <= 10, f"start={s2} len={len(g2)}")
s3, g3 = clip_gap_to_window(12, [], BS, M)
check("③ 空缺口原样返回", g3 == [] and s3 == 12, "")

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

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
