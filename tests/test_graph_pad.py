"""纯 CPU 测试（P16）：投机路径 CUDA graph 的「未精确捕获 batch size」向上取整。

为什么这些能、也应该在 CPU 上测
------------------------------
P16 新增的逻辑里真正容易错的两块 —— **桶选择** 和 **空行的元数据** ——
和 CUDA 一点关系都没有：

  · 桶选错（比如把 B=5 塞进 B=4 的图）→ 写越界；
  · 空行 slot 填了真实槽号 → 静默写坏别人的 KV；
  · 空行 block_table 留 -1 → flash-attn 读第 -1 页，非法访存；
  · 空行 cu_seqlens 给 0 或步长不齐 → 内核退化 / 形状对不上；
  · 交出去的 logits 没按真实行数切 → run_verify 的断言直接炸。

所以这里用 **CPU 张量** 把上面每一条都断言掉。真图在 CPU 上拍不了，
但 `_forward_group` 的「写静态缓冲区」这一步是纯张量操作，可以完整验证 ——
下面用 `FakeGraph` 把 replay 换成「用静态缓冲区真的跑一次 stub 前向」，
于是「pad 行有没有被写对」能被端到端看到。

跑：python tests/test_graph_pad.py
"""
import os
import sys
import types
import importlib.util

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# draft_proposer 会 import nanovllm.config（依赖 transformers）——塞占位模块，
# 和 test_batch_draft / test_spec_gate 一致。
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)
_cfg = types.ModuleType("nanovllm.config")
_cfg.Config = object
sys.modules.setdefault("nanovllm.config", _cfg)


def _load(mod_name, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gp = _load("p16_graph_pad", "nanovllm/spec_decode/graph_pad.py")
pick_bucket = _gp.pick_bucket
pad_cu_seqlens = _gp.pad_cu_seqlens
draft_pad_rows = _gp.draft_pad_rows
fill_verify_static = _gp.fill_verify_static
PAD_SLOT = _gp.PAD_SLOT
PAD_ROW_BLOCK = _gp.PAD_ROW_BLOCK
PAD_CTX_LEN = _gp.PAD_CTX_LEN
PAD_TOKEN = _gp.PAD_TOKEN

_dp = _load("p16_draft_proposer", "nanovllm/spec_decode/draft_proposer.py")
DraftModelProposer = _dp.DraftModelProposer

FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ======================================================================
#  1. 桶选择
# ======================================================================
print("=" * 70)
print("§1 桶选择：向上取整到最近的桶")
print("=" * 70)

BUCKETS = (1, 2, 4)
check("① B=1（最小桶）→ 1，pad=0",
      pick_bucket(1, BUCKETS) == 1)
check("① B=2（恰好等于桶）→ 2，pad=0",
      pick_bucket(2, BUCKETS) == 2)
check("① B=3（没捕获）→ 4，pad=1  ★ 本项优化的目标形状",
      pick_bucket(3, BUCKETS) == 4)
check("① B=4（最大桶，恰好等于）→ 4，pad=0",
      pick_bucket(4, BUCKETS) == 4)
check("① B=5（超过所有桶）→ None ⇒ 必须退回 eager",
      pick_bucket(5, BUCKETS) is None)
check("① B=0/负数 → 最小桶（不会出现在正常路径，但不能崩）",
      pick_bucket(0, BUCKETS) == 1 and pick_bucket(-3, BUCKETS) == 1)
check("① 桶集合为空 → None（没图 ⇒ 退回 eager）",
      pick_bucket(1, ()) is None)
check("① 桶乱序 / 含重复 → 仍取最小可用桶",
      pick_bucket(3, (8, 4, 1, 4, 2)) == 4 and pick_bucket(6, (8, 4, 1)) == 8)
check("① 单桶集合（如 spec_graph_bs=(1,)）：B=1 命中、B=2 无桶",
      pick_bucket(1, (1,)) == 1 and pick_bucket(2, (1,)) is None)

# ======================================================================
#  2. 空行元数据：四个安全常量 + draft 图的空行张量
# ======================================================================
print()
print("=" * 70)
print("§2 空行元数据：每一条都对应一个安全不变量")
print("=" * 70)

check("② PAD_SLOT = -1（store_kvcache 见到 -1 直接 return，不写 KV）",
      PAD_SLOT == -1, f"{PAD_SLOT}")
check("② PAD_ROW_BLOCK = 0（块 0 永远已分配；-1 会非法访存）",
      PAD_ROW_BLOCK == 0, f"{PAD_ROW_BLOCK}")
check("② PAD_CTX_LEN = 0（归零：空行读不到任何 key，与 vLLM 的 seq_lens=0 一致）",
      PAD_CTX_LEN == 0, f"{PAD_CTX_LEN}")
check("② PAD_TOKEN = 0（合法值，词表内）",
      PAD_TOKEN == 0, f"{PAD_TOKEN}")
check("② PAD_Q_LEN 默认 None（= 与真实行同宽 nq，保持 cu_seqlens_q 是等差数列）",
      _gp.PAD_Q_LEN is None, f"{_gp.PAD_Q_LEN}")

rows = draft_pad_rows(3, torch.device("cpu"))
check("② draft_pad_rows 的 key 与图静态缓冲区同名",
      set(rows) == {"input_ids", "positions", "slot_mapping", "context_lens"},
      f"{sorted(rows)}")
check("② 每个空行张量长度 = pad",
      all(v.shape == (3,) for v in rows.values()),
      f"{ {k: tuple(v.shape) for k, v in rows.items()} }")
check("② 空行 dtype：token/position int64，slot/ctx int32",
      rows["input_ids"].dtype == torch.int64 and rows["positions"].dtype == torch.int64
      and rows["slot_mapping"].dtype == torch.int32
      and rows["context_lens"].dtype == torch.int32)
check("② 空行 slot 全 -1（不写 KV）",
      bool((rows["slot_mapping"] == -1).all()),
      f"{rows['slot_mapping'].tolist()}")
check("② 空行 context_len 全 0（归零：读不到任何 key）",
      bool((rows["context_lens"] == 0).all()),
      f"{rows['context_lens'].tolist()}")
check("② 空行 token / position 全 0",
      bool((rows["input_ids"] == 0).all()) and bool((rows["positions"] == 0).all()))
check("② pad=0 → 空 dict（老路径不产生任何额外张量）",
      draft_pad_rows(0, torch.device("cpu")) == {}
      and draft_pad_rows(-1, torch.device("cpu")) == {})

# ======================================================================
#  3. 空行的 cu_seqlens
# ======================================================================
print()
print("=" * 70)
print("§3 验证图空行的 cu_seqlens：单调 + 步长自洽 + key 侧归零")
print("=" * 70)

nq = 7                                     # = 1 + k，k=6
B, pad = 3, 1
cu_q_real = torch.tensor([i * nq for i in range(B + 1)], dtype=torch.int32)
cu_k_real = torch.tensor([0, 100, 250, 700], dtype=torch.int32)   # 各请求上下文长度不齐

cu_q, cu_k = pad_cu_seqlens(cu_q_real, cu_k_real, nq, pad)
check("③ 长度 = B+pad+1",
      cu_q.numel() == B + pad + 1 and cu_k.numel() == B + pad + 1,
      f"{cu_q.numel()}")
check("③ dtype/device 与输入一致",
      cu_q.dtype == cu_q_real.dtype and cu_q.device == cu_q_real.device)
check("③ 真实前缀原样保留（切 logits 靠它）",
      torch.equal(cu_q[:B + 1], cu_q_real) and torch.equal(cu_k[:B + 1], cu_k_real))
check("③ query 步长恒为 nq（形状与捕获时逐字一致）",
      cu_q[1:].sub(cu_q[:-1]).eq(nq).all().item(),
      f"{cu_q.tolist()}")
check("③ ★ key 侧单调**不减**（cu_seqlens 是前缀和，flash-attn 的硬要求）"
      "—— 归零后是等值台阶，仍然合法",
      bool((cu_k[1:] >= cu_k[:-1]).all()),
      f"{cu_k.tolist()}")
check("③ 空行的 key 长度 = PAD_CTX_LEN = 0（归零）",
      int(cu_k[-1] - cu_k[B]) == PAD_CTX_LEN == 0,
      f"空行段长={int(cu_k[-1] - cu_k[B])}")
check("③ 多补几行也成立（pad=4：4 个等值台阶，仍单调）",
      (lambda t: bool((t[1:] >= t[:-1]).all()) and int(t[-1] - t[B]) == 0
       and t.numel() == B + 5)(pad_cu_seqlens(cu_q_real, cu_k_real, nq, 4)[1]))
check("③ pad=0 → 原样返回（同一对象，老路径零开销）",
      pad_cu_seqlens(cu_q_real, cu_k_real, nq, 0)[0] is cu_q_real)
check("③ 不引入 D2H 同步：输出 device 跟随输入（用 meta 设备验证）",
      pad_cu_seqlens(cu_q_real.to("meta"), cu_k_real.to("meta"), nq, 2)[0].device.type
      == "meta")
check("③ PAD_Q_LEN=0（vLLM 的 query_start_loc 做法）也能构造：q 侧同样是等值台阶",
      (lambda: (
          setattr(_gp, "PAD_Q_LEN", 0),
          (lambda t: bool((t[1:] >= t[:-1]).all())
           and int(t[-1] - t[B]) == 0)(pad_cu_seqlens(cu_q_real, cu_k_real, nq, 2)[0]),
          setattr(_gp, "PAD_Q_LEN", None),
      )[-2])())

# ======================================================================
#  4. 验证图静态缓冲区：真实行 + 空行
# ======================================================================
print()
print("=" * 70)
print("§4 验证图静态缓冲区：真实行原样、空行安全、返回值切到真实行")
print("=" * 70)


def mk_verify_gv(Bg, nq, W):
    """照 capture_verify_cudagraph 的写法造一份静态缓冲区（CPU 版）。"""
    return dict(
        input_ids=torch.zeros(Bg * nq, dtype=torch.int64),
        positions=torch.zeros(Bg * nq, dtype=torch.int64),
        slot_mapping=torch.zeros(Bg * nq, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([i * nq for i in range(Bg + 1)], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([i * nq for i in range(Bg + 1)], dtype=torch.int32),
        block_tables=torch.zeros(Bg, W, dtype=torch.int32),
        logits=torch.zeros(Bg * nq, 8, dtype=torch.float32),
    )


B, pad, nq, W = 3, 1, 7, 6
Bg = B + pad
gv = mk_verify_gv(Bg, nq, W)
Q = B * nq
real_ids = torch.arange(100, 100 + Q, dtype=torch.int64)
real_pos = torch.arange(1000, 1000 + Q, dtype=torch.int64)
real_slot = torch.arange(2000, 2000 + Q, dtype=torch.int32)
real_cu_q = torch.tensor([i * nq for i in range(B + 1)], dtype=torch.int32)
real_cu_k = torch.tensor([0, 5, 33, 400], dtype=torch.int32)
real_bt = torch.tensor([[1, 2, 3, 4, 5, 6],
                        [7, 8, 9, 10, 11, 12],
                        [13, 14, 15, 16, 17, 18]], dtype=torch.int32)

ret = fill_verify_static(gv, B, nq, pad, real_ids, real_pos, real_slot,
                         real_cu_q, real_cu_k, real_bt)

check("④ 返回值 = 真实行数 Q = B*nq（调用方靠它切 logits）",
      ret == Q == 21, f"{ret}")
check("④ 真实行的 input_ids / positions / slot_mapping 原样拷入",
      torch.equal(gv["input_ids"][:Q], real_ids)
      and torch.equal(gv["positions"][:Q], real_pos)
      and torch.equal(gv["slot_mapping"][:Q], real_slot))
check("④ 空行 input_ids / positions 全 0",
      bool((gv["input_ids"][Q:] == 0).all()) and bool((gv["positions"][Q:] == 0).all()))
check("④ 空行 slot_mapping 全 -1（★ 不写 KV —— 写了就是污染别人的 KV）",
      bool((gv["slot_mapping"][Q:] == -1).all()),
      f"{gv['slot_mapping'][Q:].tolist()}")
check("④ 真实行 block_table 原样拷入",
      torch.equal(gv["block_tables"][:B], real_bt))
check("④ 空行 block_table 全 = 块 0（★ 留 -1 会非法访存）",
      bool((gv["block_tables"][B:] == 0).all()),
      f"{gv['block_tables'][B:].tolist()}")
check("④ cu_seqlens_q 真实前缀保留 + 空行步长仍为 nq",
      torch.equal(gv["cu_seqlens_q"][:B + 1], real_cu_q)
      and bool(gv["cu_seqlens_q"][1:].sub(gv["cu_seqlens_q"][:-1]).eq(nq).all()),
      f"{gv['cu_seqlens_q'].tolist()}")
check("④ cu_seqlens_k 真实前缀保留 + 空行各 0 个 key + 仍单调",
      torch.equal(gv["cu_seqlens_k"][:B + 1], real_cu_k)
      and int(gv["cu_seqlens_k"][-1] - gv["cu_seqlens_k"][B]) == 0
      and bool((gv["cu_seqlens_k"][1:] >= gv["cu_seqlens_k"][:-1]).all()),
      f"{gv['cu_seqlens_k'].tolist()}")

# 反复 replay（每步都会重写缓冲区）：上一轮的脏数据不能残留
gv["block_tables"][B:] = 99
gv["slot_mapping"][Q:] = 77
fill_verify_static(gv, B, nq, pad, real_ids, real_pos, real_slot,
                   real_cu_q, real_cu_k, real_bt)
check("④ 第二步重写后无残留（空行块表不残留上一轮的 99）",
      bool((gv["block_tables"][B:] == 0).all())
      and bool((gv["slot_mapping"][Q:] == -1).all()))

# block_tables 为 None（纯 query、无前缀）时不能崩
gv2 = mk_verify_gv(Bg, nq, W)
fill_verify_static(gv2, B, nq, pad, real_ids, real_pos, real_slot,
                   real_cu_q, real_cu_k, None)
check("④ block_tables=None（无前缀缓存）时不崩，且全为块 0",
      bool((gv2["block_tables"] == 0).all()))

# 边界：B=1、pad=3（B=1 用 B=4 的图）
gv3 = mk_verify_gv(4, nq, W)
r3 = fill_verify_static(gv3, 1, nq, 3,
                        real_ids[:nq], real_pos[:nq], real_slot[:nq],
                        torch.tensor([0, nq], dtype=torch.int32),
                        torch.tensor([0, 12], dtype=torch.int32),
                        torch.tensor([[5, 6, 7, 8, 9, 10]], dtype=torch.int32))
check("④ 边界 B=1、pad=3：真实行 7 行、空行 21 行、cu 全单调",
      r3 == nq and torch.equal(gv3["input_ids"][:nq], real_ids[:nq])
      and bool((gv3["input_ids"][nq:] == 0).all())
      and bool((gv3["cu_seqlens_k"][1:] >= gv3["cu_seqlens_k"][:-1]).all()),
      f"cu_k={gv3['cu_seqlens_k'].tolist()}")
# ======================================================================
#  5. draft 图：_forward_group 的端到端 wiring
# ======================================================================
print()
print("=" * 70)
print("§5 draft 图：_forward_group 的 padding wiring（CPU 上真跑一遍）")
print("=" * 70)


class StubDraft(nn.Module):
    """确定性 stub：把每次前向看到的 context 记下来，logits = (位置+token) 的尖峰。"""

    def __init__(self, vocab=32):
        super().__init__()
        self.vocab_size = vocab
        self._bias = nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, input_ids, positions):
        from nanovllm.utils.context import get_context
        c = get_context()
        self.seen.append(dict(
            tokens=[int(x) for x in input_ids.tolist()],
            positions=[int(x) for x in positions.tolist()],
            slot=[int(x) for x in c.slot_mapping.tolist()],
            clen=[int(x) for x in c.context_lens.tolist()],
            bt=[[int(y) for y in row] for row in c.block_tables.tolist()],
        ))
        B = input_ids.shape[0]
        return torch.stack([positions.float(), input_ids.float(),
                            torch.zeros(B)], dim=-1)

    def compute_logits(self, h):
        pos = h[:, 0].long()
        tok = h[:, 1].long()
        peak = (pos + tok) % self.vocab_size
        v = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(0)
        return -50.0 * (v - peak.unsqueeze(1).float()) ** 2 + self._bias


class FakeGraph:
    """CUDA graph 的 CPU 替身：replay() = 用静态缓冲区真的跑一次前向。

    ★ 为什么需要它：真图在 CPU 上拍不了，但「pad 行有没有被写对」与 CUDA 无关
      —— 只要 replay 时真的按静态缓冲区去算一遍，就能在 CPU 上完整验证。
    """

    def __init__(self, model, gv):
        self.model = model
        self.gv = gv
        self.replays = 0

    def replay(self):
        from nanovllm.utils.context import set_context, reset_context
        set_context(False, slot_mapping=self.gv["slot_mapping"],
                    context_lens=self.gv["context_lens"],
                    block_tables=self.gv["block_tables"])
        try:
            out = self.model.compute_logits(
                self.model(self.gv["input_ids"], self.gv["positions"]))
        finally:
            reset_context()
        self.gv["logits"].copy_(out)
        self.replays += 1


def mk_graph(model, B, W, vocab=32):
    """照 capture_draft_cudagraph 的写法造一份静态缓冲区 + FakeGraph。"""
    gv = dict(
        input_ids=torch.zeros(B, dtype=torch.int64),
        positions=torch.zeros(B, dtype=torch.int64),
        slot_mapping=torch.zeros(B, dtype=torch.int32),
        context_lens=torch.zeros(B, dtype=torch.int32),
        block_tables=torch.zeros(B, W, dtype=torch.int32),
        logits=torch.empty(B, vocab),
    )
    return FakeGraph(model, gv), gv


VOCAB, W = 32, 6
tokens = torch.tensor([5, 6, 7], dtype=torch.int64)
positions = [11, 20, 31]
slot_mapping = [101, 202, 303]
context_lens = [12, 21, 32]
block_tables = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]

# ---- 5a. 老行为：pad 关着 + B 未捕获 ⇒ 退回 eager ----
m_a = StubDraft(VOCAB)
p_a = DraftModelProposer(m_a, k=4, block_size=4)
g_a, gv_a = mk_graph(m_a, 4, W, VOCAB)
p_a.bind_cudagraph({4: (g_a, gv_a)}, graph_pad=False)
out_a = p_a._forward_group(tokens, positions, block_tables, context_lens, slot_mapping)
check("⑤ 关着时 B=3 未捕获 → 退回 eager（老行为不变）",
      p_a.n_eager == 1 and p_a.n_graphed == 0 and g_a.replays == 0,
      f"eager={p_a.n_eager} graphed={p_a.n_graphed}")
check("⑤ eager 返回 [B,V] 且非静态缓冲区",
      tuple(out_a.shape) == (3, VOCAB) and out_a.data_ptr() != gv_a["logits"].data_ptr())

# ---- 5b. 开着 + B=3 用 B=4 的图 ----
m_b = StubDraft(VOCAB)
p_b = DraftModelProposer(m_b, k=4, block_size=4)
g_b, gv_b = mk_graph(m_b, 4, W, VOCAB)
p_b.bind_cudagraph({4: (g_b, gv_b)}, graph_pad=True)
out_b = p_b._forward_group(tokens, positions, block_tables, context_lens, slot_mapping)
check("⑤ 开着时 B=3 用 B=4 的图（n_graphed=1、n_padded=1）",
      p_b.n_graphed == 1 and p_b.n_padded == 1 and g_b.replays == 1,
      f"graphed={p_b.n_graphed} padded={p_b.n_padded}")
check("⑤ 只交真实行：返回 shape = [B,V]（不是 [B+pad,V]）",
      tuple(out_b.shape) == (3, VOCAB), f"{tuple(out_b.shape)}")
check("⑤ ★ 真实行 logits 与 eager 路径逐元素相同",
      torch.equal(out_b, out_a),
      f"max_abs={(out_b - out_a).abs().max().item():.3e}")

seen = m_b.seen[-1]
check("⑤ 真实行的 4 个元数据都按行写对",
      seen["tokens"][:3] == [5, 6, 7] and seen["positions"][:3] == positions
      and seen["slot"][:3] == slot_mapping and seen["clen"][:3] == context_lens,
      f"tokens={seen['tokens']} slot={seen['slot']}")
check("⑤ 空行 token/position 全 0",
      seen["tokens"][3] == 0 and seen["positions"][3] == 0)
check("⑤ 空行 slot = -1（★ 不写 KV）",
      seen["slot"][3] == -1, f"slot={seen['slot']}")
check("⑤ 空行 context_len = 0（归零：读不到任何 key）",
      seen["clen"][3] == 0, f"clen={seen['clen']}")
check("⑤ 空行 block_table 全 0 = 块 0（★ 留 -1 会非法访存）",
      seen["bt"][3] == [0] * W, f"bt[3]={seen['bt'][3]}")
check("⑤ 真实行 block_table 原样（未用列仍是 -1，与老路径一致）",
      seen["bt"][0] == block_tables[0] and seen["bt"][2] == block_tables[2])

# ---- 5c. 开着 + B 恰好等于桶 ⇒ 等价于精确命中（pad=0，不算 padded） ----
m_c = StubDraft(VOCAB)
p_c = DraftModelProposer(m_c, k=4, block_size=4)
g_c, gv_c = mk_graph(m_c, 4, W, VOCAB)
p_c.bind_cudagraph({3: (g_c, gv_c)}, graph_pad=True)
p_c._forward_group(tokens, positions, block_tables, context_lens, slot_mapping)
check("⑤ 边界 B 恰好等于桶（B=3 用 B=3 的图）：pad=0、不算 padded",
      p_c.n_padded == 0 and p_c.n_graphed == 1 and p_c.n_eager == 0,
      f"padded={p_c.n_padded} graphed={p_c.n_graphed}")

# ---- 5c'. 边界 B=max：B=4 用 B=4 的图 ----
m_c2 = StubDraft(VOCAB)
p_c2 = DraftModelProposer(m_c2, k=4, block_size=4)
g_c2, gv_c2 = mk_graph(m_c2, 4, W, VOCAB)
p_c2.bind_cudagraph({4: (g_c2, gv_c2)}, graph_pad=True)
out_c2 = p_c2._forward_group(torch.tensor([1, 2, 3, 4]), [1, 2, 3, 4],
                             block_tables[:3] + [block_tables[0]],
                             [5, 6, 7, 8], [9, 10, 11, 12])
check("⑤ 边界 B=max=4（最大桶）：pad=0、形状 [4,V]",
      p_c2.n_padded == 0 and p_c2.n_graphed == 1 and tuple(out_c2.shape) == (4, VOCAB),
      f"padded={p_c2.n_padded} shape={tuple(out_c2.shape)}")

# ---- 5d. 开着 + B 超过所有桶 ⇒ 仍然退回 eager ----
m_d = StubDraft(VOCAB)
p_d = DraftModelProposer(m_d, k=4, block_size=4)
g_d, gv_d = mk_graph(m_d, 4, W, VOCAB)
p_d.bind_cudagraph({4: (g_d, gv_d)}, graph_pad=True)
out_d = p_d._forward_group(torch.tensor([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5],
                           block_tables[:3] + [block_tables[0]] * 2,
                           [5, 6, 7, 8, 9], [9, 10, 11, 12, 13])
check("⑤ 边界 B=5 > 最大桶 4 → 退回 eager（绝不写越界）",
      p_d.n_eager == 1 and p_d.n_graphed == 0 and tuple(out_d.shape) == (5, VOCAB),
      f"eager={p_d.n_eager}")

# ---- 5e. 连续多轮 replay：上一轮的脏数据不能残留 ----
p_b._forward_group(torch.tensor([9, 8, 7]), [1, 2, 3], block_tables,
                   [2, 3, 4], [11, 22, 33])
seen2 = m_b.seen[-1]
check("⑤ 第二轮 replay：空行元数据仍正确（无上一轮残留）",
      seen2["slot"][3] == -1 and seen2["clen"][3] == 0
      and seen2["bt"][3] == [0] * W and seen2["tokens"][3] == 0)
check("⑤ 第二轮真实行已更新为新输入",
      seen2["tokens"][:3] == [9, 8, 7] and seen2["slot"][:3] == [11, 22, 33])

# ---- 5f. 块表宽度超过图的静态宽度 ⇒ 退回 eager（老保护不能失效） ----
m_f = StubDraft(VOCAB)
p_f = DraftModelProposer(m_f, k=4, block_size=4)
g_f, gv_f = mk_graph(m_f, 4, W, VOCAB)
p_f.bind_cudagraph({4: (g_f, gv_f)}, graph_pad=True)
out_f = p_f._forward_group(tokens, positions, [[1] * (W + 2)] * 3, context_lens,
                           slot_mapping)
check("⑤ 块表宽度超过静态宽度 → 退回 eager（padding 不能绕过这道保护）",
      p_f.n_eager == 1 and p_f.n_graphed == 0, f"eager={p_f.n_eager}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
