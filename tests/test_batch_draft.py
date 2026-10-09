"""纯 CPU 测试（P6）：批量 draft 提议 + draft KV 有效水位状态机。

为什么要用 stub 模型
--------------------
真机 draft 前向要 GPU，没法在 CPU 上跑。但批量提议里真正容易错的**不是**
矩阵乘法，而是「逐请求元数据怎么落到每个 batch 行上」：

  · position / context_len / slot / block_table 必须按行算对
    （一条请求的 slot 写到另一条请求的物理块 = 静默写坏别人的 KV）
  · 温度必须沿【请求维】广播（历史 bug：把 (B,) 塞进 vocab 维）
  · 补齐（catch-up）必须把缺口里的 token 全补上，且按位置顺序
  · 候选 / logits 的形状与持久化（[B,k] 与 [B,k,V]）

所以这里用一个 **确定性 stub draft 模型**：它把 context 原样记下来、
logits 是位置与 token 的尖锐函数（argmax 恒等于真峰值，采样因此可复现，
k 步不会因为随机采样而分叉）。这样上面每一条都能在 CPU 上被直接断言。

跑：python tests/test_batch_draft.py
"""
import os
import sys
import types
import importlib.util

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scheduler 会 import nanovllm.config（依赖 transformers）——塞占位模块，
# 和 test_prefix_hash / test_spec_gate 一致。
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


_dp = _load("batch_draft_proposer", "nanovllm/spec_decode/draft_proposer.py")
DraftModelProposer = _dp.DraftModelProposer
catchup_gap = _dp.catchup_gap

from nanovllm.engine.block_manager import BlockManager      # noqa: E402
from nanovllm.engine.sequence import Sequence               # noqa: E402
from nanovllm.engine.scheduler import Scheduler             # noqa: E402
from nanovllm.sampling_params import SamplingParams         # noqa: E402

BS = 4
EOS = 999999
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ======================================================================
class StubDraft(nn.Module):
    """确定性 stub：记录每次前向看到的 context，logits = 位置+token 的尖锐峰。"""

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
            is_prefill=bool(c.is_prefill),
        ))
        B = input_ids.shape[0]
        return torch.stack([positions.float(), input_ids.float(),
                            torch.zeros(B)], dim=-1)

    def compute_logits(self, h):
        pos = h[:, 0].long()
        tok = h[:, 1].long()
        peak = (pos + tok) % self.vocab_size
        v = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(0)
        # 尖锐峰：峰 0、邻项 -50 → 采样恒等于 argmax，k 步不会随机分叉
        return -50.0 * (v - peak.unsqueeze(1).float()) ** 2 + self._bias


def mk_prop(vocab=32, k=4, bs=4):
    m = StubDraft(vocab)
    p = DraftModelProposer(m, k=k, block_size=bs)
    return m, p


def mk_req(bt, ctx_len, last, temp=1.0, gap_start=None, gap=None):
    return dict(block_table=list(bt), context_len=ctx_len, last_token=last,
                temperature=temp,
                catchup_start=ctx_len - 1 if gap_start is None else gap_start,
                catchup_tokens=list(gap or []))


print("=" * 70)
print("§1 批量 draft 与逐请求 draft：同一已确认前缀上逐步 logits 一致")
print("=" * 70)

B, K = 3, 4
bt = [[11, 12, 13, 14, 15, 16]] * B          # 每个块 4 个位置
ctx = [9, 6, 12]                              # 三条请求各自的已确认长度（上下文不齐）
lasts = [5, 6, 7]
reqs = [mk_req(bt[i], ctx[i], lasts[i]) for i in range(B)]

m1, p1 = mk_prop(k=K, bs=BS)
chains_b, logits_b = p1.propose_batch(reqs)

m2, p2 = mk_prop(k=K, bs=BS)
chains_r, logits_r = [], []
for r in reqs:
    ch, lg = p2.propose_batch([r])
    chains_r.append(ch[0])
    logits_r.append(lg[0])
logits_r = torch.stack(logits_r, 0)

check("① 候选形状 [B,k]", len(chains_b) == B and all(len(c) == K for c in chains_b),
      f"{[len(c) for c in chains_b]}")
check("① logits 形状 [B,k,V]",
      tuple(logits_b.shape) == (B, K, m1.vocab_size), f"{tuple(logits_b.shape)}")
check("① 批量 vs 逐请求 逐步 logits 完全一致（同输入同前缀）",
      torch.equal(logits_b, logits_r),
      f"max_abs={(logits_b - logits_r).abs().max().item():.3e}")
check("① 批量 vs 逐请求 候选 token 完全一致",
      chains_b == chains_r, f"{chains_b} vs {chains_r}")
check("① 每轮 draft 前向次数 = k（批量）",
      p1.n_batch_forwards == K and p1.n_rounds == 1,
      f"batch_forwards={p1.n_batch_forwards} rounds={p1.n_rounds}")
check("① 逐请求是 B×k 次前向",
      p2.n_batch_forwards == B * K and p2.n_rounds == B,
      f"batch_forwards={p2.n_batch_forwards} rounds={p2.n_rounds}")

print()
print("=" * 70)
print("§2 逐请求元数据落到正确的 batch 行（position / slot / ctx / block_table）")
print("=" * 70)

ok_pos, ok_slot, ok_clen, ok_bt = True, True, True, True
for s in range(K):
    call = m1.seen[s]
    for i in range(B):
        pos = ctx[i] - 1 + s
        want_slot = bt[i][pos // BS] * BS + pos % BS
        ok_pos &= call["positions"][i] == pos
        ok_slot &= call["slot"][i] == want_slot
        ok_clen &= call["clen"][i] == pos + 1
        ok_bt &= call["bt"][i][:len(bt[i])] == bt[i]
check("② 位置按行（不同请求可以处在不同 position）", ok_pos,
      f"step0 positions={m1.seen[0]['positions']}，期望={[ctx[i]-1 for i in range(B)]}")
check("② 物理槽按行 = block_table[pos//bs]*bs + pos%bs", ok_slot,
      f"step0 slot={m1.seen[0]['slot']}")
check("② context_lens 按行 = pos+1", ok_clen, f"{m1.seen[0]['clen']}")
check("② block_table 是 [B,max_blocks] 且每行是【自己】的块表", ok_bt,
      f"{m1.seen[0]['bt']}")
check("② slot 互不相同（没有把一条请求的槽写到另一条）",
      len(set(m1.seen[0]["slot"])) == B, f"{m1.seen[0]['slot']}")
check("② 走的是 decode 路径（is_prefill=False）",
      all(call["is_prefill"] is False for call in m1.seen))

print()
print("=" * 70)
print("§3 catch-up：补齐缺口里的【全部】已确认 token，按位置顺序")
print("=" * 70)

m3, p3 = mk_prop(k=2, bs=BS)
# A：水位 3，长度 10 → 缺口 = token_ids[3..8]，共 6 个（位置 3..8）
# B：水位 == len-1 → 没有缺口
reqA = mk_req(bt[0], 10, 5, gap_start=3, gap=list(range(100, 106)))
reqB = mk_req(bt[0], 6, 6, gap_start=5, gap=[])
p3.propose_batch([reqA, reqB])

# 第 0 轮只有 A 有缺口（B 已补齐）→ 只前向 1 行；共 6 轮
check("③ 补齐轮数 = 最长缺口长度（6）", p3.n_catchup_forwards == 6,
      f"n_catchup_forwards={p3.n_catchup_forwards}")
check("③ 补齐 token 数计入统计（6）", p3.n_catchup_tokens == 6,
      f"n_catchup_tokens={p3.n_catchup_tokens}")
gap_calls = m3.seen[:6]
check("③ 补齐按位置顺序、token 一一对应",
      [c["tokens"][0] for c in gap_calls] == list(range(100, 106))
      and [c["positions"][0] for c in gap_calls] == list(range(3, 9)),
      f"tokens={[c['tokens'][0] for c in gap_calls]} pos={[c['positions'][0] for c in gap_calls]}")
check("③ 补齐只带缺口那一行（B 不进这一轮）",
      all(len(c["tokens"]) == 1 for c in gap_calls),
      f"{[c['tokens'] for c in gap_calls]}")
# 补齐之后是 k 步提议：第 0 步的位置必须是各自 len-1
main0 = m3.seen[6]
check("③ 补齐后提议从 len-1 开始（A len=10 → 9，B len=6 → 5）",
      main0["positions"] == [9, 5], f"{main0['positions']}")
check("③ 提议阶段是 2 行批量（B 一起）", len(main0["tokens"]) == 2, f"{main0['tokens']}")

# 水位刚好等于 len-1 → 不补齐
m4, p4 = mk_prop(k=1, bs=BS)
p4.propose_batch([mk_req(bt[0], 5, 9, gap_start=4, gap=[])])
check("③ 水位 == len-1（无缺口）时不补齐",
      p4.n_catchup_forwards == 0 and p4.n_catchup_tokens == 0,
      f"{p4.n_catchup_forwards}/{p4.n_catchup_tokens}")

print()
print("=" * 70)
print("§4 catchup_gap 纯逻辑（水位 → 缺口位置与 token）")
print("=" * 70)

s0, t0 = catchup_gap(0, [1, 2, 3, 4, 5])
check("④ 水位 0、长度 5 → 补 [0,4) 全部（最后一个 token 由提议第 0 步负责）",
      s0 == 0 and t0 == [1, 2, 3, 4], f"start={s0} tokens={t0}")
s1, t1 = catchup_gap(4, [1, 2, 3, 4, 5])
check("④ 水位 4、长度 5 → 无缺口", s1 == 4 and t1 == [], f"start={s1} tokens={t1}")
s2, t2 = catchup_gap(99, [1, 2, 3, 4, 5])
check("④ 水位越界时夹到 len-1（不越界读）", s2 == 4 and t2 == [], f"start={s2} tokens={t2}")
s3, t3 = catchup_gap(2, [7, 8])
check("④ 长度 2、水位 2 → 无缺口", s3 == 1 and t3 == [], f"start={s3} tokens={t3}")

print()
print("=" * 70)
print("§5 B=1 兼容入口 + 温度沿请求维广播")
print("=" * 70)

m5, p5 = mk_prop(k=3, bs=BS)
chain, logits = p5.propose(bt[0], 7, 3, 1.0)
check("⑤ propose() 返回 (k 个候选, [k,V] logits)",
      len(chain) == 3 and tuple(logits.shape) == (3, m5.vocab_size),
      f"chain={chain} shape={tuple(logits.shape)}")

# 温度广播：B=3、vocab=32（V≠B）时不能把 (B,) 塞进 vocab 维 —— 那会直接报错。
# 再用同一 seed 手工复算，验证【逐行】用的是各自的温度。
m6, p6 = mk_prop(k=1, bs=BS)
temps = [0.25, 1.0, 8.0]
reqs6 = [mk_req(bt[0], 7 + i, 3 + i, temp=temps[i]) for i in range(3)]
torch.manual_seed(20261009)
chains6, logits6 = p6.propose_batch(reqs6)
torch.manual_seed(20261009)
tcol = torch.tensor(temps, dtype=torch.float32).view(-1, 1)
p_manual = torch.softmax(logits6[:, 0].float() / tcol, dim=-1)
noise = torch.empty_like(p_manual).exponential_(1.0).clamp_min_(1e-10)
expect = (p_manual / noise).argmax(dim=-1).tolist()
check("⑤ 温度按【请求维】广播（同 seed 手工复算逐行一致）",
      [c[0] for c in chains6] == expect,
      f"got={[c[0] for c in chains6]} expect={expect}")
check("⑤ B=3 / vocab=32（V≠B）不报错", True)

print()
print("=" * 70)
print("§6 draft_valid_len 状态机：prefill / 提议 / 全拒 / 抢占 / 序列化")
print("=" * 70)

Sequence.block_size = BS
seq = Sequence(list(range(6)), SamplingParams(temperature=1.0, max_tokens=10 ** 6))
check("⑥ 新建 Sequence 的 draft_valid_len = 0", seq.draft_valid_len == 0,
      f"{seq.draft_valid_len}")

# --- scheduler 侧：postprocess_spec 把水位夹到「已确认前缀」 ---
s = Scheduler.__new__(Scheduler)
s.config = types.SimpleNamespace(spec_k=2, spec_batch_threshold=0)
s.block_manager = BlockManager(64, BS)
s.spec_proposer = None
s.token_hook = None
s.spec_k = 2
s.spec_batch_threshold = 0
s.eos = EOS
from collections import deque                                      # noqa: E402
s.waiting = deque()
s.running = deque()
seq2 = Sequence(list(range(4)), SamplingParams(temperature=1.0, max_tokens=10 ** 6))
s.block_manager.allocate(seq2, 0)
s.running.append(seq2)
seq2.num_scheduled_tokens = 4
Scheduler.postprocess(s, [seq2], [42], True)
check("⑥ prefill 后 num_cached_tokens=4", seq2.num_cached_tokens == 4, f"{seq2.num_cached_tokens}")

# 模拟 proposer 写完后把水位抬到 len-1+k（引擎里由 model_runner.run 设置）
k = 2
seq2.draft_valid_len = len(seq2) - 1 + k          # = 5-1+2 = 6
seq2.num_scheduled_tokens = 1 + k
s.block_manager.may_append(seq2, 1 + k)
Scheduler.postprocess_spec(s, [seq2], [[101, 102, 103]])   # 全接受 + bonus
# 全部接受（a=k）时，最后一个候选的 KV【从未写过】（propose 只写到 len+k-2），
# 所以真实连续有效前缀 = num_tokens-2，缺口正好是那 1 个 bonus 位。
gap_start, gap = catchup_gap(seq2.draft_valid_len, seq2.token_ids)
check("⑥ 全部接受后水位 = num_tokens-2，缺口 = 1 个位置（bonus 位）",
      seq2.draft_valid_len == seq2.num_tokens - 2 and len(gap) == 1,
      f"valid={seq2.draft_valid_len} num_tokens={seq2.num_tokens} gap={gap}")

# 部分接受：只有第 1 个候选被接受 + bonus → 交付 2 个；缺口应为 0
seq2.draft_valid_len = len(seq2) - 1 + k
seq2.num_scheduled_tokens = 1 + k
s.block_manager.may_append(seq2, 1 + k)
Scheduler.postprocess_spec(s, [seq2], [[201, 202]])
gap_start, gap = catchup_gap(seq2.draft_valid_len, seq2.token_ids)
check("⑥ 部分接受后水位 = num_tokens-1，缺口 = 0",
      seq2.draft_valid_len == seq2.num_tokens - 1 and len(gap) == 0,
      f"valid={seq2.draft_valid_len} num_tokens={seq2.num_tokens} gap={gap}")

# 一个都没接受（只有 bonus）→ 缺口也是 0（被拒候选的槽会被第 0 步覆写）
seq2.draft_valid_len = len(seq2) - 1 + k
seq2.num_scheduled_tokens = 1 + k
s.block_manager.may_append(seq2, 1 + k)
Scheduler.postprocess_spec(s, [seq2], [[301]])
gap_start, gap = catchup_gap(seq2.draft_valid_len, seq2.token_ids)
check("⑥ 全拒（仅 bonus）后水位 = num_tokens-1，缺口 = 0",
      seq2.draft_valid_len == seq2.num_tokens - 1 and len(gap) == 0,
      f"valid={seq2.draft_valid_len} num_tokens={seq2.num_tokens} gap={gap}")

# 抢占 → 归零
Scheduler.preempt(s, seq2)
check("⑥ 抢占后 draft_valid_len 归零", seq2.draft_valid_len == 0,
      f"{seq2.draft_valid_len}")

# 序列化：draft_valid_len 必须过进程边界（TP>1）
seq2.draft_valid_len = 123
import pickle                                                    # noqa: E402
seq3 = pickle.loads(pickle.dumps(seq2))
check("⑥ 序列化往返保留 draft_valid_len", seq3.draft_valid_len == 123,
      f"{seq3.draft_valid_len}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
