"""纯 CPU 回归（A 步）：draft 侧 KV 的有效性追踪 / 「前缀缓存命中 → 全量 draft 补齐」。

背景（本轮实测，3090 + Qwen3-4B/0.6B，4 条请求共享 4096-token 前缀）
------------------------------------------------------------------
前缀缓存命中时 `Sequence.draft_valid_len` 被保守地压在 0，于是 propose 之前
必须把 `[0, len(seq)-1)` 整段 token 逐个喂回 draft 模型：

    补齐 4179 次 draft 前向 / 16667 个 token，单个 decode step 17448 ms，
    占端到端墙钟 91%；吞吐 12.4 tok/s（关掉补齐的对照是 60~74 tok/s）。

为什么不能简单地「target 命中了就假定 draft 也有效」
----------------------------------------------------
`hash_to_block_id`（前缀缓存）存的是 **target 的** KV 块；draft 的 KV 写在
**另一套**物理缓冲里（每层 k_cache / v_cache 是独立张量）。两者只是共用同一份
`Sequence.block_table`（逻辑块 i → 物理块 i）。所以 target 命中时，draft 在
同一物理块里的内容是谁留下的都有可能。丢掉保守设定 → 草稿从被污染的上下文
继续预测 → 接受率悄悄变低，且【永远不报错】。

修复后的机制
------------
`Block.draft_hash` = 「draft 模型按正确前缀算过的内容」的块哈希。
  · 盖标记：`BlockManager.mark_draft_valid(seq, draft_valid_len)`
            —— 只有整块落在 draft 水位之内、且该块已登记进前缀缓存才盖。
  · 读标记：`BlockManager.draft_valid_cached_blocks(seq, num_cached_blocks)`
            —— 只认 `draft_hash == hash`，且只取【连续】前缀。
  · Scheduler 在 allocate 之后用它把 draft 水位推到连续有效块数 × block_size。

本文件覆盖这套状态机，包括【必须能失败】的负对照（§3/§4/§9）。
跑：python tests/test_draft_block_valid.py
"""
import os
import sys
import types
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scheduler / sequence 会 import nanovllm.config（依赖 transformers）——塞占位模块，
# 和 test_prefix_hash / test_spec_gate / test_batch_draft 一致。
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)
_cfg = types.ModuleType("nanovllm.config")
_cfg.Config = object
sys.modules.setdefault("nanovllm.config", _cfg)

from nanovllm.engine.block_manager import BlockManager      # noqa: E402
from nanovllm.engine.sequence import Sequence               # noqa: E402
from nanovllm.engine.scheduler import Scheduler             # noqa: E402
from nanovllm.sampling_params import SamplingParams         # noqa: E402

import importlib.util                                       # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dp_for_wm", os.path.join(ROOT, "nanovllm/spec_decode/draft_proposer.py"))
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
catchup_gap = _dp.catchup_gap

BS = 4                      # 小 block：几行就能跨块、跨水位
EOS = 999999
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ======================================================================
def mk_sched(num_blocks=64, spec_k=6):
    """绕开 Scheduler.__init__（它要真 Config），只铺 schedule()/postprocess 用到的字段。"""
    Sequence.block_size = BS
    s = Scheduler.__new__(Scheduler)
    s.config = types.SimpleNamespace(spec_k=spec_k, spec_batch_threshold=0)
    s.max_num_seqs = 8
    s.max_num_batched_tokens = 16384
    s.block_size = BS
    s.block_manager = BlockManager(num_blocks, BS)
    s.waiting = deque()
    s.running = deque()
    s.spec_batch_threshold = 0
    s.spec_proposer = None
    s.token_hook = None
    s.eos = EOS
    s.spec_k = spec_k            # property：直通 config.spec_k
    return s


def mk_seq(prompt):
    return Sequence(list(prompt), SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                 ignore_eos=True))


def emu_prefill_chunk(s, seq, chunk_end):
    """跑一个 prefill chunk 的【账务】，逐句镜像生产代码。

    镜像来源：
      model_runner.prepare_prefill  → seq.advance_draft_watermark(start, end)
      scheduler.postprocess         → num_cached_tokens += n
                                      block_manager.hash_blocks(seq, n)
                                      block_manager.mark_draft_valid(seq, 水位)
    （前向本身在 CPU 上跑不了，本测试只覆盖账务/状态机。）
    """
    start = seq.num_cached_tokens
    n = chunk_end - start
    assert n > 0, (start, chunk_end)
    seq.advance_draft_watermark(start, chunk_end)
    seq.num_cached_tokens += n
    s.block_manager.hash_blocks(seq, n)
    s.block_manager.mark_draft_valid(seq, seq.draft_valid_len)


# ======================================================================
print("§1 Block 生命周期：draft_hash 必须随块回收一起失效")
# ----------------------------------------------------------------------
bm = BlockManager(8, BS)
check("① 新建 Block 的 draft_hash = -1", bm.blocks[0].draft_hash == -1,
      f"{bm.blocks[0].draft_hash}")
bm.blocks[3].draft_hash = 12345
bm.blocks[3].reset()
check("① reset() 清掉 draft_hash", bm.blocks[3].draft_hash == -1,
      f"{bm.blocks[3].draft_hash}")

seq = mk_seq(range(3 * BS))
bm.allocate(seq, 0)
bid0 = seq.block_table[0]
bm.blocks[bid0].draft_hash = 777
bm.deallocate(seq)
b = bm._allocate_block()
check("① _allocate_block() 拿到的块 draft_hash = -1（旧标记不会漏到新用途）",
      bm.blocks[b].draft_hash == -1, f"block={b} draft_hash={bm.blocks[b].draft_hash}")

# ======================================================================
print("§2 mark_draft_valid：只有整块落在水位之内才盖")
# ----------------------------------------------------------------------
s = mk_sched()
seq = mk_seq(range(3 * BS + 1))          # 13 token → 4 个块，前 3 个整块
s.block_manager.allocate(seq, 0)
emu_prefill_chunk(s, seq, 13)            # 一步 prefill 完
bt = seq.block_table
dh = [s.block_manager.blocks[bt[i]].draft_hash for i in range(4)]
hh = [s.block_manager.blocks[bt[i]].hash for i in range(4)]
check("② 水位 = 13 → 整块 0..2 全部盖章", [d != -1 for d in dh] == [True, True, True, False],
      f"draft_hash={dh}")
check("② 章的取值就是该块的内容哈希", dh[:3] == hh[:3], f"{dh[:3]} vs {hh[:3]}")

s2 = mk_sched()
seq2 = mk_seq(range(3 * BS + 1))
s2.block_manager.allocate(seq2, 0)
emu_prefill_chunk(s2, seq2, 13)
for i in range(4):                                  # 抹掉重来，只留水位
    s2.block_manager.blocks[seq2.block_table[i]].draft_hash = -1
seq2.draft_valid_len = 10                           # 非整块倍数
s2.block_manager.mark_draft_valid(seq2, seq2.draft_valid_len)
dh2 = [s2.block_manager.blocks[seq2.block_table[i]].draft_hash for i in range(4)]
check("② 水位 10（2.5 块）→ 只盖 2 个完整块",
      [d != -1 for d in dh2] == [True, True, False, False], f"{dh2}")

seq3 = mk_seq(range(3 * BS + 1))                    # 水位 0 = 抢占后的状态
s3 = mk_sched()
s3.block_manager.allocate(seq3, 0)
emu_prefill_chunk(s3, seq3, 13)
for i in range(4):
    s3.block_manager.blocks[seq3.block_table[i]].draft_hash = -1
s3.block_manager.mark_draft_valid(seq3, 0)
check("② 水位 0 → 一个块都不盖", all(
    s3.block_manager.blocks[seq3.block_table[i]].draft_hash == -1 for i in range(4)))

# 未登记进前缀缓存的块跳过（块 2 只填了 half，hash 还是 -1）
s4 = mk_sched()
seq4 = mk_seq(range(3 * BS + 1))
s4.block_manager.allocate(seq4, 0)
seq4.num_cached_tokens = 2 * BS                     # 约定：先推进再登记（见 hash_blocks）
s4.block_manager.hash_blocks(seq4, 2 * BS)          # 只登记块 0/1，块 2 没登记
seq4.draft_valid_len = 3 * BS
s4.block_manager.mark_draft_valid(seq4, seq4.draft_valid_len)
dh4 = [s4.block_manager.blocks[seq4.block_table[i]].draft_hash for i in range(3)]
check("② 未登记进前缀缓存的块不盖（否则盖了也没人能命中）",
      [d != -1 for d in dh4] == [True, True, False], f"{dh4}")

# ======================================================================
print("§3 draft_valid_cached_blocks：只认连续有效的前缀（含负对照）")
# ----------------------------------------------------------------------
s = mk_sched()
w = mk_seq(range(3 * BS + 1))
s.block_manager.allocate(w, 0)
emu_prefill_chunk(s, w, 13)
h = mk_seq(range(3 * BS + 1))
nb = s.block_manager.can_allocate(h)
check("③ 同前缀的请求命中 3 个块", nb == 3, f"{nb}")
s.block_manager.allocate(h, nb)
check("③ 3 块全盖章 → 有效块数 = 3",
      s.block_manager.draft_valid_cached_blocks(h, nb) == 3)

# --- 负对照 1：中间一块的 draft 标记没了（draft 从没算过这一块）---
s.block_manager.blocks[h.block_table[1]].draft_hash = -1
check("③ 负对照：中间块缺标记 → 只认 1 块（不许越过缺口往后算）",
      s.block_manager.draft_valid_cached_blocks(h, nb) == 1,
      f"{s.block_manager.draft_valid_cached_blocks(h, nb)}")

# --- 负对照 2：draft 内容与块内容对不上（陈旧/污染）---
s.block_manager.blocks[h.block_table[1]].draft_hash = h.block_table[1] + 999999
check("③ 负对照：draft_hash 与块内容哈希不一致 → 判为无效",
      s.block_manager.draft_valid_cached_blocks(h, nb) == 1)
s.block_manager.blocks[h.block_table[0]].draft_hash = h.block_table[0] + 5
check("③ 负对照：首块就污染 → 有效块数 = 0（不推进水位）",
      s.block_manager.draft_valid_cached_blocks(h, nb) == 0)

# ======================================================================
print("§4 Scheduler 集成：命中 → 水位跟上 → catchup 缺口为空")
# ----------------------------------------------------------------------
s = mk_sched()
w = mk_seq(range(3 * BS + 1))
s.block_manager.allocate(w, 0)
emu_prefill_chunk(s, w, 13)
s.running.append(w)

h = mk_seq(range(3 * BS + 1))
s.waiting.append(h)
seqs, is_pref = s.schedule()
check("④ hitter 走 prefill 分支", is_pref and h in seqs, f"is_pref={is_pref}")
check("④ 命中 3 块 → draft_valid_len 先推到 12", h.draft_valid_len >= 3 * BS,
      f"{h.draft_valid_len}")
s4 = s
# 接着把 hitter 的 prefill chunk 跑完（1 个 token：位置 12）
h.advance_draft_watermark(h.num_cached_tokens, 13)
h.num_cached_tokens += 1
s.block_manager.hash_blocks(h, 1)
s.block_manager.mark_draft_valid(h, h.draft_valid_len)
check("④ prefill 完 → 水位 = 13（与无缓存路径一致）", h.draft_valid_len == 13,
      f"{h.draft_valid_len}")
st, gap = catchup_gap(h.draft_valid_len, h.token_ids)
check("④ ★ catchup 缺口为空（A 步要消灭的那笔开销）", len(gap) == 0,
      f"start={st} gap={len(gap)}")

# --- 负对照：writer 的 draft 从没跑过（标记全抹掉）---
s = mk_sched()
w = mk_seq(range(3 * BS + 1))
s.block_manager.allocate(w, 0)
emu_prefill_chunk(s, w, 13)
for i in range(len(w.block_table)):
    s.block_manager.blocks[w.block_table[i]].draft_hash = -1
s.running.append(w)
h2 = mk_seq(range(3 * BS + 1))
s.waiting.append(h2)
s.schedule()
check("④ 负对照：writer 的 draft 没跑过 → hitter 水位保持 0（回到旧的保守行为）",
      h2.draft_valid_len == 0, f"{h2.draft_valid_len}")
_, gap2 = catchup_gap(h2.draft_valid_len, h2.token_ids)
check("④ 负对照：此时缺口 = 12 个 token（说明保守设定确实还在）",
      len(gap2) == 12, f"gap={len(gap2)}")

# ======================================================================
print("§5 共享长前缀 + 后缀不同：两条请求互不污染")
# ----------------------------------------------------------------------
s = mk_sched()
P = list(range(3 * BS + 1))                    # 13 token 的共享前缀
w = mk_seq(P)
s.block_manager.allocate(w, 0)
emu_prefill_chunk(s, w, 13)
s.running.append(w)

a = mk_seq(P + [900, 901, 902, 903])           # 后缀 A
s.waiting.append(a)
s.schedule()
emu_prefill_chunk(s, a, len(a.token_ids))
b = mk_seq(P + [910, 911, 912, 913])           # 后缀 B
s.waiting.append(b)
s.schedule()
emu_prefill_chunk(s, b, len(b.token_ids))

tbl_a, tbl_b = list(a.block_table), list(b.block_table)
check("⑤ 前缀块（0..2）被两条请求共享（ref_count ≥ 2）",
      all(s.block_manager.blocks[tbl_a[i]].ref_count >= 2 for i in range(3)))
check("⑤ 后缀块各写各的物理块（不共享）",
      tbl_a[3] != tbl_b[3] and tbl_a[4] != tbl_b[4],
      f"a={tbl_a[3:]} b={tbl_b[3:]}")
check("⑤ A 的后缀内容不会被记成 B 的前缀",
      s.block_manager.blocks[tbl_a[3]].token_ids != s.block_manager.blocks[tbl_b[3]].token_ids)
# B 命中时能复用的 draft 前缀块数 = 3（共享前缀），不含任何后缀块
s2 = mk_sched()
w2 = mk_seq(P)
s2.block_manager.allocate(w2, 0)
emu_prefill_chunk(s2, w2, 13)
b2 = mk_seq(P + [910, 911, 912, 913])
nb2 = s2.block_manager.can_allocate(b2)
s2.block_manager.allocate(b2, nb2)
check("⑤ B 的可复用 draft 前缀块数 = 3（不是 5，后缀块没被算进来）",
      nb2 == 3 and s2.block_manager.draft_valid_cached_blocks(b2, nb2) == 3,
      f"nb={nb2} valid={s2.block_manager.draft_valid_cached_blocks(b2, nb2)}")

# ======================================================================
print("§6 抢占 → 归零；重新 prefill 后靠 draft 标记恢复")
# ----------------------------------------------------------------------
s = mk_sched()
w = mk_seq(P)
s.block_manager.allocate(w, 0)
emu_prefill_chunk(s, w, 13)
s.running.append(w)
h = mk_seq(P + [900, 901, 902, 903])
s.waiting.append(h)
s.schedule()
emu_prefill_chunk(s, h, len(h.token_ids))
check("⑥ 命中后水位 = 17（整段 prompt 有效）", h.draft_valid_len == 17,
      f"{h.draft_valid_len}")
Scheduler.preempt(s, h)
check("⑥ 抢占后水位归零（块已被释放）", h.draft_valid_len == 0, f"{h.draft_valid_len}")
s.waiting.append(h)
s.schedule()
# 重新调度时会命中的是【它自己】写过的块：P 的 3 个整块 + 自己后缀的 1 个整块
# = 4 块 = 16 token（第 5 块只装了 1 个 token，按约定不登记）。
check("⑥ 重新调度后靠 draft 块标记恢复到 16（连自己的后缀块也能复用）",
      h.draft_valid_len == 16, f"{h.draft_valid_len}")
emu_prefill_chunk(s, h, len(h.token_ids))
check("⑥ 重跑 prefill 后水位 = 17", h.draft_valid_len == 17, f"{h.draft_valid_len}")

# ======================================================================
print("§7 水位语义与无缓存路径一致（复用不改变水位终点）")
# ----------------------------------------------------------------------
cold = mk_seq(P)
s = mk_sched()
s.block_manager.allocate(cold, 0)
emu_prefill_chunk(s, cold, 13)          # 无缓存：start=0
check("⑦ 无缓存 prefill 的水位 = 13",
      cold.draft_valid_len == 13 and cold.num_cached_tokens == 13,
      f"wm={cold.draft_valid_len} cached={cold.num_cached_tokens}")

hot = mk_seq(P)
s = mk_sched()
w = mk_seq(P)
s.block_manager.allocate(w, 0)
emu_prefill_chunk(s, w, 13)
hot.draft_valid_len = 3 * BS            # = Scheduler 在 allocate 后推到的值
hot.advance_draft_watermark(3 * BS, 13)
check("⑦ 命中路径的水位终点同样是 13（复用不改变语义）",
      hot.draft_valid_len == 13, f"{hot.draft_valid_len}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
