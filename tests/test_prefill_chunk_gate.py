"""回归测试（纯 CPU）：分块 prefill 的两个循环不变量。

1. ★ **同一条序列一轮只能被调度一次**（P13 踩过的真 bug）
   分块后那条序列本步拼不完、没被 `popleft()`，还留在 `waiting[0]`；
   如果循环不 break，下一轮（同一个 while）会再取到**同一条**，把
   `num_scheduled_tokens` 再覆盖、再 append 一份 ⇒ `scheduled_seqs` 里出现
   同一序列的多份副本，`prepare_prefill` 把这段 chunk 重复拼 N 遍
   （实测：2048 的 chunk 被拼了 7 遍 = 14336 token）。修法是 append 后立刻 break。
   ⇒ 这条不变量值得单独钉住：**一轮里同一个 seq_id 不许出现两次。**

2. ★ P12 新增门控：**没有解码请求可让路时不切块**。
   "让路" 门控在 `bool(self.running)` 上，running 空 ⇒ 块之间不会插 decode 轮 ⇒
   分块只有代价没有收益（实测 prefill 墙钟 +20%~+70%）。所以 running 空时
   即使 `spec_prefill_chunk_tokens>0` 也不该分块。
   ★ 默认路径（chunk=0）不受影响，逐字节等价。

跑：python tests/test_prefill_chunk_gate.py
"""
import os
import sys
import types
from collections import deque
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

BS = 4
EOS = 999999
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def mk_sched(chunk, n_waiting=1, seq_len=20, n_running=0, max_bt=1024):
    """铺一个 Scheduler（绕开 __init__，与 test_spec_gate 同法）。"""
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(spec_k=0, spec_batch_threshold=0)
    s.max_num_seqs = 8
    s.max_num_batched_tokens = max_bt
    s.eos = EOS
    s.block_size = BS
    s.block_manager = BlockManager(256, BS)
    s.waiting = deque()
    s.running = deque()
    s.spec_batch_threshold = 0
    s.spec_proposer = None
    s.token_hook = None
    s.spec_k = 0
    s.prefill_chunk_tokens = chunk
    s._prefill_chunk_pending = False
    Sequence.block_size = BS
    for _ in range(n_running):
        r = Sequence([7] * (BS + 1), SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                    ignore_eos=True))
        s.block_manager.allocate(r, 0)
        r.status = 1                      # RUNNING
        s.running.append(r)
    for _ in range(n_waiting):
        w = Sequence([3] * seq_len, SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                   ignore_eos=True))
        s.waiting.append(w)
    return s


print("=" * 70)
print("§1 不变量：一轮里同一条序列不许被调度两次（真 bug 的护栏）")
print("=" * 70)

for chunk in (0, 4, 8):
    s = mk_sched(chunk, n_waiting=1, seq_len=20, n_running=1)
    scheduled, is_prefill = s.schedule()
    ids = [id(x) for x in scheduled]
    dup = len(ids) != len(set(ids))
    toks = sum(x.num_scheduled_tokens for x in scheduled)
    check(f"chunk={chunk}: 一轮内无重复序列（scheduled={len(scheduled)}）", not dup)
    check(f"chunk={chunk}: 一轮内进 token 数 ≤ chunk（进 {toks}）",
          toks <= max(chunk, 1) if chunk else toks <= 20,
          f"chunk={chunk} tokens={toks}")

# 拼不完的那条必须还留在 waiting[0]，且被标记 pending
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=1)
s.schedule()
check("chunk=8/len=20：拼不完 => 仍留在 waiting 队首", len(s.waiting) == 1
      and s.waiting[0].num_tokens == 20)
check("chunk=8/len=20：置 _prefill_chunk_pending=True", s._prefill_chunk_pending is True)

# 下一轮让路：running 里有解码请求 => 这一轮跑 decode
scheduled2, is_prefill2 = s.schedule()
check("下一轮让路给 decode（is_prefill=False）", is_prefill2 is False)
check("让路轮里 pending 标记被清掉", s._prefill_chunk_pending is False)

print()
print("=" * 70)
print("§2 P12 门控：没有解码请求可让路时不切块")
print("=" * 70)

# running 空 + chunk>0 => 一次拼完（不分块）
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=0)
scheduled, is_prefill = s.schedule()
toks = sum(x.num_scheduled_tokens for x in scheduled)
check("running 空：不分块（一次进满 20 个 token）", is_prefill is True and toks == 20,
      f"tokens={toks}")
check("running 空：序列被一次拼完、移出 waiting", len(s.waiting) == 0 and len(s.running) == 1)
check("running 空：不置 pending", s._prefill_chunk_pending is False)

# running 非空 + chunk>0 => 分块
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=1)
scheduled, is_prefill = s.schedule()
toks = sum(x.num_scheduled_tokens for x in scheduled)
check("running 非空：分块（一轮只进 chunk=8 个 token）", is_prefill is True and toks == 8,
      f"tokens={toks}")

# running 中途变空：上一轮留下的 pending 块应在下一轮拿到完整预算
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=1)
s.schedule()                       # 第 1 块（8）-> pending
check("（准备）pending=True", s._prefill_chunk_pending is True)
# 模拟 postprocess：本块写进 KV、水位推进 8
_seq = s.waiting[0]
_seq.num_cached_tokens += _seq.num_scheduled_tokens
_seq.num_scheduled_tokens = 0
s.running.clear()                  # 解码请求全部结束
scheduled, is_prefill = s.schedule()
toks = sum(x.num_scheduled_tokens for x in scheduled)
check("running 变空后：剩余 12 个 token 一次拼完（不再分块）",
      is_prefill is True and toks == 12, f"tokens={toks}")

print()
print("=" * 70)
print("§3 Scheduler 把「此刻的解码批大小」写进 config（供跳过 draft prefill 判白跑）")
print("=" * 70)

s = mk_sched(0, n_waiting=1, seq_len=8, n_running=3)
s.schedule()
check("prefill 开始时写入 _sched_running_decode_b = 本轮解码批大小",
      s.config._sched_running_decode_b == 3, f"got={s.config._sched_running_decode_b}")

s = mk_sched(0, n_waiting=1, seq_len=8, n_running=0)
s.schedule()
check("没有解码请求时写 0", s.config._sched_running_decode_b == 0)

print()
print("=" * 70)
print("§4 默认路径（chunk=0）行为不变")
print("=" * 70)

s = mk_sched(0, n_waiting=2, seq_len=20, n_running=0)
scheduled, is_prefill = s.schedule()
toks = sum(x.num_scheduled_tokens for x in scheduled)
check("chunk=0：两条短序列一轮里一起 prefill（老行为）",
      is_prefill is True and len(scheduled) == 2 and toks == 40, f"n={len(scheduled)} tokens={toks}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
