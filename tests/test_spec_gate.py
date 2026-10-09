"""回归测试（纯 CPU）：投机门控取本轮 active B 的一致快照 + spec_k 的同步关系。

真踩过的 bug
-----------
`Scheduler.spec_enabled()` 读的是 `len(self.running)`，而 decode 分支是

    while self.running and ...:
        seq = self.running.popleft()          # 队列在轮内逐渐变短
        need = 1 + (self.spec_k if self.spec_enabled() else 0)

于是同一轮里、同一条请求在不同位置被判出不同结果。实测（threshold=2、B=4）：

    逐个 pop 时看到的队列长度 = 3, 2, 1, 0
    → 决定 = 关, 开, 开, 开
    → need = 1, 3, 3, 3      ← 一轮内混着两种执行路径
    → 后处理形状分支、KV 预算、消融归因全部对不上

反过来（threshold=3、B=4）更隐蔽：轮内队列长度降到 3 时判定「不超过阈值」，
于是一轮 4 条并发全开了投机 —— 门控用的 B 根本不是本轮的 active B。

另外核对 scheduler / model_runner / config 三者对 spec_k 的读法：
  · scheduler 用 self.spec_k 预留 1+k 个 KV 槽位；
  · model_runner 用 config.spec_k 提议候选、capture_verify_cudagraph 也用它；
  · 两份副本一旦漂移（例如只改 llm.scheduler.spec_k），scheduler 侧偏小时
    会把 1+kc 个候选写进只备了 1+ks 个槽位的位置 —— CUDA 非法访存那一类崩溃。
现在 spec_k 是 scheduler 上直通 config 的 property，只有一个来源。

跑：python tests/test_spec_gate.py
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


def mk_sched(spec_k, threshold, batch):
    """铺一个「已经有 batch 条 decode 请求」的 Scheduler（绕开 __init__）。"""
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(spec_k=spec_k, spec_batch_threshold=threshold)
    s.max_num_seqs = max(batch, 8)
    s.max_num_batched_tokens = 16384
    s.eos = EOS
    s.block_size = BS
    s.block_manager = BlockManager(256, BS)
    s.waiting = deque()
    s.running = deque()
    s.spec_batch_threshold = threshold
    s.spec_proposer = None
    s.token_hook = None
    s.spec_k = spec_k                      # property：直通 config.spec_k
    Sequence.block_size = BS
    for i in range(batch):
        seq = Sequence([7] * BS, SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                ignore_eos=True))
        s.block_manager.allocate(seq, 0)
        s.running.append(seq)
    return s


def run_round(s):
    scheduled, is_prefill = s.schedule()
    return [seq.num_scheduled_tokens for seq in scheduled], is_prefill


def check_round(name, spec_k, threshold, batch, expect):
    s = mk_sched(spec_k, threshold, batch)
    needs, is_prefill = run_round(s)
    check(name,
          is_prefill is False and len(needs) == batch
          and len(set(needs)) == 1 and needs[0] == expect,
          f"need={needs}（期望全 {expect}）")
    return s


print("=" * 70)
print("§1 同一轮内决定必须一致（快照本轮 active B）")
print("=" * 70)

# 关键复现：threshold=2、B=4 -> 改动前逐个 pop 得到 need = 1,3,3,3（混着两种路径）
check_round("① B=4 超阈值 2 → 全轮都不开投机（改动前 1,3,3,3 混着）",
            spec_k=3, threshold=2, batch=4, expect=1)

# 改动前的另一个坑：轮内队列长度降到阈值以内 → 本轮 4 条并发反而全开了
check_round("① B=4、阈值 3 → 全轮都不开（改动前按轮内变短的队列判成 3,3,3,3）",
            spec_k=3, threshold=3, batch=4, expect=1)

# 边界：active B 正好等于阈值（判据是 > 阈值）→ 开
check_round("① B=3 正好等于阈值 3 → 全轮都开（边界不吃掉）",
            spec_k=3, threshold=3, batch=3, expect=4)

# 不限 batch（threshold=0）→ 开
check_round("① 阈值 0（不限）→ 全轮都开", spec_k=3, threshold=0, batch=5, expect=4)

# 更小的 B 也不该例外
check_round("① B=1 → 全轮都开", spec_k=2, threshold=8, batch=1, expect=3)


print()
print("=" * 70)
print("§2 spec_k = 0 时一律走普通 decode")
print("=" * 70)

check_round("② spec_k=0 → 每序列只留 1 个槽位", spec_k=0, threshold=0, batch=4, expect=1)


print()
print("=" * 70)
print("§3 spec_k 只有一份来源（scheduler <-> config 直通）")
print("=" * 70)

s = mk_sched(spec_k=4, threshold=0, batch=3)
check("③ 初始值两边一致（config=4）", s.spec_k == s.config.spec_k == 4,
      f"spec_k={s.spec_k} config={s.config.spec_k}")

s.spec_k = 0
check("③ 改 scheduler.spec_k → config.spec_k 跟着变（旧代码只是改副本）",
      s.config.spec_k == 0, f"config={s.config.spec_k}")
needs, _ = run_round(s)
check("③ 改完立刻生效：这一轮每序列只剩 1 个槽位", needs == [1, 1, 1], f"{needs}")

s.config.spec_k = 2
check("③ 改 config.spec_k → scheduler.spec_k 跟着变（model_runner 读的就是它）",
      s.spec_k == 2, f"scheduler.spec_k={s.spec_k}")
needs, _ = run_round(s)
check("③ 改完立刻生效：这一轮每序列预留 1+k=3", needs == [3, 3, 3], f"{needs}")

# 门控与 config 同步：把阈值调到 1，B=3 → 应关掉
s.spec_batch_threshold = 1
needs, _ = run_round(s)
check("③ 阈值也是每轮现读的：调到 1 后 B=3 立刻关掉投机",
      needs == [1, 1, 1], f"{needs}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
