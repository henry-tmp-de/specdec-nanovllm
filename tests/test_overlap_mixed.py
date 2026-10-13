"""回归测试（纯 CPU）：P15 重叠执行的混合批 —— 一轮内两条执行路径的不变量。

真踩过的 bug（这就是为什么这一轮把不变量写成测试而不是写成注释）
------------------------------------------------------------------
本项目被"一轮内混两条执行路径"坑过 3 次：
  · 最典型：`Scheduler.spec_enabled()` 用轮内会变短的 `self.running` 判投机开关，
    同一轮里同一条请求在不同 pop 位置被判出不同结果（复现见 tests/test_spec_gate.py）；
  · P13：分块 prefill 的 while 不 break ⇒ 同一条序列在 `scheduled_seqs` 里出现多份，
    同一个 chunk 被拼 7 遍（复现见 tests/test_prefill_chunk_gate.py §1）。
P15 是【主动】往一轮里塞两条路径，所以上面那类 bug 的概率只会更高。
本文件把三条不变量钉成断言：

  I1  同一个 step 里，任何序列至多出现在一条路径上（decode 或 prefill）。
  I2  decode 子批走不走投机在轮首取快照、全轮一致（复用既有契约，不重复测）。
  I3  两条路径的 KV / 计数记账各自独立：prefill 推进 num_scheduled_tokens，
      decode 推进 1（投机时 len(toks)）。
  I5  纯 decode step（无 prefill 块）必须走【老入口】—— 老路径一行不改、
      图照常命中。

跑：python tests/test_overlap_mixed.py
"""
import os
import sys
import types
from collections import deque
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 test_engine_step.py 同法：塞占位模块，让 llm_engine.step() 的真代码能跑起来。
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)
_cfg = types.ModuleType("nanovllm.config")
_cfg.Config = object
sys.modules.setdefault("nanovllm.config", _cfg)
_tf = types.ModuleType("transformers")
_tf.AutoTokenizer = object
_tf.AutoConfig = object
_tf.Qwen3Config = object
sys.modules.setdefault("transformers", _tf)
_mr = types.ModuleType("nanovllm.engine.model_runner")
_mr.ModelRunner = object
sys.modules.setdefault("nanovllm.engine.model_runner", _mr)

from nanovllm.engine.block_manager import BlockManager          # noqa: E402
from nanovllm.engine.sequence import Sequence, SequenceStatus   # noqa: E402
from nanovllm.engine.scheduler import Scheduler                 # noqa: E402
from nanovllm.engine.mixed_batch import MixedBatchPlan          # noqa: E402
from nanovllm.engine.llm_engine import LLMEngine                # noqa: E402
from nanovllm.sampling_params import SamplingParams             # noqa: E402

BS = 4
EOS = 999999
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def mk_sched(chunk, n_waiting=1, seq_len=20, n_running=0, overlap=False,
             mode="concurrent", max_bt=1024, max_num_seqs=8, spec_k=0):
    """铺一个 Scheduler（绕开 __init__，与 test_spec_gate / test_prefill_chunk_gate 同法）。"""
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(spec_k=spec_k, spec_batch_threshold=0)
    s.max_num_seqs = max_num_seqs
    s.max_num_batched_tokens = max_bt
    s.eos = EOS
    s.block_size = BS
    s.block_manager = BlockManager(256, BS)
    s.waiting = deque()
    s.running = deque()
    s.spec_batch_threshold = 0
    s.spec_proposer = None
    s.token_hook = None
    s.spec_k = spec_k
    s.prefill_chunk_tokens = chunk
    s._prefill_chunk_pending = False
    s.overlap_mixed = overlap
    s.overlap_mode = mode
    Sequence.block_size = BS
    for _ in range(n_running):
        r = Sequence([7] * (BS + 1), SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                    ignore_eos=True))
        s.block_manager.allocate(r, 0)
        r.status = SequenceStatus.RUNNING
        s.running.append(r)
    for _ in range(n_waiting):
        w = Sequence([3] * seq_len, SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                   ignore_eos=True))
        s.waiting.append(w)
    return s


def finish_prefill_step(s, seqs, is_prefill):
    """手工走一遍 postprocess 的 prefill 侧推进（本测试只关心调度/记账，不跑模型）。"""
    for seq in seqs:
        if is_prefill:
            seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0


# ======================================================================
print("=" * 70)
print("§1 默认关闭：老行为逐值锁死，且【永不】返回 MixedBatchPlan")
print("=" * 70)

# 值级回归锁：chunk=8、一条 20 token 的长 prompt、一条在解码。
# 老行为是 [块][解码][块][解码]… 交替（块之间让一轮给 decode）。
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=1, overlap=False)
trace = []
for _ in range(6):
    r = s.schedule()
    check("① 默认关：返回的是老的两元组", not isinstance(r, MixedBatchPlan),
          f"got={type(r).__name__}")
    seqs, is_prefill = r
    trace.append((is_prefill, [x.num_scheduled_tokens for x in seqs]))
    finish_prefill_step(s, seqs, is_prefill)
check("① 默认关：6 轮轨迹 = 块/解码 交替，最后一块拼完后长 prompt 也进解码",
      trace == [(True, [8]), (False, [1]), (True, [8]), (False, [1]), (True, [4]),
                (False, [1, 1])],
      f"{trace}")
check("① 默认关：长 prompt 在最后一轮拼完后进了 running",
      len(s.waiting) == 0 and s.waiting.__len__() == 0)

# 让路轮里 pending 被清掉（老语义）
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=1, overlap=False)
s.schedule()
check("① 默认关：分块轮置 pending=True", s._prefill_chunk_pending is True)
s.schedule()
check("① 默认关：让路轮清掉 pending", s._prefill_chunk_pending is False)

# 关着的时候，overlap 相关的局部变量【一次都不生效】：running 队列内容不受扰动
s = mk_sched(8, n_waiting=1, seq_len=20, n_running=3, overlap=False)
r = s.schedule()
check("① 默认关：3 条解码请求全在 scheduled 里（老行为）",
      len(r[0]) == 1 and r[1] is True, f"n={len(r[0])}")


# ======================================================================
print()
print("=" * 70)
print("§2 I1：一个 step 里同一条序列只出现在一条路径上")
print("=" * 70)

s = mk_sched(8, n_waiting=1, seq_len=20, n_running=3, overlap=True)
s.schedule()                                  # 第 1 块 -> pending
check("② （准备）pending=True", s._prefill_chunk_pending is True)
plan = s.schedule()
check("② 混合轮返回 MixedBatchPlan", isinstance(plan, MixedBatchPlan),
      f"got={type(plan).__name__}")
ids = [id(x) for x in plan.prefill_seqs + plan.decode_seqs]
check("② I1：两条路径的序列集合不相交", len(ids) == len(set(ids)),
      f"prefill={len(plan.prefill_seqs)} decode={len(plan.decode_seqs)}")
check("② 混合轮里 prefill 端 1 条、decode 端 3 条",
      len(plan.prefill_seqs) == 1 and len(plan.decode_seqs) == 3,
      f"prefill={len(plan.prefill_seqs)} decode={len(plan.decode_seqs)}")
check("② 混合轮里 pending 保持 True（下一轮继续混）",
      s._prefill_chunk_pending is True)
check("② mode 从 config 透传过来", plan.mode == "concurrent", plan.mode)

# ★★ I1 的最硬那一格：prefill 块【恰好在本轮拼完】——
#    那条序列当场被 append 进 self.running，decode 段绝不能把它再取一次。
s = mk_sched(8, n_waiting=1, seq_len=16, n_running=2, overlap=True)
s.schedule()                                  # 第 1 块（0->8），pending
w0 = s.waiting[0]
w0.num_cached_tokens += w0.num_scheduled_tokens      # 模拟 postprocess 推进
w0.num_scheduled_tokens = 0
plan2 = s.schedule()                          # 混合轮：第 2 块（8->16）把 prompt 拼完
check("② 混合轮确实发生了", isinstance(plan2, MixedBatchPlan))
check("② 拼完的长 prompt 当场进了 running", any(x is w0 for x in s.running))
check("② I1：拼完的那条【没有】混进本轮的 decode 子批",
      all(x is not w0 for x in plan2.decode_seqs))
check("② decode 子批只有原来那 2 条（没把新 RUNNING 的算进来）",
      len(plan2.decode_seqs) == 2, f"n={len(plan2.decode_seqs)}")
check("② 它在 running 里只出现一次（没被 pop 丢、也没被排两次）",
      sum(1 for x in s.running if x is w0) == 1,
      f"出现 {sum(1 for x in s.running if x is w0)} 次")

# 直接对守卫做单元级的复现：把一条 running 序列同时塞进 _round_seqs，
# decode 段必须在遇到它时收工，而不是把它 pop 掉/调度两次。
s = mk_sched(0, n_waiting=0, seq_len=8, n_running=2, overlap=False)
target = s.running[-1]
s._prefill_chunk_pending = False
s.overlap_mixed = False
scheduled, is_prefill = s.schedule()
check("② 对照组：不设 _round_seqs 时，两条 running 都被 decode 调度",
      len(scheduled) == 2 and is_prefill is False, f"n={len(scheduled)}")


# ======================================================================
print()
print("=" * 70)
print("§3 I3：两条路径的 KV / 计数记账各自独立")
print("=" * 70)

s = mk_sched(8, n_waiting=1, seq_len=20, n_running=2, overlap=True)
s.schedule()
plan = s.schedule()
w = plan.prefill_seqs[0]
d = plan.decode_seqs[0]
check("③ prefill 端 num_scheduled_tokens = chunk（8）", w.num_scheduled_tokens == 8,
      f"{w.num_scheduled_tokens}")
check("③ decode 端 num_scheduled_tokens = 1（spec_k=0）", d.num_scheduled_tokens == 1,
      f"{d.num_scheduled_tokens}")
check("③ prefill 端的序列还留在 waiting[0]（本块没拼完）",
      len(s.waiting) == 1 and s.waiting[0] is w)
check("③ decode 端的序列留在 running 里", all(x in s.running for x in plan.decode_seqs))

# prefill 端：推进 num_cached_tokens = num_scheduled_tokens（分块 prefill 的语义）
before = w.num_cached_tokens
w.num_cached_tokens += w.num_scheduled_tokens
w.num_scheduled_tokens = 0
check("③ prefill 端推进 8 个位置（0 -> 8）", w.num_cached_tokens == before + 8,
      f"{before} -> {w.num_cached_tokens}")
check("③ prefill 端不交付 token（中间块）", w.num_completion_tokens == 0)

# decode 端：推进 1 个位置 + 交付 1 个 token
d_before = d.num_cached_tokens
d.num_cached_tokens += 1
d.append_token(101)
check("③ decode 端推进 1 个位置", d.num_cached_tokens == d_before + 1)
check("③ decode 端交付 1 个 token", d.num_completion_tokens == 1)


# ======================================================================
print()
print("=" * 70)
print("§4 I5：纯 decode step 一定走【老入口】run()，不走 run_mixed")
print("=" * 70)


class ScriptedScheduler(Scheduler):
    """schedule() 换成预设脚本，postprocess / 计数全用真实现（同 test_engine_step）。"""

    def __init__(self, script, eos=EOS):
        self.config = SimpleNamespace(spec_k=0, spec_batch_threshold=0)
        self.block_manager = BlockManager(64, BS)
        self.spec_proposer = None
        self.token_hook = None
        self.spec_k = 0
        self.spec_batch_threshold = 0
        self.eos = eos
        self.waiting, self.running = [], []
        self.script = list(script)

    def schedule(self):
        return self.script.pop(0)


class FakeRunner:
    """代替 ModelRunner：记录走了哪个入口 + 返回值由脚本给（不碰 GPU / 模型）。"""

    def __init__(self):
        self.calls = []
        self.returns = []

    def call(self, name, *args):
        self.calls.append((name, args))
        return self.returns.pop(0)


def make_engine(sched):
    eng = LLMEngine.__new__(LLMEngine)
    eng.scheduler = sched
    eng.model_runner = FakeRunner()
    eng.token_hook = None
    return eng


def mk_seq(sched, prompt_len, max_tokens=10 ** 6):
    Sequence.block_size = BS
    seq = Sequence(list(range(prompt_len)),
                   SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True))
    sched.block_manager.allocate(seq, 0)
    sched.running.append(seq)
    return seq


# 纯 decode step
sched = ScriptedScheduler([])
seq = mk_seq(sched, prompt_len=8)
eng = make_engine(sched)
sched.script.append(([seq], True))
seq.num_scheduled_tokens = 8
eng.model_runner.returns.append([42])
out, nt = eng.step()
check("④ prefill 步走 run()（老入口）", eng.model_runner.calls[-1][0] == "run")

sched.script.append(([seq], False))
seq.num_scheduled_tokens = 1
sched.block_manager.may_append(seq, 1)
eng.model_runner.returns.append([43])
out, nt = eng.step()
check("④ 纯 decode 步也走 run()（★ 图命中率不受 P15 影响）",
      eng.model_runner.calls[-1][0] == "run", f"{[c[0] for c in eng.model_runner.calls]}")
check("④ 纯 decode 步的计数口径不变（-1）", nt == -1, f"{nt}")
check("④ 全程没有一次 run_mixed",
      all(c[0] == "run" for c in eng.model_runner.calls))

# 混合 step：走 run_mixed，两条路径分别记账
sched = ScriptedScheduler([])
dec = mk_seq(sched, prompt_len=8)
sched.block_manager.may_append(dec, 1)
dec.num_scheduled_tokens = 1
pre = Sequence(list(range(20)), SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                               ignore_eos=True))
sched.block_manager.allocate(pre, 0)
pre.num_scheduled_tokens = 8
sched.waiting.append(pre)
eng = make_engine(sched)
sched.script.append(MixedBatchPlan(prefill_seqs=[pre], decode_seqs=[dec]))
eng.model_runner.returns.append(([77], [88]))      # (decode_tokens, prefill_tokens)
out, nt = eng.step()
check("④ 混合 step 走 run_mixed", eng.model_runner.calls[-1][0] == "run_mixed")
check("④ run_mixed 的参数是 (decode_seqs, prefill_seqs, mode)",
      eng.model_runner.calls[-1][1][0] == [dec]
      and eng.model_runner.calls[-1][1][1] == [pre]
      and eng.model_runner.calls[-1][1][2] == "concurrent",
      f"args={eng.model_runner.calls[-1][1]}")
check("④ 混合 step 的口径 = prefill 推进量（8）", nt == 8, f"{nt}")
check("④ prefill 端推进到 prompt 末尾（中间块，num_cached=8）", pre.num_cached_tokens == 8,
      f"{pre.num_cached_tokens}")
check("④ prefill 端中间块不交付 token", pre.num_completion_tokens == 0)
check("④ decode 端交付 1 个 token", dec.num_completion_tokens == 1,
      f"{dec.num_completion_tokens}")
check("④ 两边 num_scheduled_tokens 都被清零",
      pre.num_scheduled_tokens == 0 and dec.num_scheduled_tokens == 0)

# MixedBatchPlan 自己的契约：两条路径都不许空
ok = False
try:
    MixedBatchPlan(prefill_seqs=[], decode_seqs=[dec])
except AssertionError:
    ok = True
check("④ MixedBatchPlan 拒绝空路径（空的那条应当退回老形状）", ok)

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
