"""回归测试（纯 CPU）：LLMEngine.step() 的正式吞吐口径 + token 交付 hook。

真踩过的 bug
-----------
`step()` 里 decode 分支的计数是

    num_tokens = -sum(1 + getattr(seq, "last_accepted", 1) for seq in seqs)

这不是「本步落地的 token 数」：
  · `last_accepted` 是【上一轮】落地几个的遗留字段（普通 decode 路径恒为 0，
    只有 model_runner.run_verify 会写），拿它当本步计数，滞后一轮；
  · 属性缺失时还会补 1 → 退化情况变成每序列 2；
  · 投机一轮成批交付 k+1 个时，口径和真实交付数对不上。

现在：postprocess 数出本步【实际交付的 token IDs 数】并返回，step() 直接用。
  · prefill 步：返回本步写进 KV 的 prompt token 数（分块 prefill 的中间 chunk
    不交付 token，但确实占了算力，prefill 吞吐按它算）
  · decode 步：返回实际交付数（普通 decode = 每序列 1；投机 = 每序列 len(toks)）
符号（正 = prefill / 负 = decode）、返回值形态、tqdm 语义都不变。

顺带锁住 token 交付 hook 的语义（engine/token_hook.py）：
投机一轮 k+1 个 token 是【同一时刻】可见的，记成【一批】而不是 k+1 个时刻 ——
逐 token 均摊会凭空造出到达曲线，ITL 被算小 k 倍。

跑：python tests/test_engine_step.py
"""
import os
import sys
import types
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLMEngine 的模块级 import 里有 transformers（本机纯 CPU 环境没装）和
# nanovllm.config（要真模型路径）。这里塞占位模块，让 step() 的真代码能跑起来。
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)
_cfg = types.ModuleType("nanovllm.config")
_cfg.Config = object
sys.modules.setdefault("nanovllm.config", _cfg)
_tf = types.ModuleType("transformers")
_tf.AutoTokenizer = object
_tf.AutoConfig = object
_tf.Qwen3Config = object      # model_runner -> models.qwen3 会 import 它
sys.modules.setdefault("transformers", _tf)

# llm_engine 还会 import ModelRunner，而它一路拉到 layers/attention.py 里的
# `import triton`（纯 CPU 环境没有 triton）。本测试只调 step()，不构造 runner，
# 所以直接塞一个占位模块把这个 import 掐掉 —— 不改源码，也不动 triton。
_mr = types.ModuleType("nanovllm.engine.model_runner")
_mr.ModelRunner = object
sys.modules.setdefault("nanovllm.engine.model_runner", _mr)

from nanovllm.engine.block_manager import BlockManager          # noqa: E402
from nanovllm.engine.sequence import Sequence, SequenceStatus   # noqa: E402
from nanovllm.engine.scheduler import Scheduler                 # noqa: E402
from nanovllm.engine.llm_engine import LLMEngine                # noqa: E402
from nanovllm.engine.token_hook import TokenDeliveryHook        # noqa: E402
from nanovllm.sampling_params import SamplingParams             # noqa: E402

BS = 4
EOS = 999999
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


class ScriptedScheduler(Scheduler):
    """schedule() 换成预设脚本，postprocess / 计数 / hook 全用真实现。"""

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
    """代替 ModelRunner：run() 的返回值由脚本给（不碰 GPU / 模型）。"""

    def __init__(self, returns):
        self.returns = list(returns)
        self.calls = []

    def call(self, name, seqs, is_prefill):
        self.calls.append((name, [seq.seq_id for seq in seqs], is_prefill))
        return self.returns.pop(0)


def make_engine(sched, runner_returns, hook=None):
    eng = LLMEngine.__new__(LLMEngine)          # 绕开 __init__（要真模型）
    eng.scheduler = sched
    eng.model_runner = FakeRunner(runner_returns)
    eng.token_hook = hook
    if hook is not None:
        sched.set_token_hook(hook)
    return eng


def mk_seq(sched, prompt_len, max_tokens=10 ** 6, ignore_eos=True):
    Sequence.block_size = BS
    seq = Sequence(list(range(prompt_len)),
                   SamplingParams(temperature=1.0, max_tokens=max_tokens,
                                  ignore_eos=ignore_eos))
    sched.block_manager.allocate(seq, 0)
    sched.running.append(seq)
    return seq


def prefill(sched, seq, token=42):
    """把一个序列的 prefill 走完（一步填满，cached = prompt_len）。"""
    seq.num_scheduled_tokens = seq.num_tokens
    sched.script.insert(0, ([seq], True))
    return


# ======================================================================
print("=" * 70)
print("§1 普通 decode：正式计数 = 实际交付数，不看 last_accepted")
print("=" * 70)

sched = ScriptedScheduler([])
seq = mk_seq(sched, prompt_len=4)
prefill(sched, seq)
eng = make_engine(sched, [[42]])            # prefill 那一步交付首 token
out, num_tokens = eng.step()
check("① prefill 步：正号 + 本步写进 KV 的 prompt token 数",
      num_tokens == 4, f"{num_tokens}")
check("① prefill 的 num_scheduled_tokens 已清零", seq.num_scheduled_tokens == 0)

# 关键：把 last_accepted 设成上一轮投机留下的值（3），再走一步普通 decode
seq.last_accepted = 3
sched.script.append(([seq], False))
eng.model_runner.returns.append([101])      # 普通 decode：每序列 1 个 int
sched.block_manager.may_append(seq, 1)
seq.num_scheduled_tokens = 1
out, num_tokens = eng.step()
check("① 普通 decode 一步只记 1 个（旧口径 1+last_accepted 会记成 -4）",
      num_tokens == -1, f"{num_tokens}，last_accepted={seq.last_accepted}")
check("① 交付数 == token_ids 实际增长量",
      seq.num_completion_tokens == 2, f"完成 {seq.num_completion_tokens} 个（期望 2）")
# 反面对照：旧口径就是拿上一轮的 last_accepted 当本步计数，同一份状态会记成 -4
old = -sum(1 + getattr(s, "last_accepted", 1) for s in [seq])
check("① 反面对照：旧口径 1+last_accepted 会记成 -4（确实不是落地数）",
      old == -4 and old != num_tokens, f"旧口径 {old} vs 现在 {num_tokens}")


# ======================================================================
print()
print("=" * 70)
print("§2 投机一步成批交付：计数 == len(toks)")
print("=" * 70)

sched = ScriptedScheduler([])
seq = mk_seq(sched, prompt_len=4)
prefill(sched, seq)
eng = make_engine(sched, [[42]])
eng.step()

sched.script.append(([seq], False))
seq.num_scheduled_tokens = 4               # schedule 按 1+k=4 预留（spec_k=3）
sched.block_manager.may_append(seq, 4)
eng.model_runner.returns.append([[201, 202, 203]])       # 接受 2 个 + bonus
out, num_tokens = eng.step()
check("② 投机落地 3 个 → 记 -3", num_tokens == -3, f"{num_tokens}")
check("② 落地数就是 len(toks)", seq.num_completion_tokens == 4,
      f"完成 {seq.num_completion_tokens} 个（期望 1 + 3 = 4）")


# ======================================================================
print()
print("=" * 70)
print("§3 分块 prefill：中间 chunk 不交付 token（但 prefill 吞吐照算）")
print("=" * 70)

sched = ScriptedScheduler([])
seq = mk_seq(sched, prompt_len=8)           # 2 个块
hook = TokenDeliveryHook()
hook.on_request_added(seq)                  # 模拟 LLMEngine.add_request
eng = make_engine(sched, [[11, 12], [21]], hook)

sched.script.append(([seq], True))          # 第一个 chunk：scheduled = 4
seq.num_scheduled_tokens = 4
out, num_tokens = eng.step()
check("③ 中间 chunk：吞吐按 prompt token 数算（正号）", num_tokens == 4, f"{num_tokens}")
check("③ 中间 chunk 不交付 token（num_tokens 没变长）",
      seq.num_completion_tokens == 0, f"{seq.num_completion_tokens}")
check("③ 中间 chunk 不产生首 token 记录",
      hook.requests[seq.seq_id]["first_token"] is None)

sched.script.append(([seq], True))          # 第二个 chunk：prompt 走完 + 出首 token
seq.num_scheduled_tokens = 8 - seq.num_cached_tokens
out, num_tokens = eng.step()
check("③ 收尾 chunk 仍是 prefill 口径", num_tokens == 4, f"{num_tokens}")
check("③ 收尾 chunk 交付首 token", seq.num_completion_tokens == 1,
      f"{seq.num_completion_tokens}")
check("③ 首 token 时刻被记下", hook.requests[seq.seq_id]["first_token"] is not None)


# ======================================================================
print()
print("=" * 70)
print("§4 hook：投机一轮成批交付按【批】记，不均摊")
print("=" * 70)

sched = ScriptedScheduler([])
seq = mk_seq(sched, prompt_len=4)
hook = TokenDeliveryHook()
sched.set_token_hook(hook)
sched.token_hook.on_request_added(seq)          # 模拟 LLMEngine.add_request
eng = make_engine(sched, [[42]], hook)
sched.script.append(([seq], True))
seq.num_scheduled_tokens = 4
eng.step()                                      # 首 token（prefill 收尾）

for i in range(3):                              # 3 轮投机，每轮 3 个 token
    sched.script.append(([seq], False))
    seq.num_scheduled_tokens = 4
    sched.block_manager.may_append(seq, 4)
    eng.model_runner.returns.append([[301 + i * 10 + j for j in range(3)]])
    eng.step()

rec = hook.requests[seq.seq_id]
check("④ 交付 token 数 = 1 + 3×3 = 10", rec["num_tokens"] == 10, f"{rec['num_tokens']}")
check("④ 记录成 4 批（1 次 prefill + 3 轮投机），不是 10 个时刻",
      len(rec["batches"]) == 4, f"{len(rec['batches'])} 批")
check("④ 每批的 token 数正确", [n for _, n in rec["batches"]] == [1, 3, 3, 3],
      f"{[n for _, n in rec['batches']]}")
check("④ ITL 只有 3 个间隔（批间），不是 9 个",
      len(hook.itl(seq.seq_id)) == 3, f"{len(hook.itl(seq.seq_id))}")
check("④ 时刻单调不减", all(b >= a for a, b in zip([t for t, _ in rec["batches"]],
                                                  [t for t, _ in rec["batches"]][1:])))
check("④ TPOT = (末 - 首) / (交付数 - 1)，不为负也不为 NaN",
      hook.tpot(seq.seq_id) >= 0.0 and hook.tpot(seq.seq_id) == hook.tpot(seq.seq_id),
      f"{hook.tpot(seq.seq_id):.3e}s")
check("④ TTFT 从入队时刻起算（≥ 0）", hook.ttft(seq.seq_id) >= 0.0,
      f"{hook.ttft(seq.seq_id):.3e}s")
check("④ 未结束的请求 end 为空", rec["end"] is None)


# ======================================================================
print()
print("=" * 70)
print("§5 hook 记录请求结束时刻；默认不挂 = 零开销路径")
print("=" * 70)

sched = ScriptedScheduler([])
seq = mk_seq(sched, prompt_len=4, max_tokens=1)     # 首个 token 就顶满
hook = TokenDeliveryHook()
sched.set_token_hook(hook)
hook.on_request_added(seq)
eng = make_engine(sched, [[42]], hook)
sched.script.append(([seq], True))
seq.num_scheduled_tokens = 4
out, num_tokens = eng.step()
rec = hook.requests[seq.seq_id]
check("⑤ 撞 max_tokens 结束", seq.status == SequenceStatus.FINISHED)
check("⑤ 结束时刻被记下", rec["end"] is not None)
check("⑤ 结束时刻 == 最后一次交付时刻（同一步内完成）",
      rec["end"] == rec["last_token"])
check("⑤ e2e = 结束 - 入队（≥ 0）", hook.e2e(seq.seq_id) >= 0.0,
      f"{hook.e2e(seq.seq_id):.3e}s")
check("⑤ generate() 能拿到的输出就是 seq.completion_token_ids",
      out == [(seq.seq_id, seq.completion_token_ids)] and len(out[0][1]) == 1,
      f"{out}")

sched2 = ScriptedScheduler([])
seq2 = mk_seq(sched2, prompt_len=4)
eng2 = make_engine(sched2, [[42]])                  # 不传 hook
check("⑤ 默认不挂 hook（scheduler.token_hook is None）",
      sched2.token_hook is None and eng2.token_hook is None)
sched2.script.append(([seq2], True))
seq2.num_scheduled_tokens = 4
out, num_tokens = eng2.step()
check("⑤ 不挂 hook 时计数照常", num_tokens == 4 and seq2.num_completion_tokens == 1,
      f"num_tokens={num_tokens}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
