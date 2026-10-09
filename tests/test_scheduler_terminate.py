"""回归测试（纯 CPU）：投机一步落地多个 token 时，max_tokens / EOS 必须拦得住。

真踩过的 bug
-----------
`Scheduler.postprocess_spec` 原来的判据是

    seq.num_completion_tokens == seq.max_tokens      # == 而不是 >=

投机一步会落地多个 token（被接受的候选 + 1 个 bonus，最多 k+1 个），
完成 token 数会【跳过】max_tokens：比如从 62 直接跳到 65，`== 64` 永远不成立。
后果：序列一直生成下去。实测跑了 300 步、572 个 token 还没停 ——
在 benchmark 里就表现为「draft 模式一跑就是几百秒」。

跑：python tests/test_scheduler_terminate.py
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scheduler 会 import nanovllm.config（依赖 transformers），
# 而 nanovllm/__init__.py 会拉起整个 LLM（torch.distributed 等）。
# 这个测试只用到 Scheduler.postprocess_spec，给它们塞占位模块，
# 保证纯 CPU、无重依赖就能跑。
_pkg = types.ModuleType("nanovllm")
_pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
sys.modules.setdefault("nanovllm", _pkg)
_cfg = types.ModuleType("nanovllm.config")
_cfg.Config = object
sys.modules.setdefault("nanovllm.config", _cfg)

from nanovllm.engine.sequence import Sequence, SequenceStatus   # noqa: E402
from nanovllm.engine.scheduler import Scheduler                 # noqa: E402
from nanovllm.sampling_params import SamplingParams             # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


class FakeBM:
    """只要不报错就行：本测试不关心 block 管理。

    签名要和真的 BlockManager.hash_blocks(seq, num_new_tokens) 一致 ——
    推进量是显式入参（契约见 block_manager.hash_blocks 的 docstring）。
    """
    def hash_blocks(self, seq, num_new_tokens): pass
    def mark_draft_valid(self, seq, draft_valid_len): pass
    def deallocate(self, seq): pass


class FakeSched:
    """绕过 __init__（它要真 Config），只提供 postprocess_spec 用到的字段。"""
    def __init__(self):
        self.block_manager = FakeBM()
        self.spec_proposer = None
        self.token_hook = None     # 默认不挂 hook（引擎里的默认值）
        self.running = []
        self.eos = 99          # 与情形 3 里的 eos 一致


def mk(max_tokens, ignore_eos=True, prompt_len=3):
    s = FakeSched()
    seq = Sequence([7] * prompt_len,
                   SamplingParams(temperature=1.0, max_tokens=max_tokens,
                                  ignore_eos=ignore_eos))
    s.running.append(seq)
    return s, seq


print("=" * 70)
print("测试：postprocess_spec 的终止判据")
print("=" * 70)

# ---- 情形 1：一步落地 3 个，跨过 max_tokens=4 ----
s, seq = mk(max_tokens=4)
Scheduler.postprocess_spec(s, [seq], [[10, 11, 12]])          # 完成 3，没到上限
check("未到上限时不结束", seq.status != SequenceStatus.FINISHED and seq in s.running,
      f"完成={seq.num_completion_tokens}")

Scheduler.postprocess_spec(s, [seq], [[13, 14, 15]])          # 会跨过 4
check("跨过 max_tokens 必须结束（用 >= 而不是 ==）",
      seq.status == SequenceStatus.FINISHED,
      f"完成={seq.num_completion_tokens}（期望 4）、status={seq.status.name}")
check("完成 token 数正好截到 max_tokens", seq.num_completion_tokens == 4,
      f"完成={seq.num_completion_tokens}，token_ids={seq.token_ids}")
check("多出来的 token 被丢弃", seq.token_ids[-1] == 13, f"token_ids={seq.token_ids}")
check("结束的序列从 running 摘掉", seq not in s.running)

# ---- 情形 2：一次落地就把 max_tokens 顶满 ----
s, seq = mk(max_tokens=3)
Scheduler.postprocess_spec(s, [seq], [[21, 22, 23]])
check("一步正好顶满时结束", seq.status == SequenceStatus.FINISHED
      and seq.num_completion_tokens == 3,
      f"完成={seq.num_completion_tokens}")

# ---- 情形 3：撞 EOS 要在 EOS 处截断，后面的全丢 ----
s, seq = mk(max_tokens=100, ignore_eos=False)
eos = 99
Scheduler.postprocess_spec(s, [seq], [[31, eos, 33]])
check("撞 EOS 时在 EOS 处截断", seq.status == SequenceStatus.FINISHED
      and seq.token_ids[-1] == eos,
      f"token_ids={seq.token_ids}")

# ---- 情形 4：正常多步落地，最终不会超出 max_tokens ----
s, seq = mk(max_tokens=10)
for _ in range(20):
    if seq.status == SequenceStatus.FINISHED:
        break
    Scheduler.postprocess_spec(s, [seq], [[1, 2, 3]])
check("反复落地也不会超出 max_tokens（不再无限生成）",
      seq.status == SequenceStatus.FINISHED and seq.num_completion_tokens == 10,
      f"完成={seq.num_completion_tokens}，status={seq.status.name}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
