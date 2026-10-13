"""回归测试（纯 CPU）：P14 多序列 partial prefill 轮转 + 公平准入。

为什么这四条必须钉住
--------------------
调度器是本项目最容易"静默出错"的地方 —— 错了不会崩，只会：
  · 把同一段 chunk 重复拼 N 遍（P13 实测：2048 的 chunk 拼了 7 遍 = 14336 token）；
  · 让某条请求永远排在环外（FTTF 等 3 倍时间，vLLM #57413 那类）。
所以这里用**纯列表 + 手工推进**把语义钉死，不依赖 GPU。

① 不变量：**一轮内同一条序列只能被调度一次**（P13 踩过的真 bug 的护栏）
   断言两条：`scheduled_seqs` 里 id 不重复；本轮进 token 之和 ≤ 本轮预算。
② 顺序轮转的公平性：预算不足时不许总从队首开始（每条被服务的次数差 ≤ 1）。
③ 预算不足时的行为：给的是 min(剩余预算, chunk)，不超预算、不空转。
④ 默认关闭时逐值等于老逻辑（老路径一行未改）：
   · 属性根本不存在（老调度器）= rotate=False 的行为；
   · 经典场景的期望值按老逻辑写死（chunk=0 的两条短序列一轮拼完、分块拼不完立刻收工…）。

跑：python tests/test_prefill_multiseq.py
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
from nanovllm.engine.prefill_rotation import PrefillRotation  # noqa: E402
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


def mk_sched(chunk, waiting_lens, n_running=0, max_bt=1024, rotate=False,
             with_rotate_attr=True):
    """铺一个 Scheduler（绕开 __init__，与 test_spec_gate / test_prefill_chunk_gate 同法）。

    with_rotate_attr=False 模拟【P14 之前的调度器】：连 prefill_rotate 属性都没有
    （schedule() 里用 getattr 兜底 ⇒ 行为必须与 rotate=False 逐值相同）。
    """
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(spec_k=0, spec_batch_threshold=0)
    s.max_num_seqs = 8
    s.max_num_batched_tokens = max_bt
    s.eos = EOS
    s.block_size = BS
    s.block_manager = BlockManager(512, BS)
    s.waiting = deque()
    s.running = deque()
    s.spec_batch_threshold = 0
    s.spec_proposer = None
    s.token_hook = None
    s.spec_k = 0
    s.prefill_chunk_tokens = chunk
    s._prefill_chunk_pending = False
    if with_rotate_attr:
        s.prefill_rotate = rotate
        s._rotation = None
    Sequence.block_size = BS
    seqs = []
    for ln in waiting_lens:
        w = Sequence([3] * ln, SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                              ignore_eos=True))
        s.waiting.append(w)
        seqs.append(w)
    for _ in range(n_running):
        r = Sequence([7] * (BS + 1), SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                    ignore_eos=True))
        s.block_manager.allocate(r, 0)
        r.status = 1                      # RUNNING
        s.running.append(r)
    return s, seqs


def drive(s, seqs, max_rounds=400):
    """跑到 waiting 空；返回 (prefill 轮数, {id(seq): 拼完时在第几个 prefill 轮},
    {id(seq): 累计进过的 token 数})。

    ★ 手工模拟 postprocess 的推进（只做调度测试需要的部分）—— 本文件不碰 GPU，
      也不碰 hash_blocks（它会把块登记进前缀缓存，让第二条同 prompt 的序列
      "命中缓存"、num_tokens 大变，纯属噪声）。
    """
    done, got, rnd = {}, {}, 0
    for _ in range(max_rounds):
        if not s.waiting:
            break
        scheduled, is_prefill = s.schedule()
        if is_prefill:
            rnd += 1
            for seq in scheduled:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                got[id(seq)] = got.get(id(seq), 0) + seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens >= seq.num_tokens and id(seq) not in done:
                    done[id(seq)] = rnd
        else:
            # 解码轮：什么都不推进（只为让 yield_to_decode 的门控有事可做）
            for seq in scheduled:
                seq.num_scheduled_tokens = 0
    return rnd, done, got


print("=" * 70)
print("§1 不变量：一轮内同一条序列不许被调度两次 + 进 token 数 ≤ 本轮预算")
print("=" * 70)

# 覆盖「轮转开 / 关」×「预算宽 / 紧」×「chunk 大 / 小」的组合。
CASES = [
    # (chunk, waiting_lens, n_running, max_bt, rotate)
    (8,  [32, 32, 32], 1, 1024, True),
    (8,  [32, 32, 32], 1, 16,   True),    # 预算只够 2 条
    (8,  [32, 32, 32], 1, 8,    True),    # 预算只够 1 条
    (8,  [32, 32, 32], 1, 1024, False),
    (8,  [32, 32, 32], 1, 16,   False),
    (0,  [20, 20, 20], 1, 1024, False),
    (0,  [20, 20, 20], 0, 24,   False),   # 老行为：分块关 + 预算紧
    (4,  [17, 33, 9],  2, 128,  True),    # 长度不齐
]
for (chunk, lens, nrun, maxbt, rot) in CASES:
    s, seqs = mk_sched(chunk, lens, n_running=nrun, max_bt=maxbt, rotate=rot)
    seen_rounds = 0
    ok_dup = ok_budget = ok_cap = True
    while s.waiting and seen_rounds < 200:
        scheduled, is_prefill = s.schedule()
        if not is_prefill:
            for x in scheduled:
                x.num_scheduled_tokens = 0
            continue
        seen_rounds += 1
        ids = [id(x) for x in scheduled]
        if len(ids) != len(set(ids)):
            ok_dup = False
        tot = sum(x.num_scheduled_tokens for x in scheduled)
        if tot > maxbt:
            ok_budget = False
        if chunk and any(x.num_scheduled_tokens > chunk for x in scheduled):
            ok_cap = False
        for x in scheduled:
            x.num_cached_tokens += x.num_scheduled_tokens
            x.num_scheduled_tokens = 0
    tag = f"chunk={chunk} lens={lens} run={nrun} maxbt={maxbt} rotate={rot}"
    check(f"无重复调度 ({tag})", ok_dup)
    check(f"进 token 数 ≤ 预算 {maxbt} ({tag})", ok_budget)
    if chunk:
        check(f"每序列每轮 ≤ chunk={chunk} ({tag})", ok_cap)

print()
print("=" * 70)
print("§2 轮转：3 条长序列【同一轮】推进，而不是串成 1×/2×/3× 的时间")
print("=" * 70)

# 3 条 32 token、chunk=8 ⇒ 每条要 4 块。老逻辑下三条【首尾相衔】地串行推进：
#   A 占第 1~4 个 prefill 轮；A 拼完的那一轮（第 4 轮）循环会继续往下取到 B
#   （`popleft` 之后队首换人），所以 B 从第 4 轮起跑、第 7 轮拼完，C 从第 7 轮起跑。
#   ⇒ [4, 7, 10]：最后一条等到 10 轮 = 第一条的 2.5 倍（16K/chunk=1024 时就是
#     §d.5 记的"第 1/2/3 条分别在第 16/32/48 轮完成"那个现象，这里是同一个机制的缩小版）。
s, seqs = mk_sched(8, [32, 32, 32], n_running=1, max_bt=1024, rotate=False)
_, done_off, _ = drive(s, seqs)
rounds_off = [done_off[id(x)] for x in seqs]
check("rotate=False：三条首尾相衔地串行完成 [4,7,10]（最后一条等 2.5 倍，老行为）",
      rounds_off == [4, 7, 10], f"got={rounds_off}")

s, seqs = mk_sched(8, [32, 32, 32], n_running=1, max_bt=1024, rotate=True)
_, done_on, got_on = drive(s, seqs)
rounds_on = [done_on[id(x)] for x in seqs]
check("rotate=True：三条【都在第 4 轮】完成（= 轮转的目标）",
      rounds_on == [4, 4, 4], f"got={rounds_on}")
check("rotate=True：总进 token 数不变（3×32，一分不多）",
      sum(got_on.values()) == 96 and all(got_on[id(x)] == 32 for x in seqs),
      f"got={[got_on[id(x)] for x in seqs]}")

# 4 条也能一起走
s, seqs = mk_sched(8, [24, 24, 24, 24], n_running=1, max_bt=1024, rotate=True)
_, done4, _ = drive(s, seqs)
check("rotate=True：4 条 24 token（3 块）都在第 3 轮完成",
      [done4[id(x)] for x in seqs] == [3, 3, 3, 3], f"got={[done4[id(x)] for x in seqs]}")

print()
print("=" * 70)
print("§3 公平准入：预算不够时【轮着来】，不许总从队首开始（不饿死）")
print("=" * 70)

# chunk=8、预算=16 ⇒ 每轮只装得下 2 条序列的各一块（8+8=16）。
# 三条 24 token（各 3 块）时：轮转让每条每轮都能排上 ⇒ 都在第 3 轮拼完；
# 老逻辑只在"当前这条拼不完"时才收工，另一条要等下一条……
# 3 条 × 3 块 = 9 块 / 每轮 2 块 ⇒ 最快也要 5 轮；公平性看的是【差】不是绝对轮数。
s, seqs = mk_sched(8, [24, 24, 24], n_running=1, max_bt=16, rotate=True)
_, done_rr, _ = drive(s, seqs)
rr = [done_rr[id(x)] for x in seqs]
check("rotate=True：完成轮次 [4,4,5]（差 ≤ 1 = 公平轮转；且预算用满，9 块只用 5 轮）",
      rr == [4, 4, 5], f"got={rr}")

s, seqs = mk_sched(8, [24, 24, 24], n_running=1, max_bt=16, rotate=False)
_, done_f0, _ = drive(s, seqs)
f0 = [done_f0[id(x)] for x in seqs]
check("rotate=False：同样场景下 [3,6,9]（差 6；队首独占 + 每轮白白剩 8 个 token）",
      f0 == [3, 6, 9], f"got={f0}")

# 第一条是超长请求时，老逻辑会让后面的【整轮挨饿】；轮转不会。
s, seqs = mk_sched(8, [200, 16, 16], n_running=1, max_bt=24, rotate=True)
_, done_f, _ = drive(s, seqs)
# 后两条短请求各自只要 2 块；轮转下它们在第 2 / 3 轮就该完成
check("rotate=True：队首是超长请求时，后面的短请求不被饿死（≤3 轮完成）",
      done_f[id(seqs[1])] <= 3 and done_f[id(seqs[2])] <= 3,
      f"rounds={[done_f[id(x)] for x in seqs]}")

s, seqs = mk_sched(8, [200, 16, 16], n_running=1, max_bt=24, rotate=False)
_, done_f0, _ = drive(s, seqs)
check("rotate=False：同样场景下短请求要等到长请求拼完（老行为，作对照）",
      done_f0[id(seqs[1])] > 3 and done_f0[id(seqs[2])] > 3,
      f"rounds={[done_f0[id(x)] for x in seqs]}")

print()
print("=" * 70)
print("§4 预算不足：给 min(剩余预算, chunk)，不超预算、不空转")
print("=" * 70)

# 预算 12、chunk=8、3 条 32：第 1 条拿 8，第 2 条只拿 min(剩余4, 8)=4。
s, seqs = mk_sched(8, [32, 32, 32], n_running=1, max_bt=12, rotate=True)
scheduled, is_prefill = s.schedule()
gives = [x.num_scheduled_tokens for x in scheduled]
check("预算 12/chunk 8：这一轮进 8 + 4 = 12（用满预算但不超过）",
      is_prefill is True and sum(gives) == 12, f"gives={gives}")
check("预算 12/chunk 8：没有空转的序列（每条都 >0）", all(g > 0 for g in gives))
check("预算 12/chunk 8：一轮内不重复", len(set(id(x) for x in scheduled)) == len(scheduled))

# 预算只剩 3：给 3（< chunk），不是 0、也不是 chunk
s, seqs = mk_sched(8, [32, 32, 32], n_running=1, max_bt=3, rotate=True)
scheduled, is_prefill = s.schedule()
check("预算 3/chunk 8：只服务 1 条、给它 3 个 token",
      len(scheduled) == 1 and scheduled[0].num_scheduled_tokens == 3,
      f"gives={[x.num_scheduled_tokens for x in scheduled]}")

# 预算正好够一轮全服务：不会多转一圈
s, seqs = mk_sched(8, [32, 32, 32], n_running=1, max_bt=24, rotate=True)
scheduled, _ = s.schedule()
check("预算 24 = 3×chunk 8：正好每条一块、一轮结束（不重复绕圈）",
      len(scheduled) == 3 and sum(x.num_scheduled_tokens for x in scheduled) == 24,
      f"n={len(scheduled)}")

print()
print("=" * 70)
print("§5 默认关闭时逐值等于老逻辑")
print("=" * 70)

# (a) 连 prefill_rotate 属性都没有（= P14 之前的调度器）⇒ 与 rotate=False 逐值相同
for (chunk, lens, nrun, maxbt) in [(8, [32, 32, 32], 1, 1024), (8, [32, 32, 32], 1, 16),
                                   (0, [20, 20], 0, 1024), (4, [17, 33, 9], 2, 128)]:
    s_a, sa = mk_sched(chunk, lens, n_running=nrun, max_bt=maxbt, rotate=False,
                       with_rotate_attr=False)
    s_b, sb = mk_sched(chunk, lens, n_running=nrun, max_bt=maxbt, rotate=False)
    ra = drive(s_a, sa)
    rb = drive(s_b, sb)
    same = (ra[0] == rb[0]
            and [ra[1][id(x)] for x in sa] == [rb[1][id(x)] for x in sb]
            and [ra[2][id(x)] for x in sa] == [rb[2][id(x)] for x in sb])
    check(f"属性缺失 ≡ rotate=False（chunk={chunk} lens={lens} maxbt={maxbt}）", same,
          f"rounds={ra[0]}/{rb[0]}")

# (b) 经典老场景写死的期望值（与 test_prefill_chunk_gate 的口径一致）
s, seqs = mk_sched(0, [20, 20], n_running=0, max_bt=1024, rotate=False)
scheduled, is_prefill = s.schedule()
check("chunk=0：两条短序列一轮里一起 prefill（老行为）",
      is_prefill is True and len(scheduled) == 2
      and sum(x.num_scheduled_tokens for x in scheduled) == 40,
      f"n={len(scheduled)}")

s, seqs = mk_sched(8, [20], n_running=1, max_bt=1024, rotate=False)
scheduled, is_prefill = s.schedule()
check("chunk=8/len=20：一次只拼 8 个 token，拼不完就收工（老行为）",
      is_prefill is True and len(scheduled) == 1
      and scheduled[0].num_scheduled_tokens == 8
      and s._prefill_chunk_pending is True and len(s.waiting) == 1)

s, seqs = mk_sched(8, [20], n_running=1, max_bt=1024, rotate=False)
s.schedule()
scheduled2, is_prefill2 = s.schedule()
check("chunk=8：上一轮没拼完 ⇒ 这一轮让路给 decode（老行为）",
      is_prefill2 is False and s._prefill_chunk_pending is False)

s, seqs = mk_sched(8, [20], n_running=0, max_bt=1024, rotate=False)
scheduled, is_prefill = s.schedule()
check("chunk=8 但 running 空：不分块，一次拼完（P12 门控，老行为）",
      is_prefill is True and sum(x.num_scheduled_tokens for x in scheduled) == 20)

# (c) rotate=True 但 chunk=0 ⇒ 不轮转，与老行为逐值相同
# ★ 逐值比较必须按【下标】对齐 —— 两份 scheduler 里的序列是不同对象，id() 不可比。
s_a, sa = mk_sched(0, [20, 20, 20], n_running=1, max_bt=1024, rotate=True)
s_b, sb = mk_sched(0, [20, 20, 20], n_running=1, max_bt=1024, rotate=False)
ra, rb = drive(s_a, sa), drive(s_b, sb)
check("rotate=True 但 chunk=0：不轮转（没有「每条每块」的上限就无所谓轮转）",
      (ra[0], [ra[1][id(x)] for x in sa], [ra[2][id(x)] for x in sa])
      == (rb[0], [rb[1][id(x)] for x in sb], [rb[2][id(x)] for x in sb]),
      f"rounds={ra[0]}/{rb[0]}")

print()
print("=" * 70)
print("§6 PrefillRotation 单测（策略本身，不铺 Scheduler）")
print("=" * 70)

q = deque(["A", "B", "C"])
rot = PrefillRotation()
rot.reset()
check("next() 取队首", rot.next(q) == "A")
rot.yield_turn(q, q[0])
check("yield_turn 后队列左移一位（A 到环尾）", list(q) == ["B", "C", "A"])
check("next() 取到 B（不会重复取 A）", rot.next(q) == "B")
rot.mark(q[0])
q.popleft()                       # 模拟"拼完了、移出 waiting"
check("mark 过的序列不再被取到（即使还在队列里）",
      rot.next(deque(["A", "B", "C"])) is None, "A 已服务、B 已服务 ⇒ 环头 C 应可服务")
rot2 = PrefillRotation()
rot2.reset()
rot2.mark("X")
q2 = deque(["X", "Y"])
check("served 的环头 ⇒ next() 返回 None（= 本轮结束）", rot2.next(q2) is None)
rot2.reset()
check("reset() 后重新可用", rot2.next(q2) == "X")
rot3 = PrefillRotation()
rot3.reset()
check("空队列 ⇒ next() 返回 None", rot3.next(deque()) is None)

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
