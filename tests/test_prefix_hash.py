"""回归测试（纯 CPU）：前缀缓存哈希的登记契约。

真踩过的 bug（改动前实测，本文件 §1/§2/§3 就是复现）
----------------------------------------------------
`BlockManager.hash_blocks` 内部按

    start = seq.num_cached_tokens // bs
    end   = (seq.num_cached_tokens + seq.num_scheduled_tokens) // bs

算登记区间 —— 这个写法的隐含语义是「num_cached_tokens 是【本次之前】的旧值，
num_scheduled_tokens 是本次推进量」。但两个调用点都在调用【之前】就把
num_cached_tokens 加过了（scheduler.postprocess / postprocess_spec），
传进去的是【新值】，于是整个区间后移一格。实测后果：

  ① 越界：prompt = 8、bs = 4、一步 prefill 完 → 调用时 cached=8、scheduled=8
     → start=2、end=4，而 block_table 只有 2 个元素 → IndexError。
     极端的例子（方案原文）：prompt 恰好 512、bs=256、一步 prefill 完。
  ② 半满块被当成满块登记：cached 4→6（投机落地 2 个，而 num_scheduled_tokens
     是按上限预留的 1+k=3）→ start=1、end=2 → 把只有 2 个 token 的块登记进
     前缀缓存，刚写满的 block 0 反而被跳过。
  ③ 投机没捞到候选、退回普通 decode 时，postprocess 用 num_scheduled_tokens
     （= 1+k）推进 cached —— 一步多跑 k 个 → 几步后 cached 跑到 num_tokens
     前面 → 登记区间越界 / 登记的块根本还没确认。

为什么必须错一次就完蛋：登记错的块 = 下次相同前缀命中错误缓存 →
静默输出错误 token（见 hash_blocks docstring 的红线）。

契约（改动后，所有调用点统一到这个写法）
----------------------------------------
    hash_blocks(seq, num_new_tokens)
      · num_new_tokens = 本次真正被【确认】并推进的 token 数
        （普通/分块 prefill = 本步处理的 prompt token 数；
          普通 decode = 1；投机验证 = len(toks) = 接受的候选 + bonus）
      · 调用时 seq.num_cached_tokens 必须已经推进到新值
      · 登记区间 = [旧完成块数, 新完成块数) =
            [ (cached - num_new) // bs , cached // bs )
      · num_cached_tokens 就是「有效缓存量」：它比 num_tokens 少 1，
        少掉的那一个是刚采出来、KV 还要等下一次前向（按 len-1 重算）才落地的
        token —— 所以「整块都在 cached 之内」等价于「这一块的 KV 全部有效」。

跑：python tests/test_prefix_hash.py
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scheduler 会 import nanovllm.config（依赖 transformers），nanovllm/__init__.py
# 还会拉起整个 LLM。这个测试只用 BlockManager + Scheduler 的 postprocess，
# 给它们塞占位模块就能纯 CPU、无重依赖跑起来（和 test_scheduler_terminate 一致）。
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

BS = 4                      # 用小 block，几行代码就能跨页
EOS = 999999                # 让测试里的 token 永远撞不到 eos
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def call(fn, *args, **kwargs):
    """跑一次可能抛异常的操作，把异常当证据返回（改动前这里会 IndexError）。"""
    try:
        fn(*args, **kwargs)
        return None
    except Exception as e:            # noqa: BLE001 —— 测试要的就是把异常记下来
        return f"{type(e).__name__}: {e}"


def fresh_sched(num_blocks=64, spec_k=0, threshold=0):
    """绕开 Scheduler.__init__（它要真 Config），只铺 postprocess 用到的字段。"""
    s = Scheduler.__new__(Scheduler)
    s.config = types.SimpleNamespace(spec_k=spec_k, spec_batch_threshold=threshold)
    s.block_manager = BlockManager(num_blocks, BS)
    s.spec_proposer = None
    s.spec_k = spec_k
    s.spec_batch_threshold = threshold
    s.eos = EOS
    s.waiting = []
    s.running = []
    return s


def mk_seq(s, prompt_len):
    Sequence.block_size = BS
    seq = Sequence(list(range(prompt_len)),
                   SamplingParams(temperature=1.0, max_tokens=10 ** 6))
    s.block_manager.allocate(seq, 0)
    s.running.append(seq)
    return seq


def registered(bm):
    """当前哈希表里登记着的块：(block_id, token_ids)。"""
    return [(bid, list(bm.blocks[bid].token_ids)) for bid in bm.hash_to_block_id.values()]


def true_chain(token_ids, n_blocks):
    """前缀哈希的真实链：h_i = H(h_{i-1}, tokens_i)，h_{-1} = -1。"""
    hs, h = [], -1
    for i in range(n_blocks):
        h = BlockManager.compute_hash(token_ids[i * BS:(i + 1) * BS], h)
        hs.append(h)
    return hs


def registry_matches_prefix(bm, seq):
    """哈希表必须【精确等于】真实前缀链上那些已填满的块（多一个都算错）。

    这里同时覆盖三条红线：越界块、半满块、断链（拿别的前缀的哈希顶替）。
    """
    want = seq.num_cached_tokens // BS
    if want > len(seq.block_table):
        return False, f"cached={seq.num_cached_tokens} 要求 {want} 块，block_table 只有 {len(seq.block_table)} 块"
    hs = true_chain(seq.token_ids, want)
    for i in range(want):
        b = bm.blocks[seq.block_table[i]]
        if b.token_ids != seq.token_ids[i * BS:(i + 1) * BS]:
            return False, f"block {i} 内容不对: {b.token_ids}"
        if b.hash != hs[i]:
            return False, f"block {i} 哈希不是真实前缀链"
        if bm.hash_to_block_id.get(hs[i]) != b.block_id:
            return False, f"block {i} 的哈希没指向它自己"
    if len(bm.hash_to_block_id) != want:
        return False, f"哈希表里 {len(bm.hash_to_block_id)} 条，应该 {want} 条"
    if any(len(t) != BS for _, t in registered(bm)):
        return False, f"有半满块: {[(i, len(t)) for i, t in registered(bm)]}"
    return True, want


# ======================================================================
print("=" * 70)
print("§1 一步 prefill 正好填满整数个块 —— 改动前越界")
print("=" * 70)

s = fresh_sched()
bm = s.block_manager
seq = mk_seq(s, 8)                          # 8 token = 2 整块
seq.num_scheduled_tokens = 8               # schedule() 里 chunked prefill 的推进量
err = call(Scheduler.postprocess, s, [seq], [42], True)
check("① 不越界（改动前 IndexError：start=2,end=4 但 block_table 只有 2 个）",
      err is None, err or "")
check("① 两个刚填满的块都登记了（block 0 不再被跳过）",
      bm.blocks[seq.block_table[0]].token_ids == [0, 1, 2, 3]
      and bm.blocks[seq.block_table[1]].token_ids == [4, 5, 6, 7],
      f"{bm.blocks[seq.block_table[0]].token_ids} / {bm.blocks[seq.block_table[1]].token_ids}")
ok, info = registry_matches_prefix(bm, seq)
check("① 登记结果 == 真实前缀链（哈希、内容、条数、无半满块）", ok, info)

# 再走一步 decode：cached 只推进 1，登记区间不应该再动
seq.num_scheduled_tokens = 1
bm.may_append(seq, 1)
err = call(Scheduler.postprocess, s, [seq], [43], False)
ok2, info2 = registry_matches_prefix(bm, seq)
check("① 后续 decode 步不重复登记、不污染（cached=9 → 仍只有 2 个满块）",
      err is None and ok2, err or info2)


# ======================================================================
print()
print("=" * 70)
print("§2 投机一步落地 2 个（num_scheduled_tokens 是 1+k=3）—— 改动前半满块被登记")
print("=" * 70)

s = fresh_sched()
bm = s.block_manager
seq = mk_seq(s, 4)                          # 1 个整块
seq.num_scheduled_tokens = 4
call(Scheduler.postprocess, s, [seq], [42], True)      # prefill 完成：cached=4
check("② 前置：prefill 后 block 0 已登记", bm.blocks[seq.block_table[0]].token_ids == [0, 1, 2, 3])

seq.num_scheduled_tokens = 3               # schedule() 的 need = 1 + spec_k
bm.may_append(seq, 3)                      # schedule() 会先把 1+k 个槽位备好
err = call(Scheduler.postprocess_spec, s, [seq], [[101, 102]])   # 真正确认 2 个
check("② 不抛异常", err is None, err or "")
check("② num_cached_tokens 只推进【真正确认的】len(toks)=2（不是 1+k=3）",
      seq.num_cached_tokens == 6, f"cached={seq.num_cached_tokens}，期望 6")
bad = [(bid, t) for bid, t in registered(bm) if len(t) != BS]
check("② 半满块绝不登记（改动前把只有 2 个 token 的块 1 登记了）",
      not bad, f"半满块 {[(i, t) for i, t in bad]}")
ok, info = registry_matches_prefix(bm, seq)
check("② 登记结果 == 真实前缀链（block 1 还没满，不该出现）", ok, info)


# ======================================================================
print()
print("=" * 70)
print("§3 投机没捞到候选、退回普通 decode —— 改动前 cached 每步多跑 k")
print("=" * 70)

s = fresh_sched()
bm = s.block_manager
seq = mk_seq(s, 4)
seq.num_scheduled_tokens = 4
call(Scheduler.postprocess, s, [seq], [42], True)       # prefill 完成
drift, err = [], None
for step in range(8):
    seq.num_scheduled_tokens = 3            # 投机按 1+k 预留了槽位…
    bm.may_append(seq, 3)
    e = call(Scheduler.postprocess, s, [seq], [200 + step], False)   # …但只交付 1 个
    if e is not None:
        err = e
        break
    drift.append((seq.num_cached_tokens, seq.num_tokens))
check("③ 退回普通 decode 时 cached 不漂移（恒等于 num_tokens - 1）",
      err is None and all(c == n - 1 for c, n in drift),
      err or f"逐步 (cached, num_tokens) = {drift}")
check("③ 只登记满块", all(len(t) == BS for _, t in registered(bm)),
      f"{[(i, len(t)) for i, t in registered(bm)]}")


# ======================================================================
print()
print("=" * 70)
print("§4 连续多步（普通 decode + 投机交替）后，哈希表必须精确等于真实前缀链")
print("=" * 70)

s = fresh_sched(num_blocks=128)
bm = s.block_manager
seq = mk_seq(s, 6)                          # 1.5 个块（最后一个半满）
seq.num_scheduled_tokens = 6
call(Scheduler.postprocess, s, [seq], [42], True)
plan = [2, 3, 1, 3, 1, 2, 1, 1, 3, 2, 1, 3]      # 交替模拟「投机落地多个 / 退回普通」
err = None
for i, n in enumerate(plan):
    if n == 1:
        seq.num_scheduled_tokens = 3           # 退回普通 decode：预留 1+k、只交付 1
        bm.may_append(seq, 3)
        e = call(Scheduler.postprocess, s, [seq], [300 + i], False)
    else:
        seq.num_scheduled_tokens = 3
        bm.may_append(seq, 3)
        e = call(Scheduler.postprocess_spec, s, [seq], [[400 + i * 10 + j for j in range(n)]])
    if e is not None:
        err = e
        break
ok, info = registry_matches_prefix(bm, seq)
check("④ 12 步之后哈希表 == 真实前缀链（不重不漏、无断链）", err is None and ok,
      err or info)
check("④ 期间没有登记过越界块（没有 IndexError）", err is None, err or "")


# ======================================================================
print()
print("=" * 70)
print("§5 红线：被拒的草稿候选绝不进哈希")
print("=" * 70)

s = fresh_sched()
bm = s.block_manager
seq = mk_seq(s, 4)
seq.num_scheduled_tokens = 4
call(Scheduler.postprocess, s, [seq], [42], True)
seq.draft_tokens = [777, 888]              # 提议了 2 个候选…
seq.num_scheduled_tokens = 3
bm.may_append(seq, 3)
call(Scheduler.postprocess_spec, s, [seq], [[101]])      # …只有第 1 个被接受
check("⑤ 草稿候选没进 token_ids", 777 not in seq.token_ids and 888 not in seq.token_ids,
      f"token_ids={seq.token_ids}")
check("⑤ 登记进哈希的块里没有草稿候选",
      all(777 not in t and 888 not in t for _, t in registered(bm)),
      f"{registered(bm)}")
check("⑤ 草稿用完即弃（draft_tokens 清空）", seq.draft_tokens == [],
      f"draft_tokens={seq.draft_tokens}")
ok, info = registry_matches_prefix(bm, seq)
check("⑤ 登记结果仍等于真实前缀链", ok, info)


# ======================================================================
print()
print("=" * 70)
print("§6 一个都没接受（拒绝回退）时不得登记任何东西")
print("=" * 70)

s = fresh_sched()
bm = s.block_manager
seq = mk_seq(s, 4)
seq.num_scheduled_tokens = 4
call(Scheduler.postprocess, s, [seq], [42], True)
before = len(bm.hash_to_block_id)
seq.num_scheduled_tokens = 3
bm.may_append(seq, 3)
seq.draft_tokens = [777]
call(Scheduler.postprocess_spec, s, [seq], [[]])          # toks 为空 = 全拒
check("⑥ 全拒时不新增登记", len(bm.hash_to_block_id) == before,
      f"{before} -> {len(bm.hash_to_block_id)}")
check("⑥ 全拒时 token_ids 没变长", seq.num_tokens == 5, f"num_tokens={seq.num_tokens}")
check("⑥ 全拒时没有半满块被登记",
      all(len(t) == BS for _, t in registered(bm)),
      f"{[(i, len(t)) for i, t in registered(bm)]}")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
