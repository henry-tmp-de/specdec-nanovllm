#!/usr/bin/env python
"""p14_equiv.py —— P14 的【逐值对拍】：默认关闭时必须与老代码一字不差。

为什么要有这个脚本
------------------
P14 改的是 `schedule()` 的 prefill 循环体 —— 本项目最容易"静默出错"的地方
（错了不崩，只会少调度/重复调度）。所以交付标准不是"测试过了"，而是
**同一份脚本、同一组配置，指向【基线代码】与【改动后的代码】各跑一遍，
调度轨迹逐值相同**（沿用 docs/验收 里既有的对拍手法）。

★ 逐值对拍必须跑两棵树：
    ROOT=基线   /home/ziru/nano-vllm/p14-base   (= main@001f712，P14 之前)
    ROOT=改动后 /home/ziru/nano-vllm/p14-work

两个模式
--------
MODE=cpu（默认，纯 CPU、不 import torch）—— **调度轨迹**对拍：
    绕开 __init__ 铺 Scheduler，按固定脚本逐轮 schedule()，记录
    `(is_prefill, [(序列序号, 本轮进几个 token), ...])` 的逐轮序列。
    覆盖「chunk 0/2/4/8 × 预算宽/紧 × 有/无解码请求」的矩阵。
    · rotate=False（= 老路径）  ⇒ 必须与基线【逐值相同】
    · rotate=True （= 新功能）  ⇒ 在有竞争的场景上【必须不同】（否则说明开关没生效）

MODE=engine（GPU）—— **执行路径**对拍：真引擎 + 固定种子 + 固定负载，
    比对每次前向的 (input_ids, positions, is_prefill, logits) bitwise md5
    与最终产出 token_ids 的 md5。默认路径（两个 P14 开关都不给）必须全等。

用法
----
    # 纯 CPU：两棵树各跑一遍，比 @@P14EQUIV@@ 行的 trace_md5
    python -u scripts/p14_equiv.py                 # 在各自的工作区里跑
    # GPU（每个进程一个引擎，CASE 一个进程）
    CASE=default CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2501 \
        /home/ziru/nano-vllm/venv/bin/python -u scripts/p14_equiv.py
"""
import hashlib
import json
import os
import sys
import types
from collections import deque
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BS = 4                                  # 块大小：小一点，纯 CPU 也快
EOS = 999999

# (chunk, max_num_batched_tokens, n_running, waiting 的长度列表)
CPU_CASES = [
    ("chunk0_wide",    0, 1024, 0, [20, 20]),
    ("chunk0_narrow",  0, 24,   0, [20, 20, 20]),
    ("chunk0_mixed",   0, 1024, 1, [20, 20, 20]),
    ("chunk8_wide",    8, 1024, 1, [32, 32, 32]),
    ("chunk8_narrow",  8, 16,   1, [24, 24, 24]),
    ("chunk8_tight",   8, 8,    1, [24, 24, 24]),
    ("chunk4_ragged",  4, 128,  2, [17, 33, 9]),
    ("chunk2_many",    2, 10,   1, [9, 5, 7, 13]),
]


def _stub_config():
    """只加载 nanovllm 的纯 CPU 部分（不 import torch / transformers）。"""
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [os.path.join(ROOT, "nanovllm")]
    sys.modules.setdefault("nanovllm", pkg)
    cfg = types.ModuleType("nanovllm.config")
    cfg.Config = object
    sys.modules.setdefault("nanovllm.config", cfg)


def mk_sched(chunk, lens, n_running, max_bt):
    """绕开 __init__ 铺一个 Scheduler（两棵树通用 —— 不依赖任何 P14 符号）。"""
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.sampling_params import SamplingParams

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
    # ★ 老树没有这两个属性 —— setattr 无害（它的 schedule() 根本不读），
    #   正好用来证明"老树 + rotate=True 与 rotate=False 完全一样（开关不存在）"。
    s.prefill_rotate = False
    s._rotation = None
    Sequence.block_size = BS
    seqs = []
    for ln in lens:
        w = Sequence([3] * ln, SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                              ignore_eos=True))
        s.waiting.append(w)
        seqs.append(w)
    for _ in range(n_running):
        r = Sequence([7] * (BS + 1), SamplingParams(temperature=1.0, max_tokens=10 ** 6,
                                                    ignore_eos=True))
        s.block_manager.allocate(r, 0)
        r.status = 1
        s.running.append(r)
        seqs.append(r)          # 解码序列也进索引（轨迹里会出现它们的调度）
    return s, seqs


def trace(case, rotate):
    """跑完整个场景，返回逐轮调度轨迹（纯调度语义，不含任何时间/显存）。"""
    name, chunk, max_bt, nrun, lens = case
    s, seqs = mk_sched(chunk, lens, nrun, max_bt)
    s.prefill_rotate = bool(rotate)
    idx = {id(x): i for i, x in enumerate(seqs)}
    rounds = []
    # ★ 判据是"waiting 空了"而不是"引擎空了"：本模拟里没有序列会结束，
    #   解码请求会一直留在 running 里。我们比的是【prefill 的调度过程】。
    for _ in range(500):
        if not s.waiting:
            break
        scheduled, is_prefill = s.schedule()
        rounds.append([1 if is_prefill else 0,
                       [(idx[id(x)], int(x.num_scheduled_tokens)) for x in scheduled]])
        for x in scheduled:
            if is_prefill:
                x.num_cached_tokens += x.num_scheduled_tokens
            x.num_scheduled_tokens = 0
        if len(rounds) > 400:
            break
    return rounds


def trace_md5(rounds):
    h = hashlib.md5()
    for r in rounds:
        h.update(json.dumps(r).encode())
        h.update(b"|")
    return h.hexdigest()


def run_cpu():
    _stub_config()
    out = []
    for case in CPU_CASES:
        name = case[0]
        t_off = trace(case, rotate=False)
        t_on = trace(case, rotate=True)
        row = dict(case=name, chunk=case[1], max_bt=case[2], n_running=case[3],
                   lens=case[4], rounds=len(t_off),
                   md5_rotate_off=trace_md5(t_off), md5_rotate_on=trace_md5(t_on),
                   same_on_off=(t_off == t_on),
                   trace=[r for r in t_off])
        out.append(row)
        print("@@P14CASE@@" + json.dumps(row, ensure_ascii=False), flush=True)
    print("@@P14EQUIV@@" + json.dumps(dict(root=ROOT, mode="cpu", cases=out),
                                      ensure_ascii=False), flush=True)


# ======================================================================
#  MODE=engine：真引擎 + 固定种子，逐次前向的 bitwise md5
# ======================================================================
ENGINE_CASES = {
    # 默认路径：两个 P14 开关都不给 ⇒ 必须与基线全等
    "default":      dict(chunk=0,    rotate=False, B=3, ctx=1024, out=48),
    "chunk_off":    dict(chunk=2048, rotate=False, B=3, ctx=4096, out=48),
    # 新功能（基线树没有此概念 ⇒ 允许不同，用来证明开关真的起作用）
    "rotate_on":    dict(chunk=2048, rotate=True,  B=3, ctx=4096, out=48),
}
SEED = 20261013


def run_engine():
    _stub_config()
    case = os.environ.get("CASE", "default")
    c = ENGINE_CASES[case]
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
    DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
    WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")

    kw = dict(enforce_eager=False, max_model_len=8192, max_num_batched_tokens=16384,
              max_num_seqs=8, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    # ★ 老树的 Config 里没有这两个字段 ⇒ 会被 LLMEngine 的字段过滤静默丢掉
    #   （等价于"开关关着"），所以这一个脚本能同时跑两棵树。
    kw["spec_prefill_chunk_tokens"] = c["chunk"]
    kw["spec_prefill_rotate"] = c["rotate"]
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner

    H = hashlib.md5()
    N = [0]
    orig_rm = ModelRunner.run_model

    def rm_wrap(self, input_ids, positions, is_prefill):
        out = orig_rm(self, input_ids, positions, is_prefill)
        N[0] += 1
        H.update(b"RM")
        H.update(input_ids.detach().to("cpu", torch.int64).numpy().tobytes())
        H.update(positions.detach().to("cpu", torch.int64).numpy().tobytes())
        H.update(b"P" if is_prefill else b"D")
        H.update(out.detach().to("cpu", torch.float32).numpy().tobytes())
        return out
    ModelRunner.run_model = rm_wrap

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids
    B, ctx = c["B"], c["ctx"]
    span = (len(ids) - ctx) // B
    prompts = [ids[i * span: i * span + ctx] for i in range(B)]

    torch.manual_seed(SEED)
    sp = SamplingParams(temperature=1.0, max_tokens=c["out"], ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)

    oh = hashlib.md5()
    for o in outs:
        oh.update(",".join(map(str, o["token_ids"])).encode())
        oh.update(b"|")

    print("@@P14EQUIV@@" + json.dumps(dict(
        root=ROOT, mode="engine", case=case, chunk=c["chunk"], rotate=c["rotate"],
        B=B, ctx=ctx, out=c["out"], n_forwards=N[0],
        logits_md5=H.hexdigest(), out_md5=oh.hexdigest(),
        blocks=mr.config.num_kvcache_blocks,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        port=os.environ.get("NANOVLLM_PORT")), sort_keys=True), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    if os.environ.get("MODE", "cpu") == "engine":
        run_engine()
    else:
        run_cpu()
