"""A 步正确性对拍：前缀缓存命中后的 draft KV 复用 vs 全量补齐 vs 无复用参考。

★ 为什么判据是「draft logits + 接受率」而不是「输出 token」
  本项目的验证是【拒绝采样】且 q 就取本次 propose 的 draft logits —— 也就是说
  「草稿 KV 坏掉」不会让输出分布变错，只会让 q 变差、接受率悄悄下降。
  所以对拍必须看：
    · 同一已确认前缀上、第一个 propose 的【原始 logits】（增量路径 vs 全量重算）
    · 接受率 / 提出数 / 接受数
    · 输出 token（作为粗粒度自检：位置、slot 写错这类问题会在这里露出来）
  （这正是任务书说的「接受率悄悄变低，而且永远不会报错」。）

五种模式（每种单进程 + 全新引擎，跑完落盘 logits 供 a7_compare.py 对拍）
--------------------------------------------------------------------------
REF   只有 hitter 一条请求，前缀缓存是冷的 → 自己的 prefill 全算一遍（参考）
NEW   writer 先跑（写 target 前缀缓存 + draft KV + draft 块标记），再跑 hitter
      → hitter 走【复用】路径（本轮修复的目标路径）
OLD   同 NEW，但把 BlockManager.draft_valid_cached_blocks 钉成 0
      → hitter 被迫【全量补齐】= 修复前的行为
NEG   同 NEW，但把 writer 的 draft 块标记全部抹掉，并把 writer 用过的物理块的
      draft KV 清零 —— 复现「target 命中了前缀缓存，draft 侧却没写过这段」
      （负对照：修复前会在这条路径上白付补齐，朴素修复会直接读脏 KV）
NEGX  同 NEG，但把补齐也关掉 —— 模拟「朴素修复」（认为 target 命中 ⇒ draft 有效）
      → 必须能看出接受率/logits 崩掉，证明上面那道闸是承重的

用法: python a7_verify.py <mode> <rep>
"""
import os
import sys
import json
import hashlib
import importlib.util
from time import perf_counter

CODE_ROOT = os.environ.get("NV_ROOT", "/home/ziru/nano-vllm/p1-work")
sys.path.insert(0, CODE_ROOT)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.spec_decode.draft_proposer import DraftModelProposer
from nanovllm.engine.token_hook import TokenDeliveryHook
import nanovllm.spec_decode.verify as V

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
VOCAB = 151936

_s = importlib.util.spec_from_file_location(
    "a7_quant_lib", os.path.join(CODE_ROOT, "scripts/a7_quant.py"))
_a7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_a7)
build_prompt = _a7.build_prompt


def report(d):
    print("@@B@@" + json.dumps(d, ensure_ascii=False), flush=True)


def main():
    mode = sys.argv[1].upper()
    REP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    LS = int(os.environ.get("A7_LS", "4096"))
    SUF = int(os.environ.get("A7_SUF", "64"))
    OUTTOK = int(os.environ.get("A7_OUT_TOK", "24"))
    K = int(os.environ.get("A7_K", "6"))
    MML = 4608
    OUTDIR = os.environ.get("A7_OUT", os.path.join(CODE_ROOT, "a7-runs/verify"))
    os.makedirs(OUTDIR, exist_ok=True)

    shared = build_prompt("zh", 2, LS, 70000 + LS)
    w_prompt = shared + build_prompt("code", 3, SUF, 71000 + REP)
    h_prompt = shared + build_prompt("code", 7, SUF, 72000 + REP)
    assert len(w_prompt) == len(h_prompt) == LS + SUF

    # ---------------- 探针 ----------------
    ABORT_MARKS = [mode in ("NEG", "NEGX")]     # 抹掉 writer 的 draft 块标记
    ZERO_AFTER_WRITER = [mode in ("NEG", "NEGX")]
    NO_CATCHUP = [mode == "NEGX"]
    FORCE_NO_REUSE = [mode == "OLD"]

    # ★ 用 getattr 取（A 步之前的老版本没有这两个方法）—— 这样同一个脚本既能
    #   跑当前实现，也能跑 1c1d907 的裸 nanovllm，用来做「默认档是否逐字节等价」的对拍。
    _orig_mark = getattr(BlockManager, "mark_draft_valid", None)
    _orig_dcv = getattr(BlockManager, "draft_valid_cached_blocks", None)
    _orig_req = ModelRunner._draft_request

    CFG = dict(writer_done=[False])

    def _mark(self, seq, wm):
        if _orig_mark is None or (ABORT_MARKS[0] and not CFG["writer_done"][0]):
            return
        return _orig_mark(self, seq, wm)

    def _dcv(self, seq, nb):
        if FORCE_NO_REUSE[0] or _orig_dcv is None:
            return 0
        return _orig_dcv(self, seq, nb)

    def _req(self, seq):
        d = _orig_req(self, seq)
        if NO_CATCHUP[0] and not CFG["armed_seen"][0]:
            d["catchup_tokens"] = []
        return d

    BlockManager.mark_draft_valid = _mark
    if _orig_dcv is not None or FORCE_NO_REUSE[0]:
        BlockManager.draft_valid_cached_blocks = _dcv
    ModelRunner._draft_request = _req

    ACC = {"p": 0, "a": 0}
    _orig_vb = V.verify_batch

    def _tvb(dp, tl, dt, temperatures=None, draft_is_point_mass=False):
        res = _orig_vb(dp, tl, dt, temperatures, draft_is_point_mass)
        if not draft_is_point_mass:
            ACC["p"] += int(res.n_proposed.sum())
            ACC["a"] += int(res.accept_mask.sum())
        return res

    V.verify_batch = _tvb

    CAP = {"logits": None, "arm": False, "calls": 0}
    CFG["armed_seen"] = [False]
    _orig_pb = DraftModelProposer.propose_batch

    def _pb(self, reqs):
        chains, logits = _orig_pb(self, reqs)
        if CAP["arm"]:
            CAP["calls"] += 1
            CFG["armed_seen"][0] = True
            if CAP["logits"] is None:
                CAP["logits"] = logits.detach().float().cpu().clone()
        return chains, logits

    DraftModelProposer.propose_batch = _pb

    # ---------------- 建引擎 ----------------
    hook = TokenDeliveryHook()
    llm = LLM(TARGET, token_hook=hook, max_model_len=MML, max_num_batched_tokens=16384,
              max_num_seqs=8, enforce_eager=False, spec_k=K, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0)
    mr = llm.model_runner
    prop = mr.spec_proposer
    bm = llm.scheduler.block_manager

    sp = SamplingParams(temperature=1.0, max_tokens=OUTTOK, ignore_eos=True)
    llm.generate([h_prompt], SamplingParams(temperature=1.0, max_tokens=8,
                                            ignore_eos=True), use_tqdm=False)
    # ★ 统一冷启动：warmup 用的就是 h_prompt，它会把这批块登记进前缀缓存，
    #   不清掉的话后面的每个模式都已经是「命中」状态，REF 就不再是参考了。
    bm.hash_to_block_id.clear()

    if mode != "REF":                       # writer 先跑，建立共享前缀的缓存
        llm.generate([w_prompt], sp, use_tqdm=False)
        CFG["writer_done"][0] = True
        if ZERO_AFTER_WRITER[0]:
            # 负对照：把 draft 的 KV 整片清零 —— 代表「这些物理块里存的不是
            # 这段前缀的内容」（draft 没按这个前缀写过 + 内容是脏的）。
            with torch.no_grad():
                mr.draft_kv_cache.zero_()
            torch.cuda.synchronize()

    # ---------------- 测 hitter ----------------
    hook.requests.clear()
    ACC.update(p=0, a=0)
    d0 = dict(rounds=prop.n_rounds, batch_forwards=prop.n_batch_forwards,
              catchup_forwards=prop.n_catchup_forwards,
              catchup_tokens=prop.n_catchup_tokens)
    torch.manual_seed(1234 + REP)
    torch.cuda.synchronize()
    t0 = perf_counter()
    CAP["arm"] = True
    outs = llm.generate([h_prompt], sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall = perf_counter() - t0
    d1 = dict(rounds=prop.n_rounds, batch_forwards=prop.n_batch_forwards,
              catchup_forwards=prop.n_catchup_forwards,
              catchup_tokens=prop.n_catchup_tokens)
    dc = {kk: d1[kk] - d0[kk] for kk in d0}

    s = hook.summary()
    sid = sorted(s.keys())[0]
    toks = list(outs[0]["token_ids"])
    lg = CAP["logits"]
    lg_path = os.path.join(OUTDIR, "%s_r%d_logits.pt" % (mode, REP))
    if lg is not None:
        torch.save(lg, lg_path)

    report(dict(
        mode=mode, rep=REP, L=LS + SUF, LS=LS, SUF=SUF, k=K, out=OUTTOK,
        hitter_prompt_md5=hashlib.md5(json.dumps(h_prompt).encode()).hexdigest(),
        writer_prompt_md5=hashlib.md5(json.dumps(w_prompt).encode()).hexdigest(),
        out_token_ids=toks,
        out_md5=hashlib.md5(json.dumps(toks).encode()).hexdigest(),
        n_out=len(toks),
        accept_rate=(round(ACC["a"] / ACC["p"], 6) if ACC["p"] else None),
        proposed=ACC["p"], accepted=ACC["a"],
        draft_counters=dc,
        n_propose_calls_armed=CAP["calls"],
        draft_logits_saved=(lg_path if lg is not None else None),
        draft_logits_shape=(list(lg.shape) if lg is not None else None),
        ttft_ms=round(s[sid]["ttft"] * 1000, 2),
        wall_s=round(wall, 4),
        zeroed_draft_kv=bool(ZERO_AFTER_WRITER[0]),
    ))


if __name__ == "__main__":
    main()
