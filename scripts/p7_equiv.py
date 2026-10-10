"""逐位等价对拍（M1）：同一份 dump 脚本，两个用途共用。

  用途 A  `spec_draft_window=0` vs 基线 1c1d907
          → 默认档必须逐字节不变（用户硬性要求）。
  用途 B  W=2048 vs W=0（L=1024：窗口盖住全部上下文）
          → 滑窗档必须【逐步】等价，而不是"接受率差不多"。

对拍的不是统计量，是【原始字节】：
  · 每次 draft 前向的 (tokens, positions, context_lens, block_tables, slot_mapping)
  · 该次前向返回的 logits 的 md5（bitwise，float32 原始字节）
  · 每次 propose_batch 的请求几何量 + 候选 token
  · 目标模型验证前向 logits 的 md5
  · 最终输出的 token_ids + md5
  · 前 NSAVE 次 draft 前向的原始 logits 张量（落盘 .pt，用于算 max|Δ|）

跨进程比较：两次运行的 dump.json 必须逐字段完全相同（p7_equiv_cmp.py）。

用法:
  NV_ROOT=<代码根> python scripts/p7_equiv.py <W> <L> <OUT> <B> <dump.json> [seed]

没开滑窗（W=0）时不传 spec_draft_window —— 基线 1c1d907 的 Config 没有这个字段。
"""
import os
import sys
import json
import hashlib

CODE_ROOT = os.environ.get("NV_ROOT", "/home/ziru/nano-vllm/p1-work")
sys.path.insert(0, CODE_ROOT)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.draft_proposer import DraftModelProposer

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

import importlib.util as _iu
_spec = _iu.spec_from_file_location("a7q", os.path.join(CODE_ROOT, "scripts/a7_quant.py"))
_a7 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_a7)
build_prompt = _a7.build_prompt

NSAVE = int(os.environ.get("P7_NSAVE", "6"))   # 前 N 次 draft 前向的原始 logits 落盘
NMAX = 4000        # 记录上限（防 JSON 爆掉）
DUMP_KV = os.environ.get("P7_DUMP_KV", "0") == "1"   # 记录每次前向读到的 KV 内容哈希


def md5_tensor(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _kcache(model):
    """draft KV cache 的 layer0 k 张量 (nb, bs, nkv, hd)（找不到就返回 None）。"""
    for m in model.modules():
        kc = getattr(m, "k_cache", None)
        if kc is not None and hasattr(kc, "numel") and kc.numel():
            return kc
    return None


def main():
    W = int(sys.argv[1]); L = int(sys.argv[2]); OUT = int(sys.argv[3])
    B = int(sys.argv[4]); path = sys.argv[5]
    SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 777
    K = 6

    REC = {"forwards": [], "proposes": [], "verifies": []}
    SAVED = []

    _ofg = DraftModelProposer._forward_group

    def _fg(self, tokens, positions, block_tables, context_lens, slot_mapping):
        kv_hash = ""
        if DUMP_KV:
            kc = _kcache(self.model)
            if kc is not None:
                parts = []
                for bt_i in block_tables:
                    idx = [int(b) for b in bt_i if int(b) >= 0]
                    if idx:
                        parts.append(kc[idx].float().contiguous().cpu().numpy().tobytes())
                kv_hash = hashlib.md5(b"".join(parts)).hexdigest()
        lg = _ofg(self, tokens, positions, block_tables, context_lens, slot_mapping)
        if len(REC["forwards"]) < NMAX:
            tok = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
            REC["forwards"].append(dict(
                tokens=[int(x) for x in tok],
                positions=[int(x) for x in positions],
                ctx=[int(x) for x in context_lens],
                bt=[[int(x) for x in r] for r in block_tables],
                slot=[int(x) for x in slot_mapping],
                kv_md5=kv_hash,
                logits_md5=md5_tensor(lg),
            ))
            if len(SAVED) < NSAVE:
                SAVED.append(lg.detach().float().cpu().clone())
        return lg

    DraftModelProposer._forward_group = _fg

    _opb = DraftModelProposer.propose_batch

    def _norm(r):
        """只留两边版本都有的键，并给滑窗键补默认值 —— 否则基线（1c1d907 的
        请求字典里没有 window_blocks/valid_from/draft_ring）会与当前版本
        在键集合上"假差异"。"""
        return dict(block_table=[int(x) for x in r.get("block_table", [])],
                    context_len=int(r.get("context_len", 0)),
                    last_token=int(r.get("last_token", -1)),
                    catchup_start=int(r.get("catchup_start", 0)),
                    catchup_tokens=[int(x) for x in (r.get("catchup_tokens") or [])],
                    window_blocks=int(r.get("window_blocks") or 0),
                    valid_from=int(r.get("valid_from") or 0),
                    draft_ring=[int(x) for x in (r.get("draft_ring") or [])])

    def _pb(self, reqs):
        ch, lg = _opb(self, reqs)
        REC["proposes"].append(dict(
            reqs=[_norm(r) for r in reqs],
            cands=[[int(t) for t in c] for c in ch],
            logits_md5=md5_tensor(lg),
        ))
        return ch, lg

    DraftModelProposer.propose_batch = _pb

    prompts = [build_prompt("zh" if i % 2 == 0 else "code", (5 * i) % 12, L,
                            40000 + L + i) for i in range(B)]
    kw = dict(max_model_len=4608, max_num_batched_tokens=16384, max_num_seqs=8,
              enforce_eager=False, spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    if W > 0:
        kw["spec_draft_window"] = W
    llm = LLM(TARGET, **kw)
    mr, bm = llm.model_runner, llm.scheduler.block_manager

    # 目标模型验证前向的 logits 也要对拍（输出 token 只是它的粗粒度投影）
    rvf = getattr(mr, "run_verify_forward", None)
    if rvf is not None:
        def _rvf(input_ids, positions, *a, **k):
            lg = rvf(input_ids, positions, *a, **k)
            REC["verifies"].append(dict(
                positions=[int(x) for x in positions.tolist()],
                logits_md5=md5_tensor(lg)))
            return lg
        mr.run_verify_forward = _rvf

    # 预热：与 p7_window/p7_diag 同一套（保证 RNG 消耗一致可比）
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=8,
                                              ignore_eos=True), use_tqdm=False)
    bm.hash_to_block_id.clear()
    REC["forwards"].clear(); REC["proposes"].clear(); REC["verifies"].clear()
    SAVED.clear()
    torch.manual_seed(SEED)

    outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=OUT,
                                                ignore_eos=True), use_tqdm=False)
    cands = [[int(t) for t in o["token_ids"]] for o in outs]
    d = dict(
        W=W, L=L, OUT=OUT, B=B, seed=SEED, k=K,
        window_blocks=getattr(mr, "draft_window_blocks", 0),
        num_kvcache_blocks=len(bm.blocks),
        num_draft_blocks=(len(mr.draft_kv_cache[0, 0]) if mr.draft_kv_cache is not None else 0),
        out_token_ids=cands,
        out_md5=hashlib.md5(json.dumps(cands).encode()).hexdigest(),
        n_forwards=len(REC["forwards"]),
        n_proposes=len(REC["proposes"]),
        n_verifies=len(REC["verifies"]),
        forwards=REC["forwards"], proposes=REC["proposes"], verifies=REC["verifies"],
    )
    with open(path, "w") as f:
        json.dump(d, f)
    if SAVED:
        torch.save(torch.stack(SAVED), path + ".logits.pt")
    print("@@E@@" + json.dumps({k: d[k] for k in
          ("W", "L", "OUT", "B", "window_blocks", "out_md5", "n_forwards",
           "n_proposes", "n_verifies", "num_kvcache_blocks", "num_draft_blocks")}),
          flush=True)


if __name__ == "__main__":
    main()
