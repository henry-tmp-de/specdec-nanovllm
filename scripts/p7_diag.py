"""B 步诊断：把滑窗档【实际用到的几何量】和【逐候选位的接受率】打出来。

为什么需要它
------------
W=512(M=2) 的接受率（0.24）比窗口更小的 W=256(M=1)（0.48）还差 —— 非单调。
非单调基本只可能是实现问题（窗口更大不可能更差），但光看端到端接受率看不出
根因，所以这里直接打印：

  · 每次 propose 用到的 (pos, b0, clen, block_table) —— 窗口到底给了多少上下文
  · 逐候选位 s 的接受率 accept_mask[:, s] —— 是第 0 位就差，还是后面几步才崩
  · 同一配置下 W=0（全上下文）的对照

用法: python p7_diag.py <W> <L> <B> <out_tokens> <max_log_calls>
"""
import os
import sys
import json
import hashlib

CODE_ROOT = os.environ.get("NV_ROOT", "/home/ziru/nano-vllm/p1-work")
sys.path.insert(0, CODE_ROOT)

import torch
import importlib.util as _iu
_s = _iu.spec_from_file_location("a7q", os.path.join(CODE_ROOT, "scripts/a7_quant.py"))
_a7 = _iu.module_from_spec(_s)
_s.loader.exec_module(_a7)
_a7.install_can_allocate_probe()

from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.draft_proposer import DraftModelProposer
import nanovllm.spec_decode.verify as V

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")


def main():
    W = int(sys.argv[1])
    L = int(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    K = 6
    MAXLOG = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    PROBE = os.environ.get("P7_PROBE", "0") == "1"

    GEO = []
    if PROBE:
        _og = DraftModelProposer._geom

        def _g(self, r, pos, valid_from=None):
            got = _og(self, r, pos, valid_from)
            if len(GEO) < MAXLOG:
                M = int(r.get("window_blocks") or 0)
                GEO.append(dict(pos=int(pos), M=M, valid_from=int(
                    r.get("valid_from", 0) if valid_from is None else valid_from),
                    b0=(int(pos) - got[2] + 1) // self.block_size if got[2] > 0 else -1,
                    clen=int(got[2]), bt=list(got[1])))
            return got
        DraftModelProposer._geom = _g

    PERS = [0] * K
    PRINT = [0] * K
    _ovb = V.verify_batch

    def _tvb(dp, tl, dt, temperatures=None, draft_is_point_mass=False):
        res = _ovb(dp, tl, dt, temperatures, draft_is_point_mass)
        if not draft_is_point_mass:
            m = res.accept_mask
            for s in range(m.shape[1]):
                PERS[s] += int(m[:, s].sum())
                PRINT[s] += int(m[:, s].numel())
        return res
    V.verify_batch = _tvb

    prompts = [_a7.build_prompt("zh" if i % 2 == 0 else "code", (5 * i) % 12, L,
                                40000 + L + i) for i in range(B)]
    llm = LLM(TARGET, max_model_len=4608, max_num_batched_tokens=16384, max_num_seqs=8,
              enforce_eager=False, spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0, spec_draft_window=W)
    mr, bm = llm.model_runner, llm.scheduler.block_manager
    prop = mr.spec_proposer
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=8,
                                              ignore_eos=True), use_tqdm=False)
    bm.hash_to_block_id.clear()
    if PROBE:
        GEO.clear()
    # ★ 固定采样种子：不同 W 之间要能逐位对比（否则生成文本不同，
    #   接受率会因为「跑到了素材的哪一段」而系统性漂移，看起来像 bug）。
    torch.manual_seed(4242)
    out = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=OUT,
                                               ignore_eos=True), use_tqdm=False)
    tot = sum(len(o["token_ids"]) for o in out)
    per_s = [(round(PERS[s] / PRINT[s], 4) if PRINT[s] else None) for s in range(K)]
    print("@@D@@" + json.dumps(dict(
        W=W, L=L, B=B, out=OUT, k=K,
        num_kvcache_blocks=len(bm.blocks), num_draft_blocks=len(mr.draft_kv_cache[0, 0]),
        window_blocks=mr.draft_window_blocks,
        rounds=prop.n_rounds, out_tokens=tot,
        accept_overall=(round(sum(PERS) / max(1, sum(PRINT)), 4)),
        accept_per_position_s=per_s,
        clamp_hits=mr._window_fallbacks if hasattr(mr, "_window_fallbacks") else None,
        geometry=GEO if PROBE else None,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
