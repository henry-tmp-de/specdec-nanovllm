"""B 步：draft 滑窗的 W 曲线（接受率 / 吞吐 / 显存 / 容量 / 正确性）。

W 是【运行期开关】`spec_draft_window`（0 = 全上下文 = 1c1d907 的行为），
所以全上下文与 W=256/512/1024/2048 是同一份二进制、同一个进程里翻转的 ——
测出来的差异不会混进代码版本差异。

两类负载（★ 不能只用一类：重复性素材会把接受率抬到 ~0.99）
  rep  —— p6 的「池化轮转拼接」素材（与冻结基线可比，接受率偏高）
  nat  —— 自然长文：把中英文语料做一个 seeded 随机排列后原样拼接，
          不加重号/编号、不循环 → 非周期性，接受率更低也更接近真实

记录：接受率、端到端吞吐、TTFT、每轮每序列产出 token 数
      （口径：总产出 token / (全局轮次 × B)，见任务书「测量口径的坑」）、
      KV 块数（target/draft 池）、KV 总字节、每 token 的 KV 字节、
      峰值显存、draft 滑窗回退次数。

用法: python p7_window.py <W> <workload: rep|nat> <rep> <B> <L> <OUT> <k>
"""
import os
import sys
import json
import hashlib
from time import perf_counter

CODE_ROOT = os.environ.get("NV_ROOT", "/home/ziru/nano-vllm/p1-work")
sys.path.insert(0, CODE_ROOT)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.token_hook import TokenDeliveryHook
import nanovllm.spec_decode.verify as V

import importlib.util as _iu
_s = _iu.spec_from_file_location("a7q", os.path.join(CODE_ROOT, "scripts/a7_quant.py"))
_a7 = _iu.module_from_spec(_s)
_s.loader.exec_module(_a7)
ZH, CODE, build_prompt = _a7.ZH, _a7.CODE, _a7.build_prompt

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")


def report(d):
    print("@@B@@" + json.dumps(d, ensure_ascii=False), flush=True)


def build_natural(seed, L):
    """自然长文：seeded 随机排列后原样拼接（去重号、不循环）。"""
    t = _a7.tok()
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ZH) + len(CODE), generator=g).tolist()
    pool = ([ZH[i] for i in order if i < len(ZH)]
            + [CODE[i - len(ZH)] for i in order if i >= len(ZH)])
    text, ids = "", []
    reps = 0
    while len(ids) < L and reps < 8:
        for seg in pool:
            if len(ids) >= L:
                break
            text = text + "\n" + seg
            ids = t.encode(text, add_special_tokens=False)
        reps += 1
    assert len(ids) >= L, (L, len(ids))
    return ids[:L]


def main():
    W = int(sys.argv[1])
    wl = sys.argv[2]
    REP = int(sys.argv[3])
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    L = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    OUT = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    K = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    MML = 4608

    if wl == "nat":
        prompts = [build_natural(50000 + REP * 977 + i * 31, L) for i in range(B)]
    else:
        prompts = [build_prompt("zh" if i % 2 == 0 else "code", (REP * 5 + i) % 12, L,
                                40000 + L + i + REP * 13) for i in range(B)]

    hook = TokenDeliveryHook()
    llm = LLM(TARGET, token_hook=hook, max_model_len=MML, max_num_batched_tokens=16384,
              max_num_seqs=8, enforce_eager=False, spec_k=K, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0, spec_draft_window=W)
    mr, bm = llm.model_runner, llm.scheduler.block_manager
    prop = mr.spec_proposer
    hf, dhf = mr.config.hf_config, mr.draft_hf_config

    nb = len(bm.blocks)
    nd = len(mr.draft_kv_cache[0, 0]) if mr.draft_kv_cache is not None else 0
    tb = _a7.kv_bytes(hf)
    db = _a7.kv_bytes(dhf)
    tgt_blocks_per_seq = (L + OUT + MML) // mr.block_size + 2

    sp_w = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
    llm.generate([prompts[0]], sp_w, use_tqdm=False)

    sp = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
    ACC = {"p": 0, "a": 0}
    _ovb = V.verify_batch

    def _tvb(dp, tl, dt, temperatures=None, draft_is_point_mass=False):
        res = _ovb(dp, tl, dt, temperatures, draft_is_point_mass)
        if not draft_is_point_mass:
            ACC["p"] += int(res.n_proposed.sum())
            ACC["a"] += int(res.accept_mask.sum())
        return res
    V.verify_batch = _tvb

    bm.hash_to_block_id.clear()
    hook.requests.clear()
    d0 = dict(rounds=prop.n_rounds, batch_forwards=prop.n_batch_forwards,
              catchup_forwards=prop.n_catchup_forwards, catchup_tokens=prop.n_catchup_tokens)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall = perf_counter() - t0
    d1 = dict(rounds=prop.n_rounds, batch_forwards=prop.n_batch_forwards,
              catchup_forwards=prop.n_catchup_forwards, catchup_tokens=prop.n_catchup_tokens)
    dc = {k: d1[k] - d0[k] for k in d0}

    total = sum(len(o["token_ids"]) for o in outs)
    s = hook.summary()
    ttfts = [s[i]["ttft"] for i in sorted(s) if hook.requests[i]["first_token"]]
    rounds = dc["rounds"]
    info = dict(
        W=W, workload=wl, rep=REP, B=B, L=L, out=OUT, k=K,
        prompt_md5s=[hashlib.md5(json.dumps(p).encode()).hexdigest()[:8] for p in prompts],
        output_tok_per_s=round(total / wall, 2),
        total_out_tokens=total, wall_s=round(wall, 4),
        ttft_median_ms=round(_a7.median(ttfts) * 1000, 2) if ttfts else None,
        accept_rate=(round(ACC["a"] / ACC["p"], 4) if ACC["p"] else None),
        proposed=ACC["p"], accepted=ACC["a"],
        rounds=rounds,
        # ★ 口径：全局轮次 × B（逐请求路径下 rounds 是每请求的，这里引擎走批量）
        out_tokens_per_round_per_seq=(round(total / (rounds * B), 4) if rounds else None),
        draft_counters=dc,
        # ---- 显存 / 容量 ----
        num_kvcache_blocks=nb,
        num_draft_blocks=nd,
        draft_window_blocks=mr.draft_window_blocks,
        spec_draft_window=getattr(mr.config, "spec_draft_window", None),
        target_block_kb=round(tb / 1024, 1), draft_block_kb=round(db / 1024, 1),
        kv_total_gb=round((nb * tb + nd * db) / 2**30, 4),
        kv_target_gb=round(nb * tb / 2**30, 4), kv_draft_gb=round(nd * db / 2**30, 4),
        kv_bytes_per_token=round((nb * tb + nd * db) / (nb * mr.block_size), 1),
        target_capacity_tokens=nb * mr.block_size,
        mem_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        window_fallbacks=getattr(llm.scheduler, "_window_fallbacks", 0),
    )
    report(info)


if __name__ == "__main__":
    main()
