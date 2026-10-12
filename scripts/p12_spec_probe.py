#!/usr/bin/env python
"""p12_spec_probe.py —— P12 任务 5：**分块 / 跳过 draft prefill 与投机路径的交互**。

三组实验（env TEST）：

TEST=threshold  —— 门控打架：`spec_k>0` 但 `spec_batch_threshold` 让【解码时】投机关掉。
  · 场景：N_DEC 条解码稳态 + 一条长 prompt。
  · 比 prefill 墙钟：SKIP_OFF=0（照跑 draft prefill）vs SKIP_OFF=1（跳过开关打开）。
    spec_k=6>0 ⇒ `_draft_prefill_needed()` 恒 True ⇒ 【两个条件应当逐行相等】
    ⇒ 证明"prefill 时刻不知道未来解码批"这条会导致 draft prefill 白跑。
  · 再量：这一次解码轮实际有没有开投机（`spec_enabled` 判 False）。

TEST=flip —— 运行期把 spec_k 从 0 翻回正数：draft KV 的补齐路径能不能走通。
  · 引擎带 draft、跳过开关打开、spec_k=0；先把长 prompt prefill 完（draft KV 空），
    生成若干 token 后把 spec_k 翻成 6，继续到结束。
  · 量：有没有崩、投机有没有真的开起来（接受率 > 0）、`draft_valid_len` 有没有补齐。
  · 正确性的理论依据：验证用的是【target】的 logits，草稿只影响接受率、不影响输出分布
    （见 spec_decode 的拒绝采样）。所以这里要证的是"不崩 + 能跑起来"，不是"逐字节相同"。

用法:
  W=$PWD; TEST=threshold N_DEC=3 SKIP_OFF=0 CUDA_VISIBLE_DEVICES=6 NANOVLLM_PORT=2580 \
    /home/ziru/nano-vllm/venv/bin/python -u $W/scripts/p12_spec_probe.py
env: TEST(threshold|flip) CTX(16384) N_DEC(3) OUT(48) SKIP_OFF(0) THRESH(2) FLIP_AT(16) WIN(0)
"""
import json
import os
import statistics
import sys
from time import perf_counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
WIKI = os.path.join(ROOT, "data", "wikitext2_test.txt")


def dvl_gap(seq):
    """draft KV 水位与【已确认前缀】的差：>0 表示 draft 的 KV 落后于 target。"""
    return int(seq.num_cached_tokens - seq.draft_valid_len)


def main():
    import torch
    from nanovllm import LLM, SamplingParams

    TEST = os.environ.get("TEST", "threshold")
    CTX = int(os.environ.get("CTX", "16384"))
    N_DEC = int(os.environ.get("N_DEC", "3"))
    OUT = int(os.environ.get("OUT", "48"))
    SKIP_OFF = os.environ.get("SKIP_OFF", "0") == "1"
    THRESH = int(os.environ.get("THRESH", "2"))
    FLIP_AT = int(os.environ.get("FLIP_AT", "16"))
    WIN = int(os.environ.get("WIN", "0") or 0)
    CHUNK = int(os.environ.get("CHUNK", "0") or 0)

    MAXLEN = CTX + 256
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
              max_num_seqs=8, spec_k=6, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=(THRESH if TEST == "threshold" else 0),
              spec_skip_draft_prefill_when_off=SKIP_OFF)
    if WIN:
        kw["spec_draft_window"] = WIN
    if CHUNK:
        kw["spec_prefill_chunk_tokens"] = CHUNK
    llm = LLM(TARGET, **kw)
    mr = llm.model_runner
    sched = llm.scheduler
    bm = sched.block_manager

    with open(WIKI, encoding="utf-8") as f:
        ids = llm.tokenizer(f.read()).input_ids

    # ---- 数 draft prefill 的 token 量与调用次数 ----
    dstat = {"calls": 0, "tokens": 0}
    if mr.draft_model is not None:
        orig_fwd = mr.draft_model.forward

        def d_wrap(input_ids, positions=None):
            dstat["calls"] += 1
            try:
                dstat["tokens"] += int(input_ids.numel())
            except Exception:
                pass
            return orig_fwd(input_ids, positions)
        mr.draft_model.forward = d_wrap

    print("@@SPECHEAD@@" + json.dumps(dict(
        test=TEST, ctx=CTX, n_dec=N_DEC, out=OUT, skip_off=SKIP_OFF, thresh=THRESH,
        flip_at=FLIP_AT, chunk=CHUNK, win=WIN, spec_k=mr.config.spec_k,
        dev=torch.cuda.get_device_name(0), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))), flush=True)

    dec_prompt = ids[:64]
    long_ids = ids[2000:2000 + CTX]

    if TEST == "threshold":
        sp_dec = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
        for _ in range(N_DEC):
            llm.add_request(dec_prompt, sp_dec)
        for _ in range(6):
            llm.step()
        torch.cuda.synchronize()
        # 此刻 running 里正好 N_DEC 条 → 本轮 spec_enabled(active_b) 会怎么判？
        active_b = len(sched.running)
        spec_would = sched.spec_enabled(active_b)
        d0 = dict(dstat)
        llm.add_request(long_ids, SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True))
        pf = []
        while not llm.is_finished():
            _o, nt = llm.step()
            if nt > 0:
                pf.append(nt)
            torch.cuda.synchronize()
        print("@@SPEC@@" + json.dumps(dict(
            test="threshold", skip_off=SKIP_OFF, thresh=THRESH,
            active_b_at_prefill=active_b, spec_enabled_at_that_b=spec_would,
            n_prefill_rounds=len(pf),
            draft_prefill_calls=dstat["calls"] - d0["calls"],
            draft_prefill_tokens=dstat["tokens"] - d0["tokens"],
            note="spec_enabled=False 而 draft_prefill_tokens>0 ⇒ 这趟 draft 是白跑的"),
            ensure_ascii=False), flush=True)
        # 再单独确认：解码时的 active_b 与门控判定
        reps = []
        for b in range(1, N_DEC + 2):
            reps.append(dict(b=b, spec_enabled=sched.spec_enabled(b)))
        print("@@SPECGATE@@" + json.dumps(dict(threshold=THRESH, by_b=reps)), flush=True)

    else:  # flip
        # ★ 量"投机到底有没有真的跑起来"的指标：每个解码轮**实际交付**几个 token
        #   （`llm.step()` 的第二个返回值，decode 轮 = -交付数）。
        #   k=6 且草稿健康时 >1；草稿 KV 是陈旧的 ⇒ 接受率崩 ⇒ ≈1.0。
        sched.spec_k = 0                      # 先关投机
        llm.add_request(long_ids, SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True))
        sid = sched.waiting[-1].seq_id
        flipped_at = None
        seq_seen = None
        n_round = 0
        delivered_all = []                    # 每个 decode 轮的交付数
        delivered_after = []
        while not llm.is_finished():
            n_round += 1
            _o, nt = llm.step()
            torch.cuda.synchronize()
            if nt < 0:
                delivered_all.append(-nt)
                if flipped_at is not None:
                    delivered_after.append(-nt)
            seq = None
            for s in list(sched.running):
                if s.seq_id == sid:
                    seq = s
            seq_seen = seq
            if seq is not None and flipped_at is None and seq.num_completion_tokens >= FLIP_AT:
                pre_dvl = seq.draft_valid_len
                sched.spec_k = 6              # ★ 运行期翻回正数
                flipped_at = n_round
                print("@@FLIP@@ " + json.dumps(dict(
                    at_round=n_round, token=seq.num_completion_tokens,
                    draft_valid_len_at_flip=pre_dvl, seq_len=len(seq),
                    gap=dvl_gap(seq),
                    note="翻回正数后 draft KV 必须靠 proposer 补齐（catchup）")), flush=True)
        mean_all = statistics.mean(delivered_all) if delivered_all else None
        mean_after = statistics.mean(delivered_after) if delivered_after else None
        print("@@SPEC@@ " + json.dumps(dict(
            test="flip", skip_off=SKIP_OFF, n_rounds=n_round, flipped_at=flipped_at,
            n_decode_rounds=len(delivered_all),
            spec_k_effective_from_start=sched.spec_k,
            mean_delivered_per_round_all=round(mean_all, 3) if mean_all else None,
            mean_delivered_per_round_after_flip=round(mean_after, 3) if mean_after else None,
            final_draft_valid_len=(seq_seen.draft_valid_len if seq_seen else None),
            final_seq_len=(len(seq_seen) if seq_seen else None),
            draft_prefill_calls=dstat["calls"], draft_prefill_tokens=dstat["tokens"]),
            ensure_ascii=False), flush=True)

    import atexit
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
