#!/usr/bin/env python
"""p0_weight_bw.py —— Phase 0 闸门脚本（INT8 权重量化前的上界测量）

做两件事，都用【完全不改引擎】的 monkeypatch 方式：

  A) 代理实验：同一个配置（同 B / 同 ctx / 同输出长度 / 同 graph 开关）下，
     换不同权重规模的模型（Qwen3-4B / 1.7B / 0.6B），看 decode tok/s
     是否按权重字节数成比例变化。
     → 若成比例 ⇒ decode 确实是纯权重流式瓶颈 ⇒ W8A16 的上界推算成立。

  B) device 侧 kernel 分解（B=1 与 B=4 各一份），用来算 W8A16 的 Amdahl 上界。
     GEMM 占比 f。注意 79.27% 是【B=1 的 device time 占比】，B=4 必须单独测。

口径：
  · decode-only tok/s 由 TokenDeliveryHook 的 TPOT 反推（1/TPOT），
    不含 prefill / 排队，与台账 §0 的「tok/s」口径分开记。
  · 每组时间跑 RUNS 次取中位数 + 组间范围。
  · 时间（hook）与 profiler 分开跑。

用法:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/p0_weight_bw.py
env:
  MODEL   模型路径（默认 ~/nano-vllm/models/Qwen3-4B）
  TAG     输出子目录名（默认取模型 basename）
  CTXS    逗号分隔（默认 "1024,4096"）
  BATCHES 逗号分隔（默认 "1,4"）
  OUTLEN  输出 token 数（默认 256）
  RUNS    每配置计时次数（默认 5）
  GRAPHS  逗号分隔（默认 "1"）
  PROFILE 1=也跑 profiler 分解（默认 0）
"""
import os
import sys
import gc
import json
import time
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
# from nanovllm.engine.block_manager import BlockManager  # 不再 patch
from nanovllm.engine.token_hook import TokenDeliveryHook

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/nano-vllm/models/Qwen3-4B"))
TAG = os.environ.get("TAG", os.path.basename(MODEL.rstrip("/")))
CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096").split(",")]
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4").split(",")]
GRAPHS = [bool(int(x)) for x in os.environ.get("GRAPHS", "1").split(",")]
OUTLEN = int(os.environ.get("OUTLEN", 256))
RUNS = int(os.environ.get("RUNS", 5))
PROFILE = os.environ.get("PROFILE", "0") == "1"
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 8192))
MAX_NUM_BATCHED_TOKENS = 16384
SKIP_DECODE = 8
ACTIVE_DECODE = 64
OUT_DIR = os.path.join(ROOT, "out", f"p0_{TAG}")

CAT_RULES = [
    ("kv_cache_write", ["store_kvcache"]),
    ("attention",      ["flash", "fmha", "attention", "_attn", "mha"]),
    ("sampling",       ["softmax", "argmax", "exponential", "multinomial", "topk",
                        "sampler", "logprob"]),
    ("gemm_linear",    ["cublas", "cutlass", "xmma", "ampere", "nvjet", "hmma",
                        "gemm", "sgemm", "matmul", "wgrad", "dgrad"]),
    ("norm_rope_eltwise", ["rms_norm", "rmsnorm", "silu", "rotary", "rope",
                           "elementwise", "vectorized", "triton_", "embedding",
                           "index", "gather", "concat", "split", "fill",
                           "copy", "add", "mul", "cast", "contiguous", "reshape"]),
]


def categorize(name):
    low = name.lower()
    if "memcpy" in low or "memset" in low:
        return "memcpy"
    for cat, keys in CAT_RULES:
        for k in keys:
            if k in low:
                return cat
    return "other"


def build_base_ids(tok, need):
    """足够长的分词素材（与 profile_decode_breakdown.py 同源文本）。"""
    txt = ("The quick brown fox jumps over the lazy dog. "
           "In a distant galaxy, researchers study attention kernels. ")
    base = tok.encode(txt * 4000)
    assert len(base) >= need, f"素材太短 {len(base)} < {need}"
    return base


def prompts_for(base, ctx, B, stride=64):
    """B 条【互不相同】且长度都为 ctx 的 prompt（平移窗口，避免前缀缓存共享）。"""
    need = ctx + (B - 1) * stride
    if len(base) < need:
        stride = max(1, (len(base) - ctx) // max(B - 1, 1))
    return [base[i * stride: i * stride + ctx] for i in range(B)]


def weight_bytes(model):
    """唯一权重字节数（按 storage 去重）。

    ★ Qwen3-4B `tie_word_embeddings=True`：`lm_head.weight.data` 与
      `embed_tokens.weight.data` 指向同一块存储，named_parameters() 会把它
      列两遍 → 直接求和会多算一个 embedding（约 0.78 GB）。必须按 data_ptr 去重，
      否则算出来的「有效带宽」分母偏大。
    """
    tot = 0
    seen = set()
    per = {}
    for name, p in model.named_parameters():
        key = (p.data_ptr(), p.numel(), p.element_size())
        b = p.numel() * p.element_size()
        per[name] = b
        if key in seen:
            continue
        seen.add(key)
        tot += b
    return tot, per


def kv_bytes_per_token(hf):
    n_layers = hf.num_hidden_layers
    n_kv = hf.num_key_value_heads
    hd = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    return 2 * n_layers * n_kv * hd * 2  # K+V, 2 bytes (bf16)


# ---------------------------------------------------------------- timing
def time_runs(llm, prompts, ctx, outlen, runs):
    sp = SamplingParams(temperature=1.0, max_tokens=outlen, ignore_eos=True)
    # warmup（短 prompt，把 graph / compile 走一遍）
    llm.generate([prompts[0][:64]], SamplingParams(temperature=1.0, max_tokens=8,
                                                   ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()
    recs = []
    for _ in range(runs):
        hook = TokenDeliveryHook()
        llm.token_hook = hook
        llm.scheduler.token_hook = hook
        torch.cuda.synchronize()
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.time() - t0
        ntok = sum(len(o["token_ids"]) for o in outs)
        s = hook.summary()
        tpot = statistics.median([v["tpot"] for v in s.values() if v["tpot"] == v["tpot"]])
        ttft = statistics.median([v["ttft"] for v in s.values()])
        e2e = max(v["e2e"] for v in s.values())
        recs.append({
            "wall_s": wall, "ntok": ntok,
            "wall_tok_per_s": ntok / wall,
            "decode_tok_per_s": 1.0 / tpot if tpot == tpot else None,
            "tpot_ms": (tpot * 1e3) if tpot == tpot else None,
            "ttft_ms": ttft * 1e3, "e2e_s": e2e,
        })
    llm.token_hook = None
    llm.scheduler.token_hook = None
    return recs


def med_range(vals, fmt="%.3f"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    m = statistics.median(vals)
    return {"median": m, "min": vals[0], "max": vals[-1],
            "str": (fmt % m) if vals[0] == vals[-1] else (fmt % m) + " [%s~%s]" % (fmt % vals[0], fmt % vals[-1])}


class DecodeWindow:
    def __init__(self, skip, active):
        self.skip, self.active = skip, active
        self.n_decode, self.started, self.stopped = 0, False, False
        self.prof = None

    def install(self):
        outer = self

        def run(self_runner, seqs, is_prefill):
            if (not is_prefill) and (not outer.started) and outer.n_decode >= outer.skip:
                torch.cuda.synchronize(); outer.prof.start(); outer.started = True
            out = outer._orig_run(self_runner, seqs, is_prefill)
            if not is_prefill:
                outer.n_decode += 1
                if outer.started and (not outer.stopped) and outer.n_decode >= outer.skip + outer.active:
                    torch.cuda.synchronize(); outer.prof.stop(); outer.stopped = True
            return out

        self._orig_run = ModelRunner.run
        ModelRunner.run = run

    def uninstall(self):
        if self._orig_run is not None:
            ModelRunner.run = self._orig_run
            self._orig_run = None


def profile_once(llm, prompts, outlen):
    sp = SamplingParams(temperature=1.0, max_tokens=SKIP_DECODE + ACTIVE_DECODE + 8,
                        ignore_eos=True)
    prof = profile(activities=[ProfilerActivity.CUDA], with_stack=False)
    win = DecodeWindow(SKIP_DECODE, ACTIVE_DECODE)
    win.prof = prof
    win.install()
    try:
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        win.uninstall()
        if win.started and not win.stopped:
            prof.stop()
    steps = ACTIVE_DECODE if win.stopped else max(win.n_decode - SKIP_DECODE, 0)
    cuda_dt = torch.autograd.DeviceType.CUDA
    cat, total = {}, 0.0
    for ev in prof.events():
        if getattr(ev, "device_type", None) != cuda_dt:
            continue
        d = float(getattr(ev, "device_time", 0) or 0)
        if d <= 0:
            continue
        c = cat.setdefault(categorize(ev.name), [0.0, 0])
        c[0] += d; c[1] += 1
        total += d
    out = {"profiled_decode_steps": steps, "total_device_ms": total / 1000.0,
           "device_ms_per_step": total / 1000.0 / max(steps, 1),
           "categories": {k: {"ms": v[0] / 1000.0,
                              "pct": 100 * v[0] / max(total, 1e-9), "calls": v[1]}
                          for k, v in sorted(cat.items(), key=lambda x: -x[1][0])}}
    del prof, cat
    gc.collect()
    return out


def teardown(llm):
    try:
        import atexit
        atexit.unregister(llm.exit)
    except Exception:
        pass
    try:
        llm.exit()
    except Exception as e:
        print("exit warn:", e)
    try:
        llm.exit = lambda: None
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== p0_weight_bw MODEL={MODEL} TAG={TAG} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} ===")
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}")
    # ★ 旧 profile_decode_breakdown.py 里 patch hash_blocks 是因为当时
    #   前缀缓存哈希契约有 bug（prompt>=256 崩）；那个 bug 已在阶段 0 修好，
    #   现在【不再 patch】，跑的是真实代码路径。

    results = []
    for graph_on in GRAPHS:
        kw = dict(enforce_eager=not graph_on, max_model_len=MAX_MODEL_LEN,
                  max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)
        print(f"\n#### building LLM graph_on={graph_on}")
        llm = LLM(MODEL, **kw)
        try:
            wb, _ = weight_bytes(llm.model_runner.model)
            hf = llm.model_runner.config.hf_config
            kv_pt = kv_bytes_per_token(hf)
            print(f"#### weight_bytes={wb/2**30:.3f} GiB  kv_bytes_per_token={kv_pt/1024:.1f} KiB"
                  f"  layers={hf.num_hidden_layers}")
            base = build_base_ids(llm.tokenizer, max(CTXS) + 512)
            for ctx in CTXS:
                for B in BATCHES:
                    prompts = prompts_for(base, ctx, B)
                    try:
                        recs = time_runs(llm, prompts, ctx, OUTLEN, RUNS)
                        r = {
                            "model": TAG, "graph_on": graph_on, "B": B, "ctx": ctx,
                            "outlen": OUTLEN, "weight_bytes": wb, "kv_bytes_per_token": kv_pt,
                            "decode_tok_per_s": med_range([x["decode_tok_per_s"] for x in recs], "%.2f"),
                            "wall_tok_per_s": med_range([x["wall_tok_per_s"] for x in recs], "%.2f"),
                            "tpot_ms": med_range([x["tpot_ms"] for x in recs], "%.3f"),
                            "ttft_ms": med_range([x["ttft_ms"] for x in recs], "%.3f"),
                            "e2e_s": med_range([x["e2e_s"] for x in recs], "%.4f"),
                            "ntok": recs[0]["ntok"], "runs": RUNS,
                            "decode_bw_GiB_s": (wb + kv_pt * ctx) / 2**30 /
                                               (med_range([x["tpot_ms"] for x in recs], "%.6f")["median"] / 1e3),
                        }
                        if PROFILE:
                            r["profile"] = profile_once(llm, prompts, OUTLEN)
                        results.append(r)
                        print("@@P0@@" + json.dumps(r, ensure_ascii=False))
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"@@FAIL@@ ctx={ctx} B={B}: {type(e).__name__}: {e}")
        finally:
            teardown(llm)
            del llm

    with open(os.path.join(OUT_DIR, "p0_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== SUMMARY (written {OUT_DIR}/p0_results.json) ===")
    for r in results:
        p = r.get("profile", {})
        g = p.get("categories", {}).get("gemm_linear", {}).get("pct")
        print(f"{r['model']:>10} graph={int(r['graph_on'])} B={r['B']} ctx={r['ctx']:>5} | "
              f"decode={r['decode_tok_per_s']['str']} tok/s | ttft={r['ttft_ms']['str']} ms | "
              f"bw={r['decode_bw_GiB_s']:.1f} GiB/s | gemm={g}")
    print("@@DONE@@")


if __name__ == "__main__":
    main()
