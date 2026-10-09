#!/usr/bin/env python
"""profile_decode_breakdown.py

target-only (Qwen3-4B) / B=1 / 普通 decode（非投机）的 device 侧 kernel 时间分解。

目的：回答 P1 决策问题 —— 普通 decode 中 attention 占 GPU 时间的比例 f 是多少？
      Amdahl 上界 1/(1-f)；代入历史 kernel 加速比 s=1.31/1.47/1.59 得整机加速预测。

【不修改引擎代码】：只在进程内 monkeypatch，磁盘上 nanovllm/ 一个字节都没变。
  patch 只有两处，且都【不碰计算路径】：
    1. BlockManager.hash_blocks -> no-op
       原因：现有代码对 prompt >= 256 token 必崩（见脚本内 BUG 说明），
       不 patch 就根本跑不到 ctx=1024/4096。hash_blocks 是纯 CPU 前缀缓存
       记账（xxhash + dict），不产生任何 CUDA kernel，且在 profiler 窗口
       （ModelRunner.run）之外调用 —— 关掉它不影响 f 的测量。
    2. ModelRunner.run 包一层 -> 只为框定 profiler 的起停窗口。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<p1-work> python -u scripts/profile_decode_breakdown.py
"""
import os
import sys
import gc
import json
import atexit
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 8192))  # 默认 4096 装不下 4096 prompt + 输出
MAX_NUM_BATCHED_TOKENS = 16384
CTXS = [int(x) for x in os.environ.get("CTXS", "1024,4096").split(",")]
GRAPHS = [bool(int(x)) for x in os.environ.get("GRAPHS", "1,0").split(",")]
SKIP_PROBE = os.environ.get("SKIP_PROBE", "1") == "1"
SKIP_DECODE = 8               # profiler 窗口前跳过的 decode 步
ACTIVE_DECODE = 64            # 窗口内 profile 的 decode 步
MAX_TOKENS = SKIP_DECODE + ACTIVE_DECODE + 8
N_LAYERS = 36
OUT_DIR = os.path.join(ROOT, "out")
TAG = time.strftime("%Y%m%d_%H%M%S") + f"_mml{MAX_MODEL_LEN}"


def build_prompt_ids(llm, ctx):
    tok = llm.tokenizer
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "In a distant galaxy, researchers study attention kernels. " * 700)
    assert len(base) >= ctx, f"prompt 太短: {len(base)} < {ctx}"
    return base[:ctx]


# ----------------------------------------------------------------------
# 引擎 BUG 探针：不 patch 时 prompt >= 256 token 直接崩
# ----------------------------------------------------------------------
def probe_hash_bug(llm, ctx=300):
    """返回 (crashed: bool, exc_str)。"""
    ids = build_prompt_ids(llm, ctx)
    try:
        llm.generate([ids], SamplingParams(temperature=1.0, max_tokens=4,
                                           ignore_eos=True), use_tqdm=False)
        return False, "no-crash"
    except Exception as e:
        return True, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------
# profiler 开关：框住「decode 第 SKIP+1 步 .. SKIP+ACTIVE 步」
# ----------------------------------------------------------------------
class DecodeWindow:
    def __init__(self, skip, active):
        self.skip = skip
        self.active = active
        self.n_decode = 0
        self.started = False
        self.stopped = False
        self.prof = None
        self._orig_run = None

    def install(self):
        outer = self

        def run(self_runner, seqs, is_prefill):
            if (not is_prefill) and (not outer.started) and outer.n_decode >= outer.skip:
                torch.cuda.synchronize()
                outer.prof.start()
                outer.started = True
            out = outer._orig_run(self_runner, seqs, is_prefill)
            if not is_prefill:
                outer.n_decode += 1
                if outer.started and (not outer.stopped) and \
                        outer.n_decode >= outer.skip + outer.active:
                    torch.cuda.synchronize()
                    outer.prof.stop()
                    outer.stopped = True
            return out

        self._orig_run = ModelRunner.run
        ModelRunner.run = run

    def uninstall(self):
        if self._orig_run is not None:
            ModelRunner.run = self._orig_run
            self._orig_run = None


# ----------------------------------------------------------------------
# 分类
# ----------------------------------------------------------------------
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


def categorize(name: str) -> str:
    low = name.lower()
    if "memcpy" in low or "memset" in low:
        return "memcpy"
    for cat, keys in CAT_RULES:
        for k in keys:
            if k in low:
                return cat
    return "other"


def dev_us(ev):
    v = getattr(ev, "device_time", None)
    if v is None:
        v = getattr(ev, "cuda_time", None)
    return float(v or 0.0)


def collect_kernels(prof):
    cuda_dt = torch.autograd.DeviceType.CUDA
    agg = {}
    total = 0.0
    n = 0
    for ev in prof.events():
        if getattr(ev, "device_type", None) != cuda_dt:
            continue
        dur = dev_us(ev)
        if dur <= 0:
            continue
        name = ev.name
        rec = agg.get(name)
        if rec is None:
            agg[name] = [dur, 1, categorize(name)]
        else:
            rec[0] += dur
            rec[1] += 1
        total += dur
        n += 1
    return agg, total, n


def fmt_table(rows, title):
    lines = [title, "-" * 100,
             f"{'name':<62} {'us':>10} {'cnt':>7} {'us/call':>9}"]
    for name, tot, cnt in rows:
        lines.append(f"{name[:62]:<62} {tot:>10.1f} {cnt:>7d} {tot/max(cnt,1):>9.2f}")
    return "\n".join(lines)


def run_one(llm, ctx, graph_on, tag):
    prompt_ids = build_prompt_ids(llm, ctx)
    sp = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, ignore_eos=True)

    # warmup（短 prompt，把 cuda graph / torch.compile 都走一遍）
    llm.generate([prompt_ids[:64]], SamplingParams(temperature=1.0, max_tokens=8,
                                                   ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize()

    # 干净计时
    t0 = time.time()
    out = llm.generate([prompt_ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
    ntok = len(out[0]["token_ids"])

    # profiled
    prof = profile(activities=[ProfilerActivity.CUDA], with_stack=False)
    win = DecodeWindow(SKIP_DECODE, ACTIVE_DECODE)
    win.prof = prof
    win.install()
    try:
        llm.generate([prompt_ids], sp, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        win.uninstall()
        if win.started and not win.stopped:
            prof.stop()
    profiled_steps = ACTIVE_DECODE if win.stopped else max(win.n_decode - SKIP_DECODE, 0)

    agg, total_us, n_ev = collect_kernels(prof)
    os.makedirs(os.path.join(OUT_DIR, tag), exist_ok=True)
    raw = sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda x: -x[1])

    cat_tot = {}
    for k, v in agg.items():
        c = cat_tot.setdefault(v[2], [0.0, 0])
        c[0] += v[0]
        c[1] += v[1]

    with open(os.path.join(OUT_DIR, tag, f"kernels_ctx{ctx}_graph{int(graph_on)}.txt"), "w") as f:
        f.write(fmt_table(raw[:30], f"TOP-30 kernels by device time  ctx={ctx} "
                                    f"graph={int(graph_on)} profiled_decode_steps={profiled_steps}"))
        f.write("\n\n" + fmt_table(raw, "ALL kernels") + "\n\n")
        f.write("CATEGORY SUMMARY\n" + "-" * 100 + "\n")
        for c, (t, cnt) in sorted(cat_tot.items(), key=lambda x: -x[1][0]):
            f.write(f"{c:<24} {t:>12.1f} us  {100*t/max(total_us,1e-9):>6.2f}%  calls={cnt}\n")

    attn = cat_tot.get("attention", [0.0, 0])
    attn_us, attn_calls = attn[0], attn[1]
    f_ratio = attn_us / total_us if total_us > 0 else 0.0
    dev_per_step = total_us / max(profiled_steps, 1) / 1000.0
    wall_per_step = wall / max(ntok, 1) * 1000.0
    flash_detail = [{"name": k[:70], "us_per_call": round(v[0] / v[1], 2), "calls": v[1]}
                    for k, v in sorted((k, v) for k, v in agg.items() if "flash" in k.lower())]

    res = {
        "ctx": ctx, "graph_on": bool(graph_on), "enforce_eager": not graph_on,
        "max_model_len": MAX_MODEL_LEN, "dtype": "bfloat16", "model": TARGET,
        "max_tokens_requested": MAX_TOKENS,
        "tokens": ntok, "wall_s": round(wall, 3), "tok_per_s": round(ntok / wall, 2),
        "tok_per_s_decode_only": round(ntok / wall, 2),
        "profiled_decode_steps": profiled_steps,
        "n_cuda_kernel_events": n_ev, "n_distinct_kernels": len(agg),
        "total_device_ms": round(total_us / 1000.0, 3),
        "device_ms_per_step": round(dev_per_step, 4),
        "wall_ms_per_step": round(wall_per_step, 4),
        "attention_device_ms": round(attn_us / 1000.0, 3),
        "attention_calls": attn_calls,
        "attention_calls_per_step": round(attn_calls / max(profiled_steps, 1), 2),
        "attention_us_per_call": round(attn_us / max(attn_calls, 1), 2),
        "attention_us_per_step": round(attn_us / max(profiled_steps, 1), 2),
        "attention_us_per_layer_per_step": round(attn_us / max(profiled_steps, 1) / N_LAYERS, 2),
        "flash_kernels": flash_detail,
        "max_num_blocks_graph_pad": (MAX_MODEL_LEN + 255) // 256,
        "f": round(f_ratio, 5),
        "amdahl_bound": round(1.0 / (1.0 - f_ratio), 4) if f_ratio < 1 else None,
        "S_s131": round(1.0 / ((1 - f_ratio) + f_ratio / 1.31), 4),
        "S_s147": round(1.0 / ((1 - f_ratio) + f_ratio / 1.47), 4),
        "S_s159": round(1.0 / ((1 - f_ratio) + f_ratio / 1.59), 4),
        "categories": {c: {"ms": round(v[0] / 1000.0, 3),
                           "pct": round(100 * v[0] / max(total_us, 1e-9), 2),
                           "calls": v[1]}
                       for c, v in sorted(cat_tot.items(), key=lambda x: -x[1][0])},
    }
    with open(os.path.join(OUT_DIR, tag, f"config_ctx{ctx}_graph{int(graph_on)}.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("@@R@@" + json.dumps(res, ensure_ascii=False))
    del prof, agg, cat_tot
    gc.collect()
    return res


def teardown(llm):
    try:
        atexit.unregister(llm.exit)     # 否则 atexit 持有 bound method -> 模型不释放
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
    print(f"=== profile_decode_breakdown tag={TAG} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} ===")
    print(f"target={TARGET}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}")

    kw = dict(enforce_eager=True, max_model_len=MAX_MODEL_LEN,
              max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)

    # ---------- 0) 引擎 BUG 探针（不 patch，跑一次 300-token prompt）----------
    if not SKIP_PROBE:
        print("\n#### [0] hash_blocks bug probe (unpatched, ctx=300)")
        llm = LLM(TARGET, **kw)
        crashed, msg = probe_hash_bug(llm, 300)
        print(f"@@BUGPROBE@@ crashed={crashed} detail={msg}")
        teardown(llm)
        del llm
    else:
        print("#### [0] bug probe skipped (SKIP_PROBE=1)")

    # ---------- patch ----------
    BlockManager.hash_blocks = lambda self, seq: None      # measurement-only no-op
    print("#### [patch] BlockManager.hash_blocks -> no-op (纯 CPU 前缀缓存记账，"
          "不产生 CUDA kernel，且在 profiler 窗口之外)")

    results = []
    for graph_on in GRAPHS:
        kw = dict(enforce_eager=not graph_on, max_model_len=MAX_MODEL_LEN,
                  max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)
        print(f"\n#### building LLM graph_on={graph_on} {kw}")
        llm = LLM(TARGET, **kw)
        try:
            for ctx in CTXS:
                print(f"\n---- ctx={ctx} graph_on={graph_on} ----")
                try:
                    results.append(run_one(llm, ctx, graph_on, TAG))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"@@FAIL@@ ctx={ctx} graph_on={graph_on}: {type(e).__name__}: {e}")
        finally:
            teardown(llm)
            del llm

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"ctx={r['ctx']:>5} graph={int(r['graph_on'])} | f={r['f']*100:6.2f}% | "
              f"attn={r['attention_us_per_call']:6.1f}us/call x{r['attention_calls_per_step']:.0f} "
              f"= {r['attention_us_per_step']:7.1f}us/step | "
              f"dev/step={r['device_ms_per_step']:.3f}ms wall/step={r['wall_ms_per_step']:.3f}ms | "
              f"tok/s={r['tok_per_s']:7.2f} | 1/(1-f)={r['amdahl_bound']:.4f} | "
              f"S(1.31)={r['S_s131']:.4f} S(1.47)={r['S_s147']:.4f} S(1.59)={r['S_s159']:.4f}")
    print(f"@@DONE@@ tag={TAG}")


if __name__ == "__main__":
    main()
