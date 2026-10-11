#!/usr/bin/env python
"""profile_spec_device.py —— 真实投机 decode 的「GPU 忙 vs 墙钟」对照。

不插桩、不加同步：只用 torch.profiler 框住 decode 段，然后
  · 墙钟 / 窗口内 CUDA device 时间之和  -> GPU 占空比
  · 按 kernel 名把 draft 前向 / 验证前向分开（splitkv = draft decode；
    varlen = 验证；另按调用次数核对）

用法: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<p1-work> python -u scripts/profile_spec_device.py
env: CTX OUTLEN WIN SKIP ACTIVE
"""
import os
import sys
import json
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

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
CTX = int(os.environ.get("CTX", 4096))
OUTLEN = int(os.environ.get("OUTLEN", 64))
WIN = int(os.environ.get("WIN", 0))
SKIP = int(os.environ.get("SKIP", 4))
ACTIVE = int(os.environ.get("ACTIVE", 24))


class Window:
    def __init__(self, skip, active, prof):
        self.skip, self.active, self.n, self.prof = skip, active, 0, prof
        self.started = self.stopped = False

    def install(self):
        outer = self
        orig = ModelRunner.run

        def run(self_runner, seqs, is_prefill):
            if (not is_prefill) and (not outer.started) and outer.n >= outer.skip:
                torch.cuda.synchronize()
                outer.wall0 = time.perf_counter()
                outer.prof.start()
                outer.started = True
            out = orig(self_runner, seqs, is_prefill)
            if not is_prefill:
                outer.n += 1
                if outer.started and not outer.stopped and outer.n >= outer.skip + outer.active:
                    torch.cuda.synchronize()
                    outer.wall1 = time.perf_counter()
                    outer.prof.stop()
                    outer.stopped = True
            return out

        self._orig = orig
        ModelRunner.run = run

    def uninstall(self):
        ModelRunner.run = self._orig


def main():
    print("=== profile_spec_device ctx=%d out=%d win=%d ===" % (CTX, OUTLEN, WIN))
    kw = dict(enforce_eager=False, max_model_len=8192, max_num_batched_tokens=16384,
              spec_k=6, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0,
              spec_graph_bs=(1,))
    if WIN:
        kw["spec_draft_window"] = WIN
    llm = LLM(TARGET, **kw)
    tok = llm.tokenizer
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "In a distant galaxy, researchers study attention kernels. " * 800)
    ids = base[:CTX]
    llm.generate([ids], SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True),
                 use_tqdm=False)
    torch.cuda.synchronize()

    pr = llm.model_runner.spec_proposer
    r0 = pr.n_rounds
    prof = profile(activities=[ProfilerActivity.CUDA], with_stack=False)
    win = Window(SKIP, ACTIVE, prof)
    win.prof = prof
    win.install()
    try:
        llm.generate([ids], SamplingParams(temperature=1.0, max_tokens=OUTLEN,
                                           ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        win.uninstall()
        if win.started and not win.stopped:
            prof.stop()
    rounds = pr.n_rounds - r0
    wall = (win.wall1 - win.wall0) if win.stopped else None
    cuda_dt = torch.autograd.DeviceType.CUDA
    agg, tot = {}, 0.0
    for ev in prof.events():
        if getattr(ev, "device_type", None) != cuda_dt:
            continue
        d = float(getattr(ev, "device_time", 0) or 0)
        if d <= 0:
            continue
        r = agg.setdefault(ev.name, [0.0, 0])
        r[0] += d
        r[1] += 1
        tot += d
    res = {"ctx": CTX, "outlen": OUTLEN, "win": WIN, "rounds_in_window": rounds,
           "wall_ms": round(wall * 1e3, 2) if wall else None,
           "device_ms": round(tot / 1e3, 2),
           "gpu_duty_pct": round(100 * tot / 1e3 / (wall * 1e3), 1) if wall else None,
           "wall_ms_per_round": round(wall * 1e3 / max(rounds, 1), 3) if wall else None,
           "device_ms_per_round": round(tot / 1e3 / max(rounds, 1), 3)}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("\nTOP kernels (us / calls):")
    for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[:22]:
        print("  %-58s %10.1f %7d" % (k[:58], v[0], v[1]))
    res["top"] = [{"name": k[:70], "us": round(v[0], 1), "calls": v[1]}
                  for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[:22]]
    # 分离：varlen(=验证) vs splitkv(=draft decode)
    var = [k for k in agg if "flash_fwd_kernel" in k or "varlen" in k.lower()]
    spk = [k for k in agg if "splitkv" in k.lower()]
    print("\nvarlen 类 kernel: %s" % [k[:60] for k in var])
    print("splitkv 类 kernel: %s" % [k[:60] for k in spk])
    if var:
        res["varlen_us"] = round(sum(agg[k][0] for k in var), 1)
        res["varlen_calls"] = sum(agg[k][1] for k in var)
    if spk:
        res["splitkv_us"] = round(sum(agg[k][0] for k in spk), 1)
        res["splitkv_calls"] = sum(agg[k][1] for k in spk)
    print("@@DEVR@" + json.dumps(res, ensure_ascii=False))
    try:
        import atexit
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()


if __name__ == "__main__":
    main()
