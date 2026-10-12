#!/usr/bin/env python
"""p12_run_tests.py —— 逐个独立进程跑 tests/test_*.py，统计每条 check() 的 PASS/FAIL。

为什么一个文件一个进程：实测一个进程里连跑会互相污染（test_block_alloc 会崩）。
每个 tests/test_*.py 的 check() 都会打印 `  [PASS] name` / `  [FAIL] name`，
直接数这两种标记即可（与 P13 用 sys.settrace 数的结果一致）。

用法:
  /d/Anaconda/python.exe scripts/p12_run_tests.py                      # 本机（torch cpu）
  /home/ziru/nano-vllm/venv/bin/python scripts/p12_run_tests.py        # 服务器

末行：@@TESTS@@ {"pass": N, "fail": M, "files": {...}}
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run_one(path):
    p = subprocess.run([sys.executable, path], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    out = p.stdout + "\n" + p.stderr
    npass = out.count("[PASS]")
    nfail = out.count("[FAIL]")
    if npass == 0 and nfail == 0:
        npass = 1 if p.returncode == 0 else 0
        nfail = 0 if p.returncode == 0 else 1
    return npass, nfail, out


def main():
    tests = sorted(f for f in os.listdir(os.path.join(ROOT, "tests"))
                   if f.startswith("test_") and f.endswith(".py"))
    tot_p = tot_f = 0
    files = {}
    for t in tests:
        tp, tf, out = run_one(os.path.join(ROOT, "tests", t))
        tot_p += tp
        tot_f += tf
        files[t] = [tp, tf]
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
        print(f"[{t}] PASS={tp} FAIL={tf}  | {tail[0][:80]}", flush=True)
    print("@@TESTS@@ " + json.dumps(dict(pass_=tot_p, fail=tot_f, files=files)))
    sys.exit(1 if tot_f else 0)


if __name__ == "__main__":
    main()
