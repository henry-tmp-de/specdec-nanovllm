#!/usr/bin/env python
"""check_quant_equiv.py —— 量化默认关闭时，引擎与改动前【逐字节等价】

做法：用固定 seed 跑一组固定请求，把【完整输出 token ids】打成 md5。
把同一个脚本分别指到 `p1-work-base`（改动前）与 `p1-work`（改动后）各跑一次，
md5 必须完全相同。

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<dir> python scripts/check_quant_equiv.py
"""
import os
import sys
import json
import hashlib

for p in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
MAXLEN = 8192

P = ("def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n"
     "        total += num\n    return total\n\n"
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
     "    return a\n\n")
Q = ("The history of the Roman Empire spans more than a thousand years, beginning "
     "with the founding of the city and ending with the fall of Constantinople. ")


def one(mode):
    torch.manual_seed(20261010)
    kw = dict(enforce_eager=False, max_model_len=MAXLEN, max_num_batched_tokens=16384)
    if mode == "spec":
        kw.update(spec_k=6, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
    llm = LLM(TARGET, **kw)
    sp = SamplingParams(temperature=1.0, max_tokens=256, ignore_eos=True)
    out = llm.generate([P, Q], sp, use_tqdm=False)
    ids = [o["token_ids"] for o in out]
    llm.exit = lambda: None
    del llm
    return ids


def md5_of(ids):
    h = hashlib.md5()
    for row in ids:
        h.update(",".join(map(str, row)).encode())
        h.update(b"|")
    return h.hexdigest()


if __name__ == "__main__":
    res = {}
    for mode in ("plain", "spec"):
        ids = one(mode)
        res[mode] = {"md5": md5_of(ids), "n": [len(x) for x in ids],
                     "head": [x[:8] for x in ids]}
    print("@@EQ@@" + json.dumps(res, ensure_ascii=False))
