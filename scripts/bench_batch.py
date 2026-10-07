"""batch 门控实测：并发数 B 到多少时投机解码开始不划算？

nano-vllm 的 scheduler.spec_enabled() 里已经有门控（spec_batch_threshold），
但一直没数据支撑该设多少。这里在同一进程里把门控打开/关掉各跑一次，
避免重复加载模型。

用法: python bench_batch.py <batch> <k> <max_tokens>
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

B   = int(sys.argv[1])
k   = int(sys.argv[2])
mt  = int(sys.argv[3])

llm = LLM(TARGET, enforce_eager=False, max_num_batched_tokens=16384,
          spec_k=k, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)

BASE = ("def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n"
        "        total += num\n    return total\n\n")
PROMPTS = [BASE + f"def helper_{i}(x):\n    return x * {i}\n\n" for i in range(B)]
sp = SamplingParams(temperature=1.0, max_tokens=mt, ignore_eos=True)


def run(with_spec: bool):
    # ★ 门控是每步现读的，所以同一个进程里能来回切
    llm.scheduler.spec_batch_threshold = 0 if with_spec else 0
    if not with_spec:
        llm.scheduler.spec_k = 0          # 直接关掉投机
    else:
        llm.scheduler.spec_k = k
    torch.cuda.synchronize()
    t0 = time.time()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = sum(len(o["token_ids"]) for o in outs)
    return n / dt


run(True)                                  # 预热
spec = run(True)
base = run(False)
print("@@BATCH@@" + json.dumps({"batch": B, "k": k,
                                "spec_tok_per_s": round(spec, 1),
                                "base_tok_per_s": round(base, 1),
                                "speedup": round(spec / base, 3)}))
