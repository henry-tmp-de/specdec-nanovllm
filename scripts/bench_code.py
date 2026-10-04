"""代码补全场景测投机解码 —— n-gram 重叠度最高的任务之一"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=4, spec_batch_threshold=0)

orig_verify = llm.model_runner.run_verify
S = {"verify": 0, "proposed": 0, "landed": 0, "accepted_tok": 0}
def traced(seqs):
    S["proposed"] += sum(len(s.draft_tokens) for s in seqs)
    out = orig_verify(seqs)
    S["verify"] += 1
    S["landed"] += sum(len(x) for x in out)
    S["accepted_tok"] += sum(len(x) for x in out) - len(seqs)  # 减去每步必出的1个
    return out
llm.model_runner.run_verify = traced

steps = {"n": 0}
orig_post = llm.scheduler.postprocess
def traced_post(seqs, t, pf):
    steps["n"] += 1
    return orig_post(seqs, t, pf)
llm.scheduler.postprocess = traced_post

# 场景：给出函数前半段，要求模型补全（会大量复制已有的标识符/结构）
PROMPT = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total


def calculate_max(numbers):
    maximum = numbers[0]
    for num in numbers:
        if num > maximum:
            maximum = num
    return maximum


def calculate_min(numbers):
    minimum = numbers[0]
    for num in numbers:
        if num < minimum:
            minimum = num
    return minimum


def calculate_avg(numbers):
    average = 0
    for num in numbers:
        average += num
    average = average / calculate_sum(numbers)
    return average
"""
sp = SamplingParams(temperature=0.2, max_tokens=100, ignore_eos=True)
t0 = time.time()
outs = llm.generate([PROMPT], sp, use_tqdm=False)
torch.cuda.synchronize()
dt = time.time() - t0
n_out = len(outs[0]["token_ids"])

print(f"\n{'='*60}")
print(f"耗时 {dt:.2f}s   总 step {steps['n']}   产出 {n_out} token")
print(f"verify 次数 {S['verify']}  投机占比 {S['verify']/max(steps['n'],1)*100:.0f}%")
print(f"提议 {S['proposed']}   其中被接受 {S['accepted_tok']}  "
      f"→ 接受率 {S['accepted_tok']/max(S['proposed'],1)*100:.1f}%")
print(f"平均每 step 产出 {n_out/max(steps['n'],1):.2f} token (基线是 1.0)")
print("="*60)
print("生成:", outs[0]["text"][:250])
