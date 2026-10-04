"""诊断：投机路径为什么输出重复 token。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams as SP

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)

# 直接构造一个序列，观察投机提议的内容
sp = SP(temperature=1e-4, max_tokens=8, ignore_eos=True)
prompt = llm.tokenizer.encode("The capital of France is")
print("prompt tokens:", prompt)

prop = llm.model_runner.spec_proposer
print("proposer stats:", prop.stats())

seq = Sequence(prompt, sp)
print("\n--- 提议测试 ---")
cand = prop.propose(seq.token_ids, 3)
print("从 prompt 提议:", cand)
print("prompt 里出现过 3-gram 吗")

# 打印前 20 个 3-gram
from collections import Counter
grams = Counter()
toks = prompt
for i in range(len(toks)-2):
    grams[tuple(toks[i:i+3])] += 1
print("prompt 中前 5 个 3-gram:", list(grams.items())[:5])

print("\n--- 手工走一步 decode 看提议到验证的完整链路 ---")
llm.scheduler.add(seq)
llm.add_request("The capital of France is", sp)
info = {}
orig = llm.scheduler.schedule
def traced():
    seqs, is_pf = orig()
    info["seqs"] = seqs; info["pf"] = is_pf
    return seqs, is_pf
llm.scheduler.schedule = traced

step = 0
while not llm.is_finished() and step < 6:
    info.clear()
    seqs, is_pf = llm.scheduler.schedule()
    mode = "PREFILL" if is_pf else f"DECODE(n_sched={seqs[0].num_scheduled_tokens})"
    print(f"\n[step {step}] {mode}")
    for s in seqs:
        print(f"   draft={s.draft_tokens}  num_tokens={s.num_tokens}")
    toks = llm.model_runner.call("run", seqs, is_pf)
    print(f"   -> run 返回: {toks if not is_pf else toks}")
    llm.scheduler.postprocess(seqs, toks, is_pf)
    step += 1
print("\n最终生成:", seq.completion_token_ids)
