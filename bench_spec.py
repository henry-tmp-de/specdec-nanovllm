"""投机解码端到端测试。
每个配置跑在独立子进程里（nano-vllm 的 dist.init_process_group 不可重复调用）。
用法: python bench_spec.py <spec_k> <max_tokens>
"""
import os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
TEMP = 1e-4
P = "The capital of France is"

def main():
    spec_k = int(sys.argv[1]); max_tokens = int(sys.argv[2])
    t0 = time.time()
    llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
              spec_k=spec_k, spec_batch_threshold=0)
    init_s = time.time() - t0
    sp = SamplingParams(temperature=TEMP, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize(); t1 = time.time()
    out = llm.generate([P], sp, use_tqdm=False)[0]["token_ids"]
    torch.cuda.synchronize(); gen_s = time.time() - t1
    print("@@JSON@@" + json.dumps({
        "spec_k": spec_k, "init_s": init_s, "gen_s": gen_s,
        "n_tokens": len(out), "tokens": out[:20],
    }))

main()
