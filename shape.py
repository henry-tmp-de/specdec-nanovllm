import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams
TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT  = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(TARGET, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=2, spec_method="draft", draft_model=DRAFT, spec_batch_threshold=0)
mr = llm.model_runner
for nm, mdl in [("target", mr.model), ("draft", mr.draft_model)]:
    for m in mdl.modules():
        if hasattr(m, "k_cache") and hasattr(m, "v_cache") and m.k_cache.numel():
            print(f"{nm} 第0层 k_cache shape = {tuple(m.k_cache.shape)}")
            print(f"{nm}   k_cache.stride() = {m.k_cache.stride()}")
            print(f"{nm}   k_cache 是视图? base_ptr 相同? {m.k_cache.data_ptr()}")
            break
print("\nkv_cache 整体:", tuple(mr.kv_cache.shape))
print("draft_kv_cache 整体:", tuple(mr.draft_kv_cache.shape))
print("\n第0层切片 kv_cache[0,0] 形状:", tuple(mr.kv_cache[0,0].shape))
