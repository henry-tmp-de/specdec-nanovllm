"""查提议器为什么返回空"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("np_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanovllm/spec_decode/ngram_proposer.py"))
np_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(np_mod)
NgramProposer = np_mod.NgramProposer

from transformers import AutoTokenizer
MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok.encode("The capital of France is")
print("prompt token ids:", ids, " len =", len(ids))

p = NgramProposer(n=3, window=8)
# 手动加索引（引擎里是靠 prefill 时注册的，这里先手动模拟）
p.add(ids)
print("stats after add:", p.stats())
cand = p.propose(ids, 3)
print("propose(ids, 3) ->", cand)
print()

print("=== 关键: n=3 时 key 是 (n-1)=2 元组 ===")
for L in range(2, 0, -1):
    key = tuple(ids[-L:])
    print(f"  试长度 {L}: key={key}  在索引里? {key in p._index}")
print()
print("=== 看索引里到底有什么 ===")
for k, v in list(p._index.items())[:8]:
    print(f"  {k} -> {list(v)}")
