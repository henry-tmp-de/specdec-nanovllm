"""检查 draft 模型自己的输出是否正常"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context

DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(DRAFT)
import torch.distributed as dist
torch.set_default_dtype(cfg.dtype); torch.set_default_device("cuda")
dist.init_process_group("nccl", "tcp://localhost:2399", world_size=1, rank=0)
torch.cuda.set_device(0)
m = Qwen3ForCausalLM(cfg); load_model(m, DRAFT)
tok = AutoTokenizer.from_pretrained(DRAFT)

ids = tok.encode("def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total\n\n")
print(f"prompt tokens = {len(ids)}")
dev = "cuda"

def run(toks, seqlen, nb=4):
    n = len(toks)
    t = lambda x,d: torch.tensor(x, dtype=d, device=dev)
    set_context(True, t([0,n],torch.int32), t([0,seqlen],torch.int32),
                seqlen, seqlen, t(list(range(seqlen-n, seqlen)),torch.int32),
                None, t([[0]*nb],torch.int32), is_spec_verify=True)
    with torch.inference_mode():
        o = m(t(toks,torch.int64), t(range(seqlen-n, seqlen), torch.int64))
    lg = m.compute_logits(o.clone()); reset_context()
    return lg.float()

# 正常 prefill：一次算所有 token
lg = run(ids, len(ids))
print("\n[正常 prefill] 最后位置的分布：")
p = torch.softmax(lg[-1], -1)
top = torch.topk(p, 5)
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"   {i:>6} {v:.4f}  {tok.decode([i])!r}")
print(f"   max p = {p.max():.4f}  （正常模型应 <1.0）")

# ★ 关键：单token decode 路径（draft 走的就是这条）
lg2 = run(ids[-1:], len(ids))
p2 = torch.softmax(lg2[-1], -1)
top2 = torch.topk(p2, 5)
print("\n[draft 用的单 token 路径] 分布：")
for v, i in zip(top2.values.tolist(), top2.indices.tolist()):
    print(f"   {i:>6} {v:.4f}  {tok.decode([i])!r}")
print(f"   max p = {p2.max():.4f}")
print("\n对比：若单token 路径的 max p 远高于 prefill 路径，说明该路径有 bug")
