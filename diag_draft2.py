"""draft 模型的输出分布是否退化？（只做 prefill 一条路径，最简）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.distributed as dist
dist.init_process_group("nccl", "tcp://localhost:2399", world_size=1, rank=0)
torch.cuda.set_device(0)
from transformers import AutoTokenizer, AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context

DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
cfg = AutoConfig.from_pretrained(DRAFT)
torch.set_default_dtype(cfg.dtype); torch.set_default_device("cuda")
m = Qwen3ForCausalLM(cfg); load_model(m, DRAFT)
tok = AutoTokenizer.from_pretrained(DRAFT)

ids = tok.encode("def calculate_sum(numbers):\n    total = 0\n")
L = len(ids); dev = "cuda"
print(f"draft prompt tokens = {L}")

# 标准 prefill：一次算全部（这是 draft prefill 走的路）
n = L
t = lambda x, d: torch.tensor(x, dtype=d, device=dev)
set_context(True, t([0,n],torch.int32), t([0,L],torch.int32), L, L,
            t(range(L),torch.int32), None, t([[0]*4],torch.int32), is_spec_verify=True)
with torch.inference_mode():
    o = m(t(ids,torch.int64), t(range(L),torch.int64))
lg = m.compute_logits(o.clone()); reset_context()

p = torch.softmax(lg[-1].float(), -1)
top = torch.topk(p, 5)
print("\ndraft prefill 最后位置 top5：")
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"   {i:>6} p={v:.4f}  {tok.decode([i])!r}")
print(f"   ★ max p = {p.max():.6f}")
print()
print("若 max p ≈ 1.0000 -> draft 分布退化（正常应在0.1~0.9）")
print("若 max p < 0.99     -> draft 正常，问题在别处")
