#!/usr/bin/env python
"""check_quant_direct_equiv.py —— 对拍三种「拿到 int8 权重」的方式

  A 就地量化      ：Qwen3ForCausalLM(bf16) + load_model + apply_int8_quant
                    （引擎现有路径；量化在 GPU 上做）
  B 直载/GPU 量化 ：build_model_int8(..., load_int8_values=False)
                    （meta 构造，每层单独上 GPU 用同一个 quantize_weight 算）
  C 直载/读 int8  ：build_model_int8(..., load_int8_values=True)
                    （直接读离线 checkpoint 里的 int8 + scale，= CPU 量化值）

比较项（全部 torch.equal，逐位，不是 allclose）：
  1. 量化模块的 int8 权重      2. 量化模块的 bf16 scale
  3. 四种形状的融合 GEMM 输出（M = 1/8/64/128/512，覆盖 split-K 与非 split-K）
  4. 抽查未量化参数 + lm_head/embed_tokens 权重共享是否保住

    CUDA_VISIBLE_DEVICES=0 python -u scripts/check_quant_direct_equiv.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model, build_model_int8
from nanovllm.layers.quant import apply_int8_quant
from nanovllm.config import _resolve_quant

BF16 = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
I8 = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-int8-all")

K_OF = {"model.layers.0.self_attn.qkv_proj": 1024,
        "model.layers.0.self_attn.o_proj": 2048,
        "model.layers.0.mlp.gate_up_proj": 1024,
        "model.layers.0.mlp.down_proj": 3072}
OTHER = ["model.embed_tokens.weight", "model.norm.weight",
         "model.layers.0.input_layernorm.weight",
         "model.layers.0.post_attention_layernorm.weight",
         "model.layers.5.self_attn.q_norm.weight"]


def compare(ref, other, mods, tag):
    bad_w, bad_s, bad_other, bad_gemm = [], [], [], []
    for n in mods:
        a, b = ref.get_submodule(n), other.get_submodule(n)
        if a.weight.dtype != torch.int8 or b.weight.dtype != torch.int8:
            bad_w.append((n, "dtype", str(a.weight.dtype), str(b.weight.dtype)))
        elif a.weight.shape != b.weight.shape or not torch.equal(a.weight, b.weight):
            bad_w.append((n, int((a.weight != b.weight).sum()), a.weight.numel()))
        if a.weight_scale.shape != b.weight_scale.shape \
                or not torch.equal(a.weight_scale, b.weight_scale):
            bad_s.append((n, int((a.weight_scale != b.weight_scale).sum())))
    pa, pb = dict(ref.named_parameters()), dict(other.named_parameters())
    for n in OTHER:
        x, y = pa[n], pb[n]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            bad_other.append(n)
    for M in (1, 8, 64, 128, 512):
        for name, kin in K_OF.items():
            torch.manual_seed(1234 + M)
            x = torch.randn(M, kin, dtype=torch.bfloat16, device="cuda")
            ya = ref.get_submodule(name)._apply(x, None)
            yb = other.get_submodule(name)._apply(x, None)
            if ya.shape != yb.shape or not torch.equal(ya, yb):
                bad_gemm.append((M, name, float((ya - yb).abs().max())))
    tie = (other.lm_head.weight.data.data_ptr()
           == other.model.embed_tokens.weight.data.data_ptr())
    res = dict(
        int8_weight_mismatch=len(bad_w), weight_examples=bad_w[:2],
        scale_mismatch=len(bad_s), scale_examples=bad_s[:2],
        other_param_mismatch=bad_other,
        gemm_cases=len(K_OF) * 5, gemm_mismatch=len(bad_gemm), gemm_examples=bad_gemm[:2],
        lm_head_tied=bool(tie),
    )
    print(f"@@EQ:{tag}@@" + json.dumps(res, ensure_ascii=False))
    return (not bad_w and not bad_s and not bad_other and not bad_gemm and tie), res


def main():
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
    torch.cuda.set_device(0)
    hf = AutoConfig.from_pretrained(BF16)
    torch.set_default_dtype(hf.dtype)
    torch.set_default_device("cuda")

    kind, gran, gs, mods, _qc = _resolve_quant(I8, None, "per_channel", 128)
    mods = sorted(mods)
    assert kind == "int8" and gran == "per_channel", (kind, gran)
    print(f"[equiv] 量化模块 {len(mods)} 个, granularity={gran}, group_size={gs}")

    m_a = Qwen3ForCausalLM(hf)
    load_model(m_a, BF16)
    apply_int8_quant(m_a, tuple(mods), gran, gs)
    print("[equiv] A 就地量化（GPU）完成")

    m_b, _sb = build_model_int8(Qwen3ForCausalLM, hf, I8, tuple(mods), gran, gs,
                                load_int8_values=False)
    print("[equiv] B 直载 / GPU 量化完成")

    m_c, _sc = build_model_int8(Qwen3ForCausalLM, hf, I8, tuple(mods), gran, gs,
                                load_int8_values=True)
    print("[equiv] C 直载 / 读离线 int8 完成")

    ok_b, res_b = compare(m_a, m_b, mods, "A_vs_B_gpuquant")
    ok_c, res_c = compare(m_a, m_c, mods, "A_vs_C_offlineint8")
    print("VERDICT_B(GPU量化):", "BITWISE_IDENTICAL" if ok_b else "DIFFERENT")
    print("VERDICT_C(离线int8):", "BITWISE_IDENTICAL" if ok_c else "DIFFERENT")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
