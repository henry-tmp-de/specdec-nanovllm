#!/usr/bin/env python
"""quantize_int8.py —— 离线 INT8 权重量化（W8A16：int8 存储 + bf16 计算）

把 BF16 checkpoint 转成「int8 权重 + 反量化 scale」的新 checkpoint，并打印量化统计。

设计要点（对应任务书 §Phase 1）
--------------------------------
1. 粒度：`per_channel`（每输出行一个 scale，默认）或 `per_group`（每 group_size 个
   输入通道一个 scale，默认 128）。对称量化，零点恒为 0（权重近似零均值，对称免零点）。
2. 只量化 transformer block 里的线性层：q/k/v/o/gate/up/down。
   embedding / lm_head / norm / bias 一律保持 bf16。
   理由：Qwen3 `tie_word_embeddings=True`，embed_tokens 与 lm_head 共享同一份权重，
   动它既改输入表示又改输出分布；而它在权重字节里只占约 10%。
3. 输出目录内容：
     <name>.safetensors      int8 权重（key 名不变）+ <key>_scale（bf16）
     quant_config.json       粒度 / 量化层清单 / 统计
     其余文件（config.json / tokenizer* / *.json）原样复制
4. 统计：每层 scale 范围、饱和比例（|q|==127 的占比）、相对量化误差
   ||W - dequant(W)||_F / ||W||_F。

用法:
  python scripts/quantize_int8.py --model <bf16_dir> --out <int8_dir> [--granularity per_channel]
"""
import os
import re
import json
import shutil
import argparse
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# 量化哪些层：只匹配 block 里的线性层（HF checkpoint 的 key 名）
QUANT_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
# 明确排除（双保险）
SKIP_RE = re.compile(r"(embed_tokens|lm_head|norm|bias)")
QMAX = 127


def quantize_tensor(w: torch.Tensor, granularity: str = "per_channel", group_size: int = 128):
    """对称量化。返回 (q_int8, scale)。

    w: [N, K] float/bf16
    per_channel  : scale [N, 1]
    per_group    : W 看成 [N, K//G, G]，scale [N, K//G, 1]（G 整除 K）
    """
    assert w.dim() == 2, f"只处理 2D 权重，收到 {tuple(w.shape)}"
    N, K = w.shape
    wf = w.to(torch.float32)
    if granularity == "per_channel":
        amax = wf.abs().amax(dim=1, keepdim=True)                      # [N,1]
    elif granularity == "per_group":
        assert group_size > 0 and K % group_size == 0, \
            f"per_group 需要 group_size 整除 K，收到 K={K} G={group_size}"
        amax = wf.abs().view(N, K // group_size, group_size).amax(dim=2, keepdim=True)
        amax = amax.view(N, K // group_size, 1)
    else:
        raise ValueError(granularity)
    scale = (amax / QMAX).clamp_min(1e-8)
    if granularity == "per_channel":
        q = torch.round(wf / scale).clamp(-QMAX, QMAX).to(torch.int8)
        deq = q.to(torch.float32) * scale
    else:
        q = torch.round(wf.view(N, K // group_size, group_size) / scale)
        q = q.clamp(-QMAX, QMAX).to(torch.int8).view(N, K)
        deq = q.view(N, K // group_size, group_size).to(torch.float32) * scale
        deq = deq.view(N, K)
    err = (wf - deq).norm() / wf.norm().clamp_min(1e-12)
    sat = (q.abs() == QMAX).float().mean()
    return q, scale.to(torch.bfloat16), err.item(), sat.item(), amax.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", default="per_channel",
                    choices=["per_channel", "per_group"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.model), os.path.abspath(args.out)
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob(os.path.join(src, "*.safetensors")))
    assert files, f"{src} 下没有 safetensors"

    out_tensors = {}
    stats = []
    n_quant = n_keep = 0
    bytes_q = bytes_k = 0

    for fi in files:
        with safe_open(fi, "pt", "cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if t.dtype is not torch.bfloat16 and t.dtype is not torch.float16 \
                        and t.dtype is not torch.float32:
                    t = t.to(torch.bfloat16)
                if QUANT_RE.search(key) and not SKIP_RE.search(key):
                    q, scale, err, sat, amax = quantize_tensor(
                        t, args.granularity, args.group_size)
                    out_tensors[key] = q                        # int8
                    out_tensors[key + "_scale"] = scale         # bf16
                    n_quant += 1
                    bytes_q += q.numel() * 1 + scale.numel() * 2
                    stats.append({
                        "name": key, "shape": list(t.shape),
                        "amax_mean": float(amax.mean()), "amax_max": float(amax.max()),
                        "scale_mean": float(scale.to(torch.float32).mean()),
                        "scale_min": float(scale.to(torch.float32).min()),
                        "scale_max": float(scale.to(torch.float32).max()),
                        "rel_err": err, "saturate_frac": sat,
                        "bytes_bf16": t.numel() * 2,
                        "bytes_int8": q.numel() * 1 + scale.numel() * 2,
                    })
                    del t, q, scale, amax
                else:
                    out_tensors[key] = t.to(torch.bfloat16)
                    n_keep += 1
                    bytes_k += t.numel() * 2
                    del t

    save_file(out_tensors, os.path.join(dst, "model_int8.safetensors"),
              metadata={"format": "pt"})
    del out_tensors

    # 其余文件原样复制（config.json / tokenizer* / generation_config 等）
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if os.path.isfile(p) and not name.endswith(".safetensors"):
            shutil.copy2(p, os.path.join(dst, name))

    # bytes_q  = 量化层的新字节（int8 + scale）
    # bytes_k  = 未量化层的 bf16 字节（embedding / lm_head / norm ...）
    tot_bf16 = sum(s["bytes_bf16"] for s in stats) + bytes_k
    cfg = {
        "quant_method": "int8_w8a16",
        "granularity": args.granularity,
        "group_size": args.group_size if args.granularity == "per_group" else None,
        "qmax": QMAX, "symmetric": True,
        "quantized_layers": [s["name"] for s in stats],
        "kept_bf16_layers": [],          # 见下面的普通权重清单
        "n_quantized": n_quant, "n_kept": n_keep,
        "bytes_bf16_total": tot_bf16,
        "bytes_int8_total": bytes_q + bytes_k,
        "note": ("quantized = block 线性层 (q/k/v/o/gate/up/down)；"
                 "embed_tokens / lm_head / norm / bias 保持 bf16"),
        "per_layer": stats,
    }
    with open(os.path.join(dst, "quant_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"quantized {n_quant} tensors, kept {n_keep} in bf16")
    print(f"quant  层 bf16={sum(s['bytes_bf16'] for s in stats)/2**20:.1f} MiB"
          f" -> int8={bytes_q/2**20:.1f} MiB"
          f"  ({bytes_q/max(sum(s['bytes_bf16'] for s in stats),1):.3f}×)")
    print(f"未量化 层 bf16={bytes_k/2**20:.1f} MiB")
    print(f"总计   bf16={tot_bf16/2**20:.1f} MiB -> int8={(bytes_q+bytes_k)/2**20:.1f} MiB"
          f"  ({tot_bf16/max(bytes_q+bytes_k,1):.3f}×)")
    errs = [s["rel_err"] for s in stats]
    print(f"相对量化误差 rel_err: mean={sum(errs)/len(errs):.5f} "
          f"max={max(errs):.5f} ({stats[errs.index(max(errs))]['name']})")
    print(f"饱和比例 saturate_frac: mean={sum(s['saturate_frac'] for s in stats)/len(stats):.6f}")
    print(f"@@QCFG@@" + json.dumps({k: v for k, v in cfg.items() if k != "per_layer"},
                                    ensure_ascii=False))


if __name__ == "__main__":
    main()
