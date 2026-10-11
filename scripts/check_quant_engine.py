#!/usr/bin/env python
"""check_quant_engine.py —— 验收【交付的配置开关】本身

三件事：
 1. 正面：draft_model 指向「带 quant_config.json 的目录」时，引擎**自动**量化 draft 的 FFN，
    并报出量化字节数 / KV 池 / 一次生成跑通。
 2. 对照：同配置但不带 quant_config.json 的 draft 目录，量化统计必须为空。
 3. 反面：直接把【int8 checkpoint 目录】当模型传进去，必须**报错**（不能静默装错权重）。

用法:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/check_quant_engine.py
"""
import os
import sys
import gc
import json
import atexit
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from nanovllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
D_BF16 = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
D_QDIR = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-qffn")     # bf16 权重 + quant_config.json(ffn)
D_QALL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-qall")     # bf16 权重 + quant_config.json(全线性层)
D_INT8 = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-int8-ffn")  # 真·int8 checkpoint
# 真·int8 checkpoint（全线性层 196 个张量）—— 预量化直载用
D_QALL_INT8 = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B-int8-all")

P = ("def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n"
     "        total += num\n    return total\n\n")


def build(draft, tag, direct=False):
    # 打点：KV 预算公式 = 0.9*total - used - peak + current（见 allocate_kv_cache）
    from nanovllm.engine.model_runner import ModelRunner
    _orig = ModelRunner.allocate_kv_cache

    def traced(self):
        free, total = torch.cuda.mem_get_info()
        st = torch.cuda.memory_stats()
        print(f"[kv-budget] total={total/2**30:.2f}GB used={(total-free)/2**30:.2f}GB "
              f"peak={st['allocated_bytes.all.peak']/2**30:.3f}GB "
              f"current={st['allocated_bytes.all.current']/2**30:.3f}GB")
        return _orig(self)

    ModelRunner.allocate_kv_cache = traced
    # ★ 同容量对照（P9）：把 KV 池强制到 FORCE_BLOCKS 块，两条路径在【同样大的池子】
    #   下比「capture 后 / 生成峰值」—— 排除「池子小所以总占用小」这个混淆。
    #   做法：反解利用率 u 使 budget == FORCE_BLOCKS*per_block（budget = int(T*u)-used-peak+cur）。
    _fb = int(os.environ.get("FORCE_BLOCKS", "0") or 0)
    if _fb:
        _orig2 = ModelRunner.allocate_kv_cache

        def _forced(self, _fb=_fb, _orig2=_orig2):
            c = self.config
            hf, dhf = c.hf_config, self.draft_hf_config
            bs = c.kvcache_block_size
            hd = lambda h: getattr(h, "head_dim", h.hidden_size // h.num_attention_heads)
            per = (2 * hf.num_hidden_layers * bs * hf.num_key_value_heads * hd(hf) * hf.dtype.itemsize
                   + 2 * dhf.num_hidden_layers * bs * dhf.num_key_value_heads * hd(dhf) * dhf.dtype.itemsize)
            free, total = torch.cuda.mem_get_info()
            st = torch.cuda.memory_stats()
            need = (_fb * per + (total - free) + st["allocated_bytes.all.peak"]
                    - st["allocated_bytes.all.current"] + per // 2)
            c.gpu_memory_utilization = need / total
            print(f"[force-blocks] 目标 {_fb} 块 -> util={c.gpu_memory_utilization:.6f} "
                  f"per_block={per}")
            return _orig2(self)

        ModelRunner.allocate_kv_cache = _forced
    # 定位 peak 是从哪一步涨起来的
    from nanovllm.utils import loader as _ld
    from nanovllm.layers import quant as _q
    _o_load, _o_apply = _ld.load_model, _q.apply_int8_quant

    def _snap(where):
        st = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"  [snap:{where}] peak={st['allocated_bytes.all.peak']/2**30:.3f} "
              f"current={st['allocated_bytes.all.current']/2**30:.3f} "
              f"used={(total-free)/2**30:.3f} reserved={st['reserved_bytes.all.current']/2**30:.3f}")

    def _l2(m, p):
        r = _o_load(m, p)
        _snap("after load_model")
        return r

    def _a2(m, n, *a, **k):
        _snap("before apply_int8_quant")
        r = _o_apply(m, n, *a, **k)
        _snap("after apply_int8_quant")
        return r

    _ld.load_model = _l2
    _q.apply_int8_quant = _a2
    import nanovllm.engine.model_runner as _mr
    _mr.apply_int8_quant = _a2
    # ★ 修 bug：runner 顶部 `from ...loader import load_model` 提前绑定了函数，
    #   只改 _ld.load_model 不会影响 runner 内的调用；必须同时改 module 属性。
    _mr.load_model = _l2
    llm = LLM(TARGET, enforce_eager=False, max_model_len=8192,
              max_num_batched_tokens=16384, spec_k=6, spec_method="draft",
draft_model=draft, spec_batch_threshold=0,
              quant_direct_load=direct)
    mr = llm.model_runner
    st = getattr(mr, "draft_quant_stats", [])
    # ★ 固定种子：让「就地量化 vs 直载」的生成 token 序列可比（P9 验证项）
    torch.manual_seed(20261010)
    out = llm.generate([P], SamplingParams(temperature=1.0, max_tokens=64,
                                           ignore_eos=True), use_tqdm=False)
    nb = sum(s["bytes_before"] for s in st)
    na = sum(s["bytes_after"] for s in st)
    rec = {
        "tag": tag, "draft_dir": draft,
        "draft_quant_modules": len(st),
        "draft_ffn_bytes_before": nb, "draft_ffn_bytes_after": na,
        "used_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "num_kvcache_blocks": mr.config.num_kvcache_blocks,
        "num_draft_blocks": getattr(mr.config, "num_draft_blocks", None),
        "kv_tokens_target": mr.config.num_kvcache_blocks * mr.block_size,
"gen_tokens": len(out[0]["token_ids"]),
        "gen_md5": hashlib.md5(",".join(map(str, out[0]["token_ids"])).encode()).hexdigest(),
    }
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return rec


def main():
    # ★ 每个 case 单起一个进程：engine 重建时显存释放不干净（第二遍会 "KV cache 预算不足"），
    #   这不是我们改出来的问题，是引擎按 0.9 利用率一次性吃掉整卡。
    case = os.environ.get("CASE", "bf16")
    if case == "bf16":
        print("@@C@@" + json.dumps(build(D_BF16, "draft_bf16 (无 quant_config.json)"),
                                   ensure_ascii=False))
    elif case == "qffn":
        print("@@C@@" + json.dumps(build(D_QDIR, "draft_qffn (自动识别 int8)"),
                                   ensure_ascii=False))
    elif case == "qall_direct":
        print("@@C@@" + json.dumps(build(D_QALL_INT8, "draft_int8直载 (全线性层)", direct=True),
                                   ensure_ascii=False))
    elif case == "qall":
        print("@@C@@" + json.dumps(build(D_QALL, "draft_qall (全线性层自动识别 int8)"),
                                   ensure_ascii=False))
    elif case == "int8dir":
        try:
            build(D_INT8, "int8 checkpoint (应报错)")
            print("@@NEG@@ FAIL —— 没有报错，静默装错了")
        except RuntimeError as e:
            print("@@NEG@@ OK —— " + str(e)[:220])
    print("@@DONE@@")


if __name__ == "__main__":
    main()
