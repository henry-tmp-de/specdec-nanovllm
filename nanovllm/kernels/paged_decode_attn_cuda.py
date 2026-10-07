"""CUDA 版 paged decode attention —— Python 入口（JIT 编译 .cu）。

用法
----
    from nanovllm.kernels.paged_decode_attn_cuda import paged_decode_attention_cuda
    o = paged_decode_attention_cuda(q, k_cache, v_cache, block_table,
                                    context_lens, scale, splits=16,
                                    block_n=64, warps=4, stages=3)

第一次调用会编译 paged_decode_attn_cuda.cu（几十秒），之后走缓存。

实现细节看 .cu 里的注释；这里只负责调度和缓存扩展模块。

⚠️ 扩展里还导出了一个 `ws(...)`（warp specialization 版）。**它结果不对，别用**：
命名 barrier 的生产者/消费者握手有个没解决的竞态，`ctx=1` 就能复现，详见 README 第六节。
正式数字全部来自 `paged_decode`（mma 版）。
"""

import os
import sys

# Ampere (RTX 3090 = sm_86)。必须在 import torch.utils.cpp_extension 之前设好。
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

# torch.utils.cpp_extension 会 fork 出 ninja；pip 装的 ninja 落在 venv/bin 里，
# 而 venv 没 activate 时 PATH 里没有它 —— 自己补上，免得报 "Ninja is required"。
_bin = os.path.dirname(sys.executable)
if _bin and _bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
_ext = None


def get_ext(verbose: bool = False):
    """编译（或从缓存取）CUDA 扩展。"""
    global _ext
    if _ext is None:
        _ext = load(
            name="paged_decode_cuda_ext",
            sources=[os.path.join(_DIR, "paged_decode_attn_cuda.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "--ptxas-options=-v"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=verbose,
        )
    return _ext


def paged_decode_attention_cuda(q, k_cache, v_cache, block_table, context_lens,
                                scale, splits=8, block_n=64, warps=4, stages=3,
                                fused=True):
    """分页 decode attention（CUDA 实现）。

    形状
    ----
    q            : (num_seqs, num_q_heads, head_dim)  bf16
    k_cache      : (num_blocks, block_size, num_kv_heads, head_dim)  bf16
    v_cache      : 同 k_cache
    block_table  : (num_seqs, max_blocks) int32
    context_lens : (num_seqs,) int32
    返回          : (num_seqs, num_q_heads, head_dim) bf16

    参数
    ----
    splits  : split-K 的份数（grid.y）
    block_n : 流水线每个 stage 处理多少个位置
    warps   : 每个 CTA 的 warp 数
    stages  : 流水线深度；1 = 不用 cp.async（同步 LDG->STS，作为对照）
    fused   : True = split 之间的归约由主 kernel 里最后一个到达的 CTA 顺手做掉
              （threadfence + atomic），省掉第二个 kernel 的启动；
              False = 单独起一个 combine kernel（和 Triton v3 的结构一样）
    """
    return get_ext().paged_decode(q, k_cache, v_cache, block_table, context_lens,
                                  float(scale), int(splits), int(block_n),
                                  int(warps), int(stages), bool(fused))


def paged_decode_attention_cuda_partial(q, k_cache, v_cache, block_table,
                                        context_lens, scale, splits=8,
                                        block_n=64, warps=4, stages=3, fused=None):
    """只跑主 kernel（不做 split 之间的 combine），用来把两段时间拆开量。

    主 kernel 本身不接受 fused 开关（融合归约在调用方决定），这里收下并忽略，
    这样调用方可以无脑把同一份 cfg 传给 full / partial 两个入口。
    """
    return get_ext().partial(q, k_cache, v_cache, block_table, context_lens,
                             float(scale), int(splits), int(block_n),
                             int(warps), int(stages))
