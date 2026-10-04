"""
Draft model 提议器 —— 用真实小模型自回归生成候选。

与 n-gram 提议的本质区别
------------------------
n-gram：从历史文本「捞」候选，没有真实分布
        → 接受概率退化为 min(1, p/1) = p[draft]
        → 实测自由生成场景 <1%，无收益

draft model：小模型【真实自回归】生成
        → 有真实分布 q，接受概率 min(1, p/q) 才有理论保障
        → 文献上这个值通常 0.6~0.8

前置条件
--------
draft 与 target 必须共享词表，否则无法比较同一 token 的概率。
本项目：Qwen3-0.6B (draft)配 Qwen3-4B (target)，vocab 均为 151936 ✅

代价
----
draft 需要自己的 KV cache（层数/头数都与 target 不同，不能共用）。
显存按 block_bytes 比例从总预算里切。
"""

from typing import List, Optional, Tuple

import torch
from torch import nn


class DraftModelProposer:
    """用小模型自回归生成 k 个候选，返回 token 和它们在 draft 下的真实概率。"""

    def __init__(self, model: nn.Module, k: int = 4, block_size: int = 256):
        self.model = model
        self.k = k
        self.block_size = block_size
        self.kv_cache: Optional[torch.Tensor] = None
        self.n_layers_bound = 0

    # ------------------------------------------------------------------
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> int:
        """把 draft 各层的 k_cache/v_cache 接到独立显存块上。

        与 target 的做法一致（见 model_runner.allocate_kv_cache）：
        遍历带 k_cache/v_cache 属性的子模块，按顺序绑定。
        返回绑定的层数，供调用方校验显存算得对不对。
        """
        self.kv_cache = kv_cache
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        self.n_layers_bound = layer_id
        return layer_id

    def cache_bytes(self, n_kv_heads: int, head_dim: int, dtype_itemsize: int) -> int:
        """draft 一块 KV cache 占多少字节/block（用于切分显存预算）。"""
        return (2 * self.n_layers_bound * self.block_size
                * n_kv_heads * head_dim * dtype_itemsize)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def propose(
        self,
        seq_block_table,
        context_len: int,
        last_token: int,
        temperature: float,
    ) -> Tuple[List[int], torch.Tensor]:
        """自回归生成 k 个候选。

        参数
        ----
        seq_block_table : list[int]  该序列在【draft 自己的 KV cache】里的 block 表
        context_len     : int       已确认的 token 数（draft 侧的长度）
        last_token      : int       上一个 token（draft 从这里继续）
        temperature     : float     采样温度

        返回
        ----
        chain  : List[int]           k 个候选 token
        probs  : (k, vocab) 每个候选在【draft 分布】下的完整概率向量
                 ★ 这是核心：真实分布让 min(1, p/q) 的接受率有了理论保障
        """
        device = next(self.model.parameters()).device

        from nanovllm.utils.context import set_context, reset_context

        chain: List[int] = []
        cand_probs: List[torch.Tensor] = []

        cur_token = last_token
        pos = context_len          # 下一个 token 的位置

        for step in range(self.k):
            # ---- draft 前向：每步只算1 个 token，走 decode 路径 ----
            input_ids = torch.tensor([cur_token], dtype=torch.int64, device=device)
            positions = torch.tensor([pos], dtype=torch.int64, device=device)

            # slot_mapping：draft 自己的 cache 里，这个 token 写哪个物理槽
            # 简化：draft cache 与 target 独立，但 block_table 结构相同
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            if block_idx < len(seq_block_table):
                physical = seq_block_table[block_idx] * self.block_size + offset
            else:
                physical = -1
            slot_mapping = torch.tensor([physical], dtype=torch.int32, device=device)
            cache_seqlen = torch.tensor([pos + 1], dtype=torch.int32, device=device)
            block_tables = torch.tensor([seq_block_table], dtype=torch.int32, device=device)

            set_context(False, slot_mapping=slot_mapping,
                        context_lens=cache_seqlen, block_tables=block_tables)
            try:
                logits = self.model.compute_logits(self.model(input_ids, positions))
            finally:
                reset_context()

            p = torch.softmax(logits.float().div_(temperature), dim=-1)
            noise = torch.empty_like(p).exponential_(1.0).clamp_min_(1e-10)
            tok = int((p / noise).argmax(dim=-1))

            chain.append(tok)
            cand_probs.append(p[0])
            cur_token = tok
            pos += 1

        return chain, torch.stack(cand_probs)