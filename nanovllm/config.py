import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # ---------- 投机解码 ----------
    spec_k: int = 0              # 0 = 关闭（默认，保证与原版行为一致）
    spec_ngram: int = 3          # n-gram 提议器的 n
    spec_method: str = "ngram"   # "ngram" | "draft"
    draft_model: str = ""        # draft 路线的小模型路径（spec_method="draft" 时必填）
    spec_batch_threshold: int = 0  # batch 超过此值就关闭投机，0 = 不限制
    #   ★ 动机：实测 batch 大时投机解码是净亏损（验证要付 B·(k+1) 的 target 算力，
    #     而 n-gram 的接受率随上下文多样性上升而下降）。留 0 = 不限制，
    #     但默认建议设 8，benchmark 时能看出盈亏平衡点。

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
