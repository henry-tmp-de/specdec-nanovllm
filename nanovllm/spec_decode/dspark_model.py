"""Qwen3 DSpark inference, matching DeepSpec's released block-7 model.

Derived from DeepSpec's Qwen3 DSpark implementation:
https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2
Copyright (c) 2026 The DeepSpec Authors, MIT; see THIRD_PARTY/DeepSpec-LICENSE.

The target owns its weights and paged cache.  This module owns *different*
context K/V values indexed by the same physical slots.  Speculative suffixes
are never made visible merely because they were written: the caller supplies
the confirmed context length on every proposal.
"""
from __future__ import annotations

import glob
import os

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotary(x, positions, inv_freq):
    """HF Qwen3 rotate_half convention; x is [B, positions, heads, dim]."""
    angles = positions.float().unsqueeze(-1) * inv_freq
    angles = torch.cat((angles, angles), dim=-1)
    cos = angles.cos().to(x.dtype).unsqueeze(-2)
    sin = angles.sin().to(x.dtype).unsqueeze(-2)
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, nh, nk, d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(h, nh * d, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(h, nk * d, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(h, nk * d, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(nh * d, h, bias=False)
        self.q_norm = RMSNorm(d, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(d, cfg.rms_norm_eps)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class MarkovHead(nn.Module):
    def __init__(self, vocab, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, vocab, bias=False)

    def forward(self, token_ids):
        return self.markov_w2(self.markov_w1(token_ids))


class ConfidenceHead(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.proj = nn.Linear(inputs, 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


class DSparkModel(nn.Module):
    """Inference-only model; the embedding and LM head alias target weights."""
    def __init__(self, cfg, embedding_weight, lm_head_weight):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.markov_head = MarkovHead(cfg.vocab_size, cfg.markov_rank)
        inputs = cfg.hidden_size + (cfg.markov_rank if cfg.confidence_head_with_markov else 0)
        self.confidence_head = ConfidenceHead(inputs) if cfg.enable_confidence_head else None
        # Do not register the shared target parameters again as DSpark checkpoint keys.
        self.__dict__["embedding_weight"] = embedding_weight
        self.__dict__["lm_head_weight"] = lm_head_weight
        rope = getattr(cfg, "rope_parameters", None) or {}
        theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 1_000_000))
        inv = 1.0 / (theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.cache = None
        self.block_size = None
        self.sentinel = None
        self.window_tokens = 0
        self.scratch = None

    def bind_cache(self, num_blocks, block_size, scratch_rows=0):
        c = self.cfg
        self.block_size = block_size
        self.sentinel = num_blocks * block_size
        # The last slot safely absorbs graph-padding writes; zeroed so masked
        # padding cannot inject NaNs into SDPA even on a backend that loads it.
        self.cache = torch.empty(2, c.num_hidden_layers, self.sentinel + 1,
                                 c.num_key_value_heads, c.head_dim,
                                 dtype=self.fc.weight.dtype, device=self.fc.weight.device)
        self.cache[:, :, self.sentinel].zero_()
        if scratch_rows:
            self.scratch = torch.empty(2, c.num_hidden_layers, scratch_rows,
                                       c.num_key_value_heads, c.head_dim,
                                       dtype=self.fc.weight.dtype, device=self.fc.weight.device)

    def write_context(self, features, positions, slots, scratch=False):
        """Project selected target outputs into each layer's separate context KV."""
        if self.cache is None:  # target warmup, before the cache is allocated
            return
        x = self.hidden_norm(self.fc(features))
        c = self.cfg
        if scratch:
            if self.scratch is None or features.shape[0] > self.scratch.shape[2]:
                raise RuntimeError("DSpark scratch capacity is insufficient")
        else:
            safe_slots = slots.long().clamp(min=0)
            safe_slots = torch.where(slots >= 0, safe_slots, self.sentinel)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            key = attn.k_norm(attn.k_proj(x).view(-1, c.num_key_value_heads, c.head_dim))
            key = rotary(key, positions, self.inv_freq)
            value = attn.v_proj(x).view(-1, c.num_key_value_heads, c.head_dim)
            if scratch:
                self.scratch[0, i, :features.shape[0]].copy_(key)
                self.scratch[1, i, :features.shape[0]].copy_(value)
            else:
                self.cache[0, i].index_copy_(0, safe_slots, key)
                self.cache[1, i].index_copy_(0, safe_slots, value)
        # Padding writes to the sentinel are ignored by the mask.  Restore zero
        # for the next gather; this also handles multiple padded rows safely.
        if not scratch:
            self.cache[:, :, self.sentinel].zero_()

    def commit_scratch(self, rows, slots):
        if len(rows) != len(slots):
            raise ValueError("DSpark commit row/slot count mismatch")
        if not rows:
            return
        device = self.cache.device
        src = torch.tensor(rows, device=device, dtype=torch.long)
        dst = torch.tensor(slots, device=device, dtype=torch.long)
        self.cache.index_copy_(2, dst, self.scratch.index_select(2, src))

    def forward_hidden(self, anchors, context_lens, block_tables, context_bucket):
        """One noncausal seven-position backbone forward for a whole batch."""
        c, B = self.cfg, anchors.shape[0]
        assert self.cache is not None
        positions = context_lens[:, None] + torch.arange(c.block_size, device=anchors.device)
        ids = torch.full((B, c.block_size), c.mask_token_id, dtype=torch.long, device=anchors.device)
        ids[:, 0] = anchors
        hidden = F.embedding(ids, self.embedding_weight)
        offsets = torch.arange(context_bucket, device=anchors.device)
        if self.window_tokens:
            # Window mode receives explicit slots in chronological order;
            # context_lens still holds the absolute anchor position for RoPE.
            indices = block_tables.long()
            visible_lens = context_lens.clamp(max=self.window_tokens)
            valid = (offsets[None, :] < visible_lens[:, None]) & (indices >= 0)
        else:
            logical_blocks = torch.div(offsets, self.block_size, rounding_mode="floor")
            physical = block_tables[:, logical_blocks].long()
            indices = physical * self.block_size + offsets.remainder(self.block_size)
            valid = (offsets[None, :] < context_lens[:, None]) & (physical >= 0)
        indices = torch.where(valid, indices, self.sentinel).reshape(-1)
        noise_valid = torch.ones(B, c.block_size, dtype=torch.bool, device=anchors.device)
        attn_mask = torch.cat((valid, noise_valid), dim=1)[:, None, None, :]
        for i, layer in enumerate(self.layers):
            residual = hidden
            x = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(x).view(B, c.block_size, c.num_attention_heads, c.head_dim))
            k = attn.k_norm(attn.k_proj(x).view(B, c.block_size, c.num_key_value_heads, c.head_dim))
            v = attn.v_proj(x).view(B, c.block_size, c.num_key_value_heads, c.head_dim)
            q = rotary(q, positions, self.inv_freq).transpose(1, 2)
            k = rotary(k, positions, self.inv_freq).transpose(1, 2)
            v = v.transpose(1, 2)
            ctx_k = self.cache[0, i].index_select(0, indices).view(B, context_bucket, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            ctx_v = self.cache[1, i].index_select(0, indices).view(B, context_bucket, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            keys = torch.cat((ctx_k, k), dim=2)
            values = torch.cat((ctx_v, v), dim=2)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=attn_mask,
                                                 dropout_p=0.0, is_causal=False,
                                                 enable_gqa=True)
            hidden = residual + attn.o_proj(out.transpose(1, 2).reshape(B, c.block_size, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.norm(hidden)

    def propose(self, anchors, context_lens, block_tables, context_bucket, temperatures):
        hidden = self.forward_hidden(anchors, context_lens, block_tables, context_bucket)
        base_logits = F.linear(hidden, self.lm_head_weight)
        previous = anchors
        tokens, corrected = [], []
        for i in range(self.cfg.block_size):
            logits = base_logits[:, i] + self.markov_head(previous)
            probs = torch.softmax(logits.float() / temperatures[:, None], dim=-1)
            previous = (probs / torch.empty_like(probs).exponential_(1.0)).argmax(dim=-1)
            tokens.append(previous)
            corrected.append(logits)
        return torch.stack(tokens, dim=1), torch.stack(corrected, dim=1)

    def base_logits(self, anchors, context_lens, block_tables, context_bucket):
        """Deterministic block forward; sampling is intentionally separate."""
        return F.linear(self.forward_hidden(anchors, context_lens, block_tables, context_bucket),
                        self.lm_head_weight)

    def sample_step(self, base_logits, previous, temperatures):
        """Identical conditional q and exponential-race sampler to propose()."""
        logits = base_logits + self.markov_head(previous)
        probs = torch.softmax(logits.float() / temperatures[:, None], dim=-1)
        token = (probs / torch.empty_like(probs).exponential_(1.0)).argmax(dim=-1)
        return token, logits

    def confidence(self, hidden, previous):
        if self.confidence_head is None:
            return None
        if self.cfg.confidence_head_with_markov:
            hidden = torch.cat((hidden, self.markov_head.markov_w1(previous).to(hidden.dtype)), dim=-1)
        return self.confidence_head(hidden).float()


def validate_config(cfg, target_cfg, spec_k):
    """Reject unsupported families instead of silently running a different model."""
    required = dict(block_size=7, num_hidden_layers=5, markov_rank=256,
                    markov_head_type="vanilla", mask_token_id=151669,
                    enable_confidence_head=True, confidence_head_with_markov=True)
    for key, expected in required.items():
        value = getattr(cfg, key, None)
        if value != expected:
            raise ValueError(f"DSpark config {key}={value!r}; expected {expected!r}")
    if list(cfg.target_layer_ids) != [1, 9, 17, 25, 33]:
        raise ValueError(f"unexpected target layers: {cfg.target_layer_ids}")
    if spec_k != cfg.block_size:
        raise ValueError("DSpark block-7 emits seven candidates; spec_k must be 7")
    for field in ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
        if getattr(cfg, field) != getattr(target_cfg, field):
            raise ValueError(f"DSpark and target disagree on {field}")
    if getattr(cfg, "attention_bias", False) or getattr(cfg, "hidden_act", "silu") != "silu":
        raise ValueError("unsupported DSpark Qwen3 attention/MLP configuration")


def load_official_weights(model, path):
    """Strictly load real checkpoint and verify frozen shared weights in chunks."""
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint in {path}")
    params = dict(model.named_parameters())
    shared = {"embed_tokens.weight": model.embedding_weight, "lm_head.weight": model.lm_head_weight}
    seen = set()
    with torch.no_grad():
        for filename in files:
            with safe_open(filename, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    if key in seen:
                        raise ValueError(f"duplicate DSpark parameter {key}")
                    seen.add(key)
                    if key in shared:
                        target = shared[key]
                        tensor_slice = weights.get_slice(key)
                        if list(tensor_slice.get_shape()) != list(target.shape):
                            raise ValueError(f"{key} shape does not match target")
                        for start in range(0, target.shape[0], 4096):
                            end = min(start + 4096, target.shape[0])
                            chunk = tensor_slice[start:end].to(device=target.device, dtype=target.dtype)
                            if not torch.equal(chunk, target[start:end]):
                                raise ValueError(f"{key} differs from target at rows {start}:{end}")
                    elif key in params:
                        value = weights.get_tensor(key)
                        if value.shape != params[key].shape:
                            raise ValueError(f"wrong shape for {key}: {value.shape} != {params[key].shape}")
                        params[key].copy_(value)
                    else:
                        raise ValueError(f"unexpected DSpark checkpoint parameter {key}")
    missing = (set(params) | set(shared)) - seen
    if missing:
        raise ValueError(f"missing DSpark parameters: {sorted(missing)}")
    return model.eval()
