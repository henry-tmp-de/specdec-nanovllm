"""DSpark-only W8A16 inference modules and strict, bounded-memory PTQ.

The checkpoint is loaded normally first. Selected BF16 weights are then copied
to CPU in output-row chunks, quantized with the engine's existing quantizer,
and copied to device INT8/scale buffers. No full dequantized device weight is
created or cached in the forward path.
"""
from __future__ import annotations

import torch
from torch import nn

from nanovllm.layers.quant import int8_linear, quantize_weight

PRESETS = ("none", "ffn", "attention", "all", "q_o", "down", "q_o_down")
FFN = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ATTENTION = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
NUM_LAYERS = 5


class DSparkInt8Linear(nn.Module):
    """A bias-free DSpark nn.Linear with symmetric per-output-channel INT8."""

    def __init__(self, qweight, scale):
        super().__init__()
        if (qweight.ndim != 2 or qweight.dtype != torch.int8 or
                scale.shape != (qweight.shape[0],) or scale.dtype != torch.bfloat16 or
                qweight.device != scale.device or not qweight.is_contiguous() or
                not scale.is_contiguous()):
            raise ValueError("invalid DSpark INT8 buffers")
        self.out_features, self.in_features = qweight.shape
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

    def forward(self, x):
        if (x.dtype != torch.bfloat16 or x.device != self.qweight.device or
                x.ndim < 1 or x.shape[-1] != self.in_features):
            raise ValueError("DSpark INT8 linear requires same-device BF16 activations with matching K")
        return int8_linear(x, self.qweight, self.scale, granularity="per_channel")


def expected_names(preset):
    if preset not in PRESETS:
        raise ValueError(f"invalid DSpark quantization preset: {preset!r}")
    suffixes = {"none": (), "ffn": FFN, "attention": ATTENTION, "all": FFN + ATTENTION,
                "q_o": ("self_attn.q_proj", "self_attn.o_proj"),
                "down": ("mlp.down_proj",),
                "q_o_down": ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")}[preset]
    return tuple(f"layers.{i}.{suffix}" for i in range(NUM_LAYERS) for suffix in suffixes)


def _expected_shape(cfg, suffix):
    h, inner = cfg.hidden_size, cfg.intermediate_size
    q = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_key_value_heads * cfg.head_dim
    return {
        "mlp.gate_proj": (inner, h), "mlp.up_proj": (inner, h),
        "mlp.down_proj": (h, inner), "self_attn.q_proj": (q, h),
        "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
        "self_attn.o_proj": (h, q),
    }[suffix]


def _validate_architecture(model):
    """Validate the complete candidate inventory before mutating any module."""
    if model.cfg.num_hidden_layers != NUM_LAYERS or len(model.layers) != NUM_LAYERS:
        raise ValueError("DSpark quantization requires exactly five decoder layers")
    if model.fc.weight.dtype != torch.bfloat16:
        raise ValueError("DSpark fc must remain BF16")
    device = model.fc.weight.device
    for name, param in model.named_parameters(remove_duplicate=False):
        if param.dtype != torch.bfloat16 or param.device != device:
            raise ValueError(f"DSpark parameter {name} must be BF16 on {device}")
    if any(weight.dtype != torch.bfloat16 or weight.device != device
           for weight in (model.embedding_weight, model.lm_head_weight)):
        raise ValueError("target-shared embedding and LM head must remain BF16 on the model device")
    expected = set(expected_names("all"))
    modules = dict(model.named_modules(remove_duplicate=False))
    actual = {name for name, mod in modules.items()
              if name.startswith("layers.") and isinstance(mod, (nn.Linear, DSparkInt8Linear))}
    if actual != expected:
        raise ValueError(f"unexpected DSpark linear inventory: missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")
    param_names = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        param_names.setdefault(id(param), []).append(name)
    shared_ids = {id(model.embedding_weight), id(model.lm_head_weight)}
    for name in sorted(expected):
        module = modules[name]
        suffix = name.split(".", 2)[2]
        if type(module) is not nn.Linear or module.bias is not None:
            raise TypeError(f"{name} must be an unquantized, bias-free nn.Linear")
        if tuple(module.weight.shape) != _expected_shape(model.cfg, suffix):
            raise ValueError(f"{name} has unexpected shape {tuple(module.weight.shape)}")
        if module.weight.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must have a CPU/CUDA BF16 weight")
        aliases = param_names[id(module.weight)]
        if len(aliases) != 1 or id(module.weight) in shared_ids:
            raise ValueError(f"{name} weight is shared: {aliases}")


def apply_dspark_quant(model, preset, chunk_rows=256):
    """Replace precisely the selected modules; return actual byte-level inventory.

    At most one destination INT8 matrix and one CPU BF16/FP32 row chunk are
    transient per selected layer. No GPU FP32 conversion is used. The model's
    unselected modules and target-shared weights are left in place.
    """
    names = expected_names(preset)
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if not names:
        return []
    _validate_architecture(model)
    results = []
    with torch.no_grad():
        for name in names:
            parent_name, leaf = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            old = getattr(parent, leaf)
            weight = old.weight
            n, k = weight.shape
            qweight = torch.empty((n, k), dtype=torch.int8, device=weight.device)
            scale = torch.empty((n,), dtype=torch.bfloat16, device=weight.device)
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                host = weight[start:end].detach().to(device="cpu")
                if not torch.isfinite(host).all().item():
                    raise ValueError(f"{name} contains non-finite weights at rows {start}:{end}")
                q, s = quantize_weight(host, "per_channel")
                qweight[start:end].copy_(q)
                scale[start:end].copy_(s)
            new = DSparkInt8Linear(qweight, scale)
            new.train(old.training)
            setattr(parent, leaf, new)
            results.append({"name": name, "shape": [n, k], "weight_dtype_before": "bfloat16",
                            "weight_dtype_after": "int8", "scale_dtype": "bfloat16",
                            "scale_shape": [n], "weight_bytes": qweight.numel() * qweight.element_size(),
                            "scale_bytes": scale.numel() * scale.element_size(),
                            "bytes_before": weight.numel() * weight.element_size(),
                            "bytes_after": (qweight.numel() * qweight.element_size() +
                                            scale.numel() * scale.element_size())})
            del old, weight, qweight, scale, new, host, q, s
    return results
