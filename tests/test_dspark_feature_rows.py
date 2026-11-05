"""Qwen's real collection loop selects rows before residual add and cat."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", pkg)


def stub(name, **members):
    mod = types.ModuleType(name)
    mod.__dict__.update(members)
    sys.modules[name] = mod


stub("transformers", Qwen3Config=object)
stub("nanovllm.layers.activation", SiluAndMul=object)
stub("nanovllm.layers.attention", Attention=object)
stub("nanovllm.layers.layernorm", RMSNorm=object)
stub("nanovllm.layers.linear", QKVParallelLinear=object,
     MergedColumnParallelLinear=object, RowParallelLinear=object)
stub("nanovllm.layers.rotary_embedding", get_rope=lambda *a, **kw: None)
stub("nanovllm.layers.embed_head", VocabParallelEmbedding=object, ParallelLMHead=object)
spec = importlib.util.spec_from_file_location("qwen3_feature_test", ROOT / "nanovllm/models/qwen3.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, positions, hidden, residual):
        if residual is None:
            residual = hidden.clone()
        return hidden + self.index + 1, residual + self.index + 2


class Norm(nn.Module):
    def forward(self, hidden, residual):
        return hidden + residual, residual


@pytest.mark.parametrize("rows", [[0, 3, 5], [], [5, 1]])
def test_selected_features_match_full_loop_and_preserve_target(rows):
    model = module.Qwen3Model.__new__(module.Qwen3Model)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(10, 4)
    model.layers = nn.ModuleList([Layer(0), Layer(1), Layer(2)])
    model.norm = Norm()
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    positions = torch.arange(len(ids))
    hidden, full = model(ids, positions, (0, 2))
    selected_hidden, selected = model(ids, positions, (0, 2), torch.tensor(rows, dtype=torch.long))
    torch.testing.assert_close(selected_hidden, hidden)
    torch.testing.assert_close(selected, full[rows])
    assert selected.shape == (len(rows), 8)
