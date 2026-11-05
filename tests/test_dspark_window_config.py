"""No-model configuration and prefix-policy admission checks."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("nanovllm")
pkg.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", pkg)
transformers = types.ModuleType("transformers")
class FakeAutoConfig:
    @staticmethod
    def from_pretrained(path):
        return SimpleNamespace(max_position_embeddings=4096)


transformers.AutoConfig = FakeAutoConfig
quant = types.ModuleType("nanovllm.layers.quant")
quant.hf_names_to_modules = lambda names: names
spec = importlib.util.spec_from_file_location("isolated_dspark_config", ROOT / "nanovllm/config.py")
config_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"transformers": transformers, spec.name: config_module}):
    spec.loader.exec_module(config_module)
Config = config_module.Config


@pytest.fixture(autouse=True)
def isolated_quant_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "nanovllm.layers.quant", quant)


def test_defaults_and_uniform_prefix_disable(tmp_path):
    default = Config(str(tmp_path))
    assert default.dspark_window_tokens == 0 and default.enable_prefix_cache
    assert default.dspark_window_storage == "full_cache"
    assert default.decode_graph_bs == ()
    for method in ("ngram", "draft", "dspark"):
        c = Config(str(tmp_path), spec_method=method, enable_prefix_cache=False)
        assert not c.enable_prefix_cache


@pytest.mark.parametrize("changes, message", [
    ({"dspark_window_tokens": -1}, "invalid"),
    ({"dspark_window_storage": "ring"}, "positive"),
    ({"dspark_window_tokens": 257}, "multiple"),
    ({"dspark_window_tokens": 256}, "spec_method"),
    ({"dspark_window_tokens": 256, "spec_method": "dspark", "spec_k": 7}, "prefix"),
    ({"dspark_window_tokens": 256, "spec_method": "dspark", "spec_k": 7,
      "enable_prefix_cache": False, "spec_overlap_mixed": True}, "mixed"),
    ({"dspark_window_tokens": 256, "spec_method": "dspark", "spec_k": 7,
      "enable_prefix_cache": False, "tensor_parallel_size": 2}, "TP=1"),
    ({"dspark_window_tokens": 256, "spec_method": "dspark", "spec_k": 7,
      "enable_prefix_cache": False, "spec_draft_head_blocks": 1}, "sink"),
])
def test_unsupported_configurations_fail_before_model_loading(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        Config(str(tmp_path), **changes)


@pytest.mark.parametrize("storage", ["full_cache", "ring"])
def test_valid_window_config(tmp_path, storage):
    c = Config(str(tmp_path), dspark_window_tokens=512, dspark_window_storage=storage,
               spec_method="dspark", spec_k=7, enable_prefix_cache=False)
    assert c.dspark_window_tokens == 512 and c.dspark_window_storage == storage


@pytest.mark.parametrize("buckets", [[], (0, 4), (-1, 4), (1, 5), (1, 1, 4),
                                      (1, 2, 3), (True, 4), (1.0, 4), ("1", 4)])
def test_invalid_target_decode_graph_buckets(tmp_path, buckets):
    with pytest.raises(ValueError, match="decode_graph_bs"):
        Config(str(tmp_path), max_num_seqs=4, decode_graph_bs=buckets)


def test_target_decode_graph_buckets_can_be_unsorted_and_sparse(tmp_path):
    c = Config(str(tmp_path), max_num_seqs=4, decode_graph_bs=(4, 1))
    assert c.decode_graph_bs == (4, 1)


@pytest.mark.parametrize("max_batch", [11, 15])
def test_target_decode_graph_exact_one_to_max_is_valid(tmp_path, max_batch):
    buckets = tuple(range(1, max_batch + 1))
    c = Config(str(tmp_path), max_num_seqs=max_batch, decode_graph_bs=buckets)
    assert c.decode_graph_bs == buckets


def test_target_decode_graph_buckets_above_existing_graph_limit_rejected(tmp_path):
    with pytest.raises(ValueError, match="<= 512"):
        Config(str(tmp_path), max_num_seqs=513, decode_graph_bs=(1, 513))
