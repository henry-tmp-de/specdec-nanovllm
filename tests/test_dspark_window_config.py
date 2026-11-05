"""No-model configuration and prefix-policy admission checks."""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
sys.modules["transformers"] = transformers
quant = types.ModuleType("nanovllm.layers.quant")
quant.hf_names_to_modules = lambda names: names
sys.modules["nanovllm.layers.quant"] = quant
from nanovllm.config import Config


def test_defaults_and_uniform_prefix_disable(tmp_path):
    default = Config(str(tmp_path))
    assert default.dspark_window_tokens == 0 and default.enable_prefix_cache
    assert default.dspark_window_storage == "full_cache"
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
