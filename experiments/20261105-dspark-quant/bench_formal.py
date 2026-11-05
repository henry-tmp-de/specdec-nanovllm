"""DSpark W512 quantization wrapper for the previous fixed-batch harness.

Launch only under this experiment's gpu_control.py. Raw runs include the real
module/buffer inventory; metadata inspection occurs outside the measured loop.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/20261105-dspark-window-resume"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))


def load_bench():
    spec = importlib.util.spec_from_file_location("dspark_previous_bench_formal", OLD / "bench_formal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()
    return (git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=all", "--",
                                               "nanovllm", "experiments/20261105-dspark-window-resume",
                                               "experiments/20261105-dspark-quant")))


def runtime_inventory(runner, preset):
    """Inspect actual buffers without device synchronization or copying data."""
    import torch
    from torch import nn
    from nanovllm.spec_decode.dspark_quant import DSparkInt8Linear, expected_names

    model = runner.dspark_model
    if model is None or runner.config.dspark_quant_preset != preset:
        raise RuntimeError("runner does not match requested DSpark quantization preset")
    if (model.embedding_weight is not runner.model.model.embed_tokens.weight or
            model.lm_head_weight is not runner.model.lm_head.weight):
        raise RuntimeError("DSpark no longer aliases target embedding and LM head")
    selected = set(expected_names(preset))
    candidates = []
    for name in expected_names("all"):
        module = model.get_submodule(name)
        if name in selected:
            if not isinstance(module, DSparkInt8Linear):
                raise RuntimeError(f"selected module missing INT8 wrapper: {name}")
            weight, scale = module.qweight, module.scale
            if weight.dtype != torch.int8 or scale.dtype != torch.bfloat16 or scale.shape != (weight.shape[0],):
                raise RuntimeError(f"invalid quantized tensor metadata: {name}")
            quantized = True
        else:
            if type(module) is not nn.Linear or module.weight.dtype != torch.bfloat16:
                raise RuntimeError(f"unselected module is not BF16 nn.Linear: {name}")
            weight, scale, quantized = module.weight, None, False
        if weight.device != model.fc.weight.device:
            raise RuntimeError(f"candidate device mismatch: {name}")
        weight_bytes = weight.numel() * weight.element_size()
        scale_bytes = 0 if scale is None else scale.numel() * scale.element_size()
        candidates.append({"name": name, "shape": list(weight.shape), "quantized": quantized,
                           "weight_dtype": str(weight.dtype), "weight_bytes": weight_bytes,
                           "scale_dtype": str(scale.dtype) if scale is not None else None,
                           "scale_shape": list(scale.shape) if scale is not None else None,
                           "scale_bytes": scale_bytes, "total_bytes": weight_bytes + scale_bytes})
    if {x["name"] for x in runner.dspark_quant_stats} != selected:
        raise RuntimeError("loading statistics differ from actual selected modules")
    selected_stats = {x["name"]: x for x in runner.dspark_quant_stats}
    if any(x["quantized"] and x["total_bytes"] != selected_stats[x["name"]]["bytes_after"]
           for x in candidates):
        raise RuntimeError("loading and live buffer byte counts differ")
    noncandidate = []
    for name, p in model.named_parameters():
        if name.endswith(".weight") and name[:-7] in set(expected_names("all")):
            continue
        if p.dtype != torch.bfloat16:
            raise RuntimeError(f"excluded DSpark parameter is not BF16: {name}")
        noncandidate.append({"name": name, "shape": list(p.shape), "dtype": str(p.dtype),
                             "bytes": p.numel() * p.element_size()})
    shared = [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype),
               "bytes": p.numel() * p.element_size(), "aliases_target": True}
              for name, p in (("embedding_weight", model.embedding_weight),
                              ("lm_head_weight", model.lm_head_weight))]
    if model.fc.weight.dtype != torch.bfloat16 or any(x["dtype"] != "torch.bfloat16" for x in shared):
        raise RuntimeError("fc and target-shared tensors must remain BF16")
    return {"preset": preset, "loading": "BF16 device checkpoint then CPU row-chunk PTQ",
            "quantize_host_wall_s": runner.dspark_quant_host_s,
            "quantized_modules": len(selected), "candidate_modules": candidates,
            "candidate_total_bytes": sum(x["total_bytes"] for x in candidates),
            "noncandidate_parameters": noncandidate,
            "noncandidate_parameter_bytes": sum(x["bytes"] for x in noncandidate),
            "shared_target_weights": shared,
            "shared_bytes_note": "listed for identity; already counted in target, do not add to DSpark",
            "loading_stats": runner.dspark_quant_stats}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dspark-quant-preset", choices=("none", "ffn", "attention", "all"), required=True)
    args, rest = parser.parse_known_args()
    bench = load_bench()
    original_options, original_run = bench.options_for, bench.run_once

    def options(arm, batch, ctx, output_tokens, window, *a, **kw):
        if arm != "dspark_ring" or batch not in (1, 8, 16) or ctx != 1024 or output_tokens != 128 or window != 512:
            raise ValueError("this pilot requires DSpark ring W512, ctx=1024, output=128 and B=1/8/16")
        result = original_options(arm, batch, ctx, output_tokens, window, *a, **kw)
        result["dspark_quant_preset"] = args.dspark_quant_preset
        return result

    def run_once(llm, *a, **kw):
        inventory = runtime_inventory(llm.model_runner, args.dspark_quant_preset)
        result = original_run(llm, *a, **kw)
        result["dspark_quant"] = inventory
        return result

    bench.options_for = options
    bench.run_once = run_once
    bench.git_provenance = git_provenance
    sys.argv = [sys.argv[0], *rest]
    bench.main()


if __name__ == "__main__":
    main()
