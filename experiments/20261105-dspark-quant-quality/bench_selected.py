"""Fixed-weight DSpark W512 speed pilot using the prior native engine harness.

Each process loads one preset once before graph capture. This does not use the
quality diagnostic's dual-weight bank. Launch under this round's shared guard
only after the coordinator chooses candidates; the manifest is the previous
1K workload manifest, not the teacher-forced quality manifest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "experiments/20261105-dspark-quant"
PRESETS = ("none", "q_o", "down", "q_o_down")


def load_existing_wrapper():
    spec = importlib.util.spec_from_file_location("dspark_prior_quant_bench", QUANT / "bench_formal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "nanovllm",
                "experiments/20261105-dspark-window-resume", "experiments/20261105-dspark-quant",
                "experiments/20261105-dspark-quant-quality")
    return git("rev-parse", "HEAD"), bool(dirty)


class RecordJSON:
    """Preserve the requested preset even if model loading fails."""
    def __init__(self, preset):
        self.preset = preset
        self.inventory = None

    loads = staticmethod(json.loads)

    def dumps(self, row, *args, **kwargs):
        if isinstance(row, dict) and row.get("type") in ("warmup", "run", "failure"):
            row = {**row, "dspark_quant_preset": self.preset}
            if self.inventory is not None:
                row.setdefault("dspark_quant", self.inventory)
        return json.dumps(row, *args, **kwargs)


def install_wrappers(bench, prior, preset):
    if preset not in PRESETS:
        raise ValueError("unsupported selected preset")
    original_options, original_run = bench.options_for, bench.run_once
    records = RecordJSON(preset)

    def options(arm, batch, ctx, output_tokens, window, *args, **kwargs):
        if (arm != "dspark_ring" or batch not in (1, 8) or ctx != 1024 or
                output_tokens != 128 or window != 512):
            raise ValueError("selected pilot requires DSpark ring W512, B=1/8, ctx=1024, output=128")
        result = original_options(arm, batch, ctx, output_tokens, window, *args, **kwargs)
        if result.get("gpu_memory_utilization") != 0.85:
            raise ValueError("selected pilot requires gpu_memory_utilization=0.85")
        result["dspark_quant_preset"] = preset
        return result

    def run_once(llm, *args, **kwargs):
        inventory = prior.runtime_inventory(llm.model_runner, preset)
        records.inventory = inventory
        result = original_run(llm, *args, **kwargs)
        result["dspark_quant"] = inventory
        return result

    bench.options_for, bench.run_once, bench.json = options, run_once, records
    bench.git_provenance = git_provenance
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dspark-quant-preset", choices=PRESETS, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 8), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dspark-model", required=True)
    parser.add_argument("--manifest", required=True, help="prior 1K fixed-batch workload manifest")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def native_argv(args):
    # AR is never loaded for the fixed DSpark arm; the old harness requires an
    # --ar-model argument syntactically, so the target path is an unused value.
    return ["--arm", "dspark_ring", "--phase", "pilot", "--batch", str(args.batch),
            "--ctx", "1024", "--window", "512", "--output-tokens", "128",
            "--panels", "code", "math", "general", "--repeats", "3",
            "--gpu-memory-utilization", "0.85", "--seed", "20261105",
            "--target", args.target, "--ar-model", args.target, "--dspark-model", args.dspark_model,
            "--manifest", args.manifest, "--out", args.out]


def main():
    args = parse_args()
    prior = load_existing_wrapper()
    bench = prior.load_bench()
    install_wrappers(bench, prior, args.dspark_quant_preset)
    sys.argv = [sys.argv[0], *native_argv(args)]
    bench.main()


if __name__ == "__main__":
    main()
