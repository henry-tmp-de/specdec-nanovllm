"""CPU-only tests of diagnostic thresholds and prefill-only accounting."""
import importlib.util
import math
from pathlib import Path
import json
import subprocess
import sys
import time
import types

import numpy  # noqa: F401 -- initialize before local CPU PyTorch
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("test_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("gpu_compare_prefill")
bench = load("bench_prefill")


def test_logits_kl_direction_argmax_and_margin():
    p, q = [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]
    a = torch.tensor([math.log(x) for x in p])
    b = torch.tensor([math.log(x) for x in q])
    result = compare.compare_logits(a, b, 1.0, torch)
    assert result["kl"] == pytest.approx(sum(x * math.log(x / y) for x, y in zip(p, q)), abs=1e-6)
    assert result["kl_direction"] == "reference || candidate"
    assert not result["pass"] and result["reference_argmax"] != result["candidate_argmax"]
    assert result["reference_top1_top2_margin"] > 0 and result["candidate_top1_top2_margin"] > 0
    assert compare.compare_logits(a, a.clone(), 1e-4, torch)["pass"]
    b[0] = float("nan")
    assert not compare.compare_logits(a, b, 1e-4, torch)["finite"]


def test_kv_relative_and_zero_reference_absolute_fallback():
    ref = torch.tensor([1.0, -1.0])
    assert compare.compare_kv(ref, ref + 0.001, 0.01, torch)["pass"]
    assert not compare.compare_kv(ref, ref + 0.02, 0.01, torch)["pass"]
    zeros = torch.zeros(2)
    near = compare.compare_kv(zeros, torch.tensor([0.0, 5e-7]), 0.01, torch)
    assert near["pass"] and near["relative_rms"] is None and near["zero_reference_max_abs_limit"] == 1e-6
    assert not compare.compare_kv(zeros, torch.tensor([0.0, 2e-6]), 0.01, torch)["pass"]
    assert compare.sample_positions(257) == [0, 127, 128, 255, 256]
    assert compare.sample_positions(513) == [0, 127, 128, 255, 256, 511, 512]


def row(output_tokens):
    ids = [13, 14][:output_tokens]
    requests = [{"seq_id": 1, "arrival_s": 0.0, "finish_s": float(output_tokens),
                 "deliveries": [{"at_s": float(i + 1), "count": 1} for i in range(output_tokens)],
                 "output_token_ids": ids, "output_sha256": bench.old.token_hash(ids)}]
    rounds = [{"start_s": float(i), "end_s": float(i) + 0.9, "prefill": i == 0,
               "active_b": 1, "delivered": 1,
               "sequences": [{"seq_id": 1, "delivered": 1, "proposed": 0,
                              "accepted_prefix": 0, "remaining_before": output_tokens - i}]}
              for i in range(output_tokens)]
    return {"arm": "nospec", "requested_batch": 1, "output_tokens": output_tokens,
            "wall_s": 3.0, "requests": requests, "rounds": rounds, "delivered_tokens": output_tokens,
            "ignore_eos": True, "preemptions": 0, "prefix_cached_tokens": 0,
            "prefix_lookup_count": 0, "prefix_hit_blocks": 0, "observed_prefill_lengths": [1024],
            "target_decode_graph": {"exact": 0 if output_tokens == 1 else 1, "padded": 0, "fallbacks": {}}}


def test_output_one_is_valid_without_fabricated_decode():
    result = bench.validate_result(row(1), [1024])
    assert result["valid"], result["errors"]
    assert result["input_tokens"] == 1024 and result["prefill_only_input_tps"] == pytest.approx(1024 / 3)
    assert result["decode_span_s"] is None and result["full_request_output_tps"] is None
    wrong = row(1)
    wrong["observed_prefill_lengths"] = [512]
    assert not bench.validate_result(wrong, [1024])["valid"]


def test_decode_graph_is_measured_and_fallback_rejected():
    sample = row(2)
    result = bench.validate_result(sample, [1024])
    assert result["valid"], result["errors"]
    assert result["prefill_only_input_tps"] is None and result["full_request_output_tps"] == pytest.approx(2 / 3)
    sample["target_decode_graph"]["fallbacks"] = {"missing": 1}
    assert not bench.validate_result(sample, [1024])["valid"]


def test_fake_runner_traces_16_decode_steps_and_distinct_kv_preimages(monkeypatch):
    class Params:
        def __init__(self, **kwargs):
            self.max_tokens = kwargs["max_tokens"]
    stub = types.ModuleType("nanovllm")
    stub.SamplingParams = Params
    monkeypatch.setitem(sys.modules, "nanovllm", stub)

    class Seq:
        def __init__(self, index, prompt, blocks):
            self.seq_id, self.prompt, self.block_table = index + 1, prompt, blocks
            self.num_cached_tokens, self.num_scheduled_tokens, self.num_completion_tokens = 0, len(prompt), 0
    class Runner:
        def __init__(self):
            self.block_size = 256
            self.kv_cache = torch.zeros((2, 3, 7, 256, 1, 2))
            self.config = types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=3))
            self.spec_proposer = None
            self.verify_graph_hits = self.verify_graph_padded = 0
            self.verify_graph_fallbacks = {}
            self.decode_graph_hits = self.decode_graph_exact_hits = self.decode_graph_padded_hits = 0
            self.decode_graph_fallbacks = {}
            self.sampler = lambda logits, temperatures: torch.zeros(logits.shape[0], dtype=torch.int64)
            self.preimage_verified = False
        def run(self, active, prefill):
            if prefill:
                for seq in active:
                    for index, block in enumerate(seq.block_table):
                        n = min(256, len(seq.prompt) - index * 256)
                        assert torch.isnan(self.kv_cache[:, :, block, :n]).all().item()
                        assert torch.eq(self.kv_cache[:, :, block, n:], 0).all().item()
                        self.kv_cache[:, :, block, :n].fill_(seq.seq_id)
                self.preimage_verified = True
            else:
                self.decode_graph_hits += 1
                self.decode_graph_exact_hits += 1
            logits = torch.tensor([[0.0, 1.0, 2.0]] * len(active))
            return self.sampler(logits, torch.ones(len(active))).tolist()
    class FakeLLM:
        def __init__(self):
            self.model_runner = Runner()
            self.scheduler = types.SimpleNamespace(waiting=[], running=[])
            self.next_block = 0
            self.steps = 0
        def add_request(self, prompt, params):
            needed = (len(prompt) + 255) // 256
            blocks = list(range(self.next_block, self.next_block + needed))
            self.next_block += needed
            self.scheduler.waiting.append(Seq(len(self.scheduler.waiting), prompt, blocks))
        def is_finished(self):
            return self.steps >= 17
        def step(self):
            prefill = self.steps == 0
            active = self.scheduler.waiting if prefill else self.scheduler.running
            self.model_runner.run(active, prefill)
            for seq in active:
                seq.num_completion_tokens += 1
            if prefill:
                self.scheduler.running = self.scheduler.waiting
                self.scheduler.waiting = []
            self.steps += 1
            if self.is_finished():
                self.scheduler.running = []
    llm = FakeLLM()
    original_sampler, original_run = llm.model_runner.sampler, llm.model_runner.run
    inputs = [(list(range(257)), {"actual_input_sha256": "a" * 64}),
              (list(range(513)), {"actual_input_sha256": "b" * 64})]
    result = compare.collect_case(llm, inputs, {"deadline_unix": time.time() + 30}, "full", 128, torch)
    assert llm.model_runner.preimage_verified
    assert llm.model_runner.sampler is original_sampler and llm.model_runner.run == original_run
    assert [len(x) for x in result["logits"]] == [17, 17]
    assert [len(x["positions"]) for x in result["kv"]] == [5, 7]
    assert all(torch.isfinite(t).all().item() for entry in result["kv"] for t in entry["values"])
    assert result["info"]["graph_delta"]["decode_graph_exact_hits"] == 16


def test_reduced_plan_has_only_mixed_b2_and_no_extra_timing(tmp_path):
    prompt = list(range(1024))
    metadata = {"input_sha256": compare.token_hash(prompt)}
    manifest = {"context_tokens": 1024, "panels": {"code": [
        {"token_ids": prompt, "metadata": metadata}, {"token_ids": prompt, "metadata": metadata}]}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    cases = compare.diagnostic_cases(path)
    assert list(cases) == ["b2_257_513"] and [len(ids) for ids, _ in cases["b2_257_513"]] == [257, 513]
    base = [sys.executable, "-B", str(HERE / "bench_prefill.py"), "--target", "unused",
            "--manifest", "unused", "--out", str(tmp_path / "must-not-exist"), "--mode", "full", "--budget", "0.85"]
    for forbidden in (("--batch", "1"), ("--ctx", "1024"), ("--output-tokens", "1"),
                      ("--chunk", "512"), ("--repeats", "5")):
        result = subprocess.run(base + list(forbidden), capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and "invalid choice" in result.stderr
    compare_cmd = [sys.executable, "-B", str(HERE / "gpu_compare_prefill.py"), "--target", "unused",
                   "--manifest", "unused", "--out", str(tmp_path / "must-not-exist"),
                   "--budget", "0.85", "--chunk", "512"]
    result = subprocess.run(compare_cmd, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "invalid choice" in result.stderr
    assert not (tmp_path / "must-not-exist").exists()
