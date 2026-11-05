"""CPU-only formal harness option, plan, and audit contract checks."""
import importlib.util
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bench = load("bench_formal")
plan = load("run_plan")


class FormalChecks(unittest.TestCase):
    def test_repeated_runs_reset_hooks_and_count_actual_graph_replays_on_cpu_fake(self):
        # This verifies host-side accounting and lifecycle only, not model or GPU behavior.
        class Params:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        class Hook:
            def __init__(self):
                self.requests = {}
            def on_request_added(self, seq):
                self.requests[seq.seq_id] = {"arrival": time.perf_counter(), "end": None, "batches": []}
            def on_deliver(self, seq, n, finished):
                now = time.perf_counter()
                self.requests[seq.seq_id]["batches"].append((now, n))
                if finished:
                    self.requests[seq.seq_id]["end"] = now
        class Seq:
            def __init__(self, sid, prompt, params):
                self.seq_id, self.token_ids = sid, list(prompt)
                self.max_tokens, self.num_completion_tokens = params.max_tokens, 0
                self.num_cached_tokens, self.block_table, self.draft_tokens = 0, [0], []
                self.completion_token_ids, self.is_finished = [], False
            def __len__(self):
                return len(self.token_ids)
        class Scheduler:
            def __init__(self):
                self.seqs, self.token_hook = [], None
                self.block_manager = types.SimpleNamespace(prefix_lookup_count=0, prefix_hit_blocks=0)
            def schedule(self):
                return self.seqs[:], self.seqs[0].num_completion_tokens == 0
            def preempt(self, seq):
                raise AssertionError("unexpected preempt")
            def postprocess(self, seqs, output, prefill):
                for seq, token in zip(seqs, output):
                    seq.token_ids.append(token)
                    seq.completion_token_ids.append(token)
                    seq.num_completion_tokens += 1
                    seq.is_finished = seq.num_completion_tokens == seq.max_tokens
                    self.token_hook.on_deliver(seq, 1, seq.is_finished)
                    if seq.is_finished:
                        self.seqs.remove(seq)
                return len(output)
        class Graph:
            def __init__(self):
                self.n = 0
            def replay(self):
                self.n += 1
        class Runner:
            def __init__(self):
                self.graphs, self.enforce_eager = {1: Graph()}, False
                self.spec_proposer = self.draft_model = self.dspark_model = None
                self.verify_graph_hits = self.verify_graph_padded = 0
                self.verify_graph_fallbacks = {}
                self.decode_graph_hits = self.decode_graph_exact_hits = self.decode_graph_padded_hits = 0
                self.decode_graph_bucket_histogram, self.decode_graph_fallbacks = {}, {}
                self.config = types.SimpleNamespace(dspark_window_tokens=0)
                self.dspark_window_stats = {}
            def run_model(self, ids, positions, is_prefill):
                self.graphs[1].replay()
                self.decode_graph_hits += 1
                self.decode_graph_exact_hits += 1
                self.decode_graph_bucket_histogram[1] = self.decode_graph_bucket_histogram.get(1, 0) + 1
            def run(self, seqs, prefill):
                if not prefill:
                    self.run_model(types.SimpleNamespace(shape=(len(seqs),)), None, False)
                return [100] * len(seqs)
        class FakeLLM:
            def __init__(self):
                self.model_runner, self.scheduler, self.token_hook = Runner(), Scheduler(), None
                self.next_id = 0
            def add_request(self, prompt, params):
                seq = Seq(self.next_id, prompt, params)
                self.next_id += 1
                self.token_hook.on_request_added(seq)
                self.scheduler.seqs.append(seq)
            def is_finished(self):
                return not self.scheduler.seqs
        cuda = types.SimpleNamespace(manual_seed_all=lambda seed: None, reset_peak_memory_stats=lambda: None,
                                     synchronize=lambda: None)
        fake_torch = types.SimpleNamespace(cuda=cuda, manual_seed=lambda seed: None)
        fake_modules = {"torch": fake_torch, "nanovllm": types.SimpleNamespace(SamplingParams=Params),
                        "nanovllm.engine": types.ModuleType("nanovllm.engine"),
                        "nanovllm.engine.token_hook": types.SimpleNamespace(TokenDeliveryHook=Hook)}
        llm = FakeLLM()
        original_graph = llm.model_runner.graphs[1]
        with patch.dict(sys.modules, fake_modules):
            results = [bench.run_once(llm, [[1, 2]], 4, 7, time.time() + 30, {"rounds": []}) for _ in range(2)]
        for result in results:
            record = {**result, "arm": "nospec", "requested_batch": 1, "output_tokens": 4, "ignore_eos": True}
            checked = bench.compute_run(record)
            self.assertTrue(checked["valid"], checked["errors"])
            self.assertEqual(result["target_decode_graph"]["exact"], 3)
            self.assertEqual(result["delivered_tokens"], 4)
            self.assertIsNone(result["gpu_event_ms"])
        self.assertEqual(original_graph.n, 6)
        self.assertIs(llm.model_runner.graphs[1], original_graph)
        self.assertIsNone(llm.token_hook)
        self.assertIsNone(llm.scheduler.token_hook)

    def test_five_arms_use_same_budget_model_length_and_graph_policy(self):
        for arm in ("nospec", "ar_full", "ar_window", "dspark_full", "dspark_ring"):
            w = 1024 if arm in ("ar_window", "dspark_ring") else 0
            cfg = bench.options_for(arm, 8, 4096, 256, w, "target", "ar", "ds", 32768)
            self.assertEqual(cfg["max_model_len"], 4096 + 256 + 8)
            self.assertEqual(cfg["max_num_seqs"], 8)
            self.assertEqual(cfg["gpu_memory_utilization"], 0.70)
            self.assertFalse(cfg["spec_graph_pad"])
            self.assertFalse(cfg["dspark_graph_pad"])
            self.assertEqual(cfg["spec_graph_bs"], tuple(range(1, 9)))
            self.assertEqual(cfg["decode_graph_bs"], tuple(range(1, 9)))
            self.assertFalse(cfg["enable_prefix_cache"])
            self.assertFalse(cfg["spec_overlap_mixed"])
            self.assertEqual(cfg.get("spec_k", 0), 0 if arm == "nospec" else 6 if arm.startswith("ar_") else 7)
        ar = bench.options_for("ar_window", 8, 4096, 256, 1024, "target", "ar", "ds", 32768)
        self.assertTrue(ar["spec_draft_win_align"] and ar["spec_draft_win_realign"])
        ring = bench.options_for("dspark_ring", 8, 4096, 256, 1024, "target", "ar", "ds", 32768)
        self.assertEqual(ring["dspark_window_storage"], "ring")
        self.assertFalse(ring["dspark_graph_pad"])

    def test_invalid_or_implicit_capacity_changes_fail(self):
        with self.assertRaises(ValueError):
            bench.options_for("dspark_full", 8, 4096, 256, 512, "target", "ar", "ds", 32768)
        with self.assertRaises(ValueError):
            bench.options_for("dspark_ring", 8, 4096, 256, 257, "target", "ar", "ds", 32768)
        with self.assertRaises(ValueError):
            bench.options_for("dspark_ring", 16, 4096, 256, 512, "target", "ar", "ds", 64)

    def test_capacity_budget_is_explicit_and_primary_budget_is_fixed(self):
        for arm, window in (("dspark_full", 0), ("dspark_ring", 1024)):
            options = bench.options_for(arm, 4, 16384, 256, window,
                                        "target", "ar", "ds", 4096, 0.85)
            self.assertEqual(options["gpu_memory_utilization"], 0.85)
            self.assertEqual(options["max_num_seqs"], 4)
            self.assertEqual(options["max_num_batched_tokens"], 4096)
            self.assertEqual(options["max_model_len"], 16384 + 256 + 8)
        bench.validate_budget("capacity", 0.85)
        bench.validate_budget("formal", 0.70)
        for phase in ("pilot", "formal"):
            bench.validate_budget(phase, 0.85)
            with self.assertRaises(ValueError):
                bench.validate_budget(phase, 0.80)
        for bad in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bench.validate_budget("capacity", bad)

    def test_full_plan_has_150_distinct_cells_and_rotates_arms(self):
        result = plan.build("formal", 1024, plan.PANELS, 3)
        self.assertEqual(len(result["jobs"]), 50)
        self.assertEqual(len(result["cells"]), 150)
        self.assertEqual(len({str(sorted(c.items())) for c in result["cells"]}), 150)
        self.assertEqual(result["matrix_scope"], "full minimum fixed-batch matrix")
        self.assertNotEqual(result["jobs"][0]["arm"], result["jobs"][5]["arm"])
        self.assertEqual(len(result["cells"]) * 3, 450)
        self.assertEqual({j["gpu_memory_utilization"] for j in result["jobs"]}, {0.70})
        self.assertEqual({c["gpu_memory_utilization"] for c in result["cells"]}, {0.70})
        self.assertEqual(result["gpu_memory_utilization"], 0.70)
        self.assertIn("no command is launched", result["execution"])

    def test_pilot_and_capacity_remain_separate(self):
        pilot = plan.build("pilot", 1024, ("code",), 1)
        self.assertEqual(len(pilot["jobs"]), 14)
        self.assertEqual({j["batch"] for j in pilot["jobs"]}, {2, 8})
        self.assertEqual({j["window"] for j in pilot["jobs"]}, {0, 512, 1024, 2048})
        self.assertEqual({j["gpu_memory_utilization"] for j in pilot["jobs"]}, {0.70})
        adjusted = plan.build("pilot", 1024, ("code",), 1, 0.85)
        self.assertEqual(adjusted["gpu_memory_utilization"], 0.85)
        self.assertEqual({j["gpu_memory_utilization"] for j in adjusted["jobs"]}, {0.85})
        self.assertEqual({c["gpu_memory_utilization"] for c in adjusted["cells"]}, {0.85})
        capacity = plan.build("capacity", 1024, ("code",), 1)
        self.assertEqual([(x["batch"], x["ctx"], x["max_num_batched_tokens"]) for x in capacity["jobs"]],
                         [(4, 16384, 4096), (4, 16384, 4096)])
        self.assertEqual([x["gpu_memory_utilization"] for x in capacity["jobs"]], [0.85, 0.85])
        self.assertEqual({x["gpu_memory_utilization"] for x in capacity["cells"]}, {0.85})
        self.assertEqual(capacity["gpu_memory_utilization"], 0.85)
        self.assertIn("target pool block count may differ", capacity["jobs"][0]["capacity_policy"])


if __name__ == "__main__":
    unittest.main()
