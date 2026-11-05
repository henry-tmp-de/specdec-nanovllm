"""CPU-only checks: raw audit, workload construction, and prewarm planning."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

audit = module("audit_results")
workload = module("run_workload")
smoke = module("bench_smoke")


def sample_run():
    ids = [7, 8, 9, 10]
    return {"type": "run", "arm": "dspark", "mode": "ring", "ctx": 252,
            "requested_batch": 1, "panel": "code", "repeat": 0, "output_tokens": 4,
            "window": 256, "graph_mode": "eager", "diagnostic_instrumented": True, "ignore_eos": True,
            "wall_s": 5.0, "delivered_tokens": 4, "preemptions": 0, "prefix_cached_tokens": 0,
            "requests": [{"seq_id": 1, "arrival_s": 0.1, "finish_s": 4.0,
                          "deliveries": [{"at_s": 1.0, "count": 1}, {"at_s": 3.0, "count": 2},
                                         {"at_s": 4.0, "count": 1}],
                          "output_token_ids": ids, "output_sha256": audit.token_hash(ids)}],
            "rounds": [{"start_s": 0.2, "end_s": 1.1, "prefill": True, "active_b": 1, "delivered": 1,
                        "sequences": [{"seq_id": 1, "delivered": 1, "proposed": 0, "accepted_prefix": 0}]},
                       {"start_s": 1.2, "end_s": 3.1, "prefill": False, "active_b": 1, "delivered": 2,
                        "sequences": [{"seq_id": 1, "delivered": 2, "proposed": 2, "accepted_prefix": 1, "remaining_before": 3}]},
                       {"start_s": 3.2, "end_s": 4.1, "prefill": False, "active_b": 1, "delivered": 1,
                        "sequences": [{"seq_id": 1, "delivered": 1, "proposed": 0, "accepted_prefix": 0, "remaining_before": 1}]}]}


class CPUChecks(unittest.TestCase):
    def test_raw_recomputation_and_real_delivery_gaps(self):
        result = audit.compute_run(sample_run())
        self.assertTrue(result["valid"], result["errors"])
        self.assertAlmostEqual(result["full_request_tps"], 0.8)
        self.assertAlmostEqual(result["ttft_s"]["p50"], 0.9)
        self.assertAlmostEqual(result["tpot_s"]["p50"], 1.0)
        self.assertEqual(result["delivery_batch_gap_s"], {"n": 2, "p50": 1.0, "p95": 2.0, "p99": 2.0})
        self.assertAlmostEqual(result["acceptance_fraction"], 0.5)
        self.assertAlmostEqual(result["tau_delivered_per_decode_sequence_round"], 1.5)

    def test_invalid_and_failure_are_never_hidden(self):
        bad = sample_run()
        bad["requests"][0]["deliveries"][1]["count"] = 3
        report = audit.audit([("x", 1, bad), ("x", 2, {"type": "failure", "message": "OOM"})], 1)
        self.assertFalse(report["data_valid"])
        self.assertEqual((report["invalid_runs"], len(report["failures"])), (1, 1))
        self.assertTrue(any("token count" in e for e in report["runs"][0]["metrics"]["errors"]))

    def test_diagnostic_groups_do_not_mix_windows_or_graph_modes(self):
        records = []
        for i, (w, g) in enumerate(((256, "eager"), (512, "padded"), (1024, "exact"))):
            row = sample_run()
            row.update(window=w, graph_mode=g, repeat=i)
            records.append(("x", i + 1, row))
        diagnostic = audit.audit(records, 3)
        self.assertTrue(diagnostic["data_valid"])
        self.assertFalse(diagnostic["performance_eligible"])
        self.assertEqual(len(diagnostic["groups"]), 3)
        self.assertNotIn("pass", diagnostic)
        formal = audit.audit(records, 3, mode="formal")
        self.assertFalse(formal["pass"])
        self.assertEqual(formal["invalid_runs"], 3)

    def test_audit_rejects_wrong_landed_and_proposal_counts(self):
        row = sample_run()
        row["rounds"][1]["sequences"][0]["accepted_prefix"] = 0
        self.assertFalse(audit.compute_run(row)["valid"])
        row = sample_run()
        row["rounds"][1]["sequences"][0]["proposed"] = 4
        self.assertFalse(audit.compute_run(row)["valid"])

    def test_ar_tail_acceptance_is_not_mistaken_for_delivered_tokens(self):
        row = sample_run()
        row["arm"] = "ar_full"
        last = row["rounds"][-1]["sequences"][0]
        last.update(proposed=6, accepted_prefix=3, verified_output_tokens=4, remaining_before=1)
        result = audit.compute_run(row)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["accepted_prefix_tokens"], 4)
        self.assertEqual(result["output_tokens"], 4)

    def test_formal_requires_plan_provenance_guard_and_matching_inputs(self):
        token_ids = [42] * 252
        h = audit.token_hash(token_ids)
        source_meta = {"input_sha256": h, "wiki_tokens": 249, "task_tokens": 3,
                       "task_truncated_from_left": False, "dataset": "fake", "row": 0}
        manifest = {"context_tokens": 252, "panels": {"code": [{"token_ids": token_ids,
                                                                   "metadata": source_meta}]}}
        cell = {"arm": "dspark_ring", "mode": "ring", "ctx": 252, "requested_batch": 1,
                "panel": "code", "window": 256, "graph_mode": "graph"}
        rows, ledger = [], []
        for repeat in range(3):
            ident = f"job{repeat}"
            row = sample_run()
            row.update(**cell, phase="formal", repeat=repeat, seed=3, temperature=1.0, gpu_uuid="GPU-OK",
                       engine_commit="commit", engine_dirty=False, gpu_memory_utilization=0.70,
                       max_num_seqs=1, max_model_len=264, max_num_batched_tokens=264,
                       prefix_lookup=False, graph_policy={"padding": True}, mixed_overlap=False,
                       target_config_sha256="target", draft_config_sha256="draft", guard_id=ident,
                       target_provenance={"path": "target", "weight_files": [{"name": "w", "bytes": 10}]},
                       draft_provenance={"path": "draft", "weight_files": [{"name": "w", "bytes": 10}]},
                       diagnostic_instrumented=False, spec_k=7, graph_delta={"verify_hits": 1, "graph_hits": 1}, prefix_lookup_count=0,
                       target_decode_graph={"exact": 0, "padded": 0, "fallback": 0},
                       prefix_hit_blocks=0, input_metadata=[{**source_meta, "actual_input_sha256": h,
                                                             "wiki_fraction": 249 / 252}], kv_block_size=256)
            rows.extend([("x", repeat * 2 + 1, {"type": "warmup", "guard_id": ident,
                         "prewarm_s": 1, "prewarm_shapes": [[1, 256]],
                         "expected_prewarm_shapes": [[1, 256]], "prewarm_head_batches": [1],
                         "target_decode_graph_buckets": [1], "target_verify_graph_shapes": [[1, 7]],
                         "ar_draft_graph_buckets": [], "warmup_graph_delta": {}, "memory_after_prewarm": {}}),
                         ("x", repeat * 2 + 2, row)])
            ledger.extend([{"event": "start", "id": ident, "gpu_uuid": "GPU-OK"},
                           {"event": "child", "id": ident, "gpu_uuid": "GPU-OK"},
                           {"event": "end", "id": ident, "gpu_uuid": "GPU-OK", "kind": "formal",
                            "elapsed_s": 1, "outcome": "ok", "exit_code": 0, "ever_owned_compute_observed": True}])
        report = audit.audit(rows, 3, "formal", {"cells": [cell]}, ledger, manifest)
        self.assertTrue(report["pass"], report)
        without_plan = audit.audit(rows, 3, "formal", None, ledger, manifest)
        self.assertFalse(without_plan["pass"])
        claimed_full = audit.audit(rows, 3, "formal", {"cells": [cell], "window_tokens": 256,
                                    "matrix_scope": "full minimum fixed-batch matrix"}, ledger, manifest)
        self.assertFalse(claimed_full["pass"])
        self.assertTrue(claimed_full["plan_errors"])
        unfinished = audit.audit(rows, 3, "formal", {"cells": [cell]}, ledger[:-1], manifest)
        self.assertFalse(unfinished["pass"])
        self.assertTrue(unfinished["ledger_errors"])
        rows[-1][2]["target_provenance"] = {"path": "different", "weight_files": [{"name": "w", "bytes": 10}]}
        changed_target = audit.audit(rows, 3, "formal", {"cells": [cell]}, ledger, manifest)
        self.assertFalse(changed_target["pass"])
        self.assertTrue(changed_target["inconsistent_cell_configs"])
        rows[-1][2]["target_provenance"] = {"path": "target", "weight_files": [{"name": "w", "bytes": 10}]}
        rows[-1][2]["engine_commit"] = "other"
        changed = audit.audit(rows, 3, "formal", {"cells": [cell]}, ledger, manifest)
        self.assertFalse(changed["pass"])
        self.assertTrue(changed["inconsistent_cell_configs"])

    def test_prewarm_includes_context_crossing_and_tail_batch(self):
        shapes = smoke.prewarm_shapes(3, 4, 252, 32, 0, 256, (1, 2, 4, 8, 16))
        self.assertEqual(shapes, [(1, 256), (1, 512), (2, 256), (2, 512), (4, 256), (4, 512)])
        ring_shapes = smoke.prewarm_shapes(3, 4, 252, 32, 256, 256, (1, 2, 4, 8, 16))
        self.assertEqual(ring_shapes, [(1, 256), (2, 256), (4, 256)])
        with self.assertRaises(ValueError):
            smoke.prewarm_shapes(3, 3, 252, 32, 0, 256, (1, 2, 4))

    def test_old_manifest_synthetic_mode_is_auditable(self):
        source_file = HERE.parent / "20261101-dspark-integration/evidence/workload-manifest.json"
        source = json.loads(source_file.read_text(encoding="utf-8"))
        data = workload.build(source, 4096)
        self.assertEqual(data["context_tokens"], 4096)
        for panel in ("code", "math", "general"):
            for prior, row in zip(source["panels"][panel], data["panels"][panel]):
                self.assertEqual(row["token_ids"][-1024:], prior["token_ids"])
                self.assertEqual(workload.token_hash(row["token_ids"]), row["metadata"]["input_sha256"])
                self.assertEqual(row["metadata"]["added_wiki_tokens"], 3072)
                self.assertTrue(row["metadata"]["synthetic_context"])

    def test_local_wiki_build_uses_disjoint_token_slices(self):
        class FakeTokenizer:
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3]
            def __len__(self):
                return 256
        stub = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeTokenizer()))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"transformers": stub}):
            path = Path(tmp)
            wiki = path / "wiki.txt"
            wiki.write_text("abcdefghijklmnopqrstuvwxyz" * 10, encoding="utf-8")
            for name in ("humaneval.jsonl", "gsm8k.jsonl", "alpaca.jsonl"):
                (path / name).write_text('{"turns": ["one"]}\n{"turns": ["two"]}\n', encoding="utf-8")
            data = workload.build_from_local(str(path), str(path), str(wiki), 8, 2)
            rows = data["panels"]["code"]
            self.assertEqual(rows[0]["metadata"]["wiki_slice_start"], 0)
            self.assertEqual(rows[1]["metadata"]["wiki_slice_start"], 8)
            self.assertLessEqual(rows[0]["metadata"]["wiki_slice_end"], rows[1]["metadata"]["wiki_slice_start"])
            self.assertEqual(rows[0]["token_ids"][-3:], [1, 2, 3])
            self.assertEqual(rows[0]["metadata"]["wiki_fraction"], 5 / 8)


if __name__ == "__main__":
    unittest.main()
