"""CPU-only synthetic fixtures for report accounting and evidence gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

report = load("report_results")
base = load("tests_cpu_tools")
audit = report.audit
plan_builder = load("run_plan")


def fixture():
    ids = [42] * 252
    digest = audit.token_hash(ids)
    meta = {"input_sha256": digest, "actual_input_sha256": digest, "wiki_tokens": 249,
            "wiki_fraction": 249 / 252, "task_tokens": 3, "task_truncated_from_left": False,
            "dataset": "fixture", "row": 0}
    manifest = {252: {"context_tokens": 252, "panels": {"code": [{"token_ids": ids, "metadata": meta}]}}}
    records, ledger, jobs = [], [], []
    target = {"path": "/fixture/target", "config_sha256": "target", "weight_files": [{"name": "w", "bytes": 10}]}
    draft = {"path": "/fixture/draft", "config_sha256": "draft", "weight_files": [{"name": "w", "bytes": 8}]}
    for arm, mode, window in (("dspark_full", "full", 0), ("dspark_ring", "ring", 256)):
        jobs.append({"phase": "formal", "arm": arm, "mode": mode, "ctx": 252, "batch": 1,
                     "window": window, "gpu_memory_utilization": 0.70, "panels": ["code"], "repeats": 3,
                     "output_tokens": 4, "max_num_batched_tokens": 264})
        for repeat in range(3):
            ident = f"{arm}-{repeat}"
            row = base.sample_run()
            row.update(phase="formal", arm=arm, mode=mode, window=window, ctx=252, panel="code", repeat=repeat,
                       seed=3, temperature=1.0, gpu_uuid="GPU-fixture", engine_commit="commit", engine_dirty=False,
                       gpu_memory_utilization=0.70, max_num_seqs=1, max_model_len=264, max_num_batched_tokens=264,
                       prefix_lookup=False, graph_mode="graph", graph_policy={"name": "exact_all",
                       "target_decode_padding": False, "target_verify_padding": False,
                       "draft_padding": False, "buckets": [1]}, mixed_overlap=False,
                       target_config_sha256="target", draft_config_sha256="draft", target_provenance=target,
                       draft_provenance=draft, guard_id=ident, diagnostic_instrumented=False, spec_k=7,
                       graph_delta={"verify_hits": 1, "verify_exact_hits": 1, "verify_padded_hits": 0,
                                    "verify_fallbacks": {}, "decode_graph_hits": 0, "decode_graph_exact_hits": 0,
                                    "decode_graph_padded_hits": 0, "decode_graph_bucket_histogram": {}, "decode_graph_fallbacks": {},
                                    "graph_hits": 1, "graph_exact_hits": 1, "graph_padded_hits": 0, "graph_captures": 0,
                                    "graph_runtime_captures": 0, "graph_runtime_capture_s": 0.0, "graph_runtime_shapes": [],
                                    "graph_fallbacks": {}, "active_batch_histogram": {}, "context_bucket_histogram": {},
                                    "head_graph_hits": 7, "head_graph_captures": 0, "head_runtime_captures": 0,
                                    "head_runtime_capture_s": 0.0, "head_runtime_batches": [],
                                    "input_prepare_wall_s": 0.2, "input_prepare_calls": 2},
                       target_decode_graph={"exact": 0, "padded": 0, "fallbacks": {}}, prefix_lookup_count=0,
                       prefix_hit_blocks=0, input_metadata=[meta], kv_block_size=256, target_kv_blocks=12,
                       draft_kv_blocks=1 if arm == "dspark_ring" else 0,
                       window_stats_before={"prefill_target_rows": 10, "prefill_feature_rows": 3,
                                            "max_feature_tensor_bytes": 100, "max_projection_tensor_bytes": 20,
                                            "commit_calls": 1, "commit_rows": 2, "commit_host_enqueue_s": 0.1},
                       window_stats_after={"prefill_target_rows": 14, "prefill_feature_rows": 6,
                                           "max_feature_tensor_bytes": 110, "max_projection_tensor_bytes": 22,
                                           "commit_calls": 3, "commit_rows": 4, "commit_host_enqueue_s": 0.4},
                       memory_after_run={"allocated_bytes": 1000, "reserved_bytes": 2000,
                                         "peak_allocated_bytes": 1234, "peak_reserved_bytes": 2345,
                                         "named_tensors": {"runner.kv_cache": {"bytes": 100, "shape": [1], "dtype": "bf16"},
                                                           "dspark.cache": {"bytes": 30, "shape": [1], "dtype": "bf16"},
                                                           "dspark.scratch": {"bytes": 10, "shape": [1], "dtype": "bf16"}},
                                         "unique_parameter_bytes": {"model": 90, "draft_model": 0, "dspark_model": 20},
                                         "allocator_segment_bytes_by_pool_id": {"fixture": 2000},
                                         "named_graph_pools": {"target_decode": {"reserved_bytes": 77},
                                                               "target_verify": {"reserved_bytes": 78},
                                                               "dspark_backbone": {"reserved_bytes": 79},
                                                               "dspark_head": {"reserved_bytes": 80}},
                                         "engine_graph_mem_gb": {"decode_graphs_gb": 0.01, "verify_graphs_gb": 0.02}})
            row["wall_s"] = 5.0 + repeat + (1.0 if arm == "dspark_full" else 0.0)
            if arm == "dspark_ring":
                row["requests"][0]["output_token_ids"][-1] = 11
                row["requests"][0]["output_sha256"] = audit.token_hash(row["requests"][0]["output_token_ids"])
            warm = {"type": "warmup", **{k: row[k] for k in report.WARM_IDENTITY_FIELDS}, "prewarm_s": 1.0,
                    "prewarm_shapes": [[1, 256]], "expected_prewarm_shapes": [[1, 256]],
                    "prewarm_head_batches": [1], "target_decode_graph_buckets": [1],
                    "target_verify_graph_shapes": [[1, 7]], "ar_draft_graph_buckets": [],
                    "warmup_graph_delta": {}, "memory_after_init": deepcopy(row["memory_after_run"]),
                    "memory_after_prewarm": deepcopy(row["memory_after_run"]),
                    "memory_after_warmup_requests": deepcopy(row["memory_after_run"])}
            records.extend([(f"{ident}.jsonl", 1, warm), (f"{ident}.jsonl", 2, row)])
            for event in ("start", "child", "end"):
                ev = {"event": event, "id": ident, "gpu_uuid": "GPU-fixture", "kind": "formal", "stage": ident}
                if event == "child":
                    ev.update(pid=123, start_ticks=456)
                if event == "end":
                    ev.update(elapsed_s=1.0, exit_code=0, outcome="ok", ever_owned_compute_observed=True,
                              root_identity=[123, 456])
                ledger.append(("fixture-ledger.jsonl", len(ledger) + 1, ev))
    plan = {"phase": "formal", "matrix_scope": "explicit subset", "jobs": jobs}
    return records, ledger, manifest, [("fixture-plan.json", plan)]


class ReportChecks(unittest.TestCase):
    def test_decode_span_includes_inter_round_gap_and_prefix_is_contiguous(self):
        sample = base.sample_run()
        spans = report.decode_spans(sample["rounds"], 1)
        self.assertAlmostEqual(spans["decode_span_s"], 2.9)
        self.assertAlmostEqual(spans["decode_step_sum_s"], 2.8)
        self.assertAlmostEqual(spans["decode_span_tps"], 3 / 2.9)
        self.assertAlmostEqual(spans["full_batch_prefix_span_s"], 2.9)
        rounds = [dict(start_s=1.0, end_s=2.0, prefill=False, active_b=2, delivered=4),
                  dict(start_s=3.0, end_s=4.0, prefill=False, active_b=1, delivered=1),
                  dict(start_s=5.0, end_s=6.0, prefill=False, active_b=2, delivered=2)]
        out = report.decode_spans(rounds, 2)
        self.assertEqual(out["full_batch_prefix_span_s"], 1.0)
        self.assertEqual(out["decode_span_s"], 5.0)

    def test_matched_three_repeats_and_output_hash_differences(self):
        records, ledger, manifests, plans = fixture()
        result = report.build_report(records, ledger, manifests, plans)
        self.assertEqual([x["status"] for x in result["cells"]], ["complete", "complete"])
        comp = next(x for x in result["comparisons"] if x["baseline_arm"] == "dspark_full")
        self.assertEqual(comp["status"], "matched")
        self.assertEqual(comp["full_request_ratio"]["n_available"], 3)
        self.assertEqual(len(comp["full_request_ratio"]["values"]), 3)
        self.assertEqual([x["different"] for x in comp["output_hash_comparison"]], [1, 1, 1])
        self.assertNotIn("confidence_interval", comp)
        ring = next(x for x in result["cells"] if x["scope"]["arm"] == "dspark_ring")
        runs = ring["variants"][0]["runs"]
        self.assertEqual(runs[0]["memory"]["after_run"]["named_tensors"]["dspark.scratch"]["bytes"], 10)
        self.assertAlmostEqual(runs[0]["host"]["commit_enqueue_s"], 0.3)
        self.assertEqual(runs[0]["feature_rows"]["selected_rows"], 3)
        full = next(x for x in result["cells"] if x["scope"]["arm"] == "dspark_full")
        self.assertIsNone(full["variants"][0]["runs"][0]["feature_rows"]["selected_rows"])
        self.assertIsNone(full["variants"][0]["runs"][0]["host"]["commit_enqueue_s"])
        self.assertFalse(result["summary"]["primary_summary_is_complete"])
        json.dumps(result, allow_nan=False)
        text = report.markdown(result)
        self.assertIn("decode span", text)
        self.assertIn("不插值", text)

    def test_single_repeat_pilot_keeps_observation_without_confidence_claim(self):
        records, ledger, manifests, plans = fixture()
        selected = []
        ids = set()
        for source, line, row in records:
            if row.get("type") == "run" and row.get("repeat") == 0:
                row["phase"] = "pilot"
                ids.add(row["guard_id"])
                selected.append((source, line, row))
        for source, line, row in records:
            if row.get("type") == "warmup" and row.get("guard_id") in ids:
                row["phase"] = "pilot"
                selected.append((source, line, row))
        selected_ledger = []
        for source, line, item in ledger:
            if item["id"] in ids:
                item["kind"] = "pilot"
                selected_ledger.append((source, line, item))
        plan = deepcopy(plans[0][1])
        plan["phase"] = "pilot"
        for job in plan["jobs"]:
            job.update(phase="pilot", repeats=1)
        out = report.build_report(selected, selected_ledger, manifests, [("single-pilot", plan)])
        ring = next(c for c in out["cells"] if c["scope"]["arm"] == "dspark_ring")
        self.assertEqual(ring["status"], "complete")
        self.assertEqual(len(ring["variants"][0]["metrics"]["full_request_tps"]["values"]), 1)
        pair = next(c for c in out["comparisons"] if c["candidate_arm"] == "dspark_ring"
                    and c["baseline_arm"] == "dspark_full")
        self.assertEqual(pair["status"], "not_comparable")
        self.assertTrue(any("fewer than 3" in reason for reason in pair["reasons"]))
        self.assertIsNone(pair["full_request_ratio"])

    def test_unmatched_and_failed_attempts_never_become_comparable(self):
        records, ledger, manifests, plans = fixture()
        # An otherwise identical repeat using a different source must split its group.
        target = next(r for _, _, r in records if r.get("type") == "run" and r["arm"] == "dspark_full" and r["repeat"] == 2)
        target["engine_commit"] = "different"
        failed = {"type": "failure", "phase": "formal", "arm": "dspark_ring", "mode": "ring", "ctx": 252,
                  "requested_batch": 1, "window": 256, "panel": "code", "gpu_memory_utilization": 0.70,
                  "graph_mode": "graph", "error": "OOM", "message": "fixture failure", "failure_phase": "prefill"}
        records.append(("failed.jsonl", 1, failed))
        result = report.build_report(records, ledger, manifests, plans)
        statuses = {c["scope"]["arm"]: c["status"] for c in result["cells"]}
        self.assertEqual(statuses["dspark_full"], "inconsistent_conditions")
        self.assertEqual(statuses["dspark_ring"], "failed_attempt_present")
        self.assertTrue(all(c["status"] == "not_comparable" for c in result["comparisons"]))
        self.assertEqual(result["failures"][0]["record"]["error"], "OOM")

    def test_missing_draft_or_reused_guard_identity_invalidates_run(self):
        records, ledger, manifests, plans = fixture()
        row = next(r for _, _, r in records if r.get("type") == "run" and r["arm"] == "dspark_ring" and r["repeat"] == 1)
        row["draft_provenance"] = None
        end = next(r for _, _, r in ledger if r.get("event") == "end" and r["id"] == "dspark_full-0")
        end["root_identity"] = [123, 999]
        out = report.build_report(records, ledger, manifests, plans)
        cells = {c["scope"]["arm"]: c for c in out["cells"]}
        self.assertNotEqual(cells["dspark_ring"]["status"], "complete")
        self.assertEqual(cells["dspark_full"]["status"], "unverified_or_invalid")
        comp = next(c for c in out["comparisons"] if c["baseline_arm"] == "dspark_full")
        self.assertEqual(comp["status"], "not_comparable")

    def test_global_budget_or_unfinished_guard_blocks_comparisons(self):
        records, ledger, manifests, plans = fixture()
        for _, _, entry in ledger:
            if entry["event"] == "end":
                entry["elapsed_s"] = 3000.0
        out = report.build_report(records, ledger, manifests, plans)
        self.assertEqual(out["ledger"]["completed_stage_elapsed_s"], 18000.0)
        self.assertTrue(any("14400" in item["error"] for item in out["ledger_errors"]))
        self.assertTrue(all(x["status"] == "not_comparable" for x in out["comparisons"]))
        self.assertTrue(all(not curve["segments"] for curve in out["curves"]))
        records, ledger, manifests, plans = fixture()
        for event in ("start", "child", "end"):
            item = {"event": event, "id": "smoke-over", "gpu_uuid": "GPU-fixture",
                    "kind": "smoke", "stage": "smoke-over"}
            if event == "child":
                item.update(pid=333, start_ticks=444)
            if event == "end":
                item.update(elapsed_s=1801.0, exit_code=1, outcome="child_failed_or_lingering",
                            ever_owned_compute_observed=True, root_identity=[333, 444])
            ledger.append(("fixture-ledger", len(ledger) + 1, item))
        out = report.build_report(records, ledger, manifests, plans)
        self.assertTrue(any("1800" in item["error"] for item in out["ledger_errors"]))
        self.assertTrue(all(x["status"] == "not_comparable" for x in out["comparisons"]))
        # A still-running unrelated stage is also disclosed and blocks final comparisons.
        records, ledger, manifests, plans = fixture()
        ledger.append(("fixture-ledger", len(ledger) + 1, {"event": "start", "id": "running",
                      "gpu_uuid": "GPU-fixture", "kind": "smoke", "stage": "unfinished"}))
        out = report.build_report(records, ledger, manifests, plans)
        self.assertIn("running", out["ledger"]["unfinished_guard_ids"])
        self.assertFalse(out["ledger"]["total_is_complete"])
        self.assertTrue(all(x["status"] == "not_comparable" for x in out["comparisons"]))

    def test_old_070_and_new_085_plans_and_sources_never_mix(self):
        records, ledger, manifests, plans = fixture()
        new_records = deepcopy(records)
        new_ledger = deepcopy(ledger)
        for _, _, row in new_records:
            row["guard_id"] += "-085"
            row["gpu_memory_utilization"] = 0.85
            row["engine_commit"] = "commit-085"
        for _, _, event in new_ledger:
            event["id"] += "-085"
            event["stage"] += "-085"
        new_plan = deepcopy(plans[0][1])
        new_plan["gpu_memory_utilization"] = 0.85
        for job in new_plan["jobs"]:
            job["gpu_memory_utilization"] = 0.85
        result = report.build_report(records + new_records, ledger + new_ledger, manifests,
                                     plans + [("fixture-plan-085", new_plan)])
        self.assertFalse(result["plan_issues"], result["plan_issues"])
        self.assertEqual(len(result["cells"]), 4)
        self.assertTrue(all(c["status"] == "complete" for c in result["cells"]))
        self.assertEqual({p["gpu_memory_utilization"] for p in result["plans"]}, {0.70, 0.85})
        self.assertEqual([p["coverage"]["complete"] for p in result["plans"]], [2, 2])
        pairs = [c for c in result["comparisons"] if c["candidate_arm"] == "dspark_ring"
                 and c["baseline_arm"] == "dspark_full"]
        self.assertEqual({c["scope"]["gpu_memory_utilization"] for c in pairs}, {0.70, 0.85})
        self.assertTrue(all(c["status"] == "matched" for c in pairs))
        for c in result["cells"]:
            commits = {v["configuration"]["engine_commit"] for v in c["variants"]}
            self.assertEqual(commits, {"commit-085" if c["scope"]["gpu_memory_utilization"] == 0.85 else "commit"})
        self.assertIn("| phase | 预算 |", report.markdown(result))
        partial_rows = [item for item in records + new_records
                        if not (item[2].get("type") == "run" and item[2].get("guard_id") == "dspark_full-2")]
        partial = report.build_report(partial_rows, ledger + new_ledger, manifests,
                                      plans + [("fixture-plan-085", new_plan)])
        coverage = {p["gpu_memory_utilization"]: p["coverage"] for p in partial["plans"]}
        self.assertEqual(coverage[0.70]["partial_or_failed"], 1)
        self.assertEqual(coverage[0.85]["complete"], 2)
        new_pair = next(c for c in partial["comparisons"] if c["scope"]["gpu_memory_utilization"] == 0.85
                        and c["candidate_arm"] == "dspark_ring" and c["baseline_arm"] == "dspark_full")
        self.assertEqual(new_pair["status"], "matched")

    def test_full_plan_budget_consistency_and_legacy_070(self):
        for budget in (0.70, 0.85):
            plan = plan_builder.build("formal", 1024, ("code", "math", "general"), 3, budget)
            cells, errors = report.make_planned([("plan", plan)])
            self.assertFalse(errors, errors)
            self.assertEqual(len(cells), 150)
            self.assertEqual({x["scope"]["gpu_memory_utilization"] for x in cells.values()}, {budget})
        old = plan_builder.build("formal", 1024, ("code", "math", "general"), 3, 0.70)
        old.pop("gpu_memory_utilization")
        for cell in old["cells"]:
            cell.pop("gpu_memory_utilization")
        cells, errors = report.make_planned([("old-plan", old)])
        self.assertFalse(errors, errors)
        self.assertEqual({x["scope"]["gpu_memory_utilization"] for x in cells.values()}, {0.70})
        broken = plan_builder.build("formal", 1024, ("code", "math", "general"), 3, 0.85)
        broken["cells"][0]["gpu_memory_utilization"] = 0.70
        _, errors = report.make_planned([("broken", broken)])
        self.assertTrue(any("budget differs" in x["error"] for x in errors))

    def test_warmup_identity_mismatch_blocks_the_run(self):
        records, ledger, manifests, plans = fixture()
        warm = next(r for _, _, r in records if r.get("type") == "warmup" and r["guard_id"] == "dspark_ring-0")
        warm["gpu_memory_utilization"] = 0.85
        out = report.build_report(records, ledger, manifests, plans)
        ring = next(c for c in out["cells"] if c["scope"]["arm"] == "dspark_ring")
        self.assertEqual(ring["status"], "unverified_or_invalid")
        errors = ring["variants"][0]["runs"][0]["validation_errors"]
        self.assertTrue(any("warmup/run identity mismatch" in e for e in errors))

    def test_invalid_4k_fill_never_contaminates_verified_range(self):
        records, ledger, manifests, plans = fixture()
        result = report.build_report(records, ledger, manifests, plans)
        for cell in result["cells"]:
            cell["scope"]["ctx"] = 4096
            for group in cell["variants"]:
                for run in group["runs"]:
                    run["input_metadata"] = deepcopy(run["input_metadata"])
                    run["input_metadata"][0]["wiki_fraction"] = 0.96
        bad_cell = result["cells"][0]
        bad_cell["status"] = "unverified_or_invalid"
        bad = bad_cell["variants"][0]["runs"][0]
        bad["valid"] = False
        bad["input_metadata"][0]["wiki_fraction"] = 1.2
        text = report.markdown(result)
        self.assertIn("已核验的4K请求 Wiki 补齐比例为 96.00%–96.00%", text)
        self.assertNotIn("120.00%", text)
        self.assertIn('"wiki_fractions":[1.2]', text)
        self.assertIn("unverified_or_invalid", text)

    def test_missing_runtime_counter_or_scratch_is_not_treated_as_zero(self):
        records, ledger, manifests, plans = fixture()
        row = next(r for _, _, r in records if r.get("type") == "run" and r["arm"] == "dspark_ring" and r["repeat"] == 0)
        del row["graph_delta"]["graph_runtime_captures"]
        del row["memory_after_run"]["named_tensors"]["dspark.scratch"]
        out = report.build_report(records, ledger, manifests, plans)
        ring = next(c for c in out["cells"] if c["scope"]["arm"] == "dspark_ring")
        self.assertEqual(ring["status"], "unverified_or_invalid")
        errors = ring["variants"][0]["runs"][0]["validation_errors"]
        self.assertTrue(any("graph telemetry fields missing" in e for e in errors))
        self.assertTrue(any("scratch allocation missing" in e for e in errors))
        comp = next(c for c in out["comparisons"] if c["baseline_arm"] == "dspark_full")
        self.assertEqual(comp["status"], "not_comparable")

    def test_missing_curve_points_break_segments_and_never_become_zero(self):
        cells = []
        for batch in (1, 4):
            config = {"phase": "formal", "engine_commit": "x", "engine_dirty": False,
                      "gpu_uuid": "gpu", "target_provenance": {"path": "target"}, "seed": 3,
                      "temperature": 1.0, "output_tokens": 256, "gpu_memory_utilization": 0.70,
                      "prefix_lookup": False, "mixed_overlap": False, "ignore_eos": True,
                      "graph_policy": {"name": "exact_all", "buckets": list(range(1, batch + 1))},
                      "input_hashes": [str(x) for x in range(batch)]}
            scope = {"phase": "formal", "arm": "dspark_ring", "mode": "ring", "ctx": 4096,
                     "requested_batch": batch, "panel": "code", "window": 1024,
                     "graph_mode": "graph", "gpu_memory_utilization": 0.70}
            cells.append({"scope": scope, "status": "complete", "variants": [{"configuration": config,
                          "metrics": {"full_request_tps": {"n_available": 3, "values": [1.0, 2.0, 3.0], "median": 2.0}}}]})
        curve = report.build_curves(cells)[0]
        self.assertEqual(curve["segments"], [[1], [4]])
        middle = next(x for x in curve["points"] if x["batch"] == 2)
        self.assertEqual(middle["status"], "not_planned")
        self.assertIsNone(middle["median_full_request_tps"])

    def test_curve_does_not_join_different_draft_models(self):
        cells = []
        for batch, draft in ((1, "draft-a"), (2, "draft-b")):
            cfg = {"phase": "formal", "engine_commit": "x", "engine_dirty": False,
                   "gpu_uuid": "gpu", "target_provenance": {"path": "target"}, "target_config_sha256": "t",
                   "draft_provenance": {"path": draft}, "draft_config_sha256": draft, "spec_k": 7,
                   "seed": 3, "temperature": 1.0, "output_tokens": 256, "gpu_memory_utilization": 0.70,
                   "prefix_lookup": False, "mixed_overlap": False, "ignore_eos": True,
                   "graph_policy": {"name": "exact_all", "buckets": list(range(1, batch + 1))},
                   "input_hashes": [str(x) for x in range(batch)]}
            scope = {"phase": "formal", "arm": "dspark_ring", "mode": "ring", "ctx": 4096,
                     "requested_batch": batch, "panel": "code", "window": 1024,
                     "graph_mode": "graph", "gpu_memory_utilization": 0.70}
            cells.append({"scope": scope, "status": "complete", "variants": [{"configuration": cfg,
                          "metrics": {"full_request_tps": {"n_available": 3, "values": [1.0, 2.0, 3.0], "median": 2.0}}}]})
        self.assertEqual(report.build_curves(cells)[0]["segments"], [[1], [2]])

    def test_capacity_failure_and_old_diagnostics_are_preserved(self):
        plan = {"phase": "capacity", "jobs": [{"phase": "capacity", "arm": arm, "mode": mode,
                "ctx": 16384, "batch": 4, "window": window, "gpu_memory_utilization": 0.85,
                "max_num_batched_tokens": 4096, "panels": ["code"], "repeats": 1}
                for arm, mode, window in (("dspark_full", "full", 0), ("dspark_ring", "ring", 1024))]}
        failure = {"type": "failure", "phase": "capacity", "arm": "dspark_full", "mode": "full",
                   "ctx": 16384, "requested_batch": 4, "window": 0, "graph_mode": "graph",
                   "gpu_memory_utilization": 0.85, "panel": "code", "error": "OutOfMemoryError",
                   "failure_phase": "measured_requests", "request_subphase": "prefill", "message": "fixture OOM"}
        records = [("capacity.jsonl", 1, failure), ("padding.jsonl", 1, {"type": "target_probe", "batch": 11,
                   "eager_padded_kl": 0.0201313533}), ("uuid.jsonl", 1, {"type": "failure",
                   "error": "RuntimeError", "message": "UUID preflight fixture"})]
        out = report.build_report(records, [], {}, [("capacity-plan", plan)])
        statuses = {c["scope"]["arm"]: c["status"] for c in out["cells"]}
        self.assertEqual(statuses, {"dspark_full": "failed_attempt_present", "dspark_ring": "missing"})
        self.assertEqual(len(out["failures"]), 2)
        self.assertEqual(out["diagnostic_events"][0]["record"]["eager_padded_kl"], 0.0201313533)
        self.assertEqual(out["cells"][0]["failures"][0]["request_subphase"], "prefill")

    def test_capacity_success_is_paired_separately_with_actual_pool_sizes(self):
        records, ledger, manifests, plans = fixture()
        selected_records = []
        selected_ids = set()
        for source, line, row in records:
            if row.get("type") == "run" and row.get("repeat") == 0:
                row["phase"] = "capacity"
                row["gpu_memory_utilization"] = 0.85
                row["target_kv_blocks"] = 20 if row["arm"] == "dspark_ring" else 12
                selected_ids.add(row["guard_id"])
                selected_records.append((source, line, row))
        for source, line, row in records:
            if row.get("type") == "warmup" and row.get("guard_id") in selected_ids:
                row["phase"] = "capacity"
                row["gpu_memory_utilization"] = 0.85
                selected_records.append((source, line, row))
        selected_ledger = []
        for source, line, entry in ledger:
            if entry["id"] in selected_ids:
                entry["kind"] = "capacity"
                selected_ledger.append((source, line, entry))
        plan = deepcopy(plans[0][1])
        plan["phase"] = "capacity"
        for job in plan["jobs"]:
            job.update(phase="capacity", gpu_memory_utilization=0.85, repeats=1)
        out = report.build_report(selected_records, selected_ledger, manifests, [("capacity-fixture", plan)])
        self.assertEqual([c["status"] for c in out["cells"]], ["complete", "complete"])
        self.assertEqual(out["comparisons"], [])
        pair = out["capacity_comparisons"][0]
        self.assertEqual(pair["status"], "matched")
        self.assertFalse(pair["observations"][0]["target_pool_blocks_equal"])
        self.assertNotIn("confidence_interval", pair)


if __name__ == "__main__":
    unittest.main()
