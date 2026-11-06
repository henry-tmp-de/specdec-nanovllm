"""CPU helpers shared by guarded layer-prefill experiments."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "experiments" / "20261105-dspark-window-resume"
sys.path.insert(0, str(HERE))
from gpu_identity import normalize_gpu_uuid

MODES = ("full", "serial", "pipeline")
COUNTERS = ("prefill_steps", "fallback_steps", "planned_tasks", "layer_ops_enqueued", "same_chunk_waits",
            "previous_chunk_waits", "source_waits", "join_waits", "single_chunk_steps", "completed_steps",
            "completed_tasks")


def guard_environment(env, now=None):
    keys = ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX", "CUDA_VISIBLE_DEVICES")
    if any(not env.get(k) for k in keys):
        raise RuntimeError("must run through this experiment's gpu_control.py")
    try:
        guard_id = str(uuid.UUID(env["DS_GUARD_ID"]))
        expected = normalize_gpu_uuid(env["DS_GPU_UUID"])
        visible = normalize_gpu_uuid(env["CUDA_VISIBLE_DEVICES"])
        deadline = float(env["DS_RUN_DEADLINE_UNIX"])
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid guard UUID, device UUID or deadline") from exc
    if expected != visible or not math.isfinite(deadline) or deadline <= (time.time() if now is None else now):
        raise RuntimeError("guard device mismatch or deadline elapsed")
    return {"guard_id": guard_id, "gpu_uuid": expected, "deadline_unix": deadline,
            "cuda_visible_devices": visible}


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode("ascii")).hexdigest()


def read_inputs(path, panel, lengths):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    source_ctx = payload.get("context_tokens")
    rows = payload.get("panels", {}).get(panel, [])
    if type(source_ctx) is not int or len(rows) < len(lengths) or any(type(n) is not int or not 0 < n <= source_ctx for n in lengths):
        raise ValueError("workload does not cover requested prompt lengths/batch")
    result = []
    for n, row in zip(lengths, rows):
        ids, meta = row.get("token_ids"), row.get("metadata")
        if (not isinstance(ids, list) or len(ids) != source_ctx or any(type(i) is not int or i < 0 for i in ids)
                or not isinstance(meta, dict) or token_hash(ids) != meta.get("input_sha256")):
            raise ValueError("workload prompt IDs or hash invalid")
        selected = ids[-n:]
        result.append((selected, {**meta, "source_context_tokens": source_ctx, "actual_context_tokens": n,
                                  "actual_input_sha256": token_hash(selected),
                                  "trimmed_from_left": source_ctx - n,
                                  "fill_metadata_note": ("source prompt metadata" if n == source_ctx else
                                                         "source prompt metadata; fill proportions after trimming are not inferred")}))
    return result


def options_for(mode, batch, max_prompt, output_tokens, chunk, budget, *, nvtx=False):
    if (mode not in MODES or type(batch) is not int or batch < 1 or type(max_prompt) is not int or max_prompt < 1
            or type(output_tokens) is not int or output_tokens < 1 or chunk not in (128, 512)
            or budget not in (0.70, 0.85) or type(nvtx) is not bool or (nvtx and mode == "full")):
        raise ValueError("invalid layer-prefill benchmark options")
    return {"max_num_seqs": batch, "max_model_len": max_prompt + output_tokens + 8,
            "max_num_batched_tokens": batch * max_prompt, "gpu_memory_utilization": budget,
            "tensor_parallel_size": 1, "enforce_eager": False, "decode_graph_bs": tuple(range(1, batch + 1)),
            "spec_k": 0, "spec_method": "ngram", "draft_model": "", "spec_cuda_graph": False,
            "spec_overlap_mixed": False, "spec_prefill_chunk_tokens": 0, "enable_prefix_cache": False,
            "enable_layer_prefill": mode != "full", "layer_prefill_mode": "serial" if mode == "serial" else "pipeline",
            "layer_prefill_chunk_tokens": chunk, "layer_prefill_max_inflight": 2, "layer_prefill_nvtx": nvtx}


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "nanovllm",
                "experiments/20261105-dspark-window-resume", "experiments/20261106-layer-prefill")
    return {"engine_commit": git("rev-parse", "HEAD"), "engine_dirty": bool(dirty)}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target_provenance(path):
    path = Path(path).resolve()
    return {"path": str(path), "config_sha256": file_sha256(path / "config.json"),
            "weight_files": [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(path.glob("*.safetensors"))],
            "weight_content_hash_note": "file sizes only; full contents not rehashed"}


def snapshot_layer(runner):
    stats = getattr(runner, "layer_prefill_stats", None)
    return copy.deepcopy(stats) if isinstance(stats, dict) else None


def layer_delta(before, after):
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("layer-prefill counter snapshot missing")
    delta = {}
    for key in COUNTERS:
        if type(before.get(key)) is not int or type(after.get(key)) is not int or after[key] < before[key]:
            raise ValueError(f"missing or decreasing counter: {key}")
        delta[key] = after[key] - before[key]
    for key in ("fallback_reasons", "stream_task_counts"):
        a, b = before.get(key), after.get(key)
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ValueError(f"counter dictionary missing: {key}")
        values = {}
        for entry in set(a) | set(b):
            old, new = a.get(entry, 0), b.get(entry, 0)
            if type(old) is not int or type(new) is not int or new < old:
                raise ValueError(f"counter dictionary decreased: {key}/{entry}")
            if new != old:
                values[str(entry)] = new - old
        delta[key] = values
    return delta


def validate_layer_execution(mode, before, after, prompt_lengths, layers, chunk):
    """Validate counters after sampler-to-host completion, not just launches."""
    if mode == "full":
        if before is not None or after is not None:
            raise ValueError("disabled path unexpectedly created layer-prefill state")
        return None
    if mode not in ("serial", "pipeline") or type(layers) is not int or layers <= 0:
        raise ValueError("invalid mode or layer count")
    delta = layer_delta(before, after)
    tasks = sum(math.ceil(length / chunk) for length in prompt_lengths)
    streams = 1 if mode == "serial" else 2
    used = min(streams, tasks)
    expected_reason = "single_chunk" if tasks == 1 else "serial_mode" if mode == "serial" else None
    config_ok = (after.get("mode") == mode and after.get("chunk_tokens") == chunk
                 and after.get("max_inflight") == 2 and after.get("nvtx_enabled") is False
                 and after.get("streams_created") == streams and after.get("last_streams_used") == used
                 and after.get("last_single_stream_reason") == expected_reason
                 and isinstance(after.get("stream_ids"), list) and len(after["stream_ids"]) == streams
                 and len(set(after["stream_ids"])) == streams)
    checks = (delta["prefill_steps"] == 1 and delta["fallback_steps"] == 0 and
              delta["planned_tasks"] == tasks and delta["completed_steps"] == 1 and
              delta["completed_tasks"] == tasks and delta["layer_ops_enqueued"] == tasks * layers and
              delta["same_chunk_waits"] == tasks * (layers - 1) and
              delta["previous_chunk_waits"] == (tasks - len(prompt_lengths)) * layers and
              delta["source_waits"] == streams and delta["join_waits"] == tasks and
              delta["single_chunk_steps"] == (1 if tasks == 1 else 0) and
              sum(delta["stream_task_counts"].values()) == tasks and
              all(delta["stream_task_counts"].get(str(i), 0) > 0 for i in range(used)))
    if not config_ok or not checks:
        raise ValueError("actual layer-prefill counters do not match the declared execution")
    return delta
