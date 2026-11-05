"""Guarded real-checkpoint DSpark W8A16 linear microbenchmark.

Reads only five layer-0 weights from the official safetensors checkpoint.  The
kernel is checked against FP32 x @ int8.T, then multiplied by the BF16 scale;
error against the original BF16 linear is diagnostic, not a correctness gate.
Synthetic BF16 inputs and fixed weights are prepared before graph capture.

Run this entry point only as a child of this experiment's gpu_control.py.
Timing is CUDA-event elapsed time for 100 calls, repeated three times; it is
not end-to-end model latency. Compilation, preparation and capture are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments" / "20261105-dspark-window-resume"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))
from gpu_identity import normalize_gpu_uuid, verify_visible_gpu

WEIGHTS = {
    "gate": "layers.0.mlp.gate_proj.weight",
    "down": "layers.0.mlp.down_proj.weight",
    "q": "layers.0.self_attn.q_proj.weight",
    "k": "layers.0.self_attn.k_proj.weight",
    "o": "layers.0.self_attn.o_proj.weight",
}
BATCHES = (1, 8, 16)
BLOCK = 7
WARMUP = 10
ROUNDS = 3
REPLAYS = 100
# Predeclared before GPU: one gate against the same q and BF16 scale. Eager vs
# graph additionally uses BF16 torch.testing tolerances for each operator.
QUANT_RELATIVE_RMS_LIMIT = 0.01
GRAPH_RTOL = 0.016
GRAPH_ATOL = 1e-5


def guard_environment(env, now=None):
    required = ("DS_GUARD_ID", "DS_GPU_UUID", "DS_RUN_DEADLINE_UNIX", "CUDA_VISIBLE_DEVICES")
    if any(not env.get(k) for k in required):
        raise RuntimeError("必须通过本实验 gpu_control.py 启动")
    try:
        guard_id = str(uuid.UUID(env["DS_GUARD_ID"]))
        expected = normalize_gpu_uuid(env["DS_GPU_UUID"])
        visible = normalize_gpu_uuid(env["CUDA_VISIBLE_DEVICES"])
        deadline = float(env["DS_RUN_DEADLINE_UNIX"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("守护身份、设备 UUID 或截止时间非法") from exc
    if expected != visible or not math.isfinite(deadline) or deadline <= (time.time() if now is None else now):
        raise RuntimeError("守护设备不匹配或截止时间已过")
    return {"guard_id": guard_id, "gpu_uuid": expected, "cuda_visible_devices": visible,
            "deadline_unix": deadline}


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("已达到 GPU guard 截止时间")


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()
    return {"engine_commit": git("rev-parse", "HEAD"),
            "engine_dirty": bool(git("status", "--porcelain", "--untracked-files=all", "--",
                                     "nanovllm", "experiments/20261105-dspark-quant"))}


def validate_shapes(shapes):
    if set(shapes) != set(WEIGHTS):
        raise ValueError("权重清单不完整")
    for name, shape in shapes.items():
        if len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f"{name} 权重必须是非空二维矩阵")
    inner, hidden = shapes["gate"]
    q_dim, q_in = shapes["q"]
    kv_dim, kv_in = shapes["k"]
    if (shapes["down"] != [hidden, inner] or q_in != hidden or kv_in != hidden
            or shapes["o"] != [hidden, q_dim] or kv_dim > q_dim):
        raise ValueError("层权重维度与 DSpark 结构不一致")


def checkpoint_index(directory, safe_open):
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"目录下没有 safetensors: {directory}")
    wanted = {key: name for name, key in WEIGHTS.items()}
    index, shapes = {}, {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for key in source.keys():
                if key not in wanted:
                    continue
                name = wanted[key]
                if name in index:
                    raise ValueError(f"checkpoint 出现重复权重 {key}")
                index[name] = path
                shapes[name] = list(source.get_slice(key).get_shape())
    validate_shapes(shapes)
    return index, shapes


def tensor_sha256(tensor, torch):
    if tensor.device.type != "cpu":
        raise ValueError("只哈希 CPU 张量")
    data = tensor.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def quantize_cpu(weight, quantize_weight, torch, chunk_rows=256):
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("量化输入必须为 CPU BF16 二维权重")
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.int8, device="cpu")
    scale = torch.empty((n,), dtype=torch.bfloat16, device="cpu")
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        host = weight[start:end]
        if not torch.isfinite(host).all().item():
            raise ValueError(f"权重存在非有限数: rows {start}:{end}")
        q_part, s_part = quantize_weight(host, "per_channel")
        q[start:end].copy_(q_part)
        scale[start:end].copy_(s_part)
    return q, scale


def quantized_reference(x, q, scale, torch):
    # INT8 values are exactly representable as BF16. The existing per-channel
    # kernel performs the dot in FP32 accumulation, then multiplies BF16 scale.
    return (x.float() @ q.float().transpose(0, 1)) * scale.float()


def errors(actual, reference, torch):
    if not (torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item()):
        return {"finite": False, "relative_rms": None, "max_abs": None,
                "max_abs_over_reference_rms": None, "reference_rms": None}
    difference = actual.float() - reference.float()
    ref_rms = reference.float().square().mean().sqrt().item()
    diff_rms = difference.square().mean().sqrt().item()
    max_abs = difference.abs().max().item()
    if not all(math.isfinite(x) for x in (ref_rms, diff_rms, max_abs)):
        return {"finite": False, "relative_rms": None, "max_abs": None,
                "max_abs_over_reference_rms": None, "reference_rms": None}
    relative = diff_rms / ref_rms if ref_rms else (0.0 if diff_rms == 0 else None)
    normalized_max = max_abs / ref_rms if ref_rms else (0.0 if max_abs == 0 else None)
    return {"finite": True, "relative_rms": relative, "max_abs": max_abs,
            "max_abs_over_reference_rms": normalized_max, "reference_rms": ref_rms}


def graph_close(actual, eager, torch):
    stats = errors(actual, eager, torch)
    try:
        torch.testing.assert_close(actual, eager, rtol=GRAPH_RTOL, atol=GRAPH_ATOL)
        close = stats["finite"]
    except AssertionError:
        close = False
    return {**stats, "close": close, "rtol": GRAPH_RTOL, "atol": GRAPH_ATOL}


def time_one(torch, fn, deadline):
    check_deadline(deadline)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / REPLAYS


def measure(torch, operations, deadline):
    samples = {key: [] for key in operations}
    orders = []
    keys = list(operations)
    for round_index in range(ROUNDS):
        order = keys[round_index:] + keys[:round_index]
        orders.append([f"{mode}/{kind}" for mode, kind in order])
        for key in order:
            samples[key].append(time_one(torch, operations[key], deadline))
    return samples, orders


def run_case(torch, name, batch, weight, q_cpu, scale_cpu, seed, deadline,
             DSparkInt8Linear):
    check_deadline(deadline)
    n, k = weight.shape
    m = batch * BLOCK
    generator = torch.Generator(device="cpu").manual_seed(seed + m)
    x_cpu = torch.randn((m, k), generator=generator, dtype=torch.bfloat16, device="cpu")
    input_hash = tensor_sha256(x_cpu, torch)
    x = x_cpu.to("cuda")
    base = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="meta")
    base.weight = torch.nn.Parameter(weight.to("cuda"), requires_grad=False)
    quant = DSparkInt8Linear(q_cpu.to("cuda"), scale_cpu.to("cuda"))
    base.eval()
    quant.eval()

    # All compilation, allocation setup and input preparation precede capture
    # and the timed interval. Capture creates independent graph output buffers.
    for _ in range(WARMUP):
        base(x)
        quant(x)
    torch.cuda.synchronize()
    eager_bf16 = base(x).clone()
    eager_quant = quant(x).clone()
    base_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(base_graph):
        graph_bf16_output = base(x)
    quant_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(quant_graph):
        graph_quant_output = quant(x)
    base_graph.replay()
    quant_graph.replay()
    torch.cuda.synchronize()
    graph_bf16 = graph_bf16_output.clone()
    graph_quant = graph_quant_output.clone()
    # TF32 is disabled in main; FP32 reference conversion is outside timing.
    reference = quantized_reference(x, quant.qweight, quant.scale, torch)
    q_errors = {"eager": errors(eager_quant, reference, torch),
                "graph": errors(graph_quant, reference, torch)}
    bf16_errors = {"eager": errors(eager_quant, eager_bf16, torch),
                   "graph": errors(graph_quant, graph_bf16, torch)}
    graph_errors = {"bf16": graph_close(graph_bf16, eager_bf16, torch),
                    "int8": graph_close(graph_quant, eager_quant, torch)}
    passed = (all(v["finite"] and v["relative_rms"] is not None and
                  v["relative_rms"] < QUANT_RELATIVE_RMS_LIMIT for v in q_errors.values())
              and all(v["finite"] for v in bf16_errors.values())
              and all(v["close"] for v in graph_errors.values()))
    result = {"input_sha256": input_hash, "rows": m, "batch": batch,
              "reference_order": "float32(x) @ float32(q).T, then multiply float32(BF16 scale)",
              "quant_reference": q_errors, "original_bf16_diagnostic": bf16_errors,
              "graph_vs_eager": graph_errors, "correctness_pass": passed}
    if not passed:
        return result, None, None
    ops = {("eager", "bf16"): lambda: base(x), ("eager", "int8"): lambda: quant(x),
           ("graph", "bf16"): base_graph.replay, ("graph", "int8"): quant_graph.replay}
    samples, orders = measure(torch, ops, deadline)
    return result, samples, orders


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft", required=True, help="官方 DSpark checkpoint 目录")
    p.add_argument("--out", required=True, help="新 JSONL 文件，不覆盖")
    p.add_argument("--seed", type=int, default=20261105)
    args = p.parse_args()
    if not 0 <= args.seed < 2**63 - max(BATCHES) * BLOCK:
        p.error("seed 超出范围")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {"guard_id": os.environ.get("DS_GUARD_ID"), "gpu_uuid": os.environ.get("DS_GPU_UUID")}
    current = {}
    with output.open("x", encoding="utf-8") as stream:
        try:
            identity = guard_environment(os.environ)
            import torch
            from safetensors import safe_open
            from nanovllm.layers.quant import _BLOCK, _SPLITK_MAX_M, quantize_weight
            from nanovllm.spec_decode.dspark_quant import DSparkInt8Linear, expected_names

            properties = verify_visible_gpu(torch, identity["gpu_uuid"])
            identity.update(git_identity())
            identity.update({"device_name": properties.name, "torch_version": torch.__version__,
                             "torch_cuda_version": torch.version.cuda, "draft_path": str(Path(args.draft).resolve()),
                             "seed": args.seed, "input_distribution": "synthetic torch.randn BF16 on CPU",
                             "warmup_calls_per_operator": WARMUP, "timing_rounds": ROUNDS,
                             "replays_per_round": REPLAYS, "relative_rms_limit": QUANT_RELATIVE_RMS_LIMIT,
                             "graph_rtol": GRAPH_RTOL, "graph_atol": GRAPH_ATOL})
            if not set(k[:-7] for k in WEIGHTS.values()) <= set(expected_names("all")):
                raise RuntimeError("micro 权重名称与量化 wrapper 清单不一致")
            config_path = Path(args.draft) / "config.json"
            identity["draft_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            check_deadline(identity["deadline_unix"])
            index, shapes = checkpoint_index(args.draft, safe_open)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                for name, key in WEIGHTS.items():
                    current = {"weight_name": key}
                    check_deadline(identity["deadline_unix"])
                    with safe_open(str(index[name]), framework="pt", device="cpu") as source:
                        original = source.get_tensor(key).contiguous()
                    original_dtype = str(original.dtype)
                    original_hash = tensor_sha256(original, torch)
                    weight = original.to(dtype=torch.bfloat16).contiguous()
                    if list(weight.shape) != shapes[name]:
                        raise RuntimeError("checkpoint tensor 形状与索引不一致")
                    bf16_hash = tensor_sha256(weight, torch)
                    q, scale = quantize_cpu(weight, quantize_weight, torch)
                    weight_info = {"weight_name": key, "weight_file": str(index[name].resolve()),
                                   "shape_nk": list(weight.shape), "checkpoint_dtype": original_dtype,
                                   "checkpoint_tensor_sha256": original_hash, "bf16_weight_sha256": bf16_hash,
                                   "int8_weight_sha256": tensor_sha256(q, torch),
                                   "bf16_scale_sha256": tensor_sha256(scale, torch),
                                   "bf16_weight_bytes": weight.numel() * weight.element_size(),
                                   "int8_weight_bytes": q.numel() * q.element_size(),
                                   "scale_bytes": scale.numel() * scale.element_size(),
                                   "scale_dtype": str(scale.dtype)}
                    print(f"真实权重 {name} {list(weight.shape)}，开始 B=1/8/16", flush=True)
                    for batch in BATCHES:
                        current = {"weight_name": key, "batch": batch, "rows": batch * BLOCK}
                        result, samples, orders = run_case(torch, name, batch, weight, q, scale,
                                                            args.seed, identity["deadline_unix"], DSparkInt8Linear)
                        split = int(_BLOCK["per_channel"]["split"]) if result["rows"] <= _SPLITK_MAX_M else 1
                        for mode in ("eager", "graph"):
                            bf16_ms = samples[(mode, "bf16")] if samples is not None else None
                            int8_ms = samples[(mode, "int8")] if samples is not None else None
                            row = {"type": "micro", **identity, **weight_info, "batch": batch,
                                   "rows": result["rows"], "mode": mode, "input_sha256": result["input_sha256"],
                                   "split_k_from_kernel_policy": split, "correctness_pass": result["correctness_pass"],
                                   "quant_reference": result["quant_reference"][mode],
                                   "original_bf16_diagnostic": result["original_bf16_diagnostic"][mode],
                                   "graph_vs_eager": result["graph_vs_eager"],
                                   "reference_order": result["reference_order"],
                                   "bf16_ms_per_call": bf16_ms, "int8_ms_per_call": int8_ms,
                                   "bf16_median_ms": statistics.median(bf16_ms) if bf16_ms else None,
                                   "int8_median_ms": statistics.median(int8_ms) if int8_ms else None,
                                   "timing_round_order": orders}
                            write_record(stream, row)
                        if not result["correctness_pass"]:
                            raise RuntimeError("算子数值或 graph/eager 对拍失败，停止本阶段")
                        print(f"  B={batch} M={result['rows']}：数值通过，计时完成", flush=True)
                    del original, weight, q, scale
                    torch.cuda.empty_cache()
            write_record(stream, {"type": "complete", **identity, "cases": len(WEIGHTS) * len(BATCHES),
                                  "mode_records": len(WEIGHTS) * len(BATCHES) * 2})
            print("micro 完成；完整结果仅供算子级比较", flush=True)
        except BaseException as exc:
            write_record(stream, {"type": "failure", **identity, **current,
                                  "error_type": type(exc).__name__, "error": str(exc),
                                  "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
