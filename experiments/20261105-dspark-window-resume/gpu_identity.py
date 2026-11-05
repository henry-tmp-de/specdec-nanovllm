"""Strict CUDA UUID normalization shared by guarded GPU entry points.

PyTorch 2.10 exposes an object with a 16-byte ``bytes`` attribute rather than a
useful string representation. Unknown forms fail closed.
"""
from __future__ import annotations

import uuid


def normalize_gpu_uuid(value):
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith("gpu-"):
            text = text[4:]
        try:
            return "GPU-" + str(uuid.UUID(text))
        except ValueError as exc:
            raise ValueError("invalid GPU UUID string") from exc
    raw = value if isinstance(value, (bytes, bytearray, memoryview, list, tuple)) else getattr(value, "bytes", None)
    # PyTorch 2.10.0+cu128 exposed the 16 UUID bytes as a list of unsigned
    # Python integers on the authorized machine. Reject all ambiguous values.
    if type(raw) in (list, tuple):
        if len(raw) != 16 or any(type(x) is not int or not 0 <= x <= 255 for x in raw):
            raise ValueError("CUDA UUID list must contain exactly 16 unsigned integer bytes")
        raw = bytes(raw)
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError("CUDA UUID must expose 16 unsigned bytes or be a UUID string")
    raw = bytes(raw)
    if len(raw) != 16:
        raise ValueError("CUDA UUID must contain exactly 16 bytes")
    return "GPU-" + str(uuid.UUID(bytes=raw))


def verify_visible_gpu(torch, expected):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("guard must expose exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    actual = normalize_gpu_uuid(getattr(properties, "uuid", None))
    expected_uuid = normalize_gpu_uuid(expected)
    if actual != expected_uuid:
        raise RuntimeError(f"CUDA device UUID mismatch: actual={actual}, expected={expected_uuid}")
    return properties
