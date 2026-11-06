"""Share strict GPU UUID handling with the previously audited guard."""
from __future__ import annotations

import importlib.util
from pathlib import Path

OLD = Path(__file__).resolve().parents[1] / "20261105-dspark-window-resume" / "gpu_identity.py"
spec = importlib.util.spec_from_file_location("layer_prefill_previous_gpu_identity", OLD)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
normalize_gpu_uuid = identity.normalize_gpu_uuid
verify_visible_gpu = identity.verify_visible_gpu
