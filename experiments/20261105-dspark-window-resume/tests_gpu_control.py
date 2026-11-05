"""CPU-only checks for guard identity and budget accounting; no GPU is queried."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import uuid
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("guard", HERE / "gpu_control.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
import gpu_identity


def proc(pid, parent, ticks):
    return {"pid": pid, "ppid": parent, "start_ticks": ticks, "state": "S", "session": 100}


class GuardTests(unittest.TestCase):
    def test_cuda_uuid_string_and_pybind_byte_forms(self):
        value = uuid.UUID("e0671215-034e-dd14-093f-b3755e543b6f")
        expected = "GPU-e0671215-034e-dd14-093f-b3755e543b6f"
        self.assertEqual(gpu_identity.normalize_gpu_uuid("gpu-E0671215-034E-DD14-093F-B3755E543B6F"), expected)
        self.assertEqual(gpu_identity.normalize_gpu_uuid(str(value)), expected)
        for raw in (value.bytes, bytearray(value.bytes), memoryview(value.bytes)):
            pybind_like = types.SimpleNamespace(bytes=raw)
            self.assertEqual(gpu_identity.normalize_gpu_uuid(pybind_like), expected)
        with self.assertRaises(TypeError):
            gpu_identity.normalize_gpu_uuid(types.SimpleNamespace(bytes=list(value.bytes)))
        with self.assertRaises(ValueError):
            gpu_identity.normalize_gpu_uuid(types.SimpleNamespace(bytes=b"short"))

    def test_visible_gpu_check_fails_closed(self):
        expected = "GPU-e0671215-034e-dd14-093f-b3755e543b6f"
        value = uuid.UUID(expected[4:])
        props = types.SimpleNamespace(uuid=types.SimpleNamespace(bytes=value.bytes))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1,
                                                                 get_device_properties=lambda n: props))
        self.assertIs(gpu_identity.verify_visible_gpu(torch, expected), props)
        with self.assertRaises(RuntimeError):
            gpu_identity.verify_visible_gpu(torch, "GPU-00000000-0000-0000-0000-000000000001")
        torch.cuda.device_count = lambda: 2
        with self.assertRaises(RuntimeError):
            gpu_identity.verify_visible_gpu(torch, expected)

    def test_parse_stat_handles_parentheses_and_spaces(self):
        # After state: fields 4..21 then start ticks (field 22).
        source = "99 (worker (gpu) 0) S " + " ".join(str(x) for x in range(4, 23))
        parsed = guard.parse_proc_stat(source)
        self.assertEqual((parsed["pid"], parsed["ppid"], parsed["session"], parsed["start_ticks"]),
                         (99, 4, 6, 22))

    def test_descendants_and_pid_reuse(self):
        rows = {100: proc(100, 1, 10), 101: proc(101, 100, 11),
                102: proc(102, 101, 12), 200: proc(200, 1, 20)}
        owned = {(100, 10)}
        self.assertEqual(guard.extend_owned(rows, owned), {100, 101, 102})
        # Reused root PID cannot authorize its new child.
        rows = {100: proc(100, 1, 999), 201: proc(201, 100, 21)}
        self.assertEqual(guard.extend_owned(rows, owned), set())
        self.assertNotIn((201, 21), owned)

    def test_nvml_tid_resolves_to_tgid_and_unknown_fails_closed(self):
        rows = {100: proc(100, 1, 10), 101: proc(101, 100, 11), 200: proc(200, 1, 20)}
        state = {"compute_processes": [{"reported_pid_or_tid": 1101},
                                       {"reported_pid_or_tid": 1200},
                                       {"reported_pid_or_tid": 1999}]}
        def resolve(tid):
            if tid == 1999:
                raise FileNotFoundError("thread exited")
            return {1101: 101, 1200: 200}[tid]
        with patch.object(guard, "resolve_tgid", side_effect=resolve):
            results = guard.classify_compute(state, rows, {(100, 10)})
        self.assertEqual([r["owned"] for r in results], [True, False, False])
        self.assertEqual(results[0]["tgid"], 101)
        self.assertIn("resolution_error", results[2])

    def test_ledger_counts_failures_and_retries_and_rejects_open_job(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            def add(event, ident, kind, **extra):
                guard.append(path, {"event": event, "id": ident, "kind": kind,
                                    "gpu_uuid": "GPU-A", **extra})
            add("start", "a", "smoke")
            add("child", "a", "smoke", pid=123, start_ticks=456)
            add("end", "a", "smoke", elapsed_s=125.5, outcome="timeout")
            add("start", "b", "pilot")
            add("end", "b", "pilot", elapsed_s=50.25, outcome="error")
            self.assertEqual(guard.load_ledger(path, "GPU-A"), (175.75, 125.5))
            with self.assertRaises(RuntimeError):
                guard.load_ledger(path, "GPU-B")
            add("start", "c", "smoke")
            with self.assertRaisesRegex(RuntimeError, "unfinished"):
                guard.load_ledger(path, "GPU-A")
            self.assertEqual(len([json.loads(s) for s in path.read_text().splitlines()]), 6)

    def test_pidfd_identity_checked_before_signalling(self):
        rows = {100: proc(100, 1, 22)}
        stat = "100 (worker) S " + " ".join(str(x) for x in range(4, 23))
        with patch.object(guard.os, "pidfd_open", return_value=45, create=True), \
             patch.object(guard.os, "close") as close, \
             patch.object(guard.signal, "pidfd_send_signal", create=True) as send, \
             patch.object(Path, "read_text", return_value=stat):
            events = guard.signal_owned(rows, {(100, 22)}, 15)
            send.assert_called_once_with(45, 15)
            close.assert_called_once_with(45)
            self.assertEqual(events[0]["start_ticks"], 22)
        # Even if a descriptor is obtained, changed start ticks block the signal.
        with patch.object(guard.os, "pidfd_open", return_value=45, create=True), \
             patch.object(guard.os, "close"), \
             patch.object(guard.signal, "pidfd_send_signal", create=True) as send, \
             patch.object(Path, "read_text", return_value=stat):
            guard.signal_owned({100: proc(100, 1, 11)}, {(100, 11)}, 15)
            send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
