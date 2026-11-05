"""CPU regression: a slow tokenizer vocabulary size is queried only once."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("workload", HERE / "run_workload.py")
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class WorkloadCPUChecks(unittest.TestCase):
    def test_vocab_size_is_not_recomputed_per_token(self):
        class Tokenizer:
            length_queries = 0
            def __len__(self):
                self.length_queries += 1
                return 256
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3]
        tokenizer = Tokenizer()
        fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"transformers": fake}):
            path = Path(tmp)
            wiki = path / "wiki.txt"
            wiki.write_text("abcdefghijklmnopqrstuvwxyz" * 30, encoding="utf-8")
            for name in ("humaneval.jsonl", "gsm8k.jsonl", "alpaca.jsonl"):
                (path / name).write_text('{"turns": ["one"]}\n{"turns": ["two"]}\n', encoding="utf-8")
            result = workload.build_from_local(str(path), str(path), str(wiki), 32, 2)
        self.assertEqual(tokenizer.length_queries, 1)
        self.assertEqual(len(result["panels"]["code"]), 2)
        self.assertEqual(len(result["panels"]["code"][0]["token_ids"]), 32)


if __name__ == "__main__":
    unittest.main()
