"""CPU metadata-only reproduction for a speculative-to-normal decode slot.

The real Scheduler, BlockManager, Sequence, ModelRunner.run and
ModelRunner.prepare_decode are exercised. Fake tensors only carry lists;
model/proposer/sampling are stubs, so no model, CUDA API, or GPU is used.
This checks scheduler/slot mapping, not numerical inference correctness.

Expected on the current base: control passes and the final invariant assert
fails for the reachable rejected-prefix cases. Evidence is saved before assert.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("cpu-decode-slot-repro.json")


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def cuda(self, non_blocking=False):
        return self

    def tolist(self):
        return self.data


def install_stubs():
    # Never import real torch: its CUDA installation and OpenMP runtime are
    # irrelevant to this host-only metadata contract.
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.Tensor = FakeTensor
    torch.int64 = "int64"
    torch.int32 = "int32"
    torch.tensor = lambda data, **kwargs: FakeTensor(data)
    torch.inference_mode = lambda: (lambda fn: fn)
    distributed = types.ModuleType("torch.distributed")
    torch.distributed = distributed
    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = distributed

    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [str(ROOT / "nanovllm")]
    sys.modules["nanovllm"] = pkg

    def module(name, **members):
        stub = types.ModuleType(name)
        stub.__dict__.update(members)
        sys.modules[name] = stub

    module("nanovllm.config", Config=object)
    module("nanovllm.models.qwen3", Qwen3ForCausalLM=object)
    module("nanovllm.layers.sampler", Sampler=object)
    module("nanovllm.layers.quant", apply_int8_quant=lambda *a, **kw: None)
    module("nanovllm.spec_decode.draft_proposer", DraftWindow=object,
           window_realign_target=lambda *a, **kw: None)
    module("nanovllm.utils.loader", load_model=lambda *a, **kw: None)


install_stubs()
from nanovllm.engine.model_runner import ModelRunner  # noqa: E402
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402
from nanovllm.utils.context import get_context  # noqa: E402


class FakeDSpark:
    def propose_batch(self, seqs):
        # The proposer would return 7 candidates. ModelRunner.run itself applies
        # the real remaining-token limit and drops all candidates at remaining=1.
        return [[3000 + i for i in range(7)] for _ in seqs], [[0] * 7 for _ in seqs]


def make_scheduler(max_tokens, spec_k):
    config = types.SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=8,
        spec_batch_threshold=0, spec_k=spec_k,
    )
    scheduler = Scheduler(config)
    Sequence.block_size = config.kvcache_block_size
    seq = Sequence(list(range(249)), SamplingParams(max_tokens=max_tokens, ignore_eos=True))
    scheduler.add(seq)
    batch, is_prefill = scheduler.schedule()
    assert batch == [seq] and is_prefill and seq.num_scheduled_tokens == 249
    scheduler.postprocess(batch, [999], True)  # model's first sampled token
    assert len(seq) == 250 and seq.num_cached_tokens == 249
    assert len(seq.block_table) == 1 and seq in scheduler.running
    return scheduler, seq


def run_real_fallback(scheduler, seq):
    batch, is_prefill = scheduler.schedule()
    assert batch == [seq] and not is_prefill
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = scheduler.config
    runner.block_size = scheduler.block_size
    runner.rank = 0
    runner.dspark_model = object() if scheduler.config.spec_k else None
    runner.draft_model = None
    runner.spec_proposer = FakeDSpark() if scheduler.config.spec_k else None
    runner.prepare_sample = lambda seqs: None
    recorded = {}

    def target_stub(input_ids, positions, is_prefill):
        context = get_context()
        recorded.update(
            position=positions.tolist()[0],
            actual_slot=context.slot_mapping.tolist()[0],
            target_table=context.block_tables.tolist()[0],
            context_len=context.context_lens.tolist()[0],
        )
        return "model-not-run"

    runner.run_model = target_stub
    runner.sampler = lambda logits, temperatures: FakeTensor([12345])
    result = runner.run(batch, False)
    assert result == [12345]
    pos = len(seq) - 1
    expected = seq.block_table[pos // scheduler.block_size] * scheduler.block_size + pos % scheduler.block_size
    assert recorded["position"] == pos
    assert recorded["context_len"] == len(seq)
    assert recorded["target_table"] == seq.block_table
    recorded.update(expected_slot=expected, matching=recorded["actual_slot"] == expected,
                    sequence_len=len(seq), num_cached_tokens=seq.num_cached_tokens,
                    scheduled_tokens=seq.num_scheduled_tokens,
                    remaining=seq.max_tokens - seq.num_completion_tokens)
    return recorded


def control():
    scheduler, seq = make_scheduler(max_tokens=3, spec_k=0)
    result = run_real_fallback(scheduler, seq)
    assert result["matching"] and len(result["target_table"]) == 1
    return result


def speculative_rejection(accepted):
    scheduler, seq = make_scheduler(max_tokens=accepted + 3, spec_k=7)
    batch, is_prefill = scheduler.schedule()
    assert batch == [seq] and not is_prefill and seq.num_scheduled_tokens == 8
    assert len(seq.block_table) == 2  # schedule() genuinely reserved across 256
    original = list(seq.block_table)
    proposed = [2000 + i for i in range(7)]
    seq.draft_tokens = proposed
    emitted = proposed[:accepted] + [2100]  # accepted prefix plus one target correction/bonus
    scheduler.postprocess(batch, [emitted], False)
    assert len(seq) == 250 + accepted + 1
    assert seq.num_cached_tokens == len(seq) - 1
    assert seq.block_table == original and len(seq.block_table) == 2
    assert seq.max_tokens - seq.num_completion_tokens == 1
    # Scheduler still requests k=7; the real DSpark run() sees remaining=1,
    # truncates to zero candidates, and invokes real prepare_decode().
    result = run_real_fallback(scheduler, seq)
    assert result["remaining"] == 1 and result["scheduled_tokens"] == 8
    result["accepted_candidates"] = accepted
    return result


records = {"base_commit": "cefb5a2ab2f57ad2f709fd38493ac544d6b78989",
           "scope": "CPU-only metadata; model, sampling, CUDA and GPU not run",
           "control": control(),
           "accepted_0": speculative_rejection(0),
           "accepted_2": speculative_rejection(2)}
OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(records, indent=2, ensure_ascii=False))
failures = [name for name in ("accepted_0", "accepted_2") if not records[name]["matching"]]
assert not failures, f"prepare_decode slot != logical-position slot: {failures}; evidence={OUTPUT}"
