"""Exercise mixed scheduling under real BlockManager pressure, without GPU.

Real Scheduler, Sequence, BlockManager and ModelRunner.prepare_prefill are used;
tensor operations are metadata-only stubs shared with the prior slot repro.
"""
from __future__ import annotations

import json
import argparse
import importlib
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "experiments/20261102-dspark-window/cpu_decode_slot_repro.py"))
from nanovllm.engine.mixed_batch import MixedBatchPlan
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context


scheduler_module = importlib.import_module("nanovllm.engine.scheduler")
Scheduler = scheduler_module.Scheduler


def scenario(other_decode):
    Sequence.block_size = 4
    cfg = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32, eos=-1,
                          kvcache_block_size=4, num_kvcache_blocks=3 + other_decode,
                          spec_batch_threshold=0, spec_k=0,
                          spec_prefill_chunk_tokens=4, spec_overlap_mixed=True,
                          spec_overlap_mode="serial")
    sched = Scheduler(cfg)

    def add_running(base):
        seq = Sequence([base + i for i in range(4)], SamplingParams(max_tokens=20, ignore_eos=True))
        sched.add(seq)
        seqs, prefill = sched.schedule()
        assert prefill and seqs == [seq]
        sched.postprocess(seqs, [base + 50], True)
        assert len(seq) == 5 and len(seq.block_table) == 1
        return seq

    decode = add_running(10)
    victim = add_running(100) if other_decode else None
    pre = Sequence(list(range(200, 208)), SamplingParams(max_tokens=20, ignore_eos=True))
    sched.add(pre)
    seqs, prefill = sched.schedule()
    assert prefill and seqs == [pre] and pre.num_scheduled_tokens == 4
    sched.postprocess(seqs, [0], True)
    assert pre.num_cached_tokens == 4 and not sched.block_manager.free_block_ids
    original_table = list(pre.block_table)

    plan = sched.schedule()
    if isinstance(plan, MixedBatchPlan):
        prefill_seqs, decode_seqs = plan.prefill_seqs, plan.decode_seqs
    else:
        prefill_seqs, is_prefill = plan
        assert is_prefill
        decode_seqs = []
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = cfg.kvcache_block_size
    inputs, positions = runner.prepare_prefill(prefill_seqs, advance_draft=False)
    slots = get_context().slot_mapping.tolist()
    expected = [original_table[p // 4] * 4 + p % 4 for p in range(4, 8)]
    result = dict(other_decode=bool(other_decode), plan_type=type(plan).__name__,
                  prefill_scheduled=pre.num_scheduled_tokens, prefill_status=pre.status.name,
                  prefill_original_table=original_table, prefill_current_table=list(pre.block_table),
                  prefill_in_running=pre in sched.running, prefill_in_waiting=pre in sched.waiting,
                  input_ids=inputs.tolist(), positions=positions.tolist(), slots=slots,
                  expected_slots=expected, decode_scheduled=decode in decode_seqs,
                  decode_status=decode.status.name,
                  victim_status=victim.status.name if victim else None)
    assert prefill_seqs == [pre] and pre.status == SequenceStatus.RUNNING, result
    assert pre.block_table == original_table and pre in sched.running and pre not in sched.waiting, result
    assert positions.tolist() == list(range(4, 8)) and slots == expected, result
    if victim:
        assert decode_seqs == [decode] and victim.status == SequenceStatus.WAITING, result
    else:
        assert not decode_seqs and decode.status == SequenceStatus.WAITING, result
    sched.postprocess(prefill_seqs, [299], True)
    assert pre.num_cached_tokens == 8 and pre.last_token == 299
    return result


def default_tail_victim():
    Sequence.block_size = 4
    cfg = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32, eos=-1,
                          kvcache_block_size=4, num_kvcache_blocks=2,
                          spec_batch_threshold=0, spec_k=0, spec_overlap_mixed=False)
    sched = Scheduler(cfg)
    a = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=20, ignore_eos=True))
    b = Sequence([6, 7, 8, 9], SamplingParams(max_tokens=20, ignore_eos=True))
    # Allocate one block for each prompt, then append its sampled token.
    for seq in (a, b):
        sched.block_manager.allocate(seq, 0)
        seq.append_token(10)
        seq.num_cached_tokens = 4
        seq.status = SequenceStatus.RUNNING
        sched.running.append(seq)
    seqs, prefill = sched.schedule()
    assert not prefill and seqs == [a] and b.status == SequenceStatus.WAITING
    return {"decode_scheduled": a.seq_id, "tail_victim": b.seq_id,
            "tail_status": b.status.name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="run against the original scheduler from commit 88f6861")
    args = parser.parse_args()
    if args.baseline:
        source = subprocess.check_output(
            ["git", "show", "88f6861:nanovllm/engine/scheduler.py"], cwd=ROOT, text=True,
            encoding="utf-8")
        exec(compile(source, "88f6861:nanovllm/engine/scheduler.py", "exec"), scheduler_module.__dict__)
        Scheduler = scheduler_module.Scheduler
    results = [scenario(True), scenario(False)]
    print(json.dumps(results, indent=2))
    print("default tail victim:", default_tail_victim())
    print("PASS: protected prefill blocks; eligible victim and current decode preemption")
