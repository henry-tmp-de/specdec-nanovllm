"""Layer-by-layer Qwen3 target prefill within one scheduler step.

The planner is CPU-only. GPU work remains ordinary layer/attention operators;
two worker streams and already-recorded CUDA events express the dependency DAG.
No scheduler watermark is advanced by an internal task.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.utils.context import Context, get_context, restore_context


@dataclass(frozen=True)
class ChunkTask:
    task_id: int
    sequence_index: int
    sequence_id: int
    start: int
    end: int
    flat_start: int
    flat_end: int
    block_table: tuple[int, ...]
    slots: tuple[int, ...]
    previous_task: int | None


def plan_chunks(seqs, chunk_tokens, block_size):
    """Split scheduled spans, preserving sequence order and absolute positions."""
    if type(chunk_tokens) is not int or chunk_tokens <= 0 or type(block_size) is not int or block_size <= 0:
        raise ValueError("chunk and cache block sizes must be positive integers")
    if not seqs:
        raise ValueError("prefill needs at least one sequence")
    tasks, flat_offset, ids, block_owners, write_slots = [], 0, set(), {}, set()
    for seq_index, seq in enumerate(seqs):
        sid, start, count = seq.seq_id, seq.num_cached_tokens, seq.num_scheduled_tokens
        if (type(sid) is not int or sid in ids or type(start) is not int or start < 0 or
                type(count) is not int or count <= 0 or start + count > len(seq)):
            raise ValueError("invalid or duplicate prefill sequence metadata")
        ids.add(sid)
        end = start + count
        needed = (end + block_size - 1) // block_size
        blocks = tuple(seq.block_table[:needed])
        if (len(blocks) != needed or any(type(b) is not int or b < 0 for b in blocks) or
                len(set(blocks)) != len(blocks)):
            raise ValueError("incomplete or invalid prefill block table")
        for block in blocks:
            if block in block_owners and block_owners[block] != sid:
                raise ValueError("different sequences share a KV block")
            block_owners[block] = sid
        prev = None
        for begin in range(start, end, chunk_tokens):
            stop = min(begin + chunk_tokens, end)
            slots = tuple(blocks[pos // block_size] * block_size + pos % block_size
                          for pos in range(begin, stop))
            if any(slot in write_slots for slot in slots):
                raise ValueError("two chunk tasks would write the same KV slot")
            write_slots.update(slots)
            task = ChunkTask(len(tasks), seq_index, sid, begin, stop,
                             flat_offset + begin - start, flat_offset + stop - start,
                             blocks[:(stop + block_size - 1) // block_size], slots, prev)
            tasks.append(task)
            prev = task.task_id
        flat_offset += count
    return tuple(tasks)


def wavefront(tasks, num_layers, max_inflight):
    """Return a topological launch order; all waits reference recorded events."""
    if type(num_layers) is not int or num_layers <= 0 or type(max_inflight) is not int or max_inflight not in (1, 2):
        raise ValueError("invalid layer count or inflight limit")
    operations, seen = [], set()
    for first in range(0, len(tasks), max_inflight):
        group = tasks[first:first + max_inflight]
        for diagonal in range(num_layers + len(group) - 1):
            for offset, task in enumerate(group):
                layer = diagonal - offset
                if not 0 <= layer < num_layers:
                    continue
                if layer and (task.task_id, layer - 1) not in seen:
                    raise RuntimeError("task predecessor was not launched")
                if task.previous_task is not None and (task.previous_task, layer) not in seen:
                    raise RuntimeError("same-sequence predecessor was not launched")
                operations.append((task.task_id, layer))
                seen.add((task.task_id, layer))
    return tuple(operations)


def new_stats(mode, chunk_tokens, max_inflight):
    return {"mode": mode, "chunk_tokens": chunk_tokens, "max_inflight": max_inflight,
            "prefill_steps": 0, "fallback_steps": 0, "fallback_reasons": {},
            "planned_tasks": 0, "layer_ops_enqueued": 0, "same_chunk_waits": 0,
            "previous_chunk_waits": 0, "source_waits": 0, "join_waits": 0,
            "streams_created": 0, "stream_task_counts": {}, "last_streams_used": 0,
            "stream_ids": [], "last_single_stream_reason": None,
            "single_chunk_steps": 0, "completed_steps": 0, "completed_tasks": 0}


def _record_tensor(tensor, stream):
    if tensor is not None and tensor.is_cuda:
        tensor.record_stream(stream)


def _int_tensor(values, device):
    if device.type == "cuda":
        host = torch.tensor(values, dtype=torch.int32, pin_memory=True)
        return host.to(device=device, non_blocking=True), host
    return torch.tensor(values, dtype=torch.int32, device=device), None


class LayerPrefillExecutor:
    def __init__(self, model, kv_cache, block_size, chunk_tokens, max_inflight, mode, stats,
                 cuda=None, nvtx_enabled=False):
        if mode not in ("serial", "pipeline") or type(max_inflight) is not int or max_inflight not in (1, 2):
            raise ValueError("invalid layer-prefill execution mode")
        self.model, self.kv_cache, self.block_size = model, kv_cache, block_size
        self.chunk_tokens, self.max_inflight, self.mode = chunk_tokens, max_inflight, mode
        self.stats = stats
        self.cuda = cuda if cuda is not None else torch.cuda
        self.nvtx_enabled = nvtx_enabled
        self.streams = None
        self.pending_tasks = 0
        self._retained = None
        self.failed = False

    def _ensure_streams(self, device):
        if self.streams is not None:
            return
        count = 1 if self.mode == "serial" or self.max_inflight == 1 else 2
        self.streams = [self.cuda.Stream(device=device) for _ in range(count)]
        self.stats["streams_created"] = count
        self.stats["stream_ids"] = [str(getattr(stream, "cuda_stream", "unavailable")) for stream in self.streams]
        for stream in self.streams:
            _record_tensor(self.kv_cache, stream)
            for tensor in self.model.parameters():
                _record_tensor(tensor, stream)
            for tensor in self.model.buffers():
                _record_tensor(tensor, stream)

    def mark_completed(self):
        """Call only after the existing sampler's CPU result synchronization."""
        if self.pending_tasks:
            self.stats["completed_steps"] += 1
            self.stats["completed_tasks"] += self.pending_tasks
            self.pending_tasks = 0
            self._retained = None

    def forward(self, seqs, input_ids, positions):
        if self.failed:
            raise RuntimeError("layer-prefill executor cannot resume after a failed dispatch")
        if self.pending_tasks:
            raise RuntimeError("previous layer-prefill step not completed")
        parent = get_context()
        if not parent.is_prefill or parent.is_spec_verify or parent.slot_mapping is None:
            raise RuntimeError("layer-prefill needs the original prefill context")
        if (input_ids.ndim != 1 or positions.shape != input_ids.shape or
                parent.slot_mapping.numel() != input_ids.numel()):
            raise RuntimeError("prepared prefill input, position and slot lengths differ")
        tasks = plan_chunks(seqs, self.chunk_tokens, self.block_size)
        if sum(task.end - task.start for task in tasks) != input_ids.shape[0]:
            raise RuntimeError("planned token count differs from prepared input")
        core = self.model.model
        layers = core.layers
        order = wavefront(tasks, len(layers), self.max_inflight)
        device = input_ids.device
        self._ensure_streams(device)
        source = self.cuda.current_stream(device=device)
        runtime, hosts = [], []
        for task in tasks:
            cu_q, host_q = _int_tensor([0, task.end - task.start], device)
            cu_k, host_k = _int_tensor([0, task.end], device)
            block_table, host_table = _int_tensor([list(task.block_table)], device)
            hosts.extend(x for x in (host_q, host_k, host_table) if x is not None)
            context = Context(is_prefill=True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                              max_seqlen_q=task.end - task.start, max_seqlen_k=task.end,
                              slot_mapping=parent.slot_mapping[task.flat_start:task.flat_end],
                              block_tables=block_table)
            runtime.append({"task": task, "context": context,
                            "ids": input_ids[task.flat_start:task.flat_end],
                            "positions": positions[task.flat_start:task.flat_end],
                            "hidden": None, "residual": None, "events": []})
        ready = self.cuda.Event()
        ready.record(source)
        for stream in self.streams:
            stream.wait_event(ready)
            self.stats["source_waits"] += 1
        self.stats["planned_tasks"] += len(tasks)
        if len(tasks) == 1:
            self.stats["single_chunk_steps"] += 1
        self.stats["last_streams_used"] = min(len(self.streams), len(tasks))
        self.stats["last_single_stream_reason"] = (
            "single_chunk" if len(tasks) == 1 else "serial_mode" if self.mode == "serial" else
            "max_inflight_one" if self.max_inflight == 1 else None)
        try:
            for task_id, layer_id in order:
                task = tasks[task_id]
                state = runtime[task_id]
                stream_index = task_id % len(self.streams)
                stream = self.streams[stream_index]
                with self.cuda.stream(stream):
                    if layer_id:
                        stream.wait_event(state["events"][layer_id - 1])
                        self.stats["same_chunk_waits"] += 1
                    if task.previous_task is not None:
                        stream.wait_event(runtime[task.previous_task]["events"][layer_id])
                        self.stats["previous_chunk_waits"] += 1
                    if layer_id == 0:
                        for tensor in (state["ids"], state["positions"], state["context"].cu_seqlens_q,
                                       state["context"].cu_seqlens_k, state["context"].slot_mapping,
                                       state["context"].block_tables):
                            _record_tensor(tensor, stream)
                        state["hidden"] = core.embed_tokens(state["ids"])
                        key = str(stream_index)
                        self.stats["stream_task_counts"][key] = self.stats["stream_task_counts"].get(key, 0) + 1
                    if self.nvtx_enabled:
                        self.cuda.nvtx.range_push(f"layer_prefill task={task_id} seq={task.sequence_id} layer={layer_id}")
                    try:
                        restore_context(state["context"])
                        state["hidden"], state["residual"] = layers[layer_id](
                            state["positions"], state["hidden"], state["residual"])
                        if layer_id == len(layers) - 1:
                            state["hidden"], _ = core.norm(state["hidden"], state["residual"])
                    finally:
                        restore_context(parent)
                        if self.nvtx_enabled:
                            self.cuda.nvtx.range_pop()
                    event = self.cuda.Event()
                    event.record(stream)
                    state["events"].append(event)
                    self.stats["layer_ops_enqueued"] += 1
            for state in runtime:
                source.wait_event(state["events"][-1])
                self.stats["join_waits"] += 1
                _record_tensor(state["hidden"], source)
            output = torch.cat([state["hidden"] for state in runtime], dim=0)
            self.pending_tasks = len(tasks)
            # Retain metadata, pinned source buffers and events until the
            # existing sampling-to-host synchronization finishes the step.
            self._retained = (runtime, hosts, ready)
            return output
        except BaseException:
            # Fence any already-enqueued worker operations before the original
            # stream can reuse the KV pool. No CPU synchronization is added.
            fences = []
            for stream in self.streams:
                fence = self.cuda.Event()
                fence.record(stream)
                source.wait_event(fence)
                fences.append(fence)
            self._retained = (runtime, hosts, ready, fences)
            self.failed = True
            raise
        finally:
            restore_context(parent)
