"""DSpark window geometry and allocation-lifetime invariants (CPU metadata)."""
from __future__ import annotations


def begin_owner(seq, window: int):
    if seq.dspark_owner is not None:
        raise RuntimeError("DSpark owner is already active")
    seq.dspark_generation += 1
    seq.dspark_owner = (seq.seq_id, seq.dspark_generation)
    seq.dspark_ring_blocks = tuple(seq.draft_block_table) if seq.draft_block_table else None
    # Earlier prefill chunks are intentionally not collected.
    seq.dspark_valid_start = seq.dspark_valid_end = max(0, len(seq) - window)


def check_owner(seq):
    if seq.dspark_owner != (seq.seq_id, seq.dspark_generation):
        raise RuntimeError("missing or stale DSpark window owner")
    if seq.dspark_ring_blocks is not None and tuple(seq.draft_block_table) != seq.dspark_ring_blocks:
        raise RuntimeError("DSpark owner ring block table changed")


def _slotter(seq, storage: str, block_size: int, window: int):
    """Validate a block table once, then return a cheap position mapping."""
    if storage == "ring":
        blocks = seq.draft_block_table
        n = window // block_size
        if n <= 0 or len(blocks) != n or len(set(blocks)) != n or any(b < 0 for b in blocks):
            raise RuntimeError("invalid DSpark ring block table")
        def slot(position):
            if position < 0:
                raise ValueError("negative DSpark position")
            return blocks[(position // block_size) % n] * block_size + position % block_size
        return slot
    elif storage == "full_cache":
        blocks = seq.block_table
        def slot(position):
            block = position // block_size
            if position < 0 or block >= len(blocks) or blocks[block] < 0:
                raise RuntimeError("target block table does not cover DSpark position")
            return blocks[block] * block_size + position % block_size
        return slot
    else:
        raise ValueError(f"unsupported DSpark storage: {storage}")


def physical_slot(seq, position: int, storage: str, block_size: int, window: int):
    return _slotter(seq, storage, block_size, window)(position)


def visible_slots(seq, anchor: int, storage: str, block_size: int, window: int):
    check_owner(seq)
    start = max(0, anchor - window)
    if seq.dspark_valid_end != anchor or seq.dspark_valid_start > start:
        raise RuntimeError("DSpark confirmed window has a gap or stale boundary")
    slot = _slotter(seq, storage, block_size, window)
    return [slot(p) for p in range(start, anchor)]


def selected_prefill_rows(seqs, storage: str, block_size: int, window: int):
    """Return flattened target rows, destination slots, and ranges to mark on success."""
    rows, slots, ranges = [], [], []
    base = 0
    for seq in seqs:
        check_owner(seq)
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        selected_start = max(start, seq.dspark_valid_start)
        if selected_start < end:
            if selected_start != seq.dspark_valid_end:
                raise RuntimeError("DSpark prefill selected rows are not contiguous")
            rows.extend(base + p - start for p in range(selected_start, end))
            slot = _slotter(seq, storage, block_size, window)
            slots.extend(slot(p) for p in range(selected_start, end))
            ranges.append((seq, selected_start, end))
        base += seq.num_scheduled_tokens
    return rows, slots, ranges


def mark_confirmed(seq, start: int, end: int, window: int):
    check_owner(seq)
    if start != seq.dspark_valid_end or end < start:
        raise RuntimeError("DSpark commit is not contiguous with confirmed history")
    seq.dspark_valid_end = end
    seq.dspark_valid_start = max(seq.dspark_valid_start, end - window)


def commit_mapping(seq, anchor: int, count: int, scratch_start: int,
                   scratch_limit: int, storage: str, block_size: int, window: int):
    """Plan committed scratch rows; never include unconfirmed or dummy rows."""
    check_owner(seq)
    if seq.dspark_valid_end != anchor or count < 0 or scratch_start < 0:
        raise RuntimeError("invalid DSpark commit boundary")
    if scratch_start + count > scratch_limit:
        raise RuntimeError("DSpark scratch range exceeds real query rows")
    # A tiny test window can be shorter than a fully accepted block. Only the
    # final window survives; dropping already-evicted rows also avoids duplicate
    # ring destinations in one index_copy_.
    drop = max(0, count - window)
    positions = range(anchor + drop, anchor + count)
    slot = _slotter(seq, storage, block_size, window)
    return (list(range(scratch_start + drop, scratch_start + count)),
            [slot(p) for p in positions])
