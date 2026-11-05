"""Batched DSpark proposal with independently reported graph hits/fallbacks."""
from __future__ import annotations

from time import perf_counter

import torch


def _pick_bucket(batch, buckets):
    return min((b for b in buckets if b >= batch), default=None)


class DSparkProposer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.graphs = {}
        self.graph_hits = 0
        self.graph_exact_hits = 0
        self.graph_padded_hits = 0
        self.graph_captures = 0
        self.graph_runtime_captures = 0
        self.graph_runtime_capture_s = 0.0
        self.graph_runtime_shapes = []
        self.graph_fallbacks = {}
        self.active_batch_histogram = {}
        self.context_bucket_histogram = {}
        self.input_prepare_calls = 0
        self.input_prepare_wall_s = 0.0
        self.proposals = 0
        self.head_graphs = {}
        self.head_graph_hits = 0
        self.head_graph_captures = 0
        self.head_runtime_captures = 0
        self.head_runtime_capture_s = 0.0
        self.head_runtime_batches = []
        self.backbone_pool = None
        self.head_pool = None
        self.prewarm_in_progress = False
        self.prewarmed_shapes = []
        if (getattr(config, "dspark_graph_pad", False)
                and not getattr(config, "dspark_split_graph", False)):
            raise ValueError("DSpark padding requires real-batch split sampling")

    # Scheduler's generic proposer interface. DSpark reads the target cache;
    # it does not maintain a token index or the AR draft watermark.
    def observe(self, token_ids):
        pass

    def reset_watermark(self, token_ids):
        pass

    def _fallback(self, reason):
        self.graph_fallbacks[reason] = self.graph_fallbacks.get(reason, 0) + 1

    def _inputs(self, seqs):
        started = perf_counter()
        B = len(seqs)
        bs = self.model.block_size
        lengths = [len(s) - 1 for s in seqs]  # history strictly before anchor
        window = self.model.window_tokens
        visible = [min(n, window) if window else n for n in lengths]
        bucket = max(bs, ((max(visible) + bs - 1) // bs) * bs)
        self.context_bucket_histogram[bucket] = self.context_bucket_histogram.get(bucket, 0) + 1
        nblocks = bucket // bs
        tables = []
        for seq in seqs:
            if window:
                from nanovllm.spec_decode.dspark_window import visible_slots
                slots = visible_slots(seq, len(seq) - 1, self.config.dspark_window_storage,
                                      bs, window)
                tables.append(slots + [-1] * (bucket - len(slots)))
            else:
                if len(seq.block_table) < (len(seq) - 1 + bs - 1) // bs:
                    raise RuntimeError("DSpark context block table does not cover confirmed history")
                tables.append((list(seq.block_table[:nblocks]) + [-1] * nblocks)[:nblocks])
        device = self.model.fc.weight.device
        result = (torch.tensor([s.last_token for s in seqs], device=device, dtype=torch.long),
                  torch.tensor(lengths, device=device, dtype=torch.long),
                  torch.tensor(tables, device=device, dtype=torch.int32), bucket,
                  torch.tensor([s.temperature for s in seqs], device=device, dtype=torch.float32))
        self.input_prepare_calls += 1
        self.input_prepare_wall_s += perf_counter() - started
        return result

    def _capture(self, anchors, lengths, tables, bucket, temperatures):
        # Capture one exact batch/context shape, including the sequential Markov
        # sampling. Output pointers remain owned by this graph until next replay.
        started = perf_counter()
        a, l, bt, t = (x.clone() for x in (anchors, lengths, tables, temperatures))
        self.model.propose(a, l, bt, bucket, t)  # graph warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens, logits = self.model.propose(a, l, bt, bucket, t)
        self.graph_captures += 1
        if not self.prewarm_in_progress:
            self.graph_runtime_captures += 1
            self.graph_runtime_capture_s += perf_counter() - started
            self.graph_runtime_shapes.append((int(anchors.shape[0]), int(bucket)))
        return graph, (a, l, bt, t), (tokens, logits)

    @staticmethod
    def _fill_backbone(inputs, anchors, lengths, tables):
        """Put real requests first; dummy rows have no visible context."""
        ga, gl, gbt = inputs
        B = anchors.shape[0]
        if B > ga.shape[0] or tables.shape[1] != gbt.shape[1]:
            raise ValueError("DSpark graph buffer shape mismatch")
        ga.zero_()
        gl.zero_()
        gbt.fill_(-1)
        ga[:B].copy_(anchors)
        gl[:B].copy_(lengths)
        gbt[:B].copy_(tables)

    def _capture_backbone(self, graph_b, context_bucket):
        started = perf_counter()
        model = self.model
        dev = model.fc.weight.device
        blocks = context_bucket if model.window_tokens else context_bucket // model.block_size
        anchors = torch.zeros(graph_b, device=dev, dtype=torch.long)
        lengths = torch.zeros(graph_b, device=dev, dtype=torch.long)
        tables = torch.full((graph_b, blocks), -1, device=dev, dtype=torch.int32)
        out = torch.empty(graph_b, model.cfg.block_size, model.cfg.vocab_size,
                          device=dev, dtype=model.fc.weight.dtype)
        out.copy_(model.base_logits(anchors, lengths, tables, context_bucket))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.backbone_pool):
            out.copy_(model.base_logits(anchors, lengths, tables, context_bucket))
        if self.backbone_pool is None:
            self.backbone_pool = graph.pool()
        self.graph_captures += 1
        if not self.prewarm_in_progress:
            self.graph_runtime_captures += 1
            self.graph_runtime_capture_s += perf_counter() - started
            self.graph_runtime_shapes.append((int(graph_b), int(context_bucket)))
        return graph, (anchors, lengths, tables), out

    def _capture_head(self, real_b):
        """One step graph at *real* B; padding never generates RNG draws."""
        started = perf_counter()
        model = self.model
        dev = model.fc.weight.device
        base = torch.zeros(real_b, model.cfg.vocab_size, device=dev, dtype=model.fc.weight.dtype)
        previous = torch.zeros(real_b, device=dev, dtype=torch.long)
        temperatures = torch.ones(real_b, device=dev, dtype=torch.float32)
        sampled = torch.empty_like(previous)
        corrected = torch.empty_like(base)
        tokens, logits = model.sample_step(base, previous, temperatures)
        sampled.copy_(tokens)
        corrected.copy_(logits)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.head_pool):
            tokens, logits = model.sample_step(base, previous, temperatures)
            sampled.copy_(tokens)
            corrected.copy_(logits)
        if self.head_pool is None:
            self.head_pool = graph.pool()
        self.head_graph_captures += 1
        if not self.prewarm_in_progress:
            self.head_runtime_captures += 1
            self.head_runtime_capture_s += perf_counter() - started
            self.head_runtime_batches.append(int(real_b))
        return graph, (base, previous, temperatures), (sampled, corrected)

    @torch.inference_mode()
    def prewarm(self, shapes, real_batches):
        """Capture only explicitly listed (bucket-B, ctx) and real-B shapes."""
        if not getattr(self.config, "dspark_split_graph", False):
            raise ValueError("split graph must be enabled before prewarming")
        if self.config.enforce_eager or not self.config.spec_cuda_graph:
            raise ValueError("cannot prewarm graphs when CUDA graph is disabled")
        allowed = {int(b) for b in self.config.spec_graph_bs
                   if 1 <= int(b) <= int(self.config.max_num_seqs)}
        self.prewarm_in_progress = True
        try:
            for B, ctx in shapes:
                B, ctx = int(B), int(ctx)
                if B not in allowed or ctx <= 0 or ctx % self.model.block_size:
                    raise ValueError(f"unsupported prewarm shape ({B}, {ctx})")
                key = (B, ctx)
                if key not in self.graphs:
                    self.graphs[key] = self._capture_backbone(B, ctx)
                    self.prewarmed_shapes.append(key)
            for B in real_batches:
                B = int(B)
                if not 1 <= B <= self.config.max_num_seqs:
                    raise ValueError(f"unsupported real batch {B}")
                if B not in self.head_graphs:
                    self.head_graphs[B] = self._capture_head(B)
        finally:
            self.prewarm_in_progress = False

    def _graphs_enabled(self, anchors):
        return not self.config.enforce_eager and self.config.spec_cuda_graph and anchors.is_cuda

    def _split_propose(self, anchors, lengths, tables, bucket, temperatures):
        B = anchors.shape[0]
        graph_enabled = self._graphs_enabled(anchors)
        allowed = sorted({int(x) for x in self.config.spec_graph_bs
                          if 1 <= int(x) <= int(self.config.max_num_seqs)})
        graph_b = B if B in allowed else None
        if graph_b is None and getattr(self.config, "dspark_graph_pad", False):
            graph_b = _pick_bucket(B, allowed)
        if graph_enabled and graph_b is not None:
            key = (graph_b, bucket)
            if key not in self.graphs:
                self.graphs[key] = self._capture_backbone(graph_b, bucket)
            graph, inputs, base = self.graphs[key]
            self._fill_backbone(inputs, anchors, lengths, tables)
            graph.replay()
            self.graph_hits += 1
            if graph_b == B:
                self.graph_exact_hits += 1
            else:
                self.graph_padded_hits += 1
        else:
            self._fallback("graph_disabled" if not graph_enabled else "batch_shape_not_captured")
            base = self.model.base_logits(anchors, lengths, tables, bucket)

        # Only B real rows enter the stochastic step, even when backbone ran a
        # larger graph. Thus dummy rows cannot consume random numbers.
        tokens = torch.empty(B, self.model.cfg.block_size, dtype=torch.long, device=anchors.device)
        logits_out = torch.empty(B, self.model.cfg.block_size, self.model.cfg.vocab_size,
                                 dtype=base.dtype, device=anchors.device)
        previous = anchors
        if graph_enabled:
            if B not in self.head_graphs:
                self.head_graphs[B] = self._capture_head(B)
            graph, (gb, gp, gt), (sampled, corrected) = self.head_graphs[B]
            gt.copy_(temperatures)
            for i in range(self.model.cfg.block_size):
                gb.copy_(base[:B, i])
                gp.copy_(previous)
                graph.replay()
                tokens[:, i].copy_(sampled)
                logits_out[:, i].copy_(corrected)
                previous = sampled
                self.head_graph_hits += 1
        else:
            for i in range(self.model.cfg.block_size):
                previous, corrected = self.model.sample_step(base[:B, i], previous, temperatures)
                tokens[:, i].copy_(previous)
                logits_out[:, i].copy_(corrected)
        return tokens, logits_out

    @torch.inference_mode()
    def propose_batch(self, seqs):
        if not seqs:
            raise ValueError("DSpark requires a nonempty batch")
        B = len(seqs)
        self.active_batch_histogram[B] = self.active_batch_histogram.get(B, 0) + 1
        self.proposals += B
        a, l, bt, bucket, t = self._inputs(seqs)
        if getattr(self.config, "dspark_split_graph", False):
            tokens, logits = self._split_propose(a, l, bt, bucket, t)
            return tokens.tolist(), logits
        use_graph = (not self.config.enforce_eager and self.config.spec_cuda_graph
                     and a.is_cuda and B in self.config.spec_graph_bs)
        if use_graph:
            key = (B, bucket)
            if key not in self.graphs:
                self.graphs[key] = self._capture(a, l, bt, bucket, t)
            graph, (ga, gl, gbt, gt), (tokens, logits) = self.graphs[key]
            ga.copy_(a)
            gl.copy_(l)
            gbt.copy_(bt)
            gt.copy_(t)
            graph.replay()
            self.graph_hits += 1
            self.graph_exact_hits += 1
        else:
            self._fallback("graph_disabled" if self.config.enforce_eager or not self.config.spec_cuda_graph
                           else "batch_shape_not_captured")
            tokens, logits = self.model.propose(a, l, bt, bucket, t)
        return tokens.tolist(), logits
