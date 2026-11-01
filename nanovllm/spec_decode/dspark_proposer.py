"""Batched DSpark proposal with independently reported graph hits/fallbacks."""
from __future__ import annotations

import torch


class DSparkProposer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.graphs = {}
        self.graph_hits = 0
        self.graph_captures = 0
        self.graph_fallbacks = {}
        self.active_batch_histogram = {}
        self.proposals = 0

    # Scheduler's generic proposer interface. DSpark reads the target cache;
    # it does not maintain a token index or the AR draft watermark.
    def observe(self, token_ids):
        pass

    def reset_watermark(self, token_ids):
        pass

    def _fallback(self, reason):
        self.graph_fallbacks[reason] = self.graph_fallbacks.get(reason, 0) + 1

    def _inputs(self, seqs):
        B = len(seqs)
        bs = self.model.block_size
        lengths = [len(s) - 1 for s in seqs]  # history strictly before anchor
        bucket = max(bs, ((max(lengths) + bs - 1) // bs) * bs)
        nblocks = bucket // bs
        tables = []
        for seq in seqs:
            if len(seq.block_table) < (len(seq) - 1 + bs - 1) // bs:
                raise RuntimeError("DSpark context block table does not cover confirmed history")
            tables.append((list(seq.block_table[:nblocks]) + [-1] * nblocks)[:nblocks])
        device = self.model.fc.weight.device
        return (torch.tensor([s.last_token for s in seqs], device=device, dtype=torch.long),
                torch.tensor(lengths, device=device, dtype=torch.long),
                torch.tensor(tables, device=device, dtype=torch.int32), bucket,
                torch.tensor([s.temperature for s in seqs], device=device, dtype=torch.float32))

    def _capture(self, anchors, lengths, tables, bucket, temperatures):
        # Capture one exact batch/context shape, including the sequential Markov
        # sampling. Output pointers remain owned by this graph until next replay.
        a, l, bt, t = (x.clone() for x in (anchors, lengths, tables, temperatures))
        self.model.propose(a, l, bt, bucket, t)  # graph warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens, logits = self.model.propose(a, l, bt, bucket, t)
        self.graph_captures += 1
        return graph, (a, l, bt, t), (tokens, logits)

    @torch.inference_mode()
    def propose_batch(self, seqs):
        if not seqs:
            raise ValueError("DSpark requires a nonempty batch")
        B = len(seqs)
        self.active_batch_histogram[B] = self.active_batch_histogram.get(B, 0) + 1
        self.proposals += B
        a, l, bt, bucket, t = self._inputs(seqs)
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
        else:
            self._fallback("graph_disabled" if self.config.enforce_eager or not self.config.spec_cuda_graph
                           else "batch_shape_not_captured")
            tokens, logits = self.model.propose(a, l, bt, bucket, t)
        return tokens.tolist(), logits
