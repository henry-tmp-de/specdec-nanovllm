"""M2: 在【真引擎】上审计「draft 滑窗真的只看到最近 M 个 token」。

上一轮自己承认这件事没有独立验证过。这里不靠"看公式"，而是把真引擎里
每一次 draft 前向（prefill 与 decode）的 (tokens, positions, ctx_lens,
block_tables, slot_mapping) 全部记下来逐条判：

  记录面：把每次前向写进去的 KV 打上标签  content[slot] = 该 row 的绝对位置
          —— prefill 走 draft_model.forward（不经 _forward_group），
             decode/catchup 走 _forward_group，两条都记，否则会误报"没写过"。

  判定面（对每次 decode/catchup 前向的每个 key）：
    A1 可见长度 ctx ≤ M*bs
    A2 最老可见位置 ≥ pos − M*bs + 1     ← 「只看到最近 M 个 token」的硬边界
    A3 块表恰好 M 项、且 M 个物理块互不相同（环是置换，不能别名）
    A4 没有 key 指向未来（> pos）
    A5 每个 key 读到的槽，最近一次被写进去的位置恰好等于它该是的位置
         = 没有陈旧读出、没有拿别人的 KV

  A2 是「只看到 M 个 token」的边界，A5 是「看到的确实是那些 token 的 KV」。
  两条合起来 = 滑窗语义成立。

负对照（必须失败，否则这套审计是空转）：
  negrot  环的块表旋转错一格（key index ↔ 绝对位置映射整体错位）
  negclip 把 window_request_geom 倒回修复前那版（clip 向下对齐 + valid_from
          取 max(wvf, 补齐起点) + 补齐前向硬传 base）→ 复现修复前那一类错位

用法: NV_ROOT=<代码根> python scripts/p7_window_audit.py <W> <L> <OUT> <B> [none|negrot|negclip]
"""
import os
import sys
import json

CODE_ROOT = os.environ.get("NV_ROOT", "/home/ziru/nano-vllm/p1-work")
sys.path.insert(0, CODE_ROOT)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.spec_decode.draft_proposer import DraftModelProposer, DraftWindow
from nanovllm.utils.context import get_context

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")

import importlib.util as _iu
_spec = _iu.spec_from_file_location("a7q", os.path.join(CODE_ROOT, "scripts/a7_quant.py"))
_a7 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_a7)
build_prompt = _a7.build_prompt


class Audit:
    def __init__(self, M, bs):
        self.M, self.bs = M, bs
        self.content = {}          # slot -> 最近一次写进去的绝对位置
        self.fails = []
        self.n_fwd = 0
        self.n_keys = 0
        self.max_span = 0
        self.n_writes = 0

    def record_writes(self, positions, slot):
        for i in range(len(positions)):
            s = int(slot[i])
            if s < 0:
                continue
            self.content[s] = int(positions[i])
            self.n_writes += 1

    def check_forward(self, positions, ctx, bt, slot):
        for i in range(len(positions)):
            pos, clen, table = int(positions[i]), int(ctx[i]), [int(x) for x in bt[i]]
            self.n_fwd += 1
            self.max_span = max(self.max_span, clen)
            if clen <= 0:
                self.fails.append(("A0_clen<=0", pos, clen))
                continue
            b0 = pos - clen + 1
            b0_blk = b0 // self.bs
            # A1 / A2
            if clen > self.M * self.bs:
                self.fails.append(("A1_clen>M*bs", pos, clen))
            if b0 < max(0, pos // self.bs - self.M + 1) * self.bs:
                self.fails.append(("A2_window_too_old", pos, clen, b0))
            # A3
            if len(table) != self.M:
                self.fails.append(("A3_table_len", pos, len(table), self.M))
            elif len(set(table)) != self.M:
                self.fails.append(("A3_table_aliased", pos, table))
            # 写在前、读在后（Attention.forward 里 store_kvcache 先执行）
            self.record_writes([pos], [slot[i]])
            for j in range(clen):
                want = b0 + j
                if want > pos:
                    self.fails.append(("A4_future", pos, j, want))
                t, off = divmod(j, self.bs)
                if t >= len(table):
                    self.fails.append(("A3_table_short", pos, j, want, t, len(table)))
                    continue
                slot_want = table[t] * self.bs + off
                got = self.content.get(slot_want)
                if got != want:
                    self.fails.append(("A5_stale_read", pos, j, want, got))
                self.n_keys += 1


def install_negs(mode):
    import nanovllm.spec_decode.draft_proposer as DP
    if mode == "negrot":
        _obt = DraftWindow.block_table

        def _bt(self, pos, valid_from=0):
            r = _obt(self, pos, valid_from)
            return r[1:] + r[:1]            # 旋转错一格
        DraftWindow.block_table = _bt
    elif mode == "negclip":
        def _old_clip(start, gap, block_size, window_blocks, token_ids=None):
            if window_blocks <= 0 or not gap:
                return start, gap
            end = int(start) + len(gap)
            s_min = max(0, (end // block_size) - window_blocks + 1) * block_size
            if start < s_min:
                drop = min(s_min - start, len(gap))
                start, gap = start + drop, gap[drop:]
            start = (start // block_size) * block_size      # ← 修复前那一行
            start = max(start, s_min)
            if start >= end:
                return start, []
            return start, list(gap)

        def _old_geom(dvl, token_ids, block_size, window_blocks):
            start, gap = DP.catchup_gap(dvl, token_ids)
            if window_blocks <= 0:
                return start, gap, 0
            start, gap = _old_clip(start, gap, block_size, window_blocks)
            wvf = DP.window_valid_from(dvl, block_size, window_blocks)
            return start, gap, (max(wvf, start) if gap else wvf)   # ← 修复前那一行
        DP.window_request_geom = _old_geom


def main():
    W = int(sys.argv[1]); L = int(sys.argv[2]); OUT = int(sys.argv[3])
    B = int(sys.argv[4]); MODE = sys.argv[5] if len(sys.argv) > 5 else "none"
    K = 6
    install_negs(MODE)

    llm = LLM(TARGET, max_model_len=4608, max_num_batched_tokens=16384,
              max_num_seqs=8, enforce_eager=False, spec_k=K, spec_method="draft",
              draft_model=DRAFT, spec_batch_threshold=0, spec_draft_window=W)
    mr, bm = llm.model_runner, llm.scheduler.block_manager
    AU = Audit(mr.draft_window_blocks, mr.block_size)

    # ---- 记录面：draft 模型自己的 forward（covers prefill）----
    _dmod = mr.draft_model
    _ofwd = _dmod.forward

    def _fwd(input_ids, positions, *a, **k):
        c = get_context()
        if c.slot_mapping is not None:
            try:
                AU.record_writes(positions.tolist() if isinstance(positions, torch.Tensor)
                                 else list(positions), c.slot_mapping.tolist())
            except Exception:
                pass
        return _ofwd(input_ids, positions, *a, **k)
    _dmod.forward = _fwd

    # ---- 判定面：每次 decode/catchup 前向 ----
    _ofg = DraftModelProposer._forward_group

    def _fg(self, tokens, positions, block_tables, context_lens, slot_mapping):
        AU.check_forward(positions, context_lens, block_tables, slot_mapping)
        return _ofg(self, tokens, positions, block_tables, context_lens, slot_mapping)
    DraftModelProposer._forward_group = _fg

    prompts = [build_prompt("zh" if i % 2 == 0 else "code", (5 * i) % 12, L,
                            40000 + L + i) for i in range(B)]
    llm.generate([prompts[0]], SamplingParams(temperature=1.0, max_tokens=8,
                                              ignore_eos=True), use_tqdm=False)
    bm.hash_to_block_id.clear()
    AU.content.clear(); AU.fails.clear()
    AU.n_fwd = AU.n_keys = AU.n_writes = AU.max_span = 0
    torch.manual_seed(777)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=OUT,
                                                ignore_eos=True), use_tqdm=False)
    kinds = {}
    for f in AU.fails:
        kinds[f[0]] = kinds.get(f[0], 0) + 1
    print("@@A@@" + json.dumps(dict(
        W=W, L=L, OUT=OUT, B=B, mode=MODE, M=AU.M, bs=AU.bs,
        window_tokens=AU.M * AU.bs,
        n_decode_forwards=AU.n_fwd, n_keys_checked=AU.n_keys,
        max_visible_span=AU.max_span, max_span_ok=(AU.max_span <= AU.M * AU.bs),
        n_fails=len(AU.fails), fail_kinds=kinds, fail_samples=AU.fails[:4],
        out_tokens=sum(len(o["token_ids"]) for o in outs),
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
