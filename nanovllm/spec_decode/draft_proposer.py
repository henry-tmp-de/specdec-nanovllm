"""
Draft model 提议器 —— 用真实小模型自回归生成候选（P6：批量版）。

与 n-gram 提议的本质区别
------------------------
n-gram：从历史文本「捞」候选，没有真实分布
        → 接受概率退化为 min(1, p/1) = p[draft]
        → 实测自由生成场景 <1%，无收益

draft model：小模型【真实自回归】生成
        → 有真实分布 q，接受概率 min(1, p/q) 才有理论保障
        → 文献上这个值通常 0.6~0.8

前置条件
--------
draft 与 target 必须共享词表，否则无法比较同一 token 的概率。
本项目：Qwen3-0.6B (draft)配 Qwen3-4B (target)，vocab 均为 151936 ✅

代价
----
draft 需要自己的 KV cache（层数/头数都与 target 不同，不能共用）。
显存按 block_bytes 比例从总预算里切。

P6 批量路径（本轮新增）
-----------------------
`propose_batch(reqs)` 把**同一候选位置**的 B 条请求一起前向：
    B×k 次单序列前向  →  k 次批量前向（+ 偶发的补齐前向）
k 个候选位置仍然【顺序】生成 —— 候选序列本身是自回归的，
不能把时间维 k 并行。批量只发生在「请求维 B」。

每步用**逐请求**的 position / context_len / slot / block_table：
    input_ids  [B]            positions  [B]
    slot_map   [B]            ctx_lens   [B]
    block_tbl  [B, max_blocks]（不足补 -1）
所以 B 条请求各写各的物理槽，绝不把一条请求的 slot 写到另一条。

不承诺 B 倍速度：batch 前向本身更贵（权重只读一遍但算力是按 B 放大的）。
省的是「每次前向的固定开销 × (B-1)」——kernel launch / Python dispatch /
tensor 创建，实测这部分才是 B×k 次单序列前向的大头。

图（任务 B）
-----------
`bind_cudagraph({B: (graph, vars)})` 注入按精确 B 捕获的单步 decode 图；
一轮 replay k 次。每行元数据（position/ctx_len/slot/block_table）独立更新，
持久缓冲区 `[B, k, V]` 保存每个候选位置的原始 logits。
B 不在图集合里就自动退回 eager，语义完全一致。
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

# 用 -1 填充块表里没用到的那几列（与引擎其它路径一致）
_PAD_BLOCK = -1


def window_valid_from(draft_valid_len: int, block_size: int, window_blocks: int) -> int:
    """draft 环形窗口里【最老一个仍然可信的块】的起始位置。

    draft 只保留最近 `window_blocks` 个块的 KV，而且块是按绝对块号递增写进
    环形缓冲的（绝对块号 b → ring 第 b % M 块）。所以已经处理过位置
    [0, draft_valid_len) 之后，环里最老可信的块是

        newest = (draft_valid_len - 1) // block_size
        start  = max(0, newest - M + 1)

    它同时是「查询能看到的最早位置」的下界：比它更老的块要么没写过、
    要么内容已经被环覆盖，一律不许读。

    ★ 缺口【不会】让缺口之前的内容失效 —— [wvf, draft_valid_len) 这一段是
      draft 上一次正常提议时按正确前缀写好的，而 wvf 正是按「环里还有最近的
      M 块」算出来的，所以这些块仍在环里、内容仍然有效。所以下界只需要
      wvf，不能拿「缺口起点」（它就在序列末尾附近）去收紧，否则窗口会塌缩
      到大约一个块（实测踩过：见 tests/test_draft_window.py §8）。
    """
    if draft_valid_len <= 0 or window_blocks <= 0:
        return 0
    newest = (int(draft_valid_len) - 1) // block_size
    return max(0, newest - window_blocks + 1) * block_size


def clip_gap_to_window(start: int, gap: List[int], block_size: int,
                       window_blocks: int, token_ids: Optional[List[int]] = None
                       ) -> Tuple[int, List[int]]:
    """把补齐缺口夹到「最近 window_blocks 个块」之内。

    契约（调用方依赖这一条，务必保持）
    ----------------------------------
    返回的 ``(start, gap)`` 必须满足 ``gap[s] == token_ids[start + s]``，
    且 ``len(gap) == end - start``（end = 原 start + 原 len(gap)）。
    `propose_batch` 的补齐阶段就是这么配对的::

        positions = [base[i] + s for s in range(len(gap[i]))]
        tokens    = [gap[i][s] for s in range(len(gap[i]))]

    所以 `start` 与 `gap` 必须【一起动】。任何只改 `start` 不改 `gap`
    （或反之）的写法都会把 token 写到错位的位置上，静默覆盖环里正确的 KV。

    ★ 只允许【前移】，绝不允许后移
      · 前移（drop 分支）：缺口伸到窗口之外的更老位置，补了也会立刻被写掉。
        此时起点前移到窗口最老的块边界，并【同步裁掉 gap 的等长前缀】。
        被丢掉的那些位置保持陈旧 —— 它们 < start，而 b0 ≥ start//bs，
        内核根本不会读它们。
      · 后移（把 start 向下取整到块边界）是【错的】，已删除：
        原始 gap 是从原来的 start 起算的，更早的位置没有对应 token 可用；
        后移等于给整段 token 重新贴位置标签。而且"首块必须整块全新"这个
        动机本身不成立 —— 缺口【之前】的内容是 draft 上一次按正确前缀写好的，
        仍然有效（见 ModelRunner._draft_request 里 valid_from 的推导）。
    """
    if window_blocks <= 0 or not gap:
        return start, gap
    end = int(start) + len(gap)                 # = len(token_ids) - 1
    eb = end // block_size
    s_min = max(0, eb - window_blocks + 1) * block_size
    if start < s_min:
        drop = min(s_min - start, len(gap))
        start, gap = start + drop, gap[drop:]
    if start >= end:
        return start, []
    # ★ 传了 token_ids 就按新的 start 重新切一次：配对关系由构造保证，
    #   不依赖上面每一步是否恰好同步（防御未来改动破坏契约）。
    if token_ids is not None:
        gap = list(token_ids[start:end])
    return start, list(gap)


class DraftWindow:
    """draft 滑窗的纯算术（无 torch，可在 CPU 上单测）。

    bs = block_size，M = window_blocks（环里的块数），ring = M 个物理块 id。

      · 写槽      slot(p)   = ring[(p // bs) % M] * bs + (p % bs)
      · 窗口起点  b0(p)     = max(valid_from // bs, p // bs - M + 1, 0)
      · 可见长度  clen(p)   = p - b0(p)*bs + 1      （≤ M*bs）
      · 块表      bt(p)     = [ ring[(b0 + t) % M] for t in range(M) ]
        必须【从 b0 起、按绝对块号递增】—— key index j ↔ 绝对位置 b0*bs + j，
        causal 结构才对。因为 softmax 对 key 的顺序不敏感，只要「集合恰好是
        窗口」就等价；但 causal 掩码是按 index 比较的，顺序错就会看到未来。

    ★ RoPE：positions 张量照旧传【绝对位置】，丢的只是 KV，不是位置编号。
    """

    def __init__(self, ring, block_size: int, window_blocks: int):
        self.ring = [int(x) for x in ring]
        self.bs = int(block_size)
        self.M = int(window_blocks)
        assert self.M >= 1 and len(self.ring) == self.M, (self.ring, self.M)

    def b0(self, pos: int, valid_from: int = 0) -> int:
        return max(int(valid_from) // self.bs, int(pos) // self.bs - self.M + 1, 0)

    def slot(self, pos: int) -> int:
        return self.ring[(int(pos) // self.bs) % self.M] * self.bs + int(pos) % self.bs

    def block_table(self, pos: int, valid_from: int = 0) -> List[int]:
        b0 = self.b0(pos, valid_from)
        return [self.ring[(b0 + t) % self.M] for t in range(self.M)]

    def ctx_len(self, pos: int, valid_from: int = 0) -> int:
        return int(pos) - self.b0(pos, valid_from) * self.bs + 1


def catchup_gap(draft_valid_len: int, token_ids: List[int]) -> Tuple[int, List[int]]:
    """算出 draft KV 的缺口：要从哪个位置开始补、补哪些 token。

    参数
    ----
    draft_valid_len : draft KV 里「从位置 0 起连续有效」的个数（有效水位）
    token_ids       : 该请求【已确认】的全部 token

    返回
    ----
    (start, tokens)：
        start  = 缺口第一个位置 = min(draft_valid_len, len(token_ids)-1)
        tokens = token_ids[start : len(token_ids)-1]

    ★ 为什么是 len(token_ids)-1 而不是 len(token_ids)：
      最后一个已确认 token（位置 len-1）由 propose 的第 0 步负责写 ——
      它既是「读」的起点也是「写」的目标，所以不算缺口。
      缺口是中间那些【draft 没跑过、target 却已经生成了】的 token
      （gate 关闭期间的输出、前缀缓存命中的块）。
      ★ 只喂最后一个 token 恢复不了缺失前缀 —— 必须把窗口内的 token 全部补上。
    """
    end = len(token_ids) - 1
    start = min(int(draft_valid_len), end)
    return start, (list(token_ids[start:end]) if end > start else [])


def window_request_geom(draft_valid_len: int, token_ids: List[int],
                        block_size: int, window_blocks: int
                        ) -> Tuple[int, List[int], int]:
    """滑窗档下 `ModelRunner._draft_request` 的【纯算术】部分（无 torch，可 CPU 单测）。

    返回 ``(catchup_start, catchup_tokens, valid_from)``。

    · window_blocks <= 0（全上下文档）：缺口原样返回、valid_from = 0。
      调用方那时 window_blocks=0，根本不读 valid_from（走老路径），
      所以这条分支与改动前的行为逐值一致。
    · window_blocks  > 0（滑窗档）：
        start, gap = clip_gap_to_window(缺口, bs, M, token_ids)   ← 只前移、且
                     返回的 gap 恒等于 token_ids[start:end]
        valid_from = window_valid_from(draft_valid_len, bs, M)     ← 只认 wvf

      契约（调用方 propose_batch 依赖）：``catchup_tokens[s]`` 落在位置
      ``catchup_start + s`` 上。任何破坏它的写法都会静默写错 KV。

    ★ 为什么 valid_from 不取 max(wvf, 补齐起点)：
      补齐起点 = 「最早缺失的位置」，就在 len(seq) 附近；拿它当可见下界会把
      窗口砍成大约一个块（与 M 无关）。而缺口之前的内容并没有失效，见
      window_valid_from 的说明。补齐起点被夹到窗口外时也不需要兜：
      b0 的自然下界 (pos//bs - M + 1) 恒 ≥ start//bs（因为 pos ≥ end）。
    """
    start, gap = catchup_gap(draft_valid_len, token_ids)
    if window_blocks <= 0:
        return start, gap, 0
    start, gap = clip_gap_to_window(start, gap, block_size, window_blocks, token_ids)
    end = len(token_ids) - 1
    assert len(gap) == end - start, (
        f"窗口补齐区间与起点不自洽: start={start} len(gap)={len(gap)} end={end}")
    return start, gap, window_valid_from(draft_valid_len, block_size, window_blocks)


class DraftModelProposer:
    """用小模型自回归生成 k 个候选，返回 token 和它们在 draft 下的真实概率。

    对外入口就一个：`propose_batch(reqs)`（`propose` 是它的 B=1 兼容包装）。
    """

    def __init__(self, model: nn.Module, k: int = 4, block_size: int = 256):
        self.model = model
        self.k = k
        self.block_size = block_size
        self.kv_cache: Optional[torch.Tensor] = None
        self.n_layers_bound = 0
        # 由 ModelRunner 注入的静态形状 CUDA graph：{batch -> (graph, graph_vars)}
        self._graphs: Dict[int, tuple] = {}

        # ---------- 统计（结构验收用：必须能看出「每轮是 k 次批量前向」）----------
        self.n_graphed = 0            # 走图的前向次数
        self.n_eager = 0              # 退回 eager 的前向次数
        self.n_rounds = 0             # propose_batch 被调用的轮数
        self.n_batch_forwards = 0     # 提议阶段的前向次数（理想 = 轮数 × k）
        self.n_catchup_forwards = 0   # 补齐阶段的前向次数
        self.n_catchup_tokens = 0     # 补齐写进 draft KV 的 token 数（计入运行时间）
        self.max_batch_seen = 0       # 观察到过的最大请求数

    # ------------------------------------------------------------------
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> int:
        """把 draft 各层的 k_cache/v_cache 接到独立显存块上。

        与 target 的做法一致（见 model_runner.allocate_kv_cache）：
        遍历带 k_cache/v_cache 属性的子模块，按顺序绑定。
        返回绑定的层数，供调用方校验显存算得对不对。
        """
        self.kv_cache = kv_cache
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        self.n_layers_bound = layer_id
        return layer_id

    def cache_bytes(self, n_kv_heads: int, head_dim: int, dtype_itemsize: int) -> int:
        """draft 一块 KV cache 占多少字节/block（用于切分显存预算）。"""
        return (2 * self.n_layers_bound * self.block_size
                * n_kv_heads * head_dim * dtype_itemsize)

    def reset_watermark(self, token_ids):
        """draft 路线不需要 n-gram 索引。

        Scheduler 会对两种提议器统一调用这两个方法（n-gram 需要维护索引，
        draft 不需要），所以这里给个空实现保持接口一致。
        """
        pass

    def observe(self, token_ids):
        """同上：无索引可维护。"""
        pass

    # ------------------------------------------------------------------
    def bind_cudagraph(self, graphs: Dict[int, tuple]) -> None:
        """接入 ModelRunner 拍好的 draft decode 图集合：{batch -> (graph, vars)}。

        ★ 为什么必须有这一步（实测，不是理论）：
            draft 模型 0.6B，权重 1.2 GB。按 3090 的 936 GB/s 算，
            一次前向【读一遍权重】只要 1.3 ms。
            但 eager 实测 24.8 ms/步 —— 多出来的 23 ms 全是
            kernel launch + Python dispatch + 各种 tensor 创建。
            draft 要串行跑 k 次，这笔开销要乘 k，直接吃掉全部收益。

            拍成图之后，这 k 次前向各塌缩成一次 graph.replay()。
            批量版更进一步：一次 replay 就把 B 条请求一个候选位置算完。
        """
        self._graphs = dict(graphs or {})

    # ==================================================================
    #  对外入口
    # ==================================================================
    @torch.inference_mode()
    def propose(
        self,
        seq_block_table,
        context_len: int,
        last_token: int,
        temperature: float,
    ) -> Tuple[List[int], torch.Tensor]:
        """单请求提议（B=1 兼容入口，等价于 propose_batch([req])）。

        注意：这个入口不携带 draft KV 的有效水位信息，因此不做补齐。
        引擎路径统一走 propose_batch（那里会传 catchup_start/catchup_tokens）。
        """
        req = dict(block_table=list(seq_block_table), context_len=int(context_len),
                   last_token=int(last_token), temperature=float(temperature),
                   catchup_start=int(context_len) - 1, catchup_tokens=[])
        chains, logits = self.propose_batch([req])
        return chains[0], logits[0]

    @torch.inference_mode()
    def propose_batch(self, reqs: List[dict]) -> Tuple[List[List[int]], torch.Tensor]:
        """批量提议：B 条请求，每个候选位置一次批量前向，共 k 次。

        参数
        ----
        reqs : List[dict]，每个 dict 的键：
            block_table    : list[int]  该请求在【draft 自己的 KV cache】里的块表
            context_len    : int        已确认的 token 数（= len(seq)）
            last_token     : int        最后一个已确认 token
            temperature    : float      采样温度（逐请求）
            catchup_start  : int        draft KV 里从哪个位置开始不可信（有效水位）
            catchup_tokens : list[int]  要补写进 draft KV 的已确认 token
                                        （位置 = catchup_start .. catchup_start+len-1）

        返回
        ----
        chains : List[List[int]]        B × k 个候选 token
        logits : (B, k, vocab) 张量     每个候选位置在【draft 模型】下的原始 logits

        ★★★ logits 必须是【原始 logits】，不能是 softmax 之后的概率！
             原因：verify_batch 内部会自己做 softmax(logits / temperature)
             得到 q。如果这里先 softmax 了、那边再 softmax 一次，
             等于验证用的 q̃ = softmax(q) —— 会被压成接近【均匀分布】
             （实测 0.6B 的 151936 词表上，q̃ 的最大值只有 1.2e-5，
              而 1/vocab = 6.6e-6）。
             后果：草稿 token 是从真分布 q 采的，验证却用 q̃，
             拒绝采样的一致性前提被破坏 —— 输出分布不再等于目标分布。
        """
        B = len(reqs)
        assert B >= 1, "propose_batch 至少要 1 条请求"
        k = self.k
        dev = next(self.model.parameters()).device
        temps = torch.tensor([float(r["temperature"]) for r in reqs],
                             dtype=torch.float32, device=dev).view(B, 1)

        # ---------- 阶段 0：补齐 draft KV 缺口（gate 关闭若干步后重启才会发生）----------
        # 每轮把「还有缺口」的请求挑出来，按缺口的下标逐层补齐。
        # 同一条请求内部必须顺序补（位置 p 依赖 p-1 的 KV），
        # 不同请求之间可以并行 —— 所以按「第 s 个缺口位」分组批量前向。
        gaps = [list(r.get("catchup_tokens") or []) for r in reqs]
        max_gap = max((len(g) for g in gaps), default=0)
        if max_gap:
            base = [int(r.get("catchup_start", 0)) for r in reqs]
            for s in range(max_gap):
                act = [i for i in range(B) if s < len(gaps[i])]
                if not act:
                    break
                positions = [base[i] + s for i in act]
                # 补齐期间，环里可信的区间从补齐起点 base[i] 开始（见 clip_gap_to_window）
                geo = [self._geom(reqs[i], positions[j], valid_from=base[i])
                       for j, i in enumerate(act)]
                self._forward_group(
                    tokens=[gaps[i][s] for i in act],
                    positions=positions,
                    block_tables=[g[1] for g in geo],
                    context_lens=[g[2] for g in geo],
                    slot_mapping=[g[0] for g in geo],
                )
                self.n_catchup_forwards += 1
                self.n_catchup_tokens += len(act)

        # ---------- 阶段 1..k：k 个候选位置，逐个批量前向 ----------
        # ★ 关键：采样结果【留在 GPU 上】直接喂给下一步的输入缓冲区，
        #   中间不做 .item() —— 否则每步一次 device→host 同步，
        #   k 步就是 k 次 pipeline stall，图带来的收益会被同步开销吃掉一半。
        #   只在最后 .tolist() 时同步一次。
        vocab = self._vocab_size()
        tokens = torch.tensor([int(r["last_token"]) for r in reqs],
                              dtype=torch.int64, device=dev)      # (B,)
        start = [int(r["context_len"]) - 1 for r in reqs]          # (B,)
        out_logits = torch.empty(B, k, vocab, dtype=torch.float32, device=dev)
        cands = torch.empty(B, k, dtype=torch.int64, device=dev)

        for s in range(k):
            positions = [start[i] + s for i in range(B)]
            geo = [self._geom(reqs[i], positions[i]) for i in range(B)]
            logits = self._forward_group(
                tokens=tokens,
                positions=positions,
                block_tables=[g[1] for g in geo],
                context_lens=[g[2] for g in geo],
                slot_mapping=[g[0] for g in geo],
            )
            self.n_batch_forwards += 1
            # ★ 必须立刻落进持久缓冲区：走图时 logits 是静态缓冲区，下一次 replay 会被覆写
            out_logits[:, s].copy_(logits.float())
            # ★★ 温度必须沿【请求维】广播：(B,V) / (B,1) —— 不能把 (B,) 塞进 vocab 维
            p = torch.softmax(logits.float() / temps, dim=-1)
            noise = torch.empty_like(p).exponential_(1.0).clamp_min_(1e-10)
            tokens = (p / noise).argmax(dim=-1)                    # 留在 GPU 上，不 sync
            cands[:, s] = tokens

        self.n_rounds += 1
        self.max_batch_seen = max(self.max_batch_seen, B)
        # 全程唯一一次同步
        return cands.tolist(), out_logits

    # ==================================================================
    #  内部：一次批量 decode 前向（有图走图，没图走 eager，语义一致）
    # ==================================================================
    def _vocab_size(self) -> int:
        head = getattr(self.model, "lm_head", None)
        if head is not None and hasattr(head, "weight"):
            return head.weight.shape[0]
        # 兜底：从 logits 推（只在没有 lm_head 的包装模型上会走到）
        return getattr(self.model, "vocab_size", 151936)

    def _slot(self, block_table: List[int], pos: int) -> int:
        """位置 pos 的物理槽 = 块表[pos//bs] * bs + pos%bs（块表不够就给 -1）。"""
        bi, off = divmod(int(pos), self.block_size)
        if 0 <= bi < len(block_table):
            return int(block_table[bi]) * self.block_size + off
        return -1

    def _geom(self, r: dict, pos: int, valid_from: Optional[int] = None):
        """位置 pos 的 (slot, block_table, context_len)。

        窗口关闭（window_blocks == 0）时逐字等价于原来的写法：
            slot = block_table[pos//bs]*bs + pos%bs，context_len = pos + 1，
            block_table 原样传（= target 的块表）。
        窗口打开时走 DraftWindow 的环形算术（含 valid_from 下界保护）。
        """
        M = int(r.get("window_blocks") or 0)
        bs = self.block_size
        if M <= 0:
            bt = list(r["block_table"])
            return self._slot(bt, pos), bt, int(pos) + 1
        w = DraftWindow(r["draft_ring"], bs, M)
        vf = int(r.get("valid_from", 0)) if valid_from is None else int(valid_from)
        return w.slot(pos), w.block_table(pos, vf), w.ctx_len(pos, vf)

    def _forward_group(
        self,
        tokens,                 # (B,) device tensor 或 list[int]
        positions: List[int],
        block_tables: List[List[int]],
        context_lens: List[int],
        slot_mapping: List[int],
    ) -> torch.Tensor:
        """一次批量 decode 前向，返回 (B, vocab) logits。B = len(positions)。

        ★ 走图时返回的是【静态缓冲区】，调用方必须立刻 copy 走。
        """
        B = len(positions)
        entry = self._graphs.get(B)
        if entry is not None:
            graph, gv = entry
            width = gv["block_tables"].size(1)
            # 块表宽度超过图的静态宽度 → 退回 eager（正常不会发生）
            if all(len(bt) <= width for bt in block_tables):
                dev = gv["input_ids"].device
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.int64, device=dev)
                gv["input_ids"].copy_(tokens)
                gv["positions"].copy_(torch.tensor(positions, dtype=torch.int64, device=dev))
                gv["slot_mapping"].copy_(torch.tensor(slot_mapping, dtype=torch.int32, device=dev))
                gv["context_lens"].copy_(torch.tensor(context_lens, dtype=torch.int32, device=dev))
                gv["block_tables"].fill_(_PAD_BLOCK)
                for i, bt in enumerate(block_tables):
                    if bt:
                        gv["block_tables"][i, :len(bt)] = torch.tensor(
                            bt, dtype=torch.int32, device=dev)
                graph.replay()
                self.n_graphed += 1
                return gv["logits"]

        # ---------- eager ----------
        from nanovllm.utils.context import set_context, reset_context

        dev = next(self.model.parameters()).device
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64, device=dev)
        width = max(1, max(len(bt) for bt in block_tables))
        padded = [list(bt) + [_PAD_BLOCK] * (width - len(bt)) for bt in block_tables]
        set_context(
            False,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=dev),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=dev),
            block_tables=torch.tensor(padded, dtype=torch.int32, device=dev),
        )
        try:
            logits = self.model.compute_logits(
                self.model(tokens, torch.tensor(positions, dtype=torch.int64, device=dev)))
        finally:
            reset_context()
        self.n_eager += 1
        return logits
