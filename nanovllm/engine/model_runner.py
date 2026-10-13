import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.quant import apply_int8_quant


def _log_quant(which, path, stats):
    b = sum(s["bytes_before"] for s in stats)
    a = sum(s["bytes_after"] for s in stats)
    print(f"[int8] {which} 量化 {len(stats)} 个模块 "f"({os.path.basename(path.rstrip('/'))}): "
          f"{b/2**20:.1f} -> {a/2**20:.1f} MiB ({a/max(b,1):.3f}x), "
          f"rel_err max={max(s['rel_err'] for s in stats):.5f}")
# =====================================================================
# P9：预量化直载的统一入口（默认关闭 → 逐字节等于老路径）
# =====================================================================
def _log_quant_direct(which, path, stats):
    b = sum(s["bytes_before"] for s in stats)
    a = sum(s["bytes_after"] for s in stats)
    print(f"[int8] {which} 预量化直载 {len(stats)} 个模块 "
          f"({os.path.basename(path.rstrip('/'))}): "
          f"{b/2**20:.1f} -> {a/2**20:.1f} MiB ({a/max(b,1):.3f}x)  [无 GPU 侧 bf16]")


def _make_model(model_cls, hf_config, path, quant_weights, quant_modules,
                granularity, group_size, direct_load):
    """direct_load=False 时与「构造 + load_model + apply_int8_quant」逐字节等价。

    返回 (model, stats, direct)。direct=True 表示权重是【直载】进来的
    （已经跳过 bf16 阶段），调用方不要再跑 apply_int8_quant。
    """
    if quant_weights == "int8" and direct_load:
        from nanovllm.utils.loader import build_model_int8
        model, stats = build_model_int8(model_cls, hf_config, path, quant_modules,
                                        granularity, group_size)
        return model, stats, True
    model = model_cls(hf_config)
    load_model(model, path)
    # ★ 量化不由这里做 —— 调用方的 `if config.draft_quant_weights == "int8"`
    #   分支会照原样跑 apply_int8_quant。这里多做一次会把已经变成 int8 的
    #   权重再量化一遍（apply_int8_quant 直接断言拒绝）。
    return model, [], False
from nanovllm.spec_decode.draft_proposer import DraftWindow, window_realign_target
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
import json


# ===========================================================================
#  显存归因打点（P9）—— 只在 NANOVLLM_MEM_TRACE=1 时生效。
#  ★ 不设该环境变量时 _mem_snap 立即 return，不调用任何 CUDA API，
#    默认路径与改动前逐字节一致（老路径一行没删）。
#  ★ 全部打印【未舍入的原始字节数】，便于逐字节复算预算公式。
# ===========================================================================
_MEM_TRACE = os.environ.get("NANOVLLM_MEM_TRACE") == "1"


def _mem_snap(tag):
    """记一行 JSON 快照。used=total-free 是设备全局量（含 allocator 保留空间）；
    I=reserved-current（分配器内保留空闲）；O=used-reserved（分配器之外）。"""
    if not _MEM_TRACE:
        return None
    try:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        s = torch.cuda.memory_stats()
        c = s["allocated_bytes.all.current"]
        r = s["reserved_bytes.all.current"]
        p = s["allocated_bytes.all.peak"]
        rec = dict(tag=tag, free=int(free), total=int(total), used=int(total - free),
                   current=int(c), reserved=int(r), peak=int(p),
                   retained_idle=int(r - c), outside_allocator=int(total - free - r),
                   inactive_split=s.get("inactive_split_bytes.all.current"),
                   allocator=str(torch.cuda.memory.get_allocator_backend()))
        print("[mem] " + json.dumps(rec), flush=True)
        return rec
    except Exception as exc:                     # 打点失败绝不影响引擎
        print("[mem] " + json.dumps(dict(tag=tag, error=repr(exc))), flush=True)
        return None
# 分布式 rendezvous 端口。默认 "2333" —— 与原版逐字节一致（默认行为一行不变）。
# ★ 只有为了「同一台机器上并行跑多个互相独立的引擎实例」才需要覆盖它：
#   原版硬编码 tcp://localhost:2333，两个实例会抢同一个端口，只能串行。
#   给每个实例一个端口即可并行（不同 CUDA_VISIBLE_DEVICES + 不同 NANOVLLM_PORT）。
_RENDEZVOUS_PORT = os.environ.get("NANOVLLM_PORT", "2333")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", f"tcp://localhost:{_RENDEZVOUS_PORT}",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        _mem_snap("target_loaded")
        # ★ INT8：加载完之后【就地】替换（按 quant_config.json 点名的模块）。
        #   必须在 allocate_kv_cache 之前 —— KV 池大小按剩余显存算。
        self.quant_stats = []
        if config.quant_weights == "int8":
            self.quant_stats = apply_int8_quant(
                self.model, config.quant_modules,
                config.quant_granularity, config.quant_group_size)
            _log_quant("target", config.model, self.quant_stats)
        self.sampler = Sampler()
        # ---------- 投机解码 ----------
        self.spec_proposer = None
        self.draft_model = None
        self.draft_hf_config = None
        self._last_cu_seqlens_q = None
        # 投机路径的静态形状 CUDA graph（见 capture_*_cudagraph）
        #   draft 图：{B: (graph, vars)}，B = 精确 batch（一条请求一个 token）
        #   verify 图：{(B, k): (graph, vars)}，query 数 = B*(k+1)
        self.draft_graphs = {}
        self.verify_graphs = {}
        # B 步：draft 滑窗的块数 M（0 = 关闭，draft 与 target 共用 block_table）。
        # 真正的取值在 allocate_kv_cache 里按预算定；warmup 会先用旧路径跑一遍。
        self.draft_window_blocks = 0
        # P7-D：开头常驻段块数 P（0 = 不常驻 = 老行为）。同样在 allocate_kv_cache 定。
        self.draft_head_blocks = 0
        self._verify_extra = None
        # P6 消融开关（默认走批量；设 False 退回逐请求/仅 B=1 图）
        self.batch_draft = config.spec_batch_draft
        self.batch_verify_graph = config.spec_batch_verify_graph
        # P15：混合批的并发编排器（惰性建，只在真的用到时才 import torch.cuda.Stream）
        self._mixed = None
        # 图命中 / 退回统计（结构验收：B>1 的已覆盖形状必须【真的 replay 了图】）
        self.verify_graph_hits = 0
        self.verify_graph_fallbacks = {}
        # 图/缓冲区显存（capture 后量一次，供报告单列）
        self.graph_mem = {}

        if config.spec_k > 0 and config.spec_method == "ngram":
            from nanovllm.spec_decode.ngram_proposer import NgramProposer
            self.spec_proposer = NgramProposer(n=config.spec_ngram)
        elif config.spec_k > 0 and config.spec_method == "draft":
            # ★ draft model 路线：加载第二个模型
            #   前提：draft 与 target 必须共享词表，否则无法比较同一 token 的概率
            from transformers import AutoConfig
            from nanovllm.spec_decode.draft_proposer import DraftModelProposer
            self.draft_hf_config = AutoConfig.from_pretrained(config.draft_model)
            assert self.draft_hf_config.vocab_size == hf_config.vocab_size, (
                f"draft/target 词表不一致: {self.draft_hf_config.vocab_size} vs {hf_config.vocab_size}"
            )
            self.draft_model, self.draft_quant_stats, _direct = _make_model(
                Qwen3ForCausalLM, self.draft_hf_config, config.draft_model,
                config.draft_quant_weights, config.draft_quant_modules,
                config.draft_quant_granularity, config.draft_quant_group_size,
                config.quant_direct_load)
            _mem_snap("draft_loaded_before_quant")
            # ★ draft 的 INT8（与 target 独立）：量化 draft 在数学上完全安全 ——
            #   拒绝采样会把草稿猜错的地方修正回来，"相对 BF16 target 无损" 仍严格成立。
            if _direct:
                # ★ 直载：权重已经是 int8 了，绝不能再跑一次就地量化
                _log_quant_direct("draft", config.draft_model, self.draft_quant_stats)
            else:
                self.draft_quant_stats = []
                if config.draft_quant_weights == "int8":
                    self.draft_quant_stats = apply_int8_quant(
                        self.draft_model, config.draft_quant_modules,
                        config.draft_quant_granularity, config.draft_quant_group_size)
                    _log_quant("draft", config.draft_model, self.draft_quant_stats)
                _mem_snap("draft_after_quant")
            self.spec_proposer = DraftModelProposer(self.draft_model, k=config.spec_k,
                                                    block_size=self.block_size)
            print(f"[spec] draft model: {config.draft_model} "
                  f"({self.draft_hf_config.num_hidden_layers} 层, "
                  f"vocab {self.draft_hf_config.vocab_size})")

        self.warmup_model()
        # ★ 只在开了量化时做：把 warmup 期间 caching allocator 攒下的
        #   「已 free 但仍占着」的块还给驱动。allocate_kv_cache 的预算是
        #   `0.9*total - used - peak + current`，而 used 取的是【驱动级】
        #   已用量（含 allocator 缓存）—— 不还回去，量化省下的显存会被它吃掉。
        #   默认（bf16）路径一行不变。
        if config.quant_weights == "int8" or config.draft_quant_weights == "int8":
            torch.cuda.empty_cache()
        self.allocate_kv_cache()
        _mem_snap("after_allocate_kv_cache")
        if not self.enforce_eager:
            self.capture_cudagraph()
            # ★ 投机路径的两张图。必须放在 allocate_kv_cache 之后
            #   —— 图在 capture 时就把 k_cache/v_cache 的显存地址烘进去了。
            if self.config.spec_cuda_graph and self.spec_proposer is not None:
                if self.draft_model is not None:
                    self.capture_draft_cudagraph()
                self.capture_verify_cudagraph()
        torch.set_default_device("cpu")
        _mem_snap("after_capture")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _mem_snap("warmup_after_reset")
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        _mem_snap("warmup_after_prefill")
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        _mem_snap("kv_budget_entry")
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        budget = int(total * config.gpu_memory_utilization - used - peak + current)
        _mem_snap("kv_budget_entry_raw")

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        tgt_block_bytes = (2 * hf_config.num_hidden_layers * self.block_size
                           * num_kv_heads * head_dim * hf_config.dtype.itemsize)

        # ---------- 投机解码：draft 需要自己的 KV cache ----------
        # ★ 关键约束：draft 与 target 必须有【相同数量】的 block，
        #   因为 Sequence.block_table 是共享的 —— 逻辑 block i 指向物理块 i，
        #   两边的 cache 都用这个 i，只是落在不同的显存区。
        #   所以不能简单地"给 draft 分一块"，而要按【字节数】反推块数：
        #       N = 总预算 / (target_block_bytes + draft_block_bytes)
        draft_block_bytes = 0
        if self.draft_model is not None:
            d_hf = self.draft_hf_config
            d_kv_heads = d_hf.num_key_value_heads // self.world_size
            d_head_dim = getattr(d_hf, "head_dim", d_hf.hidden_size // d_hf.num_attention_heads)
            draft_block_bytes = (2 * d_hf.num_hidden_layers * self.block_size
                                 * d_kv_heads * d_head_dim * d_hf.dtype.itemsize)

        # ---------- B 步：draft 滑窗（新增分支，默认关闭）----------
        # 默认（spec_draft_window == 0）走下面这条【原有的】分法：
        #   draft 与 target 共用 block_table → 两边块数必须相同 → 按字节数反推。
        # 打开滑窗后走另一条：draft 只要 max_num_seqs * M 块（每序列私有的
        #   块级环形缓冲），target 拿走剩下的全部预算 → 容量上升。
        W = self._window_tokens()
        P = self._head_blocks()
        if draft_block_bytes and W > 0:
            assert W % self.block_size == 0, (
                f"spec_draft_window={W} 必须是 kvcache_block_size={self.block_size} 的整数倍")
            M = W // self.block_size
            # ★ P7-D：每条序列拿 (P + M) 块 —— 开头 P 块常驻 + 最近 M 块的环。
            #   P=0 时就是原来的 M*max_num_seqs，逐字节等价。
            draft_pool = (M + P) * config.max_num_seqs
            room = budget - draft_pool * draft_block_bytes
            assert room >= tgt_block_bytes, (
                f"draft 滑窗池放不下：{draft_pool} 块 x {draft_block_bytes/2**20:.0f} MiB "
                f"= {draft_pool*draft_block_bytes/2**30:.2f} GB，而 KV 预算只有 "
                f"{budget/2**30:.2f} GB（max_num_seqs={config.max_num_seqs}, M={M}, P={P}）。"
                f" 减小 spec_draft_window / spec_draft_head_blocks / max_num_seqs "
                f"或调大 gpu_memory_utilization")
            config.num_kvcache_blocks = int(room // tgt_block_bytes)
            config.num_draft_blocks = draft_pool
            config.draft_window_blocks = M
            config.draft_head_blocks = P
            self.draft_window_blocks = M
            self.draft_head_blocks = P
            print(f"[spec] draft 滑窗开启: W={W} token = {M} 块/序列 + 常驻 {P} 块, "
                  f"draft 池 {draft_pool} 块 ({(draft_pool*draft_block_bytes)/2**30:.2f} GB), "
                  f"target 池 {config.num_kvcache_blocks} 块")
        else:
            per_block_total = tgt_block_bytes + draft_block_bytes
            config.num_kvcache_blocks = budget // per_block_total
            config.num_draft_blocks = 0
            config.draft_window_blocks = 0
            config.draft_head_blocks = 0
            self.draft_window_blocks = 0
            self.draft_head_blocks = 0
        if _MEM_TRACE:
            _pb = int(tgt_block_bytes + draft_block_bytes)
            print("[mem] " + json.dumps(dict(
                tag="kv_budget_computed",
                gpu_memory_utilization=float(config.gpu_memory_utilization),
                total=int(total), used=int(used), peak=int(peak), current=int(current),
                budget=int(budget),
                tgt_block_bytes=int(tgt_block_bytes),
                draft_block_bytes=int(draft_block_bytes),
                per_block_total=_pb,
                blocks_floor=int(budget // _pb) if _pb else 0,
                budget_mod_block=int(budget % _pb) if _pb else 0,
                num_kvcache_blocks=int(config.num_kvcache_blocks))), flush=True)
        assert config.num_kvcache_blocks > 0, "KV cache 预算不足"

        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        # ---------- draft 的 KV cache ----------
        if self.draft_model is not None:
            d_hf = self.draft_hf_config
            d_kv_heads = d_hf.num_key_value_heads // self.world_size
            d_head_dim = getattr(d_hf, "head_dim", d_hf.hidden_size // d_hf.num_attention_heads)
            # 滑窗打开时 draft 池比 target 池小得多；关闭时两者相等（原有行为）。
            n_draft = config.num_draft_blocks or config.num_kvcache_blocks
            self.draft_kv_cache = torch.empty(
                2, d_hf.num_hidden_layers, n_draft,
                self.block_size, d_kv_heads, d_head_dim)
            dl = 0
            for module in self.draft_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, dl]
                    module.v_cache = self.draft_kv_cache[1, dl]
                    dl += 1
            assert dl == d_hf.num_hidden_layers, f"draft 层数不匹配: {dl} vs {d_hf.num_hidden_layers}"
            print(f"[spec] draft KV cache: {dl} 层 x {n_draft} blocks, "
                  f"每block {draft_block_bytes/1024:.0f} KB")
        n_draft_used = config.num_draft_blocks or config.num_kvcache_blocks
        kv_total_gb = (config.num_kvcache_blocks * tgt_block_bytes
                       + n_draft_used * draft_block_bytes) / 2**30
        print(f"[spec] blocks={config.num_kvcache_blocks}, "
              f"target 每block {tgt_block_bytes/1024:.0f} KB, "
              f"合计 {kv_total_gb:.2f} GB")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence], advance_draft: bool = True):
        """把本轮的 prefill 输入拼好、并设好 attention context。

        advance_draft=False（P12）：**不要**推进 draft KV 的有效水位 —— 调用方
        已经决定这轮不跑 draft 前向（跳过开关 + 确定不投机），水位一推就是
        "盖了个没写过的章"，之后翻回投机时 proposer 会拿脏 KV 提候选。
        默认 True ⇒ 老路径逐字节不变（warmup / 滑窗重建都走默认值）。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # ---------- P6 任务 D：维护 draft KV 的有效水位 ----------
            # draft_valid_len = 「从位置 0 起连续有效的 draft KV 个数」。
            # 只有【与已有有效前缀连续】的新写入才延长它：
            #   · 新请求（start=0）→ 直接延长到 end；
            #   · 分块 prefill 的后续 chunk（start == draft_valid_len）→ 延长到 end；
            #   · 前缀缓存命中（start > draft_valid_len）→ 水位【不动】。
            #     draft 侧的物理块可能残留「被拒绝候选」的 KV（见 12.5 任务 D），
            #     不能仅凭 target 前缀命中就假定 draft 侧也有效；
            #     保守留成缺口，让 proposer 在下次提议前补齐（补齐计入运行时间）。
            # ★ P12：本轮的 draft 前向被跳过时不盖章（见 advance_draft 的说明）。
            if advance_draft:
                seq.advance_draft_watermark(start, end)

            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ==================================================================
    #  投机解码：验证阶段
    # ==================================================================
    def prepare_verify(self, seqs: list[Sequence]):
        """一次 forward 算「1 个本该 decode 的 token + k 个待验证候选」。

        ★ 为什么可以复用 prefill 那条因果路径（不需要自定义mask）：
          这 k+1 个 token 在序列里是【连续的一段】，位置是 len-1 .. len+k-1。
          候选 i 能看到「已确认的部分 + 候选 0..i-1 + 自己」
          —— 这恰好就是标准的因果顺序（下标序）。
          所以 causal=True 天然正确，attention.py 一行都不用改。

          ★★ KV cache 的关键性质：
          候选的 KV 也会被写进 cache，但它们是【逻辑上死的】——
          下一步 slot_mapping = f(positions, block_table) 会重算，
          同一批物理 slot 被新内容覆写。所以【不需要真正的 KV 回滚】，
          这也是为什么 seq.draft_tokens 绝对不能进 token_ids。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            # ★ 实际送几个 = 「实际捞到的草稿数 + 1」，不是 num_scheduled_tokens
            #   （后者是按 spec_k 上限分配的 block 槽位数）。
            #   两者不一致会让 input_ids / positions / cu_seqlens_q 三者长度错位，
            #   RoPE 就会用越界的 positions 去索引 -> 输出乱码或重复 token。
            #   这正是 arXiv 2510.22876 说的「静默错误但速度正常」。
            n = 1 + len(seq.draft_tokens)
            start = len(seq) - 1                  # 最后一个已确认 token 的位置
            end = start + n                       # 验证到 len-1+n
            toks = seq.tokens_in(start, end)      # 已确认的 1个 + 草稿 k 个
            assert len(toks) == n, (f"toks={len(toks)} n={n} draft={seq.draft_tokens}")

            input_ids.extend(toks)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + n)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)     # ★ 含前缀，所以不等长
            max_seqlen_q = max(n, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)

            if not seq.block_table:              # warmup 路径
                continue
            # slot_mapping：这 n 个 token 各写到哪个物理槽
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 在把 python list 变成 tensor 之前先留一份标量：query 总数（= B*(k+1)）。
        # 图路径用它校验静态形状，且在 CPU 上算，不会引入 D2H 同步。
        total_q = cu_seqlens_q[-1]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 存一份给 run_verify 切分 logits 用
        self._last_cu_seqlens_q = cu_seqlens_q.tolist()
        # 存一份给「验证前向走 CUDA graph」用：
        #   图有静态缓冲区，这些张量要原样拷进去再 replay。
        #   ★ 索引怎么算只在这里算一次 —— 图路径和 eager 路径共用同一份计算，
        #     避免两条路径各算一套导致错位（那就是「不报错但结果错」）。
        # ★ 批量验证图（P6 任务 C）只在「所有参与请求都有完整 k 个候选」时可用：
        #   固定 k → query 数恒为 B*(k+1)，形状才是静态的。候选变短（n_i 不一致）
        #   会让 cu_seqlens_q 的步长不齐，那种形状退回 eager 并记录原因，
        #   不用补零 logits/概率强行凑图（补零会污染拒绝采样）。
        kk = self.config.spec_k
        full_k = all(len(s.draft_tokens) == kk for s in seqs)
        self._verify_extra = dict(
            n=cu_seqlens_q.numel() - 1,
            k=kk, full_k=full_k,
            total_q=total_q,            # python int（上面已留），不引入 D2H 同步
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping, block_tables=block_tables,
        )
        # ★ is_prefill=True —— 让 attention 走 flash_attn_varlen_func(causal=True)，
        #   这正是我们要的因果路径；不能用 decode 那条（它只处理 1 个 query）。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                     slot_mapping, None, block_tables, is_spec_verify=True)
        return input_ids, positions

    def _window_tokens(self) -> int:
        """配置里要求的 draft 滑窗宽度 W（0 = 关闭 = 全上下文 draft）。

        ★ 现场读 config：同一份二进制里翻转它就能做「全上下文 vs W」的消融，
          不需要切 git 版本（否则测出来的差异会混进代码版本差异）。
        """
        w = int(getattr(self.config, "spec_draft_window", 0) or 0)
        if w <= 0 or self.draft_model is None:
            return 0
        return w

    def _head_blocks(self) -> int:
        """P7-D：draft 上下文里【开头常驻】的块数 P（0 = 不常驻 = 老行为）。

        ★ 和 spec_draft_window 不同，这个开关决定池子大小（(P+M)*max_num_seqs），
          所以它在 init 期定死、运行期不再翻；`allocate_kv_cache` 会把它存进
          `self.draft_head_blocks`，运行路径只读那个字段。
        ★ 只在滑窗开着（W>0）时才有效。
        """
        if self._window_tokens() <= 0:
            return 0
        return max(0, int(getattr(self.config, "spec_draft_head_blocks", 0) or 0))

    def _run_draft_prefill(self, seqs, input_ids, positions) -> bool:
        """prefill 时也把 draft 模型跑一遍，填它自己的 KV cache。

        ★ 全上下文路径（默认，spec_draft_window == 0）：逐字保留原实现 ——
          直接复用当前 context（prepare_prefill 已经设好了），跑完【不恢复】，
          因为 target 的 prefill 紧接着会自己再 set_context 一次。
          返回 False 表示「没动 context」。

        ★ 滑窗路径（spec_draft_window > 0 且这批序列都已经拿到滑窗块）：
          转给 _run_draft_prefill_window，它会自己 set/reset context，
          所以返回 True，调用方要重建 target 的 context。
        ★ 优雅回退（两边都能退）：只要有一条序列没有滑窗块（例如池子分配失败、
          或运行期把开关翻回去），整批就走全上下文路径 —— 不会半新半旧。
        """
        # ★ 还要现场读一次 spec_draft_window：把它翻回 0（全上下文）时，
        #   已经分到的滑窗块就不再被使用 —— 开关在两个方向上都能即时生效。
        if (self.draft_window_blocks and self._window_tokens() > 0
                and seqs and all(s.draft_block_table for s in seqs)):
            # ★ 分块 prefill（P13，默认关闭）：只有【这条序列本步就把 prompt 拼完了】
            #   才值得跑窗口 prefill —— 窗口路径按 token_ids 重算整个窗口，
            #   中间块每块都重跑一遍 = N 倍浪费；而 draft 的 KV 只有在这条序列
            #   真正开始解码时才用得上（那时 prompt 已经拼完）。
            #   ★ 全部拼完时（= 不分块时的常态）走的是原分支，逐字节不变。
            ready = [s for s in seqs
                     if s.num_cached_tokens + s.num_scheduled_tokens >= s.num_tokens]
            if len(ready) == len(seqs):
                self._run_draft_prefill_window(seqs)
                return True
            if not ready:
                return False                      # 中间块：先不建 draft KV
            self._run_draft_prefill_window(ready)  # 只给拼完的那些建
            return True
        with torch.inference_mode():
            self.draft_model(input_ids, positions)
        return False

    def _draft_prefill_needed(self) -> bool:
        """本批 prefill 还要不要建 draft KV（P13，默认关闭 = 恒 True，老路径不变）。

        打开后：`spec_k <= 0` ⇒ 本引擎【这一轮绝不会投机】⇒ 这趟 0.6B 前向是纯浪费。
        （`model_runner.run()` 原来那句 `if self.draft_model is not None` 不看 spec_k，
        只要进程加载了 draft 就白跑一遍 —— 见 docs/端到端验收.md §6 与本轮上界表。）

        ★ P12 补：`spec_batch_threshold` 那种按【解码批大小】判的门控，在 prefill
          时刻确实看不到**未来**的解码批 —— 但看得到【此刻已经在解码的批】，
          而下一步解码批 ≥ 它。所以只要它已经 > 阈值，下一步的解码轮一定还关着
          投机 ⇒ 这趟 draft prefill 是确定白跑的，一起跳过。
          （Scheduler 每轮把它写进 `config._sched_running_decode_b`。）
          ★ 边界（诚实）：抢占（preempt）可能让解码批临时缩回阈值以内，那时投机
            会重新打开 —— 但这只是"少跳过一趟"，draft KV 由 proposer 正常 catchup
            补齐（P12 已把"跳过时不许盖章"修掉），不会出错。
        """
        if not getattr(self.config, "spec_skip_draft_prefill_when_off", False):
            return True
        if self.config.spec_k <= 0:
            return False
        thr = int(getattr(self.config, "spec_batch_threshold", 0) or 0)
        if thr and int(getattr(self.config, "_sched_running_decode_b", 0) or 0) > thr:
            return False
        return True

    @torch.inference_mode()
    def _run_draft_prefill_window(self, seqs):
        """滑窗路径的 draft prefill：跑「开头 P 块（常驻） + 最近 M 个块」。

        ★ 为什么不是整个 prompt：滑窗只保留最近 M 个块，更老的 token 反正会被
          环覆盖，跑它们纯属浪费。按【块边界】对齐地取最后 M 个块 [b0*bs, end)，
          一次 varlen 前向算完。
        ★ P7-D（P>0）：再多喂开头 P 个块 [0, P*bs) —— 它常驻、永不轮转，
          给 draft 一段「全局锚点」。两段按【绝对位置递增】顺序拼成一个 chunk，
          chunk 内的 local-causal 掩码于是恰好等于「开头 ∪ 尾部窗口」的可见集：
          尾部 query 能看到开头 + 它之前的尾部，开头 query 只能看到更早的开头。
          ★ 段间若有空档（b0 > P），chunk 里把空档【跳过】——被跳过的位置既不读
            也不写（它们的 KV 是陈旧的，但 b0 ≥ P 保证谁也读不到它们）。
        ★ 块表必须旋转：paged 解码要求 key index j ↔ block_table[j//bs]，
          且必须【按绝对块号递增】，causal 掩码才对（见 DraftWindow）。
        ★ RoPE：positions 用【绝对位置】—— 丢的是 KV，不是位置。
        ★ 诚实的近似：这一趟是「把超出窗口的上下文直接截掉」，
          所以窗口里靠前的那些位置拿到的是缩短了的上下文（比流式滑窗更短）。
          每向前解码一步窗口就往前挪，这些位置很快就会滑出去。
        """
        bs = self.block_size
        M = self.draft_window_blocks
        P = max(0, int(getattr(self, "draft_head_blocks", 0) or 0))
        ids, poss, slots, cu = [], [], [], [0]
        maxlen = 0
        for seq in seqs:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            if end <= 0:
                continue
            w = DraftWindow(seq.draft_block_table, bs, M, P)
            if getattr(self.config, "spec_draft_win_align", False):
                # ★ 修复开关（默认关）：尾部窗口对齐到【第一个生成位置 end】
                #   所在的块 —— 也就是解码期绝大多数查询会用的那个 b0，
                #   而不是最后一个 prompt 位置 end-1 的 b0（两者在 end 是
                #   块边界时差一个块；差异正是「滑动」这一次对齐漂移）。
                b0 = w.b0(end)
                seq.draft_win_start = b0 * bs
            else:
                b0 = w.b0(end - 1)               # 尾部窗口起点块（恒 ≥ P）
            head_end = min(P * bs, end)          # 常驻段覆盖 [0, head_end)
            w0 = b0 * bs                         # 尾部窗口起点
            # 两段：[0, head_end) 与 [w0, end)；w0 ≥ P*bs 保证两段不重叠。
            segs = []
            if head_end > 0:
                segs.append((0, head_end))
            if w0 < end:
                segs.append((w0, end))
            if not segs:
                continue
            for (a, b) in segs:
                ids.extend(seq.token_ids[a:b])
                poss.extend(range(a, b))
                slots.extend(w.slot(p) for p in range(a, b))
                cu.append(cu[-1] + (b - a))
                maxlen = max(maxlen, b - a)
            # 窗口是整段重算的 → 从 0 到 end 都算「已处理」（常驻段 + 尾部都写了）
            seq.draft_valid_len = end
        if not ids:
            return
        dev = next(self.draft_model.parameters()).device
        cu = torch.tensor(cu, dtype=torch.int32, device=dev)
        # ★★★ 这里刻意【不】传 block_tables，让 attention 走「本地 k/v + 因果」那条路：
        #   · 这一趟的 chunk 就是「最近 M 个块」，长度 ≤ M*bs，正好是窗口的全部内容；
        #   · 本地因果注意力给每个 query 的可见区间 = [w0, q]，与「窗口内因果」等价，
        #     而且索引顺序天然对齐（不需要旋转块表，也没有 key index ↔ 槽位的映射）；
        #   · store_kvcache 仍然按 slot_mapping 把 KV 写进环形缓冲，后续 decode
        #     照常通过块表读 —— 写和读的职责分开了。
        #   实机教训：之前用带 block_table 的 paged 路径跑这一趟，M=8（窗口盖住整个
        #   prompt、本该与全上下文档逐步等价）的接受率却只有 0.68 vs 0.88 ——
        #   说明 paged 路径在这一处与本地路径不等价。走本地路径后该不该等价是可判定的。
        set_context(True, cu, cu, maxlen, maxlen,
                    torch.tensor(slots, dtype=torch.int32, device=dev), None, None)
        try:
            self.draft_model(torch.tensor(ids, dtype=torch.int64, device=dev),
                             torch.tensor(poss, dtype=torch.int64, device=dev))
        finally:
            reset_context()

    # ------------------------------------------------------------------
    # ★ spec_draft_win_realign（默认关）：第二次及以后的「对齐漂移」修复
    # ------------------------------------------------------------------
    def _maybe_realign_draft_window(self, seqs) -> int:
        """检测并修掉「第二次及以后的漂移」（默认关 = 一行不执行）。

        `spec_draft_win_align` 只把【第一次】漂移对齐了：prefill 写下环里那些块时
        用的起点 = 第一个生成位置所在块的 b0。但解码继续往前、序列跨进下一个块时，
        `DraftWindow.b0` 的自然项 `p//bs − M + 1` 会超过那次对齐的地板 ——
        环里被保留的块仍编码着「更早一个块」的上下文，与读端声明的窗口不一致。
        实测（8K/out=512，引擎内确定性 α）：ALIGN=1, P=0, M=5 从 **0.644 → 0.458**。

        修法：一旦检测到自然起点要前进，就用【新起点】把这扇窗口重算一遍
        （一次 local-causal 前向，≤ M*bs 个 token），并把地板推进到新起点 ⇒
        「环里的内容 = 读端声明的窗口」这条不变量在【每个】块上都成立，而不是只在第一块上。

        ★ 默认关（`spec_draft_win_realign=False`）时整个函数第一行就返回 0，
          老路径一个字节都不动。返回本次重算的序列条数（供校验/计数）。
        """
        if not getattr(self.config, "spec_draft_win_realign", False):
            return 0
        if not self.draft_window_blocks or self._window_tokens() <= 0:
            return 0
        M = self.draft_window_blocks
        P = max(0, int(getattr(self, "draft_head_blocks", 0) or 0))
        bs = self.block_size
        todo = []
        for seq in seqs:
            if not seq.draft_block_table:
                continue
            end = len(seq)
            need, new_start = window_realign_target(
                getattr(seq, "draft_win_start", 0), end, bs, M, P)
            if not need:
                continue
            seq.draft_win_start = new_start
            w = DraftWindow(seq.draft_block_table, bs, M, P)
            todo.append((seq, w, new_start // bs, end))
        if todo:
            self._realign_draft_windows(todo)
        return len(todo)

    @torch.inference_mode()
    def _realign_draft_windows(self, todo):
        """按【新的对齐起点】把若干条序列的滑窗重算一遍（只被上面的开关调用）。

        与 `_run_draft_prefill_window` 同一套写法（本地 k/v + 因果、按 slot 写入环），
        唯一区别是起点由调用方给定（新的 b0 块），而不是 `w.b0(end-1)`。
        ★ 这个函数【只在 spec_draft_win_realign 打开时才会被调用】，默认路径不碰它。
        """
        bs = self.block_size
        ids, poss, slots, cu = [], [], [], [0]
        maxlen = 0
        for (seq, w, b0, end) in todo:
            P = w.P
            head_end = min(P * bs, end)
            w0 = b0 * bs
            segs = []
            if head_end > 0:
                segs.append((0, head_end))
            if w0 < end:
                segs.append((w0, end))
            if not segs:
                continue
            for (a, b) in segs:
                ids.extend(seq.token_ids[a:b])
                poss.extend(range(a, b))
                slots.extend(w.slot(p) for p in range(a, b))
                cu.append(cu[-1] + (b - a))
                maxlen = max(maxlen, b - a)
            # 整段重算 ⇒ 从 0 到 end 都算「已处理」（与窗口 prefill 的语义一致）
            seq.draft_valid_len = end
        if not ids:
            return
        dev = next(self.draft_model.parameters()).device
        cu = torch.tensor(cu, dtype=torch.int32, device=dev)
        set_context(True, cu, cu, maxlen, maxlen,
                    torch.tensor(slots, dtype=torch.int32, device=dev), None, None)
        try:
            self.draft_model(torch.tensor(ids, dtype=torch.int64, device=dev),
                             torch.tensor(poss, dtype=torch.int64, device=dev))
        finally:
            reset_context()

    def _draft_request(self, seq: Sequence) -> dict:
        """把一条请求的 draft 提议元数据打包给 proposer（P6 任务 D 的接口）。

        catchup 的语义：draft KV 里 [draft_valid_len, len(seq)-1) 这一段是缺口
        —— 通常是 gate 关闭的若干步里 target 自己生成的 token（draft 没跑），
        也可能来自前缀缓存命中的块。补齐必须【喂这些 token 本身】，
        只喂最后一个 token 恢复不了缺失前缀。补齐费用计入运行时间。

        ★ 滑窗档的算术全部收在 draft_proposer.window_request_geom 里
          （纯函数、可 CPU 单测），这里只负责把整条序列的滑窗块表拿对。
        """
        from nanovllm.spec_decode.draft_proposer import window_request_geom
        # ---------- 滑窗（B）：draft 的块表是每序列私有的环形缓冲 ----------
        # ★ 判据是【这条序列有没有滑窗块表】，而不是读配置：池子分配失败、或者
        #   运行期把开关翻回全上下文时，这里自动退回老路径（两边都能退）。
        # ★ 现场读开关：spec_draft_window 翻回 0 时，就算池子里还给这条序列留着
        #   滑窗块，也一律退回全上下文路径（两个方向都能即时生效）。
        # ★ P7-D：块表现在是 (P+M) 项 —— P 段常驻 + M 段环。M 从 self 读（不是
        #   len(table)），否则会把常驻段数当成环长。
        M = self.draft_window_blocks if self._window_tokens() > 0 else 0
        P = max(0, int(getattr(self, "draft_head_blocks", 0) or 0)) if M else 0
        start, catchup, valid_from = window_request_geom(
            seq.draft_valid_len, seq.token_ids, self.block_size, M, P)
        if M and getattr(self.config, "spec_draft_win_align", False):
            # ★ 窗口下界的地板 = prefill 那一刻写下环里那些块的起点
            #   ⇒ 读端声明的可见区间与那些块被写下时的上下文【同一个对齐】。
            #   默认关时这一行不执行，老路径逐字节不变。
            valid_from = max(valid_from, int(getattr(seq, "draft_win_start", 0)))
        return dict(block_table=list(seq.block_table),
                    draft_ring=list(seq.draft_block_table) if M else [],
                    window_blocks=M, head_blocks=P, valid_from=valid_from,
                    context_len=len(seq),
                    last_token=seq.last_token, temperature=seq.temperature,
                    catchup_start=start, catchup_tokens=catchup)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _prefill_forward(self, seqs: list[Sequence]):
        """prefill 的【发射半段】：把前向与 lm_head 排进【当前 stream】，不回落到 CPU。

        ★ 为什么要拆（P15，见 engine/mixed_batch.py）：要让 prefill 与 decode 真重叠，
          必须在【任何一个 .tolist() 之前】把两条路径的 kernel 都发射出去 ——
          .tolist() 会同步当前 stream，把它放在发射阶段就等于自己取消重叠。
          所以 prefill 拆成"发射"（本函数）与"采样"（调用方）：
            · 老路径 `run()` 紧跟着调用采样 ⇒ 与拆分前逐字节一致；
            · 重叠路径把本函数放进 aux stream，跑完 decode 再来采样。
        返回 (logits, temperatures)；rank != 0 时 temperatures 为 None（与老路径一致）。
        ★ 本函数【故意不带】@torch.inference_mode()：原来的 prefill 分支没有它
          （只有 run_model / 两张图的 capture 有），加了会让 prepare_prefill
          造出来的 input_ids / temperatures 变成 inference tensor，被
          `self.sampler(...)`（在 inference_mode 之外）原地算子碰到会直接报错。
          不改它 = "老路径逐字节不变"这句话不用打折扣。
        """
        # ★★ P12 修：`prepare_prefill` 里的 `advance_draft_watermark` 会把
        #   draft KV 的"有效水位"推到本 chunk 的末尾 —— 它是个**盖章**，
        #   前提是"这趟 draft 前向跑了"。P13 的跳过开关让 draft 前向不跑，
        #   但这个章还是照盖：水位被推到 end，而 draft 的 KV 一个字没写。
        #   后果（实测）：运行期把 spec_k 从 0 翻回正数后，proposer 相信
        #   水位、不做补齐，拿【未初始化】的 draft KV 提候选 ⇒ 接受率崩
        #   （每轮交付 2.77 → 1.21 个 token），而 verify 前向照付 ⇒ 比不投机还慢。
        #   ⇒ 跳过 draft prefill 的那一轮，**不盖这个章**（水位不动），
        #     之后真正需要时由 proposer 的正常 catchup 路径补齐。
        #   ★ 默认路径（开关关）`_draft_prefill_needed()` 恒 True ⇒ 与老行为逐字节一致。
        draft_needed = self._draft_prefill_needed()
        input_ids, positions = self.prepare_prefill(seqs, advance_draft=draft_needed)
        if self.draft_model is not None:
            # ★ draft 模型也要跑一遍 prefill，把它的 KV cache 建起来
            #   （关闭滑窗时 draft 与 target 共用 block_table，但各自写进自己的 cache；
            #    打开滑窗时 draft 用自己的环形块表，见 _run_draft_prefill_window）
            #   ★ P13：`_draft_prefill_needed()` 默认恒 True（老路径一行不变）；
            #     打开开关且本引擎确定不投机（spec_k<=0）时整趟跳过。
            if draft_needed and \
                    self._run_draft_prefill(seqs, input_ids, positions):
                # 滑窗路径 set 过自己的 context 又 reset 了 → 重建 target 的
                input_ids, positions = self.prepare_prefill(seqs)
        elif self.spec_proposer is not None:
            # n-gram 路线：把 prompt 的 n-gram 建进索引
            for seq in seqs:
                self.spec_proposer.reset_watermark(seq.token_ids)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, True)
        return logits, temperatures

    def run_mixed(self, decode_seqs: list[Sequence], prefill_seqs: list[Sequence],
                  mode: str):
        """P15：一个 step 内跑两条执行路径。返回 (decode_token_ids, prefill_token_ids)。

        ★ 为什么单独一个入口而不是给 `run()` 加参数：`run()` 是纯 decode / 纯 prefill
          的既有热路径（纯 decode step 必须逐字节不变、图照常命中，见 I5）。
          把混合逻辑挡在它外面，`run()` 就一行都不用改。
        """
        if self._mixed is None:
            from nanovllm.engine.mixed_batch import MixedForward
            self._mixed = MixedForward(self)
        return self._mixed.run(decode_seqs, prefill_seqs, mode)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list:
        """返回值的形态随路径不同：
           prefill        -> list[int]              每序列 1 个
           普通 decode    -> list[int]              每序列 1 个
           投机验证       -> list[list[int]]        每序列若干个
        """
        if is_prefill:
            # ★★ P15：prefill 拆成"发射"与"采样"两半。老路径连着调用两半，
            #   与拆分前逐字节一致；重叠路径把"发射"放进 aux stream、把"采样"
            #   推迟到 join 之后（任何在发射阶段的 .tolist() 都会同步当前流，
            #   自己把重叠取消掉）。理由与不变量见 engine/mixed_batch.py。
            logits, temperatures = self._prefill_forward(seqs)
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            return token_ids
        elif self.spec_proposer is not None and any(s.num_scheduled_tokens > 1 for s in seqs):
            # 先提议，再决定走验证还是退回普通 decode。
            k = self.config.spec_k
            to_prop = [s for s in seqs if s.num_scheduled_tokens > 1]
            for seq in seqs:
                if seq.num_scheduled_tokens <= 1:
                    # 本轮不参与投机的请求：草稿与 logits 都要清干净，否则会串用上一轮
                    seq.draft_tokens = []
                    seq.draft_logits = None
            if self.draft_model is not None and to_prop:
                # ★ spec_draft_win_realign（默认关）：先把「第二次及以后的漂移」
                #   修掉（检测 → 按新起点重算窗口 → 推进地板），再取请求元数据，
                #   否则 valid_from 读到的是旧地板。默认关时该函数第一行就返回 0，
                #   老路径一个字节不动。
                self._maybe_realign_draft_window(to_prop)
                reqs = [self._draft_request(s) for s in to_prop]
                if self.batch_draft:
                    # ---------- P6 任务 A：批量 draft ----------
                    # 同一候选位置 B 条请求一起前向 → k 次批量前向（原来是 B×k 次单序列）
                    chains, logits = self.spec_proposer.propose_batch(reqs)
                else:
                    # 消融开关：逐请求循环（= C 组行为，B×k 次单序列前向）
                    chains, logits = [], []
                    for r in reqs:
                        ch, lg = self.spec_proposer.propose_batch([r])
                        chains.append(ch[0])
                        logits.append(lg[0])
                for seq, ch, lg in zip(to_prop, chains, logits):
                    seq.draft_tokens = ch
                    seq.draft_logits = lg
                    # draft KV 有效水位 = 「最后已确认 token 的位置 + 1」再往后 k 个写入位置
                    # （propose 写了位置 len-1 .. len+k-2，所以有效个数 = len-1+k）
                    seq.draft_valid_len = len(seq) - 1 + len(ch)
            elif self.spec_proposer is not None:
                for seq in to_prop:
                    chains = self.spec_proposer.propose(seq.token_ids, k)
                    # 线性链一次 forward 只能验证一条 —— 要同时验证多条就得做树状
                    # 注意力（需要自定义 mask + 换 SDPA，见 README 的取舍说明）。
                    # 这里取【最长】的那条：长度直接决定能省几次 forward。
                    seq.draft_tokens = max(chains, key=len) if chains else []
            # ★ 全部都没捞到候选 -> 干净退回普通 decode。
            #   不退回的话会白白付一次 verify 的多位置 forward 成本，更慢。
            if all(len(s.draft_tokens) == 0 for s in seqs):
                for seq in seqs:
                    seq.draft_tokens = []
                input_ids, positions = self.prepare_decode(seqs)
                temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
                logits = self.run_model(input_ids, positions, False)
                token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
                reset_context()
                return token_ids
            return self.run_verify(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def run_verify(self, seqs: list[Sequence]) -> list[list[int]]:
        """投机验证：一次 forward 算 k+1 个位置，然后决定每条序列接受几个。

        流程
        ----
        1. n-gram 提议：从前缀里捞候选，塞进 seq.draft_tokens
        2. 一次 forward：算出这 1+k 个位置在【目标模型】下的分布
        3. 拒绝采样：草稿分布用「本该 decode 的那一步的分布」近似
           （真正的实现里草稿需要一次独立的 forward，这里为省一次前向，
             用 n-gram 提议 + 上一步分布作为近似 —— 见 README 的说明）
        """
        from nanovllm.spec_decode.verify import verify_batch

        # 注意：draft_tokens 已在 run() 里填好（且已增量入索引），这里不再重复提议。

        input_ids, positions = self.prepare_verify(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # ---------- 2. 目标模型一次 forward ----------
        logits = self.run_verify_forward(input_ids, positions)
        reset_context()

        if self.rank != 0:
            return None

        # ---------- 3. 逐序列做拒绝采样 ----------
        out: list[list[int]] = []
        cu = self._last_cu_seqlens_q
        # 切分边界自检：cu 的总长应等于 logits 的行数
        assert cu is not None and cu[-1] == logits.shape[0], (
            f"cu[-1]={cu[-1] if cu else None} vs logits rows={logits.shape[0]}, cu={cu}")
        for i, seq in enumerate(seqs):
            # ★ 必须和 prepare_verify 里用的是同一个 n，否则切片错位
            n = 1 + len(seq.draft_tokens)
            if n <= 1:
                # 没捞到候选：退回普通 decode，只取第 1 个位置
                sub = logits[cu[i]:cu[i] + 1]
                t = torch.tensor([seq.temperature], device=sub.device)
                tok = self.sampler(sub, t).tolist()[0]
                out.append([tok])
                continue
            # 该序列的 (1+k) 个位置的 logits：(1, 1+k, vocab)
            sub = logits[cu[i]:cu[i] + n].unsqueeze(0)
            draft_tok = torch.tensor([seq.draft_tokens], device=sub.device)   # (1, k)

            # ★★草稿分布怎么来的 —— 这是 n-gram 投机的核心设计
            #
            # n-gram 提议【不是一个模型】，它没有真实的输出分布。
            # 所以我们把它建模成【单点分布】：候选 token 的概率为 1，其余为 0。
            #
            # 这样接受概率 min(1, p_target / q_draft) = min(1, p_target / 1) = p_target
            #   → 接受与否【只取决于目标模型自己对该token 的置信度】
            #   → 完全符合直觉：目标模型很确定就接受，不确定就拒绝
            #
            # ❌ 之前的错误做法：用 sub[:, :1, :] 当草稿分布
            #    → q == p，接受概率恒等于 1 → 候选【全部被接受】
            #    → 输出退化成 n-gram 里捞到的原始片段，出现大量重复 token
            #
            # ⚠️ 严格版应该让草稿模型独立跑一次 forward 拿到真实 q。
            #    这里 n-gram 没有 forward 可跑，单点分布是数学上正确的建模。
            V = sub.shape[-1]
            kk = len(seq.draft_tokens)

            if seq.draft_logits is not None:
                # ---------- draft model 路线：真实分布 ----------
                # ★ 这才是标准投机解码：接受概率 min(1, p_target / q_draft)，
                #   q 是 draft 模型真实的输出分布。
                #   实测 n-gram 路线的接受率 <1%，根因就是 q 退化成了单点分布
                #   （接受概率 = p[draft]，通常 0.001~0.06）。
                #
                # ★★ 这里传的是【原始 logits】，不是概率。
                #   verify_batch 内部会做 q = softmax(logits / temperature)。
                #   如果传已经 softmax 过的概率，会被再 softmax 一次：
                #   151936 词表上 q̃ 会被压成近似均匀分布（max 只有 1.2e-5），
                #   而草稿 token 是从真分布 q 采的 —— 一致性前提被破坏，
                #   无损性不再成立（实测 q≡p 时 TV 从 0.003 涨到 0.108）。
                draft_q = seq.draft_logits.unsqueeze(0).float()   # (1, kk, V) 原始 logits
                t = torch.tensor([seq.temperature], device=sub.device)
                # draft 已经按 temperature 采过样了，验证时目标侧也要用同一个温度
                res = verify_batch(draft_q, sub, draft_tok, t)
            else:
                # ---------- n-gram 路线：单点分布（无真实分布可用）----------
                #必须是真正的概率分布，不能用 logits 表达单点分布：
                #   logits=[30,-30,...] 过 softmax 后是 0.9999 而非 1.0，
                #   残余概率会污染修正分布 max(0, p−q)。
                draft_probs = torch.zeros((1, kk, V), device=sub.device, dtype=torch.float32)
                draft_probs.scatter_(2, draft_tok.unsqueeze(-1), 1.0)
                t = torch.tensor([seq.temperature], device=sub.device)
                res = verify_batch(draft_probs, sub, draft_tok, t, draft_is_point_mass=True)

            accepted = int(res.accepted[0])
            toks = list(seq.draft_tokens[:accepted])
            # ★ bonus 每轮必有：k 个候选全被接受时，它是 target 第 k 行
            #   已经算好的下一个 token（不取就白算一行，而且每步只能出 k 个 token）。
            toks.append(int(res.bonus[0]))
            if not toks:
                sub1 = logits[cu[i]:cu[i] + 1]
                toks = [self.sampler(sub1, t).tolist()[0]]
            seq.last_accepted = len(toks)
            out.append(toks)
        return out

    # ==================================================================
    #  投机路径的两张 CUDA graph
    # ==================================================================
    @torch.inference_mode()
    def run_verify_forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """验证阶段那一次目标模型前向：能进图就进图，否则退回 eager。

        ★ 必须带 @torch.inference_mode()：静态缓冲区是在 inference_mode 里建的
          （capture 方法带装饰器），属于 inference tensor，
          在 InferenceMode 之外原地写会直接报错。run_model 也是同样的写法。

        ★ 为什么要单独拍一张图：
          验证前向走的是 varlen（1+k 个 query 一次算完），而 run_model 里
          is_prefill=True 会强制走 eager 分支 —— 实测一次 36 ms。
          但它的【形状其实是固定的】：batch=1、query 数恒等于 1+k、
          block 表宽度固定。形状固定就能拍图。
          （对比：同一模型的普通 decode 拍图后是 12.8 ms/步。）

        条件不满足时（batch>1、或草稿数不足 k）自动退回 eager，行为不变。
        """
        ex = self._verify_extra
        if ex is not None and self.verify_graphs:
            B, k = ex["n"], ex["k"]
            entry = None
            reason = None
            # ★ S2 消融开关：batch_verify_graph=False 时只有 B=1 走图（= C 组行为）
            if B > 1 and not self.batch_verify_graph:
                reason = "batch_verify_graph_off"
            elif not ex["full_k"]:
                # 候选变短：cu_seqlens_q 步长不齐，形状不是静态的
                reason = "ragged_draft_len"
            else:
                entry = self.verify_graphs.get((B, k))
                if entry is None:
                    reason = f"shape_not_captured(B={B},k={k})"
            if entry is not None:
                graph, gv = entry
                bt = ex["block_tables"]
                if input_ids.numel() == gv["input_ids"].numel() and \
                        (bt is None or bt.size(1) <= gv["block_tables"].size(1)):
                    gv["input_ids"].copy_(input_ids)
                    gv["positions"].copy_(positions)
                    gv["slot_mapping"].copy_(ex["slot_mapping"])
                    gv["cu_seqlens_q"].copy_(ex["cu_seqlens_q"])
                    gv["cu_seqlens_k"].copy_(ex["cu_seqlens_k"])
                    gv["block_tables"].fill_(-1)
                    if bt is not None:
                        gv["block_tables"][:, :bt.size(1)].copy_(bt)
                    graph.replay()
                    self.verify_graph_hits += 1
                    # 静态缓冲区，replay 完就是这一步的结果
                    return gv["logits"]
                reason = "static_shape_mismatch"
            if reason is not None:
                self.verify_graph_fallbacks[reason] = \
                    self.verify_graph_fallbacks.get(reason, 0) + 1
        return self.run_model(input_ids, positions, True)

    def _graph_bs(self) -> list[int]:
        """要捕获的精确 batch 桶（P6 任务 B）：只覆盖实际会用的规模。

        12.2 的要求：新增图先只覆盖精确 B=2/4，保留 B=1 原路径；
        不捕获默认 512 的全范围（那会白吃一大块图池显存）。
        这里再按 max_num_seqs 截断，并去重排序。
        """
        bs = sorted({int(b) for b in self.config.spec_graph_bs
                     if 1 <= int(b) <= self.config.max_num_seqs})
        return bs or [1]

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        """draft 模型的 decode 图集合：{B: graph}，每次 1 个 token、B 条请求。

        实测收益：eager 24.8 ms/次 -> 拍图后 ~2 ms/次。
        k 步串行，所以这笔节省要乘以 k；批量之后，B 条请求共享一次 replay。

        每张图用**自己的**持久缓冲区，各行的元数据（position / context_len /
        slot / block_table）在 replay 前独立更新，所以 B 条请求可以处在
        不同的位置、写不同的物理槽。
        """
        d_hf = self.draft_hf_config
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        before = torch.cuda.memory_allocated()
        graphs = {}
        for B in self._graph_bs():
            input_ids = torch.zeros(B, dtype=torch.int64)
            positions = torch.zeros(B, dtype=torch.int64)
            slot_mapping = torch.zeros(B, dtype=torch.int32)
            context_lens = torch.zeros(B, dtype=torch.int32)
            block_tables = torch.zeros(B, max_num_blocks, dtype=torch.int32)
            logits = torch.empty(B, d_hf.vocab_size)
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                        block_tables=block_tables)
            logits.copy_(self.draft_model.compute_logits(self.draft_model(input_ids, positions)))
            with torch.cuda.graph(graph):
                logits.copy_(self.draft_model.compute_logits(self.draft_model(input_ids, positions)))
            reset_context()
            torch.cuda.synchronize()
            graphs[B] = (graph, dict(
                input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables, logits=logits))
        self.draft_graphs = graphs
        self.spec_proposer.bind_cudagraph(graphs)
        self.graph_mem["draft_graphs_gb"] = round(
            (torch.cuda.memory_allocated() - before) / 2**30, 4)
        print(f"[spec] draft CUDA graph 就绪 (bs={sorted(graphs)}, 1 token/行)")

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        """验证前向的图集合：{(B, k): graph}，query 数 = B*(k+1)，走 varlen 路径。

        ★ 为什么可以批量化：验证送进去的是【变长】的请求（每条 n_i = 1+k 个
          query token，但前缀长度 len_i 各不相同）—— 这正是 varlen 的形态。
          固定 k 时 query 总数恒为 B*(k+1)，形状是静态的，所以能拍图。
          cu_seqlens_q/k、slot、block_table 在 replay 前原样拷进静态输入，
          logits 按请求边界切分（run_verify 用 _last_cu_seqlens_q 切）。

        ★ max_seqlen_k 在 capture 时就被烘进 kernel 参数了，所以只能给上界
          （max_model_len）。它只是给 flash-attn 分块用的提示，给大了不影响正确性
          —— 真实可见长度由 cu_seqlens_k 决定。

        ★ 只用「所有参与请求都有完整 k 个候选」的形状；其它形状走 eager 并记录原因，
          不补零 logits 强行凑图。
        """
        hf_config = self.config.hf_config
        k = self.config.spec_k
        n = k + 1
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        before = torch.cuda.memory_allocated()
        graphs = {}
        # 从大到小捕获：先用最大的 B 建池，其余复用同一个池，避免每张图各吃一块
        pool = None
        for B in sorted(self._graph_bs(), reverse=True):
            Q = B * n
            input_ids = torch.zeros(Q, dtype=torch.int64)
            positions = torch.zeros(Q, dtype=torch.int64)
            slot_mapping = torch.zeros(Q, dtype=torch.int32)
            # capture 时给一组【合法】的 dummy 值（步长恒为 n），别用全 0
            # —— varlen 内核拿到 cu_seqlens=[0,0] 会退化成长度 0。
            cu_seqlens_q = torch.tensor([i * n for i in range(B + 1)], dtype=torch.int32)
            cu_seqlens_k = torch.tensor([i * n for i in range(B + 1)], dtype=torch.int32)
            block_tables = torch.zeros(B, max_num_blocks, dtype=torch.int32)
            logits = torch.empty(Q, hf_config.vocab_size)
            graph = torch.cuda.CUDAGraph()
            set_context(True, cu_seqlens_q, cu_seqlens_k, n, self.config.max_model_len,
                        slot_mapping, None, block_tables, is_spec_verify=True)
            logits.copy_(self.model.compute_logits(self.model(input_ids, positions)))
            with torch.cuda.graph(graph, pool):
                logits.copy_(self.model.compute_logits(self.model(input_ids, positions)))
            reset_context()
            torch.cuda.synchronize()
            if pool is None:
                pool = graph.pool()
            graphs[(B, k)] = (graph, dict(
                input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                block_tables=block_tables, logits=logits))
        self.verify_graphs = graphs
        self.graph_mem["verify_graphs_gb"] = round(
            (torch.cuda.memory_allocated() - before) / 2**30, 4)
        print(f"[spec] verify CUDA graph 就绪 (shapes={sorted(graphs)}, "
              f"{n} query/请求, 共享图池)")

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
