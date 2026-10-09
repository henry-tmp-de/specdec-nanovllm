import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.spec_decode.draft_proposer import DraftWindow
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
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
        self._verify_extra = None
        # P6 消融开关（默认走批量；设 False 退回逐请求/仅 B=1 图）
        self.batch_draft = config.spec_batch_draft
        self.batch_verify_graph = config.spec_batch_verify_graph
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
            self.draft_model = Qwen3ForCausalLM(self.draft_hf_config)
            load_model(self.draft_model, config.draft_model)
            self.spec_proposer = DraftModelProposer(self.draft_model, k=config.spec_k,
                                                    block_size=self.block_size)
            print(f"[spec] draft model: {config.draft_model} "
                  f"({self.draft_hf_config.num_hidden_layers} 层, "
                  f"vocab {self.draft_hf_config.vocab_size})")

        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            # ★ 投机路径的两张图。必须放在 allocate_kv_cache 之后
            #   —— 图在 capture 时就把 k_cache/v_cache 的显存地址烘进去了。
            if self.config.spec_cuda_graph and self.spec_proposer is not None:
                if self.draft_model is not None:
                    self.capture_draft_cudagraph()
                self.capture_verify_cudagraph()
        torch.set_default_device("cpu")
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
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        budget = int(total * config.gpu_memory_utilization - used - peak + current)

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
        if draft_block_bytes and W > 0:
            assert W % self.block_size == 0, (
                f"spec_draft_window={W} 必须是 kvcache_block_size={self.block_size} 的整数倍")
            M = W // self.block_size
            draft_pool = M * config.max_num_seqs
            room = budget - draft_pool * draft_block_bytes
            assert room >= tgt_block_bytes, (
                f"draft 滑窗池放不下：{draft_pool} 块 x {draft_block_bytes/2**20:.0f} MiB "
                f"= {draft_pool*draft_block_bytes/2**30:.2f} GB，而 KV 预算只有 "
                f"{budget/2**30:.2f} GB（max_num_seqs={config.max_num_seqs}, M={M}）。"
                f" 减小 spec_draft_window / max_num_seqs 或调大 gpu_memory_utilization")
            config.num_kvcache_blocks = int(room // tgt_block_bytes)
            config.num_draft_blocks = draft_pool
            config.draft_window_blocks = M
            self.draft_window_blocks = M
            print(f"[spec] draft 滑窗开启: W={W} token = {M} 块/序列, "
                  f"draft 池 {draft_pool} 块 ({(draft_pool*draft_block_bytes)/2**30:.2f} GB), "
                  f"target 池 {config.num_kvcache_blocks} 块")
        else:
            per_block_total = tgt_block_bytes + draft_block_bytes
            config.num_kvcache_blocks = budget // per_block_total
            config.num_draft_blocks = 0
            config.draft_window_blocks = 0
            self.draft_window_blocks = 0
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
        print(f"[spec] blocks={config.num_kvcache_blocks}, "
              f"target 每block {tgt_block_bytes/1024:.0f} KB, "
              f"合计 {(per_block_total*config.num_kvcache_blocks)/2**30:.2f} GB")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
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
        if self.draft_window_blocks and seqs and all(s.draft_block_table for s in seqs):
            self._run_draft_prefill_window(seqs)
            return True
        with torch.inference_mode():
            self.draft_model(input_ids, positions)
        return False

    @torch.inference_mode()
    def _run_draft_prefill_window(self, seqs):
        """滑窗路径的 draft prefill：只跑「最近 M 个块」，用 draft 自己的块表。

        ★ 为什么不是整个 prompt：滑窗只保留最近 M 个块，更老的 token 反正会被
          环覆盖，跑它们纯属浪费。按【块边界】对齐地取最后 M 个块 [b0*bs, end)，
          一次 varlen 前向算完。
        ★ 块表必须旋转：paged 内核要求 key index j ↔ block_table[j//bs]，
          且必须【从窗口最老的块开始递增】，causal 掩码才对（见 DraftWindow）。
        ★ RoPE：positions 用【绝对位置】range(b0*bs, end) —— 丢的是 KV，不是位置。
        ★ 诚实的近似：这一趟是「把超出窗口的上下文直接截掉」，
          所以窗口里靠前的那些位置拿到的是缩短了的上下文（比流式滑窗更短）。
          每向前解码一步窗口就往前挪，这些位置很快就会滑出去。
        """
        bs = self.block_size
        M = self.draft_window_blocks
        ids, poss, slots, bts, cu = [], [], [], [], [0]
        maxlen = 0
        for seq in seqs:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            if end <= 0:
                continue
            b0 = max(0, (end - 1) // bs - M + 1)
            w0 = b0 * bs
            if w0 >= end:
                continue
            w = DraftWindow(seq.draft_block_table, bs, M)
            ids.extend(seq.token_ids[w0:end])
            poss.extend(range(w0, end))
            slots.extend(w.slot(p) for p in range(w0, end))
            bts.append(w.block_table(w0))
            cu.append(cu[-1] + (end - w0))
            maxlen = max(maxlen, end - w0)
            # 窗口是按 token_ids 整体重算的 → 从 0 到 end 都算「已处理」
            # （比它更老的 token 本来就滑出窗口了，水位语义不受影响）
            seq.draft_valid_len = end
        if not ids:
            return
        dev = next(self.draft_model.parameters()).device
        cu = torch.tensor(cu, dtype=torch.int32, device=dev)
        width = max(len(b) for b in bts)
        bt = torch.tensor([b + [-1] * (width - len(b)) for b in bts],
                          dtype=torch.int32, device=dev)
        set_context(True, cu, cu, maxlen, maxlen,
                    torch.tensor(slots, dtype=torch.int32, device=dev), None, bt)
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
        """
        from nanovllm.spec_decode.draft_proposer import (
            catchup_gap, clip_gap_to_window, window_valid_from)
        start, catchup = catchup_gap(seq.draft_valid_len, seq.token_ids)
        # ---------- 滑窗（B）：draft 的块表是每序列私有的环形缓冲 ----------
        # ★ 判据是【这条序列有没有滑窗块表】，而不是读配置：池子分配失败、或者
        #   运行期把开关翻回全上下文时，这里自动退回老路径（两边都能退）。
        M = len(seq.draft_block_table)
        valid_from = 0
        if M:
            bs = self.block_size
            start, catchup = clip_gap_to_window(start, catchup, bs, M)
            wvf = window_valid_from(seq.draft_valid_len, bs, M)
            # 有缺口时环里更老的内容可能已被覆盖 → 下界取「环的自然窗口起点」
            # 与「补齐起点」中更靠后的那个；没缺口时就用窗口起点。
            valid_from = max(wvf, start) if catchup else wvf
        return dict(block_table=list(seq.block_table),
                    draft_ring=list(seq.draft_block_table),
                    window_blocks=M, valid_from=valid_from,
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

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list:
        """返回值的形态随路径不同：
           prefill        -> list[int]              每序列 1 个
           普通 decode    -> list[int]              每序列 1 个
           投机验证       -> list[list[int]]        每序列若干个
        """
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
            if self.draft_model is not None:
                # ★ draft 模型也要跑一遍 prefill，把它的 KV cache 建起来
                #   （关闭滑窗时 draft 与 target 共用 block_table，但各自写进自己的 cache；
                #    打开滑窗时 draft 用自己的环形块表，见 _run_draft_prefill_window）
                if self._run_draft_prefill(seqs, input_ids, positions):
                    # 滑窗路径 set 过自己的 context 又 reset 了 → 重建 target 的
                    input_ids, positions = self.prepare_prefill(seqs)
            elif self.spec_proposer is not None:
                # n-gram 路线：把 prompt 的 n-gram 建进索引
                for seq in seqs:
                    self.spec_proposer.reset_watermark(seq.token_ids)
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
