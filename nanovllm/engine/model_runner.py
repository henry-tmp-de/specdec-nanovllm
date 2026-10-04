import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
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

        per_block_total = tgt_block_bytes + draft_block_bytes
        config.num_kvcache_blocks = budget // per_block_total
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
            self.draft_kv_cache = torch.empty(
                2, d_hf.num_hidden_layers, config.num_kvcache_blocks,
                self.block_size, d_kv_heads, d_head_dim)
            dl = 0
            for module in self.draft_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, dl]
                    module.v_cache = self.draft_kv_cache[1, dl]
                    dl += 1
            assert dl == d_hf.num_hidden_layers, f"draft 层数不匹配: {dl} vs {d_hf.num_hidden_layers}"
            print(f"[spec] draft KV cache: {dl} 层 x {config.num_kvcache_blocks} blocks, "
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

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 存一份给 run_verify 切分 logits 用
        self._last_cu_seqlens_q = cu_seqlens_q.tolist()
        # ★ is_prefill=True —— 让 attention 走 flash_attn_varlen_func(causal=True)，
        #   这正是我们要的因果路径；不能用 decode 那条（它只处理 1 个 query）。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                     slot_mapping, None, block_tables, is_spec_verify=True)
        return input_ids, positions

    def _run_draft_prefill(self, seqs, input_ids, positions):
        """prefill 时也把 draft 模型跑一遍，填它自己的 KV cache。

        ★ 最简实现：直接复用当前 context（prepare_prefill 已经设好了），
          跑完【不恢复】—— 因为 target 的 prefill 紧接着会自己再
          set_context 一次（run() 里prepare_prefill 在前、target 前向在后），
          这里只需要保证 draft 跑到时 context 是对的。
        """
        with torch.inference_mode():
            self.draft_model(input_ids, positions)

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
                #   （draft 与 target 共用 block_table，但各自写进自己的 cache）
                self._run_draft_prefill(seqs, input_ids, positions)
            elif self.spec_proposer is not None:
                # n-gram 路线：把 prompt 的 n-gram 建进索引
                for seq in seqs:
                    self.spec_proposer.reset_watermark(seq.token_ids)
        elif self.spec_proposer is not None and any(s.num_scheduled_tokens > 1 for s in seqs):
            # 先提议，再决定走验证还是退回普通 decode。
            k = self.config.spec_k
            for seq in seqs:
                if seq.num_scheduled_tokens > 1:
                    if self.draft_model is not None:
                        # draft 路线：小模型自回归 k步，产出候选 + 真实概率
                        chain, probs = self.spec_proposer.propose(
                            seq.block_table, len(seq), seq.last_token, seq.temperature)
                        seq.draft_tokens = chain
                        seq.draft_probs = probs
                    else:
                        chains = self.spec_proposer.propose(seq.token_ids, k)
                        # 线性链一次 forward 只能验证一条 —— 要同时验证多条就得做树状
                        # 注意力（需要自定义 mask + 换 SDPA，见 README 的取舍说明）。
                        # 这里取【最长】的那条：长度直接决定能省几次 forward。
                        seq.draft_tokens = max(chains, key=len) if chains else []
                else:
                    seq.draft_tokens = []
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
        logits = self.run_model(input_ids, positions, True)
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

            if seq.draft_probs is not None:
                # ---------- draft model 路线：真实分布 ----------
                # ★ 这才是标准投机解码：接受概率 min(1, p_target / q_draft)，
                #   q 是 draft 模型真实的输出分布。
                #   实测 n-gram 路线的接受率 <1%，根因就是 q 退化成了单点分布
                #   （接受概率 = p[draft]，通常 0.001~0.06）。
                draft_probs = seq.draft_probs.unsqueeze(0).float()   # (1, kk, V)
                t = torch.tensor([seq.temperature], device=sub.device)
                # draft 已经按 temperature 采过样了，验证时目标侧也要用同一个温度
                res = verify_batch(draft_probs, sub, draft_tok, t)
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
            if int(res.bonus[0]) >= 0:
                toks.append(int(res.bonus[0]))
            if not toks:
                sub1 = logits[cu[i]:cu[i] + 1]
                toks = [self.sampler(sub1, t).tolist()[0]]
            seq.last_accepted = len(toks)
            out.append(toks)
        return out

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
