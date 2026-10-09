from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # ---------- 投机解码 ----------
        # ★ spec_k 不在这里拷一份，而是走下面的 property 直接读写 config ——
        #   因为 schedule() 按 self.spec_k 预留 1+k 个槽位，而 model_runner
        #   提议候选、拍验证图用的是 config.spec_k。两份副本一旦漂移：
        #     · scheduler 侧偏小 → 只备了 1+ks 个槽位，却被喂进 1+kc 个候选，
        #       prepare_verify 会往没分配的槽位写 KV（CUDA 非法访存那一类崩溃）；
        #     · 旧代码里 `llm.scheduler.spec_k = 0` 这种「只改一个副本」的写法
        #       不一定会生效，正是同步关系容易踩空的地方。
        #   合成一个之后，两处读到的永远是同一个值，改哪边都生效。
        self.spec_batch_threshold = config.spec_batch_threshold
        # n-gram 提议器。由 ModelRunner 注入（引擎建好后才有），
        # 这样 token_ids 更新后能立刻把新 token 加进索引。
        self.spec_proposer = None
        # token 交付 hook（TTFT/TPOT/ITL 用，见 engine/token_hook.py）。
        # ★ 默认 None：热路径上只多一个 `is not None` 判断，不启用就零开销，
        #   正式测性能时不挂它。
        self.token_hook = None

    @property
    def spec_k(self) -> int:
        """投机候选数。唯一来源是 config.spec_k（见 __init__ 的说明）。"""
        return self.config.spec_k

    @spec_k.setter
    def spec_k(self, value: int):
        self.config.spec_k = value

    def set_spec_proposer(self, proposer):
        self.spec_proposer = proposer

    def set_token_hook(self, hook):
        self.token_hook = hook

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                # ---------- 前缀缓存命中时，draft 侧能不能跟着复用？----------
                # ★ 不能拿 num_cached_blocks 直接当 draft 的有效量：前缀缓存
                #   （hash_to_block_id）存的是 target 的 KV 块，draft 的 KV 在
                #   另一套物理缓冲里，命中 target 缓存不代表 draft 那块也有这段
                #   前缀的内容。所以只认 draft 自己的块标记（Block.draft_hash）。
                #   连续有效多少块，draft 水位就推到哪；其余留给 proposer 补齐。
                nb = self.block_manager.draft_valid_cached_blocks(seq, num_cached_blocks)
                if nb:
                    seq.draft_valid_len = max(seq.draft_valid_len, nb * self.block_size)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # ★★ 门控在【本轮开始时】取一次快照，全轮复用同一个决定。
        #    下面的 while 用 popleft() 逐条取序列，self.running 在轮内会变短；
        #    每取一条重新判断 len(self.running)，同一轮里靠后的请求就会拿到
        #    不同的决定（前面开投机、后面退回普通 decode），一轮内混着两种
        #    执行路径 —— 后处理的形状分支、KV 预算、消融归因全都对不上。
        #    注意这里取的是「本轮 active decode B」，不是「队列剩几个」。
        spec_on = self.spec_enabled(len(self.running))
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # ---------- 投机解码：这一步要给 k+1 个 token 分配槽位 ----------
            # 之所以是 k+1：1 个是本该decode 的 token，k 个是待验证的候选。
            # 但只有「本该 decode 的那个」的 KV 是最终有效的，
            # 候选的 KV 写入后靠 slot 覆写回收（见 model_runner.prepare_verify）
            need = 1 + (self.spec_k if spec_on else 0)
            while not self.block_manager.can_append(seq, need):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = need
                seq.is_prefill = False
                self.block_manager.may_append(seq, need)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def spec_enabled(self, active_b: int | None = None) -> bool:
        """本轮是否启用投机解码（含 batch 门控）。

        active_b = 本轮的 active decode B，必须由调用方在【轮开始时】取快照传入。
        ★ 不要传「当前队列长度」：decode 分支用 popleft() 逐条取走队列，
          队列长度在轮内递减，同一条请求在轮内不同位置被判出不同结果。
          默认值 len(self.running) 只适合「轮外一次性判断」的场合
          （例如 benchmark 里预估盈亏平衡点），调度路径上必须显式传快照。
        """
        if self.spec_k <= 0:
            return False
        if active_b is None:
            active_b = len(self.running)
        if self.spec_batch_threshold and active_b > self.spec_batch_threshold:
            return False
        return True

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        # ★ P6 任务 D：块被释放 → draft KV 里对应物理块的内容不再属于这条请求，
        #   有效水位必须归零。重新 prefill 后会由 prepare_prefill 重新建立。
        seq.draft_valid_len = 0
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list, is_prefill: bool) -> int:
        """收尾，并返回【本步实际交付给用户的 token 数】。

        token_ids 的形态：
           普通 decode   -> [int]          每序列 1 个
           投机验证       -> list[list[int]] 每序列若干个（已接受的 + bonus）
           分块 prefill   -> 本 chunk 不交付 token（返回 0）

        ★ 返回值就是正式吞吐口径（LLMEngine.step 用它）：一个 step 落地几个
          token 就记几个，投机一轮成批交付的 k+1 个不会被拆成别的凑数。
        """
        if not is_prefill and token_ids and isinstance(token_ids[0], list):
            return self.postprocess_spec(seqs, token_ids)
        delivered = 0
        for seq, token_id in zip(seqs, token_ids):
            # ★★ 推进量必须是【本步真正被确认的】token 数，不能拿
            #   num_scheduled_tokens 顶替（它是「本步要写几个位置」，投机时会
            #   按上限预留 1+k，而真正落地的只有 1 个）：
            #     · prefill：本 chunk 的每个位置都被 prepare_prefill 写过 → 推 chunk 大小
            #     · 普通 decode（含投机没捞到候选时的退回）：每序列只交付 1 个 token
            #   旧写法在退回路径上每步多推 k 个，几步后 num_cached_tokens 就跑到
            #   num_tokens 前面，登记区间随之越界 —— 复现见 tests/test_prefix_hash.py §3。
            #   契约「旧有效缓存量 → 新有效缓存量」见 BlockManager.hash_blocks。
            num_new = seq.num_scheduled_tokens if is_prefill else 1
            seq.num_cached_tokens += num_new
            # ★ 顺序：先推进 num_cached_tokens，再把它【已经推进过的值】连同
            #   推进量一起交给 hash_blocks。两者必须一致 —— 只推进不传量
            #   （或传了 num_scheduled_tokens 当推进量），登记区间就会整体后移一格：
            #   刚写满的块被跳过、半满块被当成满块登记。复现见 §1/§2。
            self.block_manager.hash_blocks(seq, num_new)
            # ★ draft 侧的有效性盖章要在同一步做：此刻 token_ids / 块内容 /
            #   draft_valid_len 三者都已最终化，盖出来的标记才不会有偏差。
            self.block_manager.mark_draft_valid(seq, seq.draft_valid_len)
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # ★ 新生成的 token 必须入 n-gram 索引，否则下一步提不出候选。
            #   普通 decode 路径也要做 —— 大部分 step 其实走的是这条（候选为空时退回），
            #   漏了它会让索引永远追不上，导致投机占比恒为 0。
            if self.spec_proposer is not None:
                self.spec_proposer.observe(seq.token_ids)
            finished = (not seq.ignore_eos and token_id == self.eos) \
                or seq.num_completion_tokens == seq.max_tokens
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            # 交付计数 / 计时：走到这里才是真的产出了一个 token
            # （分块 prefill 的中间 chunk 在上面就 continue 了，不计、不记时）
            delivered += 1
            if self.token_hook is not None:
                self.token_hook.on_deliver(seq, 1, finished)
        return delivered

    def postprocess_spec(self, seqs: list[Sequence], accepted_tokens: list[list[int]]) -> int:
        """投机验证后的收尾：一次落地多个 token，返回本步实际交付的 token 数。

        ★ 三件事必须做对，漏了任何一条都不会报错，只会静默出错：
          ① num_cached_tokens 只推进【真正确认的】部分，不能把被拒候选算进去
          ② hash_blocks 必须在 token_ids 更新【之后】调用，且草稿不在 token_ids 里
          ③ EOS 和 max_tokens 要在【每个】落地的 token 上检查，而不是只看最后一个
        """
        delivered = 0
        for seq, toks in zip(seqs, accepted_tokens):
            seq.draft_tokens = []# 草稿用完即弃，绝不进 token_ids
            seq.draft_logits = None        # ★ 同理，draft 分布也必须清，否则会串用上一轮

            if not toks:
                # ★ 全拒：不推进（本步没有任何 token 被确认落地）。
                #
                # 【当前引擎路径不可达】—— 只有测试直接传 [[]] 才会走到这里。
                # run_verify 的两条分支都保证 out 里至少有一个 token：
                #   · n > 1（有候选）→ 无条件 append bonus
                #     （verify_batch 每轮必给 bonus，全接受时退化成 target 第 k 行）；
                #   · n <= 1（没捞到候选）→ 显式 append 一个采样 token。
                # 所以 out 永远不含空列表。
                #
                # 旧写法 `cached -= num_scheduled_tokens`（退 1+k）语义【过头】：
                # 既然一个 token 都没落地，num_cached_tokens 就该原地不动。
                # P6 会重写 run_verify 的批量路径，这个分支【可能变成可达的】，
                # 所以这里先把语义改正，别留一个「看起来正常但退多了」的隐患。
                seq.num_scheduled_tokens = 0
                continue

            # ★★ 必须先记下 append 之前的完成 token 数：
            #   终止判据要用「base + i + 1 >= max_tokens」，不能等 append 完
            #   再拿 num_completion_tokens 去比。
            base_completion = seq.num_completion_tokens
            # ① 先记已确认的 token（此刻token_ids 还没变）
            confirmed = seq.num_tokens
            seq.append_tokens(toks)
            # num_scheduled_tokens 是 1+k，只有被接受的那部分对应真实位置
            seq.num_cached_tokens += len(toks)
            # ★ P6 任务 D：propose 写完 draft KV 后水位是「最后已确认位置 + k」，
            #   但真正被确认的只有 len(toks) 个 —— 水位要夹到「已确认前缀」，
            #   否则下次提议会以为某些位置已经缓存好了。
            #   （全部接受时 len(toks)=k+1，夹到 num_tokens-1 会留下 1 个位置
            #     的缺口，正好是要补写的 bonus 位 —— 见 model_runner 的推导。）
            seq.draft_valid_len = min(seq.draft_valid_len, seq.num_tokens - 1)

            # ② token_ids 已更新，现在才能安全地做前缀缓存哈希。
            #   ★ 推进量必须显式传 len(toks)：这里的 num_scheduled_tokens 是 1+k
            #     （schedule 按上限预留的槽位数），与被确认的 len(toks) 不相等，
            #     让 hash_blocks 自己去猜就会算错登记区间（半满块会被当成满块）。
            self.block_manager.hash_blocks(seq, len(toks))
            # ★ draft 侧盖章：必须在上面夹完水位之后（see the clamp above）。
            self.block_manager.mark_draft_valid(seq, seq.draft_valid_len)
            # ★ 把新生成的 token 增量加入 n-gram 索引，
            #   否则下一步提不出以这些新 token 结尾的候选
            #   （必须放在 append_tokens 之后 —— 那之前 token_ids 还没这些 token）
            if self.spec_proposer is not None:
                self.spec_proposer.observe(seq.token_ids)

            seq.num_scheduled_tokens = 0

            # ③逐个检查终止条件（中途撞 EOS / 撞 max_tokens 就截断，后面的全部丢弃）
            #
            # ★★ 这里原来是 `seq.num_completion_tokens == seq.max_tokens`，是错的：
            #   投机一步落地多个 token 时，完成数会【跳过】max_tokens ——
            #   比如从 62 直接跳到 65，`== 64` 永远不成立，
            #   序列会一直生成下去（实测跑到 572 个 token、300 步还不结束，
            #   这也是「draft 模式 benchmark 卡住几百秒」的真正原因）。
            #   正确判据是「base + i + 1 >= max_tokens」，即这一位会不会越界。
            newly = toks
            landed = len(newly)          # 本步真正交付给用户的 token 数
            finished = False
            for i, t in enumerate(newly):
                if (not seq.ignore_eos and t == self.eos) or base_completion + i + 1 >= seq.max_tokens:
                    # 截断：把这一步多写的 token 砍掉
                    extra = len(newly) - i - 1
                    if extra > 0:
                        del seq.token_ids[len(seq.token_ids) - extra:]
                        seq.num_tokens -= extra
                        seq.last_token = seq.token_ids[-1]
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    landed = i + 1       # 被砍掉的那些没交付，吞吐不能算进去
                    finished = True
                    break
            delivered += landed
            if self.token_hook is not None:
                # ★ 成批交付：这一轮的 landed 个 token 是【同一时刻】可见的，
                #   交给 hook 记成一批（不是 landed 个独立时刻）。
                self.token_hook.on_deliver(seq, landed, finished)
        return delivered
