from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.mixed_batch import MixedBatchPlan
from nanovllm.engine.prefill_rotation import PrefillRotation


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
            num_draft_blocks=getattr(config, "num_draft_blocks", 0),
            draft_window_blocks=getattr(config, "draft_window_blocks", 0),
            draft_head_blocks=getattr(config, "draft_head_blocks", 0))
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
        # ---------- P13：分块 prefill 让路 ----------
        #   prefill_chunk_tokens = 0（默认）⇒ 老行为：prefill 排他 + 只对第一条序列分块。
        #   >0 ⇒ 每轮 prefill 最多这么多 token，块之间让一轮给 decode（见 schedule()）。
        self.prefill_chunk_tokens = int(getattr(config, "spec_prefill_chunk_tokens", 0) or 0)
        # 上一轮是否留下了"还没拼完的 prefill 块"。默认路径只写不读，行为逐字节不变。
        self._prefill_chunk_pending = False
        # ---------- P15：重叠执行的混合批 ----------
        #   spec_overlap_mixed=False（默认）⇒ 老行为：一轮内只有一种执行路径。
        #   True ⇒ 有"待推进的 prefill 块"且还有请求在解码时，【同一个 step 里】
        #          既排 decode 子批、又排一块 prefill，返回 MixedBatchPlan。
        #   ★ 注意：混合调度本身【不含】重叠 —— 重叠在 ModelRunner/MixedForward 里。
        #     开关只决定"要不要把两件事排进同一步"，spec_overlap_mode 决定
        #     "两个前向是串着跑还是并发跑"。两者正交，消融才干净。
        self.overlap_mixed = bool(getattr(config, "spec_overlap_mixed", False))
        self.overlap_mode = str(getattr(config, "spec_overlap_mode", "concurrent")
                                or "concurrent")
        # ---------- P14：多序列轮转准入 ----------
        #   False（默认）⇒ 老行为：prefill 循环只看队首，分块拼不完就 break。
        #   True ⇒ 本轮让每条待拼序列各拿一块、没拼完的移到队尾（环状轮转 + 公平准入）。
        #   ★ 只在 prefill_chunk_tokens > 0 时生效（没有"每条每块"的上限就无所谓轮转）。
        #   机制与依据见 nanovllm/engine/prefill_rotation.py 与 config.spec_prefill_rotate。
        self.prefill_rotate = bool(getattr(config, "spec_prefill_rotate", False))
        self._rotation = None     # 惰性建；每轮 reset，见 schedule()
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

    def schedule(self) -> tuple[list[Sequence], bool] | MixedBatchPlan:
        """排一个 step。

        返回形状有两种，调用方按类型分派：
          · `(seqs, is_prefill)`  —— 老形状，一轮内【只有一条】执行路径。
          · `MixedBatchPlan`      —— 仅当 `spec_overlap_mixed=True` 时出现，
                                     一轮内有 decode 子批 + prefill 块两条路径。
        ★ 默认路径（开关关）永远只返回第一种形状，老代码一行不用改。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ★★ P13 分块 prefill 让路（默认关闭，prefill_chunk_tokens=0 时这一整段都是老行为）：
        #   上一轮留下了一个"还没拼完的 prefill 块"，且此刻还有请求在解码
        #   → 这一轮先给 decode，别让一条长 prompt 把解码请求饿死整段 prefill。
        #   实测：3 条解码 + 突来一条 16K prompt，解码那一轮的 ITL 尖刺 249×（无 draft）/312×（有 draft），
        #   而且那一轮解码请求拿到的 token 数 = 0（被完全饿死）。
        #   ★ 一轮内仍然只有一种执行路径（要么全是 prefill、要么全是 decode），不混路径
        #     —— 这个项目被"一轮内混路径"坑过 3 次（最典型：spec_enabled 门控快照 bug）。
        # ★ getattr 兜底：部分 CPU 单测用 `Scheduler.__new__` 绕开 __init__ 铺对象
        #   （见 tests/test_spec_gate.py:mk_sched），不会带这两个新字段。
        chunk = getattr(self, "prefill_chunk_tokens", 0) or 0
        # ★★ P12：没有可让路的解码请求时，分块【只有代价、没有收益】。
        #   "让路"这一步（yield_to_decode）本来就门控在 `bool(self.running)` 上：
        #   running 空 ⇒ 块与块之间不会插入任何 decode 轮 ⇒ 分块只是把同一份
        #   prefill 拆成 N 个小批去做，白付"小批效率低"的代价，没人受益。
        #   实测（无解码请求、ctx=16384、单条长 prompt）：C=256 prefill 墙钟 +24.5%、
        #   C=512 +20.3%、C=128 +70.4%（对照：C=4096 只 +4.3%）。
        #   ⇒ 只在"确实有解码请求可让路"时才切块；一旦 running 变空，
        #     上一轮留下的 pending 块会在下一轮拿到完整预算、一次拼完。
        #   ★ 默认路径（prefill_chunk_tokens=0）行为逐字节不变。
        if chunk and not getattr(self, "running", None):
            chunk = 0
        yield_to_decode = (chunk > 0
                           and getattr(self, "_prefill_chunk_pending", False)
                           and bool(self.running))
        # ---------- P15：同 step 双路径（默认关 —— 下面这几行一个字节都不生效）----------
        #   让路式每两轮才推进一块 prefill（[解码][块] 交替）⇒ decode 的 ITL 里
        #   含着【整块 prefill 的墙钟】。P15 让同一轮里既解码、又推进一块 prefill。
        #   ★ 前提（缺一不可）：overlap 开关开（前提是 chunk>0）、上一轮留了 pending
        #     块、此刻确有解码请求 —— 三条与 yield_to_decode 同源，只是换个动作。
        #   ★★ 这个项目被"一轮内混两条执行路径"坑过 3 次（最典型：spec_enabled 门控
        #     快照 bug，复现见 tests/test_spec_gate.py）⇒ 这里【不靠"小心一点"】：
        #     两条路径各有独立的列表，并用 `_round_seqs` 把 I1（一轮内同一条序列
        #     只能被调度一次）钉成代码里的守卫。
        mixed = (getattr(self, "overlap_mixed", False) and yield_to_decode
                 and bool(self.waiting))
        # ★ I1 的载体：本步【已经】被调度走的序列。prefill 段可能把一条序列推成
        #   RUNNING 并 append 进 self.running，decode 段若再 pop 到它，就是
        #   "一轮内同一条序列被调度两次"—— P13「同一个 chunk 被拼 7 遍」的镜像。
        _round_seqs: set = set()
        prefill_seqs: list = []
        if self.waiting and (mixed or not yield_to_decode):
            self._prefill_chunk_pending = False
            # ★ P12：把「此刻已经在解码的批大小」记到 config 上，供 ModelRunner 的
            #   `_draft_prefill_needed()` 判"这趟 draft prefill 会不会白跑"
            #   （spec_batch_threshold 门控在 prefill 时刻看不到未来，但能看到现在）。
            #   getattr/setattr 兜底：CPU 单测用 SimpleNamespace 铺 config。
            try:
                self.config._sched_running_decode_b = len(self.running)
            except AttributeError:
                pass
            # ---------- P14：多序列轮转准入（默认关，rot=None ⇒ 老路径一行未改）----------
            #   rot 非 None 时，循环取的不再是"队首"，而是"本轮还没服务过的环头"，
            #   并且没拼完的序列会让位（移到队尾）而不是 break 收工。
            #   ★ 只在 chunk > 0 时建：没有"每条每轮一块"的上限就没有轮转可言。
            rot = None
            if chunk and getattr(self, "prefill_rotate", False):
                rot = getattr(self, "_rotation", None)
                if rot is None:
                    rot = PrefillRotation()
                    self._rotation = rot
                rot.reset()      # ★ 每轮一份新的"本轮已服务"（见 reset 的 docstring）
            # prefill
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                seq = rot.next(self.waiting) if rot is not None else self.waiting[0]
                if seq is None:      # ★ 轮转：整环绕过一圈了，本轮到此为止
                    break
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0:
                    break
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        if rot is not None:
                            # ★ 公平准入：一条分配不到块（可能只是它特别长）不该挡住
                            #   整条队列的其他人。本轮跳过它、让位给后面；它没被移出
                            #   waiting，下一轮仍会被取到（只是排到本轮其他人的后面）。
                            rot.skip(self.waiting, seq)
                            continue
                        break
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                # ★ 轮转时这条守卫必须放行：chunk>0 本来就允许把一条序列切开，
                #   剩下的预算哪怕不够一整块也该给出去（给的是 min(剩余, chunk)）。
                if remaining < num_tokens and scheduled_seqs and rot is None:
                    break
                # ★ 本轮的 prefill 预算上限：0 = 不分块（= remaining，老行为）
                budget = min(remaining, chunk) if chunk else remaining
                if not seq.block_table:
                    self.block_manager.allocate(seq, num_cached_blocks)
                    # ---------- 前缀缓存命中时，draft 侧能不能跟着复用？----------
                    # ★ 不能拿 num_cached_blocks 直接当 draft 的有效量：前缀缓存
                    #   （hash_to_block_id）存的是 target 的 KV 块，draft 的 KV 在
                    #   另一套物理缓冲里，命中 target 缓存不代表 draft 那块也有这段
                    #   前缀的内容。所以只认 draft 自己的块标记（Block.draft_hash）。
                    #   连续有效多少块，draft 水位就推到哪；其余留给 proposer 补齐。
                    # ★ 窗口模式（B）下不适用：draft 的 KV 在自己的小池子里，跟 target
                    #   的前缀缓存块不再对应；窗口会在 draft prefill 时按 token_ids
                    #   整体重算（见 ModelRunner._run_draft_prefill_window）。
                    if not self.block_manager.draft_window_blocks:
                        nb = self.block_manager.draft_valid_cached_blocks(seq, num_cached_blocks)
                        if nb:
                            seq.draft_valid_len = max(seq.draft_valid_len, nb * self.block_size)
                    # 窗口模式：给这条序列分配它自己的滑窗块（池子够大就不会失败；
                    # 不够说明 max_num_seqs * W/block_size 超过了预算，配置问题要显式报出来）
                    if self.block_manager.draft_window_blocks and not self.block_manager.draft_acquire(seq):
                        # ★ 优雅回退：池子不够就【整条序列】退回全上下文 draft
                        #   （draft_block_table 留空 → _draft_request 自然走老路径）。
                        #   半新半旧才是危险的，这里宁可退一整条。
                        self._window_fallbacks = getattr(self, "_window_fallbacks", 0) + 1
                        if self._window_fallbacks in (1, 2, 10, 100):
                            print(f"[spec] draft 滑窗池不足，第 {self._window_fallbacks} 次"
                                  f"回退到全上下文（seq {seq.seq_id}）")
                seq.num_scheduled_tokens = min(num_tokens, budget)
                num_batched_tokens += seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                    scheduled_seqs.append(seq)
                    if rot is not None:
                        rot.mark(seq)    # ★ 拼完的也要记：同一轮里不许再碰它
                elif rot is not None:
                    # ★★ P14 轮转：这条拼不完 ≠ 本轮到此为止。
                    #   记下"本轮已服务"并把队首移到【环尾】—— 后面的序列顶上继续拼。
                    #   没有 mark 这一下，转完一圈后 while 会再取到【同一条】，
                    #   把 num_scheduled_tokens 又覆盖一遍、再 append 一份
                    #   （P13 "同一个 chunk 被拼 7 遍" 的复现条件，见
                    #     tests/test_prefill_multiseq.py §1 与本模块 docstring）。
                    self._prefill_chunk_pending = True
                    scheduled_seqs.append(seq)
                    rot.yield_turn(self.waiting, seq)
                    continue
                else:
                    # ★★ 分块：这条本步拼不完 ⇒ 【本轮到此为止】，必须立刻 break。
                    #   它没被 popleft，还留在 self.waiting[0]；不 break 的话这个 while
                    #   会再取到【同一条】，把它的 num_scheduled_tokens 又覆盖一遍、
                    #   再 append 一份 —— 于是 scheduled_seqs 里出现同一序列的多份副本，
                    #   prepare_prefill 会把这段 chunk 重复拼 N 次。
                    #   （实测：max_num_batched_tokens=16640 时一份 2048 的 chunk 被拼了 7 遍 =
                    #    14336 token，单个 prefill 轮 2055ms；而 max_num_batched_tokens=4352
                    #    恰好被原来那句 `remaining < num_tokens and scheduled_seqs` 的守卫
                    #    兜住，所以只在"预算充裕"的配置下暴露。）
                    self._prefill_chunk_pending = True
                    scheduled_seqs.append(seq)
                    break

            if mixed and scheduled_seqs:
                # ★ 不在 prefill 段 return：本步的 prefill 块先记到 prefill_seqs，
                #   下面接着排 decode 子批，最后一起作为 MixedBatchPlan 交出去。
                #   ★ scheduled_seqs 清空是为了让下面的 decode 循环从零计数
                #     （它的循环条件是 `len(scheduled_seqs) < max_num_seqs`）。
                prefill_seqs, scheduled_seqs = scheduled_seqs, []
                _round_seqs = set(prefill_seqs)
            elif scheduled_seqs:
                return scheduled_seqs, True
            else:
                # 连一块 prefill 都排不出来（block_manager 拒绝分配 / 预算为 0）
                # ⇒ 混合名不副实，退回"这一轮只跑 decode"的老形状。
                mixed = False
        # 走到这里：要么 waiting 空、要么这一轮让给了 decode ⇒ 清掉分块标记。
        # ★ mixed 时【不能清】：那是"下一轮继续混"的依据（本轮的 prefill 段
        #   已经把它设成正确的值 —— 块拼完了就是 False）。
        if not mixed:
            self._prefill_chunk_pending = False

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
            # ★★ I1（P15）：本步 prefill 段刚推成 RUNNING 的序列，绝不能在【同一轮】
            #   再被 decode 段取走 —— 一轮内一条序列只能走一条执行路径。
            #   放回队尾并收工（不 continue：continue 会把它从 running 里弄丢）。
            #   ★ 默认路径（overlap 关）_round_seqs 恒为空集 ⇒ 一行不生效。
            if _round_seqs and seq in _round_seqs:
                self.running.append(seq)
                break
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
        if mixed and not scheduled_seqs:
            # ★ 极端兜底：decode 端一条都没排上（KV 预算把整批都抢占了）。
            #   这时"混合"名不副实，退回老形状：本轮只有 prefill 块。
            return prefill_seqs, True
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        if mixed:
            # ★★ 唯一的出口形状：显式契约（两条路径各自的容器 + 发射方式），
            #    而不是把一个布尔塞回调用方让它猜。见 mixed_batch.MixedBatchPlan。
            return MixedBatchPlan(prefill_seqs=prefill_seqs,
                                  decode_seqs=scheduled_seqs,
                                  mode=getattr(self, "overlap_mode", "concurrent"))
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
