"""P15 —— 同一个 step 内「decode 走 CUDA Graph ∥ prefill 走 eager 分片」的并发编排。

为什么单独一个模块
------------------
"两条 CUDA stream 上的两条执行路径"是本仓库里【唯一】一处并发概念。
把它摊进 `ModelRunner.run()` 会变成一堆 `if mode == ...`；
收成一个有名有姓的类，改动面就只剩 `ModelRunner.run_mixed()` 一个入口。

★ 收益账（docs/优化报告.md §d.5 方向 2）
    [解码 12.1][块 284] 让路式（现状 spec_prefill_chunk_tokens>0）
                                            ⇒ decode 用户每 296 ms 拿一个字
    [解码 12.1 + 块 284] 同 step、两个前向【串着跑】
                                            ⇒ 296 ms   ← ★ 收益 0
    [解码 12.1 ∥ 块 284] 同 step、两条 CUDA stream
                                            ⇒ 284 ms   ← 省 12.1 ms
  收益 100% 来自【重叠】。所以 `serial` 不是"实现了一半"，而是**必须被测出来的那个 0**：
  它证明「混进同一个 step」本身一分钱不值，值钱的是那 12.1 ms 的重叠。

★★ 一个 step 内两条执行路径的不变量（全部写成断言级契约，不靠"小心一点"）
    I1  ★★ 同一条序列在一个 step 里【只能】出现在一条路径上。
        `Scheduler.schedule()` 里 prefill 段可能把一条序列从 waiting 推成 RUNNING，
        decode 段若再取一次就是"一轮内同一条序列被调度两次"——
        它是 P13「同一个 chunk 被拼 7 遍」那个真 bug 的镜像（见
        tests/test_prefill_chunk_gate.py §1）。守卫在 `schedule()` 的 `_round_seqs`。
    I2  decode 子批走不走投机，在【轮首】取一次快照、全轮一致（复用
        `spec_enabled(active_B)` 的既有契约，见 tests/test_spec_gate.py）。
    I3  两条路径的 KV / 计数记账【各自独立】走 `Scheduler.postprocess`：
        prefill 推进 `num_scheduled_tokens` 个位置，decode 推进 1（投机时 len(toks)）。
        绝不能让两边共用一次 postprocess 调用。
    I4  两条路径共享的只有【只读权重】与【全局 context 单例】（utils/context.py）。
        ⇒ context 必须在【各自的 kernel 发射之前】设置好；kernel 一旦发射，
          它读到的就是发射时刻的那份 context，之后 Python 侧怎么改都不影响它。
          这正是"两个前向可以交替发射、但必须在发射前各设一次 context"的依据。
    I5  纯 decode step（本轮没有 prefill 块）必须与旧实现逐字节一致：
        `schedule()` 返回老的两元组，`ModelRunner.run()` 一行不改，图照常命中。
        ★ 本模块的代码在那种 step 上【一次都不会被执行】。
★ 导入约定：本模块【顶层不 import torch】。
  `Scheduler` 要 import `MixedBatchPlan`，而调度器的纯 CPU 回归测试
  （tests/test_prefill_chunk_gate.py 等）刻意绕开 torch；把 torch 留在函数体内，
  这条依赖就只在真的跑 GPU 时才成立。
"""


class MixedBatchPlan:
    """一个 step 内同时承载两条执行路径的执行计划（Scheduler -> LLMEngine 的契约）。

    ★ 为什么不复用 `is_prefill` 这个布尔：它只有两种取值，而本步要同时表达
      「谁走 decode 图」和「谁走 eager 分片」。硬塞进一个布尔就是 12.2 那次
      `spec_enabled` 门控快照 bug 的同款错误 —— 用隐式约定代替显式契约
      （见 docs/优化报告.md §d.2 与 tests/test_spec_gate.py）。
      所以这里给两条路径各自一个明确的容器，外加一个 mode 字段。
    """

    __slots__ = ("prefill_seqs", "decode_seqs", "mode")

    def __init__(self, prefill_seqs, decode_seqs, mode="concurrent"):
        assert prefill_seqs and decode_seqs, (
            "混合批的两条路径都必须非空 —— 空的那一条应当退回老的两元组形状")
        self.prefill_seqs = prefill_seqs
        self.decode_seqs = decode_seqs
        self.mode = mode

    @property
    def seqs(self):
        """两条路径的并集（只会用在"哪些序列本步结束了"这种只读统计上）。"""
        return self.prefill_seqs + self.decode_seqs


class MixedForward:
    """把「decode 子批 + prefill 块」真正跑起来（串行 / 并发两种发射方式）。

    参数
    ----
    runner : ModelRunner
        必须提供 `run(seqs, is_prefill)`、`_prefill_forward(seqs)`、`sampler`、`rank`。

    ★ 为什么要一条【独立】的 aux stream 而不是复用当前流：
      当前流是 PyTorch 的默认流，prefill 的 eager 前向与 decode 的图 replay
      若都在它上面排队，就是"两个前向串着跑"（= 收益 0 的那一档）。
      PyTorch 建的 stream 默认带 `cudaStreamNonBlocking`，与默认流之间【没有】
      CUDA 遗留默认流那种隐式同步 ⇒ 真并发。
    """

    def __init__(self, runner):
        self.runner = runner
        self._stream = None
        # 结构验收打点（探针读它）：本进程跑过多少个混合 step、各是哪种模式。
        self.stats = {"serial_steps": 0, "concurrent_steps": 0, "prefill_tokens": 0}

    @property
    def stream(self):
        """惰性建 aux stream（建 stream 要 CUDA 上下文，不能放在 __init__ 里猜）。"""
        if self._stream is None:
            import torch
            # ★ 低优先级：prefill 是算力型长任务（284 ms），decode 是带宽型短任务
            #   （12.1 ms）。让 prefill 拿低优先级，空闲 SM 优先派给 decode 的
            #   kernel —— 我们要的是"decode 不被 prefill 挡住"，不是反过来。
            try:
                lo, hi = torch.cuda.Stream.priority_range()   # (最小优先级值, 最大优先级值)
                prio = hi                                     # 数值越大 = 优先级越低
            except Exception:
                prio = 0
            self._stream = torch.cuda.Stream(priority=prio)
        return self._stream

    def run(self, decode_seqs, prefill_seqs, mode):
        """跑一个混合 step，返回 (decode_token_ids, prefill_token_ids)。"""
        if mode == "serial":
            return self._run_serial(decode_seqs, prefill_seqs)
        if mode == "concurrent":
            return self._run_concurrent(decode_seqs, prefill_seqs)
        raise ValueError(f"未知的 spec_overlap_mode: {mode!r}（只认 serial / concurrent）")

    # ---------- P15-串行：同 step、两个前向串着跑（★ 对照组，必须测出收益 0）----------
    def _run_serial(self, decode_seqs, prefill_seqs):
        """★ 这一档【故意】不做任何重叠：两个前向按顺序发到同一条流上。

        它的意义是把「混在同一个 step」这个动作【单独】隔离出来：
        总工作量不变、发射顺序不变，只有"调度器把两件事排进了同一步"。
        如果它与让路式的 ITL 一模一样（预期如此），就证明了收益不来自混批本身。
        """
        self.stats["serial_steps"] += 1
        dec = self.runner.run(decode_seqs, False)
        pf = self.runner.run(prefill_seqs, True)
        self.stats["prefill_tokens"] += sum(s.num_scheduled_tokens for s in prefill_seqs)
        return dec, pf

    # ---------- P15-并发：两条 stream，prefill 提前发射、decode 不等它 ----------
    def _run_concurrent(self, decode_seqs, prefill_seqs):
        import torch
        from nanovllm.utils.context import reset_context

        r = self.runner
        aux = self.stream
        cur = torch.cuda.current_stream()

        # aux 必须先等默认流上的【上一步收尾】（KV 写入 / H2D 拷贝）落地，
        # 否则这一步的 prefill 可能在上一步的 K/V 还没写完时就去读自己的 KV。
        # （wait_stream 只插一个事件，不阻塞 Python —— 这正是能重叠的前提。）
        aux.wait_stream(cur)

        # ★★ 顺序是刻意的：先把 prefill 的 kernel【全部发射出去】（不采样、不 .tolist），
        #    再跑 decode。理由见 I4 —— kernel 一旦发射就与 Python 侧的 context 解绑，
        #    所以两条路径可以各自 set_context 后发射，互不干扰。
        #    ★ 反过来说：任何在发射阶段的 .tolist() 都会同步当前流，把重叠自己取消掉。
        #      所以 prefill 的采样必须被推迟到 join 之后（run_verify 里的
        #      `_last_cu_seqlens_q.tolist()` 是 CPU 张量，不在这个约束里）。
        with torch.cuda.stream(aux):
            logits, temps = r._prefill_forward(prefill_seqs)

        # decode 走当前流（照常吃 CUDA Graph），它内部的 .tolist() 只同步当前流，
        # 不会把 aux stream 上的 prefill 一起等掉。
        dec = r.run(decode_seqs, False)

        # ---------- 汇合 ----------
        # 采样放在 aux stream 上做：logits 是 aux stream 上分配的，让它整条生命
        # 周期都待在 aux stream，不需要跨流 record_stream（少一个出错面）。
        with torch.cuda.stream(aux):
            pf = r.sampler(logits, temps).tolist() if r.rank == 0 else None
        # 让默认流等 aux 排空：这一步的 prefill KV 写入必须先落地，
        # 下一步的 H2D 拷贝 / 解码才会有正确的可见性。
        cur.wait_stream(aux)
        # 与老路径一致：一轮结束就重置 context（prepare_prefill 设过它）。
        reset_context()

        self.stats["concurrent_steps"] += 1
        self.stats["prefill_tokens"] += sum(s.num_scheduled_tokens for s in prefill_seqs)
        return dec, pf
