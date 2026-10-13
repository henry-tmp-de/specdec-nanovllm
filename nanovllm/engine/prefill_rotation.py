"""P14：多序列 partial prefill 的轮转准入（Round-Robin Admission）。

要解决的问题（已量化）
----------------------
`Scheduler.schedule()` 的 prefill 循环原来**只看队首**，且分块拼不完就 `break`
⇒ 3 条长 prompt 同时到达时，只有第 1 条在推进。
实测（3 条各 16K、chunk=1024）：第 1 条在第 16 轮完成、第 2 条第 32 轮、
**第 3 条第 48 轮** —— 最后一条等 3 倍时间（见 docs/优化报告.md §d.5 方向 1）。
★ 总工作量完全一样（48×1024 个 token 一分不多）⇒ 这是**纯粹的调度损失**，
不是算力不够。

外部依据（三个独立的生产 issue 指向同一个根因：瓶颈是调度准入，不是显存）
------------------------------------------------------------------------
· SGLang issue #32549 —— 标题实测 "Decode starved to ~1 batch per 24s under
  sustained chunked-prefill load"
· SGLang issue #35537 —— 单槽卡住，KV 只用 ≤4% 也让请求等 4.5 分钟
· vLLM  issue #57413（open）—— TTFT p95 = 76 秒，而 KV < 40%、抢占 0 次
· LMDeploy 已经这么做 —— `_should_try_short_prefill_first()` 默认开，
  且两个长 chunk 之间至少隔 3 轮（比我们保守）

机制：把 waiting 队列当成一个【环】
-----------------------------------
每轮从环头起，给每条待拼序列最多 chunk 个 token，**一条只服务一次**；
本轮没拼完的序列**移到环尾**。于是：
· 公平 —— 环头永远是"最久没被服务"的那条 ⇒ 预算不够时不会总从同一条开始（不饿死）；
· 轮转 —— N 条 partial 在环上均分，每条每绕一圈至少拿到一次 chunk；
· 总工作量不变 —— 只是把同一份 token 重新分配，一分不多。

★★ 不变式：一轮内同一条序列只能被调度一次
------------------------------------------
这是 P13 踩过的**真 bug**：分块拼不完的序列没被 `popleft()`、还留在 `waiting[0]`，
循环不 `break` 就会再取到【同一条】，把 `num_scheduled_tokens` 又覆盖一遍、再 append
一份 ⇒ `scheduled_seqs` 里出现同一序列的多份副本，`prepare_prefill` 把同一段 chunk
重复拼 N 遍（实测 `round 0: step_tokens=14336` = 7 份 2048 的 chunk）。
★ 改成轮转之后循环【必须继续往下走】，这个坑只会更容易踩 —— 所以本类把
"本轮已服务"做成**显式状态**，而不是靠调用方自觉。

为什么单独成模块 / 类
---------------------
"本轮谁上、谁让位"是一个**独立概念**，与 BlockManager 的块分配、Sequence 的状态
推进完全解耦。单列出来有两个好处：
① 用纯列表就能单测（tests/test_prefill_multiseq.py 不必铺 Scheduler 就能钉住
   "不重复调度"与"公平性"两条不变量）；
② `schedule()` 的热路径上只多两句调用，不必堆一串 if。
"""
from collections import deque


class PrefillRotation:
    """环状轮转准入的状态机（不持有任何 Sequence 的业务语义）。

    典型用法（见 Scheduler.schedule 的 prefill 段）::

        rot = PrefillRotation()
        while waiting and len(scheduled_seqs) < max_num_seqs:
            rot.reset()                       # ★ 每轮只做一次，见 reset 的说明
            ...
            seq = rot.next(waiting)           # None = 本轮结束（空队列 / 已绕完一圈）
            if seq is None:
                break
            ...
            rot.mark(seq)                     # 拼完了 → 只记一下（它已被移出 waiting）
            rot.yield_turn(waiting, seq)      # 没拼完 → 记一下 + 移到环尾
            rot.skip(waiting, seq)            # 拿不到块 → 记一下 + 让位给后面

    ★ 本类**不判断**"还剩多少预算"、"这条还需要多少 token"—— 那是 schedule() 的事。
      它只回答一个问题：**这一轮的下一条该是谁，以及谁已经上过了。**
    """

    def __init__(self):
        # 本轮已经服务过的序列（按 id 记，见 next/mark 的说明）
        self._served: set[int] = set()

    def reset(self):
        """开始新一轮：清空"本轮已服务"。

        ★ 必须在**每轮的 prefill 段开头**调用一次（且只一次）：清早了会把上一轮的
          记录带进来（后面的序列整轮没机会），清晚了会把同一轮切成两半
          （一条被服务两次 —— 正是 §不变式 要防的）。
        """
        self._served.clear()

    def next(self, waiting: deque):
        """本轮该服务的下一条序列；`None` = 本轮到此结束。

        返回 `None` 的两种情况，语义完全不同，调用方按"直接 break"处理即可：
        · `waiting` 空 —— 没有待拼的请求了；
        · 环头这条**本轮已经服务过** —— 说明整环已经绕过一圈（`yield_turn` 把
          没拼完的逐条移到环尾之后，环头又转回了本轮第一条）。
        """
        if not waiting:
            return None
        seq = waiting[0]
        if id(seq) in self._served:
            return None
        return seq

    def mark(self, seq):
        """记下"这条本轮已经服务过"（拼完了、已被 `popleft()` 移出 waiting）。

        ★ 拼完的也要记：它可能在同一轮里被别的路径再次取到（例如 `waiting` 里
          还有同一条的引用），记了才不会重复拼。
        """
        self._served.add(id(seq))

    def yield_turn(self, waiting: deque, seq):
        """没拼完 → 记下已服务 + 把队首移到**环尾**，让位给后面的序列。

        ★ 这就是"轮转"的全部实现：只动 deque 的顺序，不碰序列状态 ——
          它仍留在 waiting 里，下一轮轮到它时从 `num_cached_tokens` 接着往后拼。
        ★ 移尾而不是插回队首：插回队首会让同一条在预算不足时反复抢占环头，
          后面的请求被永远饿死（vLLM #57413 那类"KV 没满却在等"的现象）。
        """
        self._served.add(id(seq))
        waiting.rotate(-1)

    def skip(self, waiting: deque, seq):
        """这条本轮拿不到块（`can_allocate() == -1`）→ 记下已服务 + 让位给后面。

        ★ 不能让"一条分配不到块"挡住整条队列的其他人（预算/显存紧张时前者可能
          只是一条特长的请求）。它没有被移出 waiting，下一轮仍会被 `next()` 取到
          —— 顺序上只是排到了本轮其他人的后面。
        """
        self._served.add(id(seq))
        waiting.rotate(-1)
