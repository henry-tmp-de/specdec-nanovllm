"""token 交付 hook —— TTFT / TPOT / ITL 的接入点（默认关闭，零开销）。

为什么必须按「批」记
--------------------
投机解码一个 step 会一次落地 k+1 个 token（被接受的候选 + bonus），
它们在【同一时刻】才对用户可见（同一次前向、同一次返回）。
若把它们当成 k+1 个独立的交付时刻去均摊或插值，就会凭空造出一条
「逐 token 均匀到达」的曲线 —— ITL 被算小 k 倍、TPOT 跟着失真。
所以这里记的是 (时刻, 本批 token 数)，【一批一条】。

两个容易做错的口径
------------------
· 计时用 time.perf_counter()（单调钟），不是 time.time() —— 后者会被
  NTP 校时 / 夏令时跳变污染。
· TTFT 的参考点是 on_request_added 的【入队时刻】（用户视角），
  不是第一次 forward 的时刻 —— 排队等待也是用户等的时间。

怎么用（只在测量时打开，正式跑性能时不挂）
------------------------------------------
    hook = TokenDeliveryHook()
    llm = LLM(model, ..., token_hook=hook)
    llm.generate(prompts, sp)
    print(hook.summary())

热路径上的开销只有一个 `is not None` 判断：不挂 hook 时不取时间、不建对象。
"""

from time import perf_counter


class TokenDeliveryHook:
    """记录每个请求的 token 交付时刻。"""

    def __init__(self):
        self.requests: dict[int, dict] = {}

    # ==================================================================
    #  写入端（引擎在交付点调用）
    # ==================================================================
    def on_request_added(self, seq):
        self.requests[seq.seq_id] = {
            "arrival": perf_counter(),   # 入队时刻
            "first_token": None,         # 首个 token 交付时刻（成批时=这一批的时刻）
            "last_token": None,          # 最后一次交付时刻
            "end": None,                 # 请求结束时刻
            "num_tokens": 0,             # 累计交付的 token 数
            "batches": [],               # [(时刻, 本批 token 数)]，投机一轮一条
        }

    def on_deliver(self, seq, num_tokens: int, finished: bool = False):
        """一步交付一批：普通 decode 是 1 个，投机一轮是 k+1 个。"""
        record = self.requests.get(seq.seq_id)
        if record is None:
            # hook 中途才挂上（这些请求入队时还没记时刻）→ 补一条，别丢数据
            self.on_request_added(seq)
            record = self.requests[seq.seq_id]
        if num_tokens <= 0:
            return
        now = perf_counter()
        if record["first_token"] is None:
            record["first_token"] = now
        record["last_token"] = now
        record["num_tokens"] += num_tokens
        record["batches"].append((now, num_tokens))
        if finished:
            record["end"] = now

    # ==================================================================
    #  读端（只在测量脚本里调，不在热路径上）
    # ==================================================================
    def ttft(self, seq_id: int) -> float:
        """首 token 延迟 = 首个 token 交付时刻 - 入队时刻。"""
        r = self.requests[seq_id]
        if r["first_token"] is None:
            return float("nan")
        return r["first_token"] - r["arrival"]

    def e2e(self, seq_id: int) -> float:
        """端到端耗时 = 结束时刻 - 入队时刻（还没结束就用最后一次交付时刻）。"""
        r = self.requests[seq_id]
        end = r["end"] if r["end"] is not None else r["last_token"]
        if end is None:
            return float("nan")
        return end - r["arrival"]

    def tpot(self, seq_id: int) -> float:
        """平均每输出 token 耗时 = (最后交付 - 首交付) / (交付数 - 1)。

        ★ 分母只用【首 token 之后】的 token：TTFT 是排队 + prefill 的钱，
          混进 TPOT 会把首 token 的延迟摊到后面每个 token 上（序列越长摊得越薄，
          看起来越"快"）。成批交付时也只按交付数算，不按批数算。
        """
        r = self.requests[seq_id]
        n = r["num_tokens"]
        if r["first_token"] is None or n <= 1:
            return float("nan")
        return (r["last_token"] - r["first_token"]) / (n - 1)

    def itl(self, seq_id: int) -> list:
        """批间间隔（秒）：相邻两次交付的时刻差。

        投机一轮 k+1 个 token 只产生【一个】间隔 —— 见文件头的说明。
        """
        ts = [t for t, _ in self.requests[seq_id]["batches"]]
        return [b - a for a, b in zip(ts, ts[1:])]

    def summary(self) -> dict:
        """逐请求的派生指标，直接可打表 / 写 json。"""
        return {
            sid: {
                "num_tokens": r["num_tokens"],
                "n_batches": len(r["batches"]),
                "ttft": self.ttft(sid),
                "tpot": self.tpot(sid),
                "e2e": self.e2e(sid),
                "finished": r["end"] is not None,
            }
            for sid, r in sorted(self.requests.items())
        }
