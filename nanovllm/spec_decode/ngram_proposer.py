"""
n-gram 提议器 —— 投机解码的「草稿来源」。

核心思想（不训练任何模型）：
    从【已经出现过的文本】里捞出最可能的后续 token 作为候选。
    已出现过的文本 = 当前请求的 prompt + 已生成的 token
                 + 历史请求的输出（global 池）

这叫 prompt lookup / n-gram 投机：
    prompt 里写过「综上所述」，后面写到「综上所述」时
    就把提示词里「综上所述」后面跟的那个词捞出来当候选。

参考实现：vLLM 的 ngram_proposer.py（KMP / lps 翻转向量找最长匹配后缀）。
本文件是按同一算法重写的精简版，只处理「线性链」，不构造树。
"""

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


class NgramProposer:
    """维护 n-gram 索引，在其中检索候选链。

    索引结构：每个 (n-1) 元组映射到其后所有出现过的 token 列表。
    例：「综上所述」→ 它后面跟过的 token 有 ['，', '。', '我们']

    参数
    ----
    n : int
        候选链的构造方式由 n 决定最大长度（见 propose 的 k）。
        n=3 表示用 3-gram：匹配到「综上所述」+ 后一个字后，继续往后接 1 个。
    window : int
        每个 key 最多保留多少个后继。越大覆盖越全，内存也越大。
        默认 32，对应一般任务足够。
    """

    def __init__(self, n: int = 3, window: int = 32):
        if n < 2:
            raise ValueError("n 必须 >= 2")
        self.n = n
        self.window = window
        # (n-1) 元组 -> 后继 token 的队列
        self._index: Dict[Tuple[int, ...], Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    # ------------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------------
    def add(self, token_ids: Sequence[int]) -> None:
        """把一段 token 加入索引。重复调用安全（幂等，且自动去重去旧）。"""
        n = self.n
        for i in range(len(token_ids) - n + 1):
            key = tuple(token_ids[i:i + n - 1])
            nxt = token_ids[i + n - 1]
            self._index[key].append(nxt)

    def add_global(self, token_ids: Sequence[int]) -> None:
        """历史请求输出进global 池。"""
        self.add(token_ids)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def propose(
        self,
        recent: Sequence[int],
        k: int,
        prompt_len: int = 0,
    ) -> List[int]:
        """从 recent（当前已有的 token，通常是 prompt + 已生成）里检索候选链。

        返回长度 <= k 的候选 token 列表。
        返回空列表表示「没找到任何匹配」——调用方应跳过这一步投机。

        策略
        ----
        1. 取 recent 的【最长】可用后缀去匹配：
           匹配越长的 pattern，候选越可靠（与 SuffixDecoding 的实测一致）。
        2. 命中后，沿着索引一路往后接，最多接 k 个。
        3. 一路接不上就停，返回已接到的部分。

        ★ 重要特性：短 prompt 时几乎必然返回空。
          因为索引里的 (n-1) 元组需要有一个「后继 token」才能建立，
          而 prompt 末尾的 n-gram 天然没有后继。所以：
            - prompt 只有几十 token  -> 大部分步都提不出候选
            - 文本长到几百 token 以上 -> 索引丰富，接受率才上得去
          ★ 这正是检索式投机解码对【任务类型】敏感的原因：
            高 n-gram 重叠的任务（代码编辑/摘要/RAG）效果好，
            低重叠的任务（创意写作）几乎没收益甚至负收益。
        """
        n = self.n
        if len(recent) < n - 1:
            return []

        # 从最长 pattern 往下试，直到命中
        for length in range(n - 1, 0, -1):
            if len(recent) < length:
                continue
            key = tuple(recent[-length:])
            if key in self._index:
                return self._walk(key, k)
        return []

    def observe(self, token_ids: Sequence[int]) -> None:
        """把【刚刚生成】的 token 增量加入索引。

        ★ 这是 n-gram 投机能不能真正work 的关键：
          add() 是全量建索引（用于 prompt 和历史输出），
          但生成出来的 token 如果不入索引，下一步就永远提不出以它结尾的候选。
          实测：只add(prompt) 时，5-token 的 prompt 提议结果恒为空。

        只需要加入「刚好够构成 n-1 元组」的那一小段，避免 O(n²)。
        """
        n = self.n
        need = n - 1
        if len(token_ids) < need:
            return
        # 只取尾部 need 个 token + 它前面的那一个，构成最后一个 key
        start = len(token_ids) - need - 1
        if start < 0:
            start = 0
        self.add(token_ids[start:])

    def _walk(self, key: Tuple[int, ...], k: int) -> List[int]:
        """从命中的 key 开始，沿着索引一路往后接。"""
        out: List[int] = []
        cur = key
        for _ in range(k):
            cand = self._index.get(cur)
            if not cand:
                break
            # 取最常见的那个后继（deque 尾部是最近的；这里用 FIFO 语义最稳）
            nxt = cand[-1]
            out.append(nxt)
            # 下一个 key = 去掉头部，加上新 token
            cur = (cur[1:] + (nxt,)) if len(cur) > 1 else (nxt,)
        return out

    # ------------------------------------------------------------------
    # 统计（用于实验记录）
    # ------------------------------------------------------------------
    @property
    def num_keys(self) -> int:
        return len(self._index)

    def stats(self) -> str:
        total = sum(len(v) for v in self._index.values())
        return f"n={self.n} window={self.window} keys={self.num_keys} edges={total}"

    def clear(self) -> None:
        self._index.clear()