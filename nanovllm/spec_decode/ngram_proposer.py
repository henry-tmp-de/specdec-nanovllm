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
        # 已入索引的位置水位线（observe 靠它做增量）
        self._indexed_upto = 0

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
        n_candidates: int = 4,
    ) -> List[List[int]]:
        """检索候选链，返回【多条】候选（不是一个 token 的列表）。

        ★★ 为什么返回多条而不是一条：
          同一个 n-gram 在历史上出现过很多次，每次的后继可能不同。
          只返回「最近一次的后继」，一旦它不对，accept rate 就直接归零。
          vLLM 的 ngram_proposer 也是返回多条候选让验证去挑
          （默认 k=4），实测这对accept rate 影响很大。

        返回：最多 n_candidates 条候选链，每条长度 <= k。
              没有匹配时返回空列表。

        策略
        ----
        1. 取 recent 的【最长】可用后缀去匹配（匹配越长候选越可靠）。
        2. 命中的每一条历史出现，各接出一条链 —— 这就是候选多样性来源。
        3. 优先【最近】出现的那些。
        """
        n = self.n
        if len(recent) < n - 1:
            return []

        # 从最长 pattern 往下试，收集【所有】命中的 key（长的优先）
        hit_keys: List[Tuple[int, ...]] = []
        for length in range(n - 1, 0, -1):
            if len(recent) < length:
                continue
            key = tuple(recent[-length:])
            if key in self._index:
                hit_keys.append(key)
                # 更长的 key（更长的 length）已经找到，短的可能也没必要再试，
                # 但为了多样性，继续收集 1 级的。
                if length > 1:
                    continue

        if not hit_keys:
            return []

        out: List[List[int]] = []
        for key in hit_keys:
            for chain in self._walk_multi(key, k, n_candidates):
                out.append(chain)
                if len(out) >= n_candidates:
                    return out
        return out

    def observe(self, token_ids: Sequence[int]) -> None:
        """把【刚刚生成】的 token 增量加入索引。

        ★ 这是 n-gram 投机能不能真正 work 的关键：
          生成出来的 token 如果不入索引，下一步就永远提不出以它结尾的候选。

        ★★ 踩过的坑：早先的实现只取「尾部 n 个 token」建索引，
          结果每次只新增1~2 条边、且 key 与上次的对不齐，
          索引里堆满了 (0,0) -> [0,0,...] 这类垃圾，
          真正的连续链条建不起来 -> 提议恒为空。

        正确做法：用「已建索引到哪里」的水位线（_indexed_upto），
        每次把 [水位线-n, 当前长度) 这段【全部】补进去。
        水位线本身落后 n-1 个 token，保证跨越边界的 n-gram 也能被建到。
        """
        n = self.n
        if len(token_ids) < n:
            return

        start = max(0, self._indexed_upto - (n - 1))
        end = len(token_ids)
        if end <= start:
            return

        # add() 内部会跳过最后不足 n 个的位置，正好是留给下次的
        self.add(token_ids[start:end])
        self._indexed_upto = end

    def reset_watermark(self, token_ids: Sequence[int]) -> None:
        """整段重建索引后，同步水位线（用于 prompt 首次入索引）。"""
        self.add(token_ids)
        self._indexed_upto = len(token_ids)

    def _walk(self, key: Tuple[int, ...], k: int, branch: int = 0) -> List[int]:
        """从命中的 key 出发接一条链。

        branch: 选第几条历史分支。
          0 = 最近一次出现（最常见也最可靠）
          1 = 倒数第二次
          ...
        枚举不同分支是候选多样性的主要来源。
        """
        out: List[int] = []
        cur = key
        for step in range(k):
            cands = self._index.get(cur)
            if not cands:
                break
            # 该 key 有多个历史后继，取第 branch 个（越靠后越久远）
            if branch >= len(cands):
                break
            nxt = cands[len(cands) - 1 - branch]      # deque 尾部是最近的
            out.append(nxt)
            cur = (cur[1:] + (nxt,)) if len(cur) > 1 else (nxt,)
        return out

    def _walk_multi(self, key: Tuple[int, ...], k: int, n: int) -> List[List[int]]:
        """从同一个 key 接出多条不同分支的链。"""
        out = []
        for branch in range(n):
            chain = self._walk(key, k, branch)
            if chain:
                out.append(chain)
            else:
                break        # 该 key 的历史分支不够了
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
        self._indexed_upto = 0