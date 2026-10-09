"""A 步量化：前缀缓存命中引发的「全量 draft 补齐」到底花多少时间。

背景
----
`Sequence.draft_valid_len`（draft 侧 KV 的有效水位）在前缀缓存命中时被保守地
压在 0（见 model_runner.prepare_prefill），于是 propose 前必须把
`[0, len(seq)-1)` 整段 token 逐个喂回 draft 模型 —— 命中 4096 token 前缀
就要补 ~4096 次前向。本脚本量化这笔开销。

三组负载（同样的 4160-token prompt，max_model_len=4608）
------------------------------------------------------
HOT   「写者」请求先跑完，把共享前缀写进（target 的）前缀缓存与（draft 的）KV；
      再一次性提交 N 条共享该前缀、仅后缀不同的请求 —— 前缀缓存命中。
HOTX  与 HOT 逐字相同，但探针侧把 catchup_tokens 置空（monkeypatch，不改生产
      代码）→ 因果对照；HOT − HOTX 就是补齐的净代价。
COLD  N+1 条前缀互不相同的请求一次性提交 —— 没有任何缓存命中，作为「没有这笔
      开销」的参照。

★ 计时口径
  · 逐 step 墙钟单独记录（不挂 profiler），补齐那一 step 就是补齐的耗时上界;
    每前向耗时 = 补齐 step 墙钟 / catchup_forwards。
  · 前缀缓存冷热状态逐条记录（can_allocate 的命中块数），不靠推测。
  · 每 rep 独立进程 + 全新引擎（run_a7.sh）。

用法: python a7_quant.py <group: HOT|HOTX|COLD> <N> <LS> <SUF> <OUT> <REP> [k]
"""
import os
import sys
import json
import hashlib
from time import perf_counter

CODE_ROOT = os.environ.get("NV_ROOT", "/home/ziru/nano-vllm/p1-work")
sys.path.insert(0, CODE_ROOT)

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.token_hook import TokenDeliveryHook

TARGET = os.path.expanduser("~/nano-vllm/models/Qwen3-4B")
DRAFT = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
VOCAB = 151936

# ---- 与 p6_bench.py 逐字相同的语料（保证与冻结基线可比）----
ZH = [
    "在机器学习中，梯度下降是一种常用的优化算法，它的基本思想是沿着损失函数下降最快的方向不断调整参数。学习率过大可能导致震荡甚至发散，过小则收敛缓慢，因此常常需要配合学习率调度策略使用。",
    "分布式训练需要解决通信瓶颈问题。常见的并行方式包括数据并行、张量并行和流水线并行。数据并行通过梯度同步来保持各副本一致，通信量随参数量线性增长，因此梯度压缩和通信重叠成为关键优化点。",
    "操作系统中的虚拟内存机制把物理内存抽象成连续的地址空间，通过页表完成地址翻译。缺页中断发生时，内核需要从磁盘换入页面，如果此时空闲页框不足，就要按照置换策略驱逐某些页面。",
    "数据库的索引结构决定了查询性能。B+ 树通过多路平衡保持较低的树高，叶子节点之间用链表连接，从而支持高效的范围扫描。相比之下，哈希索引只适合等值查询，无法处理排序和区间条件。",
    "编译器的中间表示是连接前端和后端的桥梁。静态单赋值形式让每个变量只被赋值一次，从而简化了数据流分析。基于此可以方便地实现常量传播、死代码消除和公共子表达式消除等优化。",
    "在网络协议栈中，拥塞控制需要在吞吐量和公平性之间取得平衡。慢启动阶段指数增长发送窗口，到达阈值后转为线性增长。快速重传与快速恢复通过重复确认来推断丢包，避免等待超时。",
    "现代处理器的流水线依靠分支预测来维持指令吞吐。预测错误会导致流水线冲刷，代价通常相当于十几个周期。因此编译器会尽量消除分支，或者使用条件传送指令来减少不可预测的跳转。",
    "推荐系统通常分为召回和排序两个阶段。召回阶段从海量候选中快速筛选出数百条，排序阶段再用复杂模型精细打分。特征工程中，用户行为序列的建模对效果影响最大，常使用注意力机制来聚合。",
    "强化学习中的信用分配问题是核心难点。时序差分方法用自举的方式估计价值函数，偏差小而方差大；蒙特卡洛方法无偏但方差高。广义优势估计通过参数在两者之间插值，取得折中。",
    "文件系统的日志机制保证了崩溃一致性。写操作先追加到日志区，提交后再写回数据区，最后清理日志。这种预写日志的方式把随机的元数据更新变成顺序写，显著提升了可靠性。",
    "注意力机制的计算复杂度随序列长度平方增长，这限制了长文本场景的应用。稀疏注意力和线性注意力通过限制感受野或核函数近似来降低复杂度，但往往以表达能力的下降为代价。",
    "量化是模型压缩的重要手段。训练后量化把权重映射到低位宽整数，配合逐通道缩放因子来减小误差。量化感知训练在训练时模拟量化噪声，通常能获得比训练后量化更好的精度。",
]
CODE = [
    "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n    left = merge_sort(items[:mid])\n    right = merge_sort(items[mid:])\n    return merge(left, right)\n",
    "class RingBuffer:\n    def __init__(self, capacity):\n        self.buf = [None] * capacity\n        self.head = 0\n        self.size = 0\n    def push(self, value):\n        idx = (self.head + self.size) % len(self.buf)\n        self.buf[idx] = value\n",
    "def dijkstra(graph, source):\n    dist = {node: float('inf') for node in graph}\n    dist[source] = 0\n    visited = set()\n    while len(visited) < len(graph):\n        node = min((n for n in graph if n not in visited), key=lambda n: dist[n])\n",
    "async def fetch_all(session, urls):\n    tasks = [session.get(url) for url in urls]\n    results = await asyncio.gather(*tasks, return_exceptions=True)\n    return [r for r in results if not isinstance(r, Exception)]\n",
    "struct LRUCache {\n    map: HashMap<K, usize>,\n    entries: Vec<(K, V)>,\n}\nimpl LRUCache {\n    fn insert(&mut self, key: K, value: V) {\n        if self.map.contains_key(&key) { self.touch(&key); }\n",
    "def quicksort(arr, lo, hi):\n    if lo >= hi:\n        return\n    pivot = arr[hi]\n    i = lo - 1\n    for j in range(lo, hi):\n        if arr[j] <= pivot:\n            i += 1\n            arr[i], arr[j] = arr[j], arr[i]\n    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]\n",
    "SELECT u.name, COUNT(o.id) AS orders\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\nWHERE o.created_at >= '2024-01-01'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5\nORDER BY orders DESC\nLIMIT 20;\n",
    "export function useDebounced(value, delay) {\n  const [debounced, setDebounced] = useState(value);\n  useEffect(() => {\n    const timer = setTimeout(() => setDebounced(value), delay);\n    return () => clearTimeout(timer);\n  }, [value, delay]);\n  return debounced;\n}\n",
    "class Matrix:\n    def __init__(self, rows, cols):\n        self.data = [[0.0] * cols for _ in range(rows)]\n    def matmul(self, other):\n        n, m, p = len(self.data), len(other.data), len(other.data[0])\n        out = Matrix(n, p)\n        for i in range(n):\n            for k in range(m):\n",
    "func worker(id int, jobs <-chan int, results chan<- int) {\n    for j := range jobs {\n        results <- j * j\n    }\n}\n\nfunc main() {\n    jobs := make(chan int, 100)\n    results := make(chan int, 100)\n    for w := 1; w <= 3; w++ {\n        go worker(w, jobs, results)\n    }\n}\n",
    "def tokenize_batch(texts, vocab, max_len):\n    out = []\n    for text in texts:\n        ids = [vocab.get(ch, vocab['<unk>']) for ch in text]\n        if len(ids) > max_len:\n            ids = ids[:max_len]\n        else:\n            ids += [vocab['<pad>']] * (max_len - len(ids))\n        out.append(ids)\n    return out\n",
    "impl fmt::Display for Config {\n    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n        write!(f, \"Config lr {} batch {} epochs {}\", self.lr, self.batch, self.epochs)\n    }\n}\n\nfn load_config(path: &Path) -> Result<Config, io::Error> {\n    let text = fs::read_to_string(path)?;\n",
]

_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TARGET, use_fast=True)
    return _TOK


def build_prompt(kind, pool_idx, L, lead_seed):
    """lead(4 个唯一 token) +（轮转语料 + 递增编号打破周期性），凑成恰好 L 个 token。"""
    pool = ZH if kind == "zh" else CODE
    g = torch.Generator().manual_seed(lead_seed)
    lead = torch.randint(1, VOCAB, (4,), generator=g).tolist()
    need = L - len(lead)
    t = tok()
    text, ids, i = "", [], 0
    while len(ids) < need and i < 2000:
        j = (pool_idx + i) % len(pool)
        sep = ("\n// sec %d\n" % i) if kind != "zh" else ("\n[第%d段] " % i)
        text = text + sep + pool[j]
        ids = t.encode(text, add_special_tokens=False)
        i += 1
    assert len(ids) >= need, (kind, L, len(ids))
    return lead + ids[:need]


# ------------------------------------------------------------------
# 探针
# ------------------------------------------------------------------
HITS = []
STEPS = []
_orig_step = None


def install_can_allocate_probe():
    """记录每条请求命中了多少块 —— 只在 main() 里装，避免 import 时污染别处。"""
    _orig_ca = BlockManager.can_allocate

    def _ca(self, seq):
        r = _orig_ca(self, seq)
        HITS.append([seq.seq_id, r])
        return r

    BlockManager.can_allocate = _ca


def report(d):
    print("@@B@@" + json.dumps(d, ensure_ascii=False), flush=True)


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else None


def kv_bytes(hf):
    nh = hf.num_key_value_heads
    hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    return 2 * hf.num_hidden_layers * 256 * nh * hd * hf.dtype.itemsize


def main():
    group = sys.argv[1]
    N = int(sys.argv[2])          # 命中前缀缓存的请求数（不含写者）
    LS = int(sys.argv[3])         # 共享前缀长度
    SUF = int(sys.argv[4])        # 后缀长度
    OUT = int(sys.argv[5])
    REP = int(sys.argv[6])
    K = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    L = LS + SUF
    MML = 4608
    install_can_allocate_probe()

    no_catchup = group == "HOTX"
    if no_catchup:
        _o_req = ModelRunner._draft_request

        def _strip(self, seq):
            d = _o_req(self, seq)
            d["catchup_tokens"] = []
            return d

        ModelRunner._draft_request = _strip

    # ---------- 构造 prompt ----------
    # 共享段（LS 个 token）：hot 组所有请求逐字相同；cold 组各不相同
    shared = build_prompt("zh", 2, LS, 70000 + LS)
    prom, meta = [], []
    writer = shared + build_prompt("code", 3, SUF, 71000 + REP)
    prom.append(writer)
    meta.append(dict(role="writer", n=len(writer), md5=hashlib.md5(
        json.dumps(writer).encode()).hexdigest()))
    for i in range(N):
        if group == "COLD":
            # 无复用：每条前缀都不一样（换 pool_idx + seed + 语言）
            kind = "zh" if i % 2 == 0 else "code"
            head = build_prompt(kind, (i * 5 + REP) % 12, LS, 90000 + i * 977 + REP)
        else:
            head = shared
        p = head + build_prompt("code", (i + 5) % 12, SUF, 72000 + i * 131 + REP * 17)
        prom.append(p)
        meta.append(dict(role="hitter" if group != "COLD" else "unique",
                         n=len(p), md5=hashlib.md5(json.dumps(p).encode()).hexdigest()))

    # max_num_batched_tokens 与 p6_bench 保持一致（16384）—— 它决定 warmup 的
    # 激活峰值，进而决定 num_kvcache_blocks，换了口径块数就不可比。
    kw = dict(max_model_len=MML, max_num_batched_tokens=16384,
              max_num_seqs=8, enforce_eager=False,
              spec_k=K, spec_method="draft", draft_model=DRAFT,
              spec_batch_threshold=0)
    hook = TokenDeliveryHook()
    llm = LLM(TARGET, token_hook=hook, **kw)
    mr = llm.model_runner
    prop = mr.spec_proposer
    bm = llm.scheduler.block_manager
    hf = mr.config.hf_config

    # 逐 step 计时（含 schedule + run + postprocess 全过程，不挂 profiler）
    from nanovllm.engine.llm_engine import LLMEngine
    global STEPS
    STEPS = []

    def _step(self):
        t = perf_counter()
        r = _orig_step(self)
        STEPS.append(perf_counter() - t)
        return r

    _orig_step = LLMEngine.step
    LLMEngine.step = _step

    sp_warm = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
    llm.generate([prom[0]], sp_warm, use_tqdm=False)   # 让引擎热起来（图捕获等）

    # ---------- 正式测量 ----------
    sp = SamplingParams(temperature=1.0, max_tokens=OUT, ignore_eos=True)
    bm.hash_to_block_id.clear()      # ★ 冷启动：清空前缀缓存哈希表
    hook.requests.clear()
    HITS.clear()
    d0 = dict(rounds=prop.n_rounds, batch_forwards=prop.n_batch_forwards,
              graphed=prop.n_graphed, eager=prop.n_eager,
              catchup_forwards=prop.n_catchup_forwards,
              catchup_tokens=prop.n_catchup_tokens)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = perf_counter()

    # 第一波：只跑「写者」，把共享前缀写进缓存
    h0 = len(HITS)
    llm.generate([prom[0]], sp, use_tqdm=False)
    wave1_wall = perf_counter() - t0
    hits_w1 = list(HITS[h0:])
    # 第二波：N 条请求同时提交 → 前缀缓存命中（hot/cold 的差别只在这里）
    t1 = perf_counter()
    h1 = len(HITS)
    outs = llm.generate(prom[1:], sp, use_tqdm=False)
    wave2_wall = perf_counter() - t1
    hits_w2 = list(HITS[h1:])
    torch.cuda.synchronize()
    wall = perf_counter() - t0

    total_out = sum(len(o["token_ids"]) for o in outs)
    s = hook.summary()
    ttfts, details = [], []
    for sid in sorted(s.keys()):
        r = s[sid]
        has_first = hook.requests[sid]["first_token"] is not None
        if has_first:
            ttfts.append(r["ttft"])
        details.append(dict(sid=sid, n_tokens=r["num_tokens"], n_batches=r["n_batches"],
                            ttft=round(r["ttft"], 5) if has_first else None,
                            tpot=round(r["tpot"], 6) if r["tpot"] == r["tpot"] else None))

    d1 = dict(rounds=prop.n_rounds, batch_forwards=prop.n_batch_forwards,
              graphed=prop.n_graphed, eager=prop.n_eager,
              catchup_forwards=prop.n_catchup_forwards,
              catchup_tokens=prop.n_catchup_tokens)
    d = {kk: d1[kk] - d0[kk] for kk in d0}

    # 补齐发生在哪一 step？取「第二波里最慢的一 step」作为补齐 step。
    # 第二波起始 step 索引 = 第一波跑完的 step 数（逐 step 列表是全局的）
    step_times = list(STEPS)
    top = sorted(range(len(step_times)), key=lambda i: -step_times[i])[:3]
    slow = [dict(step=i, ms=round(step_times[i] * 1000, 2)) for i in top]

    print("[diag] wave2 cache_hits=%s  catchup_tokens=%s  catchup_fwd=%s  steps=%d  max_step_ms=%.1f"
          % ([h[1] for h in hits_w2], d["catchup_tokens"], d["catchup_forwards"],
             len(step_times), (max(step_times) * 1000 if step_times else -1)), flush=True)

    info = dict(
        group=group, no_catchup=no_catchup, N=N, LS=LS, SUF=SUF, L=L, out=OUT, rep=REP, k=K,
        prompts=meta, num_kvcache_blocks=len(bm.blocks),
        target_kv_block_kb=round(kv_bytes(hf) / 1024, 1),
        draft_kv_block_kb=round(kv_bytes(mr.draft_hf_config) / 1024, 1),
        step_count=len(step_times),
        wave1_wall_s=round(wave1_wall, 4), wave2_wall_s=round(wave2_wall, 4),
        wall_s=round(wall, 4), total_out_tokens=total_out,
        output_tok_per_s=round(total_out / wall, 2),
        wave2_tok_per_s=round(total_out / wave2_wall, 2) if wave2_wall else None,
        ttft_median_ms=round(median(ttfts) * 1000, 2) if ttfts else None,
        ttft_min_ms=round(min(ttfts) * 1000, 2) if ttfts else None,
        ttft_max_ms=round(max(ttfts) * 1000, 2) if ttfts else None,
        slowest_steps=slow,
        draft_counters=d,
        catchup_ms_est=(round(step_times[top[0]] * 1000, 2) if top else None),
        cache_hits_wave1=hits_w1,
        cache_hits_wave2=hits_w2,
        mem_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        per_request=details,
    )
    report(info)


if __name__ == "__main__":
    main()
