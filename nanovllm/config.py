import os
import json
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 由 ModelRunner.allocate_kv_cache 按预算算出（dataclass 是 slots，字段必须
    # 在此声明，否则运行时赋值会 AttributeError）。
    num_draft_blocks: int = 0        # draft 池的物理块数（滑窗档下 << num_kvcache_blocks）
    draft_window_blocks: int = 0     # 每条序列滑窗的块数 M（0 = 未启用滑窗）
    draft_head_blocks: int = 0       # 每条序列【常驻的开头块】数 P（0 = 不常驻，= 老行为）

    # ---------- 投机解码 ----------
    spec_k: int = 0              # 0 = 关闭（默认，保证与原版行为一致）
    spec_ngram: int = 3          # n-gram 提议器的 n
    spec_method: str = "ngram"   # "ngram" | "draft"
    draft_model: str = ""        # draft 路线的小模型路径（spec_method="draft" 时必填）
    spec_cuda_graph: bool = True   # 给 draft 前向 / 验证前向也拍 CUDA graph
    #   ★ 实测（RTX 3090, Qwen3-4B）：同样是「每步一次前向」，
    #     eager 33.2 ms/步，拍图后 12.8 ms/步 —— 差 2.6 倍，
    #     而这 20 ms 里没有一个字节是权重读取，全是 kernel launch / Python dispatch。
    #     0.6B 的 draft 更极端：权重 1.2 GB（理论 1.3 ms），eager 实测 24.8 ms/步。
    #     draft 要串行跑 k 次，不开图这部分开销会被放大 k 倍，直接吃掉全部收益。
    #     设为 False 可退回纯 eager（只在排查问题时用）。
    spec_batch_threshold: int = 0  # batch 超过此值就关闭投机，0 = 不限制
    #   ★ 动机：实测 batch 大时投机解码是净亏损（验证要付 B·(k+1) 的 target 算力，
    #     而 n-gram 的接受率随上下文多样性上升而下降）。留 0 = 不限制，
    #     但默认建议设 8，benchmark 时能看出盈亏平衡点。

    # ---------- P6：批量 draft 与批量验证图 ----------
    spec_batch_draft: bool = True
    #   True  = 同一候选位置把 B 条请求一起前向（k 次批量前向）
    #   False = 逐请求循环（B×k 次单序列前向）—— 消融 S1 的对照开关（= C 组行为）
    spec_batch_verify_graph: bool = True
    #   True  = B>1 的验证前向走按 (B,k) 捕获的图
    #   False = 只有 B=1 走图，B>1 退回 eager —— 消融 S2 的对照开关
    # ---------- P7 / B：draft 只保留滑窗（**新增选项**，不是替换） ----------
    spec_draft_window: int = 0
    #   0（默认，= 1c1d907 的行为，逐字节等价）
    #        = 关闭滑窗：draft 与 target 共用 block_table，跑【全上下文】draft。
    #         这一条路径的代码原样保留、随时可回退（`BlockManager` 的 draft 池
    #         保持空池，`Sequence.draft_block_table` 恒为空列表）。
    #   W>0  = 打开滑窗：draft 只保留最近 W 个 token 的 KV。
    #         W 必须是 kvcache_block_size 的整数倍；实际窗口 = 最近 W/bs 个块，
    #         所以有效窗口在 W-bs+1 ~ W 个 token 之间。
    #   ★ 打开后 draft 有【自己的一套小池子 + 自己的块表】，不再共用 block_table ——
    #     这是省显存的前提（共用块表时两边块数必须相同，一字节都省不出来）。
    #     代价：draft 不再直接继承别的请求（前缀缓存命中）的 draft KV，
    #     命中时改为自己重算最近 M 个块的窗口（一次前向，不是 W 次）。
    #   ★ draft 池 = max_num_seqs * W/block_size 块，max_num_seqs 越大越吃预算。
    #   ★ 开关在【每次用到 draft 时现场读】，所以同一份二进制里翻转它就能做
    #     全上下文 vs W 的消融；见 ModelRunner._window_blocks / _draft_request。
    # ---------- P7-D：滑窗 + 开头常驻段（**新增选项**，默认 0 = 老行为）----------
    spec_draft_head_blocks: int = 0
    #   >0 = draft 的上下文 = 【开头 P 个块（常驻）】 + 【最近 M 个块（滑窗）】。
    #        纯滑窗（本字段 0）只留最近 M 块，序列【最开头】那几个块对整串有定位
    #        作用，小窗口下丢了它们接受率会掉；常驻段把这段拿回来。
    #   ★ 只影响 draft：target 一个字节不动，"相对 BF16 target 无损"不受影响。
    #   ★ draft 池随之从 M*max_num_seqs 变成 (P+M)*max_num_seqs 块 —— 容量优势会
    #     变小，所以判据仍是 (P+M) * max_num_seqs < 普通投机的池块数。
    #   ★ 与 spec_draft_window 一样是【init 期定池子大小】的开关：翻转要换进程。
    #   ★ 只有 M>0（滑窗开着）时才有效；序列短到 M 段已经盖住开头时，
    #     P 段与 M 段不重叠（见 DraftWindow.b0 的 max(P, ...)）。
    # ---------- INT8 权重量化（W8A16）----------
    # target 与 draft 各有独立开关。默认都是 None = 原 BF16 路径、逐字节等价、
    # 老代码一行不删。模型目录里有 quant_config.json 时【自动开启】并按其中
    # 列出的层名选层，不需要用户逐层指定。
    quant_weights: str | None = None          # target
    quant_granularity: str = "per_channel"
    quant_group_size: int = 128
    quant_modules: tuple = ()                 # target：要量化的引擎模块名
    quant_config: dict | None = None          # quant_config.json 原始内容
    draft_quant_weights: str | None = None    # draft（Qwen3-0.6B）
    draft_quant_granularity: str = "per_channel"
    draft_quant_group_size: int = 128
    draft_quant_modules: tuple = ()
    draft_quant_config: dict | None = None
    # ---------- 预量化直载（P9，只对开了 int8 的模型生效）----------
    quant_direct_load: bool = False
    #   True  = 量化层【在构造期】直接按 int8 分配权重、直接建 bf16 scale，
    #           从「int8 checkpoint（带 <key>_scale 的 safetensors）」直载；
    #           **GPU 上从头到尾不出现那一份 bf16 权重**。
    #   False = 默认，老路径：BF16 载入 -> apply_int8_quant 就地替换（逐字节等价）。
    #   ★ 动机（实测见 docs/显存归因排查报告-20261011.md §2.12）：
    #     就地替换 + 量化时的 fp32 临时量会在 PyTorch 分配器里留下
    #     0.589 GiB 的 inactive_split（不可回收碎片），设备已用量 used 反而变大，
    #     KV 池因此少 4 块。两条路径产出的 (int8 权重, scale) 已对拍为逐位相同。
    #   ★ 路径里带显式断言，只支持 TP=1 + per_channel + 有真 int8 checkpoint 的情况。

    # ---------- P13：调度 / 延迟侧 ----------
    spec_skip_draft_prefill_when_off: bool = False
    #   True  = 本引擎这一轮【绝不会投机】时（spec_k <= 0），prefill 阶段不再白跑 draft 前向。
    #   False = 默认，老行为：只要进程加载了 draft，prefill 就无条件多跑一遍 0.6B
    #           （model_runner.run 里那句 `if self.draft_model is not None` 不看 spec_k）。
    #   ★ 依据（实测）：docs/端到端验收.md §6 + 本轮 p13_prefill_probe —— 冷启动下
    #     draft prefill 让 target prefill 多花 20%~26%（16K 时 +739 ms），
    #     热缓存（只补未命中的尾巴）下是恒定 +31.5 ms。
    spec_prefill_chunk_tokens: int = 0
    #   >0 = 把 prefill 切成每轮最多这么多 token 的块，并在两次块之间【让一轮给 decode】。
    #        一条长 prompt 到场时，正在解码的请求不再被饿死整段 prefill。
    #        ★ 一轮内仍然只有一种执行路径（要么 prefill 要么 decode），不混路径。
    #   0  = 默认，老行为：prefill 排他，且只对第一条序列分块（`schedule()` 原逻辑）。
    #   ★ 依据（实测 p13_itl_probe，3 条解码 + 突来一条长 prompt）：
    #     解码请求那一轮的 ITL 尖刺 = 10.8×（1K）→ 249.4×（16K）【无 draft】，
    #     15.9× → 312.0×（带 draft），且那一轮解码请求拿到的 token 数 = 0。
    #     停止条件（方案 §6.3）是"只有 2–3 倍就收手" ⇒ 远未触及，值得做。

    spec_graph_bs: tuple = (1, 2, 4)
    #   捕获 CUDA 图的精确 batch 桶（draft 单步 decode 图 / 验证图共用）。
    #   按 12.2「新增图先只覆盖精确 B=2/4，保留 B=1 原路径」——
    #   不捕获默认 512 的全范围，未覆盖的 B 走 eager 并记录原因。

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # ---------- INT8 权重量化：自动识别（不需要用户逐层指定）----------
        (self.quant_weights, self.quant_granularity, self.quant_group_size,
         self.quant_modules, self.quant_config) = _resolve_quant(
            self.model, self.quant_weights, self.quant_granularity,
            self.quant_group_size)
        if self.draft_model:
            (self.draft_quant_weights, self.draft_quant_granularity,
             self.draft_quant_group_size, self.draft_quant_modules,
             self.draft_quant_config) = _resolve_quant(
                self.draft_model, self.draft_quant_weights,
                self.draft_quant_granularity, self.draft_quant_group_size)


def _resolve_quant(model_dir, kind, granularity, group_size):
    """模型目录里有 quant_config.json 就自动启用，并按其中列出的层选层。"""
    from nanovllm.layers.quant import hf_names_to_modules
    qc_path = os.path.join(model_dir, "quant_config.json")
    if not os.path.isfile(qc_path):
        if kind == "int8":
            raise ValueError(f"quant_weights='int8' 但 {model_dir} 下没有 quant_config.json"
                             f"（先用 scripts/quantize_int8.py 生成）")
        return None, granularity, group_size, (), None
    with open(qc_path) as f:
        qc = json.load(f)
    assert qc.get("quant_method") == "int8_w8a16", \
        f"不支持的量化格式: {qc.get('quant_method')}"
    mods = tuple(hf_names_to_modules(qc["quantized_layers"]))
    gran = qc.get("granularity", granularity)
    gs = qc.get("group_size") or group_size
    return "int8", gran, gs, mods, qc
