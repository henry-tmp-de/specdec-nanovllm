import json
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# ---------------------------------------------------------------------------
# 预量化直载（P9）
# ---------------------------------------------------------------------------
# 打包模块里每个 shard 在【输出行】上的顺序 —— 必须与 weight_loader 的
# shard_offset 完全一致（QKVParallelLinear: q→0, k→num_heads*hd, v→…；
# MergedColumnParallelLinear: gate→0, up→output_sizes[0]）。
_SHARD_ORDER = {"q": 0, "k": 1, "v": 2, 0: 0, 1: 1}


def _first_ckpt(path: str) -> str | None:
    fs = sorted(glob(os.path.join(path, "*.safetensors")))
    return fs[0] if fs else None


def _has_int8_ckpt(path: str) -> bool:
    fp = _first_ckpt(path)
    if fp is None:
        return False
    with safe_open(fp, "pt", "cpu") as h:
        return any(k.endswith("_scale") for k in h.keys())


def build_model_int8(model_cls, hf_config, path, quant_modules,
                     granularity="per_channel", group_size=128, device="cuda",
                     load_int8_values=False):
    """预量化直载：量化层【在构造期】直接建 int8 权重 + bf16 scale。

    与「BF16 载入 → apply_int8_quant 就地替换」的差别是**分配顺序**：
    这里 GPU 上从头到尾不出现【一整套】bf16 草稿权重 —— 未量化参数直接
    materialize 成 bf16，量化参数直接建 int8 + scale。

    量化值的来源两档：
      · load_int8_values=False（默认）：把每层 bf16 权重【单个】搬上 GPU，
        用引擎自己的 `quantize_weight` 在 GPU 上算 —— 与 apply_int8_quant
        **逐位相同**（同一函数、同一设备、同一输入）。
      · load_int8_values=True：直接读 checkpoint 里现成的 int8 + scale。
        它的值等于【离线 CPU 量化】。实测与 GPU 量化版在 112/112 层上都有
        约 0.006% 的元素差 1 LSB（scale 逐位相同）—— 这是 CPU 与 GPU 的
        fp32 除法/取整在 .5 边界上差 1 ULP 造成的。**不是逐位相同**，谨慎使用。
    """
    import torch.distributed as dist
    from nanovllm.layers.linear import LinearBase
    from nanovllm.layers.quant import quantize_weight

    # ---- 0) 不支持的组合【显式报错】，绝不静默按错规则装 ----
    assert dist.get_world_size() == 1, (
        "预量化直载当前只支持 tensor_parallel_size=1："
        "TP>1 时 scale 要按输出行分片，拼接规则与单卡不同（未验证）")
    assert granularity == "per_channel", (
        f"预量化直载当前只支持 per_channel，收到 {granularity}："
        f"per_group 的 scale 是 [N, K//G]，拼接规则不同（未验证）")

    qc = {}
    qc_path = os.path.join(path, "quant_config.json")
    if os.path.isfile(qc_path):
        with open(qc_path) as fh:
            qc = json.load(fh)
    has_i8 = _has_int8_ckpt(path)
    main_file = _first_ckpt(path)                 # 未量化参数 + 判定量化层名
    assert main_file is not None, f"{path} 下没有 safetensors"
    if load_int8_values:
        assert has_i8, f"{path} 下没有带 *_scale 的 int8 张量，不能用 load_int8_values"
        quant_file = main_file
    elif not has_i8:
        quant_file = main_file                    # 本目录自己就是 bf16 源
    else:
        quant_file = _first_ckpt(qc.get("source_model") or "")
        assert quant_file is not None, (
            f"量化层的 BF16 来源找不到（quant_config.json 的 source_model="
            f"{qc.get('source_model')!r}）—— 要么用 load_int8_values=True，"
            f"要么让 source_model 指向原始 BF16 目录")

    # ---- 1) meta 构造：这一刻不占任何一份 bf16 权重显存 ----
    prev_dev = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        model = model_cls(hf_config)
    finally:
        torch.set_default_device(prev_dev)

    packed = getattr(model, "packed_modules_mapping", {})
    quant_set = set(quant_modules)

    def _resolve(wname):
        """HF 权重名 -> (引擎参数名, shard_id)，规则与 load_model 一致。"""
        for k, (v, sid) in packed.items():
            if k in wname:
                return wname.replace(k, v), sid
        return wname, None

    # ---- 2) 找出会被量化替换的参数（这些【不】materialize，省掉那份 bf16）----
    replaced = set()
    with safe_open(main_file, "pt", "cpu") as f:
        for wk in f.keys():
            if wk.endswith("_scale") or not wk.endswith(".weight"):
                continue
            pname, _sid = _resolve(wk)
            if pname[:-len(".weight")] in quant_set:
                replaced.add(pname)

    # ---- 3) materialize 其余参数（整对象换 Parameter；保住 weight_loader）----
    #    ★ 不能用 `meta_param.data = cuda_tensor` —— meta 与 cuda 的 Variable
    #      类型不兼容，torch 会直接报 "incompatible tensor type"。
    tie = bool(getattr(hf_config, "tie_word_embeddings", False))
    for mod_name, mod in model.named_modules():
        for pname in list(mod._parameters.keys()):
            p = mod._parameters[pname]
            if p is None:
                continue
            full = f"{mod_name}.{pname}" if mod_name else pname
            if full in replaced:
                continue
            if tie and full == "lm_head.weight":
                continue    # 与 embed_tokens 共享同一份存储，只建一份
            new = nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device=device),
                               requires_grad=p.requires_grad)
            if hasattr(p, "weight_loader"):
                new.weight_loader = p.weight_loader
            setattr(mod, pname, new)
    for name, b in model.named_buffers():
        assert not b.is_meta, (
            f"buffer {name} 在 meta 构造后仍是 meta —— 该模型不能走预量化直载，"
            f"请把 quant_direct_load 关掉走原路径")
    if tie:
        # 复原 Qwen3ForCausalLM.__init__ 里的权重共享（否则会多占一整份词表嵌入）。
        # ★ 直接把同一个 Parameter 对象挂过去；不能用 `lm_head.weight.data = x`
        #   —— lm_head.weight 此刻还是 meta 张量，set_data 会报 incompatible tensor type。
        model.lm_head.weight = model.get_submodule("model.embed_tokens").weight

    # ---- 4) 未量化参数照原规则拷（来源与老 load_model 相同：用户指的目录）----
    with safe_open(main_file, "pt", "cpu") as f:
        for wk in f.keys():
            if wk.endswith("_scale"):
                continue
            pname, sid = _resolve(wk)
            if pname in replaced:
                continue
            param = model.get_parameter(pname)
            wl = getattr(param, "weight_loader", default_weight_loader)
            if sid is None:
                wl(param, f.get_tensor(wk))
            else:
                wl(param, f.get_tensor(wk), sid)

    # ---- 5) 量化参数：攒 shard -> 拼成引擎模块的 int8 权重 + scale ----
    shards = {}
    with safe_open(quant_file, "pt", "cpu") as f:
        for wk in f.keys():
            if wk.endswith("_scale"):
                continue
            pname, sid = _resolve(wk)
            if pname not in replaced:
                continue
            w = f.get_tensor(wk)
            if w.dtype == torch.int8:
                s = f.get_tensor(wk + "_scale")
                assert s.dtype == torch.bfloat16, f"{wk}_scale 不是 bf16（{s.dtype}）"
                q = w
            else:
                # ★ 在 GPU 上用引擎自己的 quantize_weight —— 与就地量化逐位相同。
                #   只搬【一层】上卡，算完立刻拿回 CPU 拼，GPU 上没有整套 bf16。
                q, s = quantize_weight(w.to(device), granularity, group_size)
                q, s = q.cpu(), s.cpu()
            shards.setdefault(pname[:-len(".weight")], {})[sid] = (q, s)

    stats = []
    for modname, sh in shards.items():
        mod = model.get_submodule(modname)
        assert isinstance(mod, LinearBase), f"{modname} 不是 LinearBase（{type(mod).__name__}）"
        order = list(sh) if len(sh) == 1 else sorted(sh, key=lambda s: _SHARD_ORDER[s])
        q = torch.cat([sh[s][0] for s in order], dim=0).contiguous()
        sc = torch.cat([sh[s][1].reshape(-1) for s in order], dim=0).contiguous()
        want = tuple(mod.weight.shape)
        assert tuple(q.shape) == want, f"{modname} 直载 int8 形状 {tuple(q.shape)} != 引擎 {want}"
        assert sc.numel() == want[0], f"{modname} scale 个数 {sc.numel()} != 输出行 {want[0]}"
        mod.weight = nn.Parameter(q.to(device), requires_grad=False)
        mod.weight_scale = nn.Parameter(sc.to(device), requires_grad=False)
        mod.quant_granularity = granularity
        mod.quant_group_size = group_size
        stats.append({"name": modname, "shape": list(want), "rel_err": None,
                      "bytes_before": want[0] * want[1] * 2,
                      "bytes_after": q.numel() + sc.numel() * 2})

    missing = quant_set - {s["name"] for s in stats}
    assert not missing, f"这些量化模块没在 checkpoint 里找到：{sorted(missing)}"

    # ---- 6) 收尾自检：不许有 meta 残留 ----
    for name, p in model.named_parameters():
        assert not p.is_meta, f"参数 {name} 仍是 meta —— 直载漏了"
    return model, stats


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # ★ 防呆：int8 checkpoint（含 *_scale）不能直接喂给 bf16 参数，
            #   `copy_` 会悄悄做 dtype 转换、把 int8 数值当成权重装进去 —— 静默出错。
            #   默认路径的 checkpoint 里永远没有 `_scale` 键，所以这条不影响原行为。
            qkeys = [k for k in f.keys() if k.endswith("_scale")]
            if qkeys:
                raise RuntimeError(
                    f"{os.path.basename(file)} 看起来是 INT8 checkpoint（含 {qkeys[0]} 等 "
                    f"{len(qkeys)} 个 scale）。引擎的量化是【从 BF16 权重就地量化】的："
                    f"请把 model/draft_model 指到【BF16 原始 checkpoint】目录，"
                    f"并让该目录里有 quant_config.json（或用 quant_weights='int8' 显式开启）。")
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
