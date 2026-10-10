import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


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
