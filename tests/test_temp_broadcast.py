"""回归测试（纯 CPU）：逐请求温度必须沿【请求维】广播（不是 vocab 维）。

真踩过的 bug
-----------
`verify._softmax_with_temp` 原来是

    shape = [1] * (x.dim() - 1) + [-1]        # (B,k,V) -> [1,1,-1]
    x = x / temperatures.reshape(shape)

对 3 维输入 logits (B, k, V)，这个 shape 是 [1, 1, -1] —— B 个温度被塞进
【最后一维】也就是 vocab 维。B=1 时长度 1 的维广播无差别，问题被完全掩盖
（引擎当前就是逐序列调用，所以没暴露）；B>1 时：

  · V ≠ B：reshape/广播直接 RuntimeError；
  · V == B：每个词的概率被除以【别人的】温度 → q 与 p 一起错位 →
    min(1, p/q) 算错 → 无损性破裂。

正确形状是 (B, 1, ..., 1)：温度对齐它所属的那条请求。
这是【正确性修复】，不是性能优化。

跑：python tests/test_temp_broadcast.py
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name: str, rel_path: str):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_verify = _load("verify", "nanovllm/spec_decode/verify.py")
_softmax_with_temp = _verify._softmax_with_temp
verify_batch = _verify.verify_batch

FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def call(fn, *a, **kw):
    """跑一次可能抛异常的操作，把异常当证据返回。"""
    try:
        return None, fn(*a, **kw)
    except Exception as e:                    # noqa: BLE001
        return f"{type(e).__name__}: {e}", None


torch.manual_seed(0)

# ======================================================================
print("=" * 70)
print("§1 三维 (B, k, V)：逐行结果必须与单序列调用（B=1）对拍")
print("=" * 70)

B, K, V = 3, 2, 5
x = torch.randn(B, K, V) * 3.0
temps = torch.tensor([0.5, 1.0, 2.5])         # 三条请求温度各不相同

err, out = call(_softmax_with_temp, x, temps)
check("① 批量调用不抛异常（改动前 V≠B 直接 RuntimeError）", err is None, err or "")
if err is None:
    for i in range(B):
        ref = _softmax_with_temp(x[i:i + 1], temps[i:i + 1])     # 单序列调用
        check(f"① 第 {i} 行（温度 {temps[i].item()}）与单序列调用一致",
              torch.allclose(out[i:i + 1], ref, atol=1e-7, rtol=0),
              f"最大差 {(out[i:i+1] - ref).abs().max().item():.3e}")
    check("① 每行温度确实生效（不同温度给出不同分布）",
          not torch.allclose(out[0], out[1], atol=1e-6)
          and not torch.allclose(out[1], out[2], atol=1e-6))
    check("① 概率仍是一行归一（sum=1）",
          torch.allclose(out.sum(-1), torch.ones(B, K), atol=1e-6))

# 二维 (B, V) 也必须对
x2 = torch.randn(B, V) * 3.0
err2, out2 = call(_softmax_with_temp, x2, temps)
check("① 二维 (B,V) 同样沿请求维广播", err2 is None, err2 or "")
if err2 is None:
    ok = all(torch.allclose(out2[i:i + 1],
                            _softmax_with_temp(x2[i:i + 1], temps[i:i + 1]), atol=1e-7)
             for i in range(B))
    check("① 二维逐行与单序列调用一致", ok)

# ---- 反面对照：把旧写法在本文件里重算一遍，证明它确实是错的 ----
B4, K4, V4 = 4, 1, 4                          # V == B 的退化情形
x4 = torch.randn(B4, K4, V4) * 3.0
t4 = torch.tensor([0.3, 0.8, 1.5, 3.0])
bad = torch.softmax(x4 / t4.reshape([1] * (x4.dim() - 1) + [-1]), dim=-1)
good = _softmax_with_temp(x4, t4)
check("① 反面对照：旧写法（[1,1,-1]）在 V==B 时不报错但结果不同",
      not torch.allclose(bad, good, atol=1e-4),
      f"最大差 {(bad - good).abs().max().item():.3e}")
B5, V5 = 3, 5                                  # V != B
x5 = torch.randn(B5, 1, V5) * 3.0
errb, _ = call(lambda: torch.softmax(
    x5 / torch.tensor([0.5, 1.0, 2.0]).reshape([1] * (x5.dim() - 1) + [-1]), dim=-1))
check("① 反面对照：旧写法在 V≠B 时直接 RuntimeError", errb is not None, errb or "")


# ======================================================================
print()
print("=" * 70)
print("§2 verify_batch 端到端：三条请求、温度各异，接受率必须各自对得上")
print("=" * 70)
print("  构造：draft 用【单点分布】（n-gram 路线），于是")
print("        接受概率 = min(1, p_target[cand] / 1) = p_target[cand]")
print("        p_target 用各自温度算 —— 温度播错，这一行的接受率就不对。")
print()

V6 = 5
B6 = 3
N6 = 30000
p_base = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
logits = p_base.log()
temps6 = torch.tensor([0.25, 1.0, 0.5])       # 三条请求温度各不相同（顺序刻意打乱）
cand_tok = 1                                   # 三条请求都提议 token 1
cand = torch.full((N6, 1), cand_tok, dtype=torch.long)
draft = torch.zeros(N6, 1, V6)
draft.scatter_(2, cand.unsqueeze(-1), 1.0)                    # 单点分布
target = logits.view(1, 1, V6).expand(N6, 2, V6).contiguous()
tiled_temps = temps6.repeat(N6 // B6)

def _run_batch():
    torch.manual_seed(7)                # 固定 RNG，让 §2 不受上面用例消耗随机数的影响
    return verify_batch(draft, target, cand, tiled_temps, draft_is_point_mass=True)


err6, res = call(_run_batch)
check("② B>1 的 verify_batch 能跑（改动前 RuntimeError）", err6 is None, err6 or "")

if err6 is None:
    acc = res.accept_mask.float().reshape(N6 // B6, B6).mean(dim=0)
    want = [float(torch.softmax(logits / t, dim=-1)[cand_tok]) for t in temps6.tolist()]
    for i in range(B6):
        check(f"② 温度 {temps6[i].item()} 那行的接受率 ≈ p[cand]（{want[i]:.3f}）",
              abs(float(acc[i]) - want[i]) < 0.02,
              f"实测 {float(acc[i]):.4f}")
    check("② 三行温度确实给出三个不同的接受率（不是同一个数）",
          len({round(w, 3) for w in want}) == 3, f"{[round(w, 3) for w in want]}")

# ======================================================================
print()
print("=" * 70)
print("§3 回归：B=1（引擎当前路径）行为不变")
print("=" * 70)

torch.manual_seed(1)
x1 = torch.randn(1, 3, V6) * 2.0
t1 = torch.tensor([1.0])
o = _softmax_with_temp(x1, t1)
ref_plain = torch.softmax(x1.float(), dim=-1)
check("③ 温度 = 1 时与不加温度完全一致", torch.allclose(o, ref_plain, atol=1e-7))
o2 = _softmax_with_temp(x1, None)
check("③ temperatures=None 时与不加温度一致", torch.allclose(o2, ref_plain, atol=1e-7))
o3 = _softmax_with_temp(x1, torch.tensor(2.0))           # 标量温度
check("③ 标量温度也能用", o3.shape == x1.shape and o3.sum(-1)[0, 0].item() > 0.99)

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
