"""
spec_decode 核心模块的正确性测试（纯 CPU，不依赖引擎和 GPU）。

跑：python tests/test_spec_decode.py
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接按文件路径加载模块，绕开 nanovllm/__init__.py（它会 import transformers 等重依赖）
import importlib.util


def _load(mod_name: str, rel_path: str):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ngram = _load("ngram_proposer", "nanovllm/spec_decode/ngram_proposer.py")
_verify = _load("verify", "nanovllm/spec_decode/verify.py")

NgramProposer = _ngram.NgramProposer
verify_batch = _verify.verify_batch

torch.manual_seed(0)

FAILED = []


def check(name, cond, extra=""):
    mark = "PASS" if cond else "FAIL"
    if not cond:
        FAILED.append(name)
    print(f"  [{mark}] {name}" + (f"   {extra}" if extra else ""))


# ======================================================================
print("=" * 70)
print("测试 1：NgramProposer —— 能否从已有文本捞出正确的后继")
print("=" * 70)

p = NgramProposer(n=3, window=8)
seq = [10, 11, 12, 13, 14, 99, 99]
p.reset_watermark(seq)
cands = p.propose([10, 11], k=3)
check("返回多条候选（列表的列表）", isinstance(cands, list) and len(cands) > 0
      and isinstance(cands[0], list), f"得到 {cands}")
check("最优候选接出后继", cands[0] == [12, 13, 14], f"第一条 {cands[0]}")
check("前缀不存在时返回空", p.propose([77, 88], k=3) == [])

# 多样性：同一个 key 多次出现，后继不同 -> 应该给出多条不同分支
p2 = NgramProposer(n=3, window=16)
p2.reset_watermark([1, 2, 9, 1, 2, 7, 1, 2, 5])
c2 = p2.propose([1, 2], k=2, n_candidates=4)
uniq = {tuple(c) for c in c2}
check("多条候选互不相同（多样性）", len(uniq) > 1, f"得到 {[list(u) for u in uniq]}")
check("最近一次出现的分支排最前", c2[0] == [5, ...][:len(c2[0])] if c2[0] else False,
      f"第一条 {c2[0]}")

# 水位线：只 add 一次后，继续 observe 应能建出连续链条
p3 = NgramProposer(n=3, window=32)
full = [1, 2, 3, 4, 5, 6, 7, 8]
p3.reset_watermark(full)
for end in range(5, 9):
    p3.observe(full[:end])
check("水位线增量建索引后可提议",
      len(p3.propose(full[:5], k=2)) > 0, f"{p3.stats()}")

# ======================================================================
print()
print("=" * 70)
print("测试 2：verify_batch —— 全部接受 / 部分拒绝 / 全部拒绝")
print("=" * 70)

B, K, V = 1, 3, 20

# --- 情形 A：q 与 p 完全一致 → 应该全部接受 ---
draft_logits = torch.randn(B, K, V)
target_logits = draft_logits.clone()          # 完全相同
# 注意 multinomial 对 2D 输入返回 (n, num_samples)，所以要先把 (K,V) 压成 (V,)
# —— 这里是【逐位置】采一个 token：第 j 个候选位置用第 j 行的分布采
probs = torch.softmax(draft_logits[0], dim=-1)          # (K, V)
draft_tokens = torch.stack([
    torch.multinomial(probs[j], 1).squeeze(0) for j in range(K)
]).unsqueeze(0)                                          # (1, K)
assert draft_tokens.shape == (B, K), draft_tokens.shape
r = verify_batch(draft_logits, target_logits, draft_tokens)
check("q==p 时全部接受", bool((r.accepted == K).all()),
      f"accepted={r.accepted.tolist()} (期望 {K})")

# --- 情形 B：target 极度尖锐，draft 完全不同 → 应该几乎全拒 ---
torch.manual_seed(1)
draft_logits = torch.zeros(B, K, V)
draft_tokens = torch.zeros(B, K, dtype=torch.long)
target_logits = torch.full((B, K + 1, V), -50.0)
target_logits[:, :, 7] = 50.0                # target 只认 token 7
# 但第 0 位让 draft 也猜 7，保证第 0 位能接受
draft_logits[:, 0, 7] = 50.0
r = verify_batch(draft_logits, target_logits, draft_tokens)
check("target 尖锐时接受长度很短", bool((r.accepted <= 1).all()),
      f"accepted={r.accepted.tolist()}")

# --- 情形 C：bonus token 必须落在 p−q 的支撑集里 ---
# 构造：target 极度偏向 token 5；草稿提议 token 3（必然被拒）
# -> bonus 必须落在 5上
print("\n【C】被拒时 bonus 落在 target 偏好的 token 上")
N_C, K_C, V_C = 4000, 1, 20
probs_c = torch.full((V_C,), 1e-6)
probs_c[5] = 1.0# target 只认5（极尖锐）
logits_c = probs_c.log().view(1, 1, V_C).expand(N_C, 2, V_C).contiguous()

draft_c = torch.full((N_C, K_C), 3, dtype=torch.long)      # 草稿提议 3
dq_c = torch.zeros(N_C, K_C, V_C)
dq_c.scatter_(2, draft_c.unsqueeze(-1), 1.0)                # 单点分布

rc = verify_batch(dq_c, logits_c, draft_c, draft_is_point_mass=True)
bonus_counts = torch.bincount(rc.bonus[rc.bonus >= 0], minlength=V_C)
top = int(bonus_counts.argmax())
check("被拒时 bonus 落在 target 偏好的 token(5)", top == 5,
      f"最常落在 {top}，计数 {int(bonus_counts[5])}/{int((rc.bonus >= 0).sum())}")
check("被拒时 accepted 长度 = 0", bool((rc.accepted == 0).all()),
      f"accepted={rc.accepted.unique().tolist()}")

# ======================================================================
print()
print("=" * 70)
print("测试 3：★ 无损性（最关键）")
print("=" * 70)
print("  场景：n-gram 草稿是【单点分布】，target 分布任意 p。")
print("  严格推导：")
print("    接受概率= p[d] / q[d] = p[d] / 1 = p[d]")
print("    被拒时 bonus ~ normalize(max(0, p - q))")
print("    => 输出分布应严格等于 p")
print()
print("  ★ 两个曾经踩过的坑（都表现为 TV 偏高，但都不是实现的错）：")
print("    ① 把概率当 logits 传 -> softmax([0.5,0.2,...]) 会变成 [0.266,...]")
print("    ② 参照组用错-> 必须拿 target 分布 p 采样，不是拿草稿采样当参照")

V3 = 6
probs = torch.tensor([0.5, 0.2, 0.15, 0.05, 0.05, 0.05])
logits = probs.log()                       # target_logits 期望的是 logits


def _tv_at(N3: int, seed: int) -> float:
    torch.manual_seed(seed)
    draft_tok3 = torch.multinomial(probs.expand(N3, V3), 1, replacement=True)
    dq = torch.zeros(N3, 1, V3)
    dq.scatter_(2, draft_tok3.unsqueeze(-1), 1.0)                 # 单点分布
    plg3 = logits.view(1, 1, V3).expand(N3, 2, V3).contiguous()  # (N, K+1, V)
    r3 = verify_batch(dq, plg3, draft_tok3, draft_is_point_mass=True)
    final = torch.where(r3.bonus >= 0, r3.bonus, draft_tok3.squeeze(1))
    c3 = torch.bincount(final, minlength=V3).float()
    c3 /= c3.sum()
    return 0.5 * (c3 - probs).abs().sum().item(), float((r3.accepted == 1).float().mean())


print("  ★ 判据不用固定阈值，而是看【TV 是否随样本量收敛到 0】：")
print("    若实现有偏，TV 会稳定在某个非零值；")
print("    若无偏，TV 应按 1/sqrt(N) 下降（样本量 ×5 -> TV 约 /2.2）")
print()
tvs_small, _ = [], []
for seed in range(5):
    tv, _a = _tv_at(40_000, seed)
    tvs_small.append(tv)
tvs_large, acc = [], 0.0
for seed in range(3):
    tv, a = _tv_at(400_000, seed)
    tvs_large.append(tv)
    acc = a

avg_small = sum(tvs_small) / len(tvs_small)
avg_large = sum(tvs_large) / len(tvs_large)
ratio = avg_small / avg_large
print(f"    N=40,000  平均 TV = {avg_small:.5f}")
print(f"    N=400,000 平均 TV = {avg_large:.5f}")
print(f"    比值 = {ratio:.2f}（理论上 1/sqrt(10) = 0.32，TV 比值应约 3.16）")
check("无损性：TV 随样本量收敛到 0（无偏）", ratio > 2.0 and avg_large < 0.004,
      f"比值 {ratio:.2f}, N=400k 时 TV={avg_large:.5f}")
print(f"    接受率 = {acc:.3f}（理论 p[d] 均值 = {float((probs * probs).sum()):.3f}）")

# ======================================================================
print()
print("=" * 70)
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
else:
    print("✓ 全部通过")

# ======================================================================
# 测试 4：draft model 路线的接受率（真实分布 vs 单点分布）
# ======================================================================
print()
print("=" * 70)
print("测试 4：真实分布 vs 单点分布 —— 接受率的差距")
print("=" * 70)
print("  n-gram 路线：q 是单点分布 -> 接受概率 = p[draft]")
print("  draft model 路线：q 是真实分布 -> 接受概率 = min(1, p/q)")
print()

V4 = 50
torch.manual_seed(0)
N4 = 40000

# 模拟：draft 模型和 target 高度相似但不完全相同（这是真实的 draft-target 关系）
p_logits = torch.randn(N4, V4)
# draft 的 logits = target + 扰动（越接近，接受率越高）
for noise in [0.0, 0.5, 1.0, 2.0]:
    q_logits = p_logits + torch.randn(N4, V4) * noise
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)

    # draft 采样候选
    cand = torch.multinomial(q, 1, replacement=True).squeeze(1)
    # 接受概率 min(1, p[cand]/q[cand])
    pc = p.gather(1, cand.unsqueeze(1)).squeeze(1)
    qc = q.gather(1, cand.unsqueeze(1)).squeeze(1)
    ratio = torch.clamp(pc / qc.clamp_min(1e-10), max=1.0)
    acc_real = float(ratio.mean())

    # 单点分布情况：接受概率 = p[cand]
    acc_point = float(pc.mean())

    print(f"  draft 与 target 差异 noise={noise}:")
    print(f"    真实分布接受率 = {acc_real:.3f}    单点分布接受率 = {acc_point:.3f}"
          f"    提升 {acc_real/max(acc_point,1e-9):.1f}x")
