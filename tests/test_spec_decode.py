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
# 构造一个有规律的前缀：「A B C D E」
seq = [10, 11, 12, 13, 14, 99, 99]
p.add(seq)
# 用「A B」检索，应能接出 C D E
cand = p.propose([10, 11], k=3)
check("从已知前缀接出后继", cand == [12, 13, 14], f"得到 {cand}")

# 检索不存在的前缀
cand = p.propose([77, 88], k=3)
check("前缀不存在时返回空", cand == [], f"得到 {cand}")

# 索引统计
check("索引统计可读", "keys" in p.stats(), p.stats())

# 加长一点，看能否接更长的链
p2 = NgramProposer(n=2, window=16)
p2.add([1, 2, 3, 4, 5, 6, 7, 8])
cand2 = p2.propose([1, 2], k=4)
check("n=2 时可接更长链", cand2[:4] == [3, 4, 5, 6], f"得到 {cand2}")

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
# 构造：target 在 token 5 上概率远高于 draft → bonus 应偏向 5
draft_logits = torch.full((B, K, V), -10.0)
draft_logits[:, 0, 3] = 10.0                  # draft 猜 3
draft_tokens = torch.full((B, K), 3, dtype=torch.long)
target_logits = torch.full((B, K + 1, V), -10.0)
target_logits[:, 0, 5] = 20.0                 # target 第 0 位极度偏向 5
target_logits[:, 1:, 9] = 20.0
# 强制第 0 位被拒：u 会小于 ratio=1 所以会接受，改为让 p(3) 极小
target_logits[:, 0, 3] = -50.0
bonus_counts = torch.zeros(V)
N = 2000
for _ in range(N):
    rr = verify_batch(draft_logits, target_logits, draft_tokens)
    if int(rr.accepted[0]) < K:
        bonus_counts[int(rr.bonus[0])] += 1
top = bonus_counts.argmax().item()
check("bonus 偏向 target 分布的高概率 token", top == 5,
      f"bonus 最常落在 token {top} (期望 5)，计数 {bonus_counts[top]:.0f}/{N}")

# ======================================================================
print()
print("=" * 70)
print("测试 3：★ 无损性（最关键）")
print("=" * 70)
print("  无损性的准确含义：连续生成时【输出序列】的分布与原模型一致。")
print("  正确检验：N 条独立序列各生成 M 步，比较「走投机」与「直接用 target 采样」")
print("  的 token 频率。")
print()
print("  ⚠️ 上一版测试拿 bonus 和 p[0] 比 —— bonus 来自修正分布 max(0, p−q)，")
print("     本来就不等于 p。那是【检验方法】错，不是实现错。")

N = 20000          # 独立序列条数
M = 4              # 每条生成 4 步
V2 = 6

# 参照组：完全不用投机，每步直接从 target 分布采
p_all = torch.randn(N, M, V2) + 0.5
tgt_probs = torch.softmax(p_all, dim=-1)                     # (N, M, V)
ref_tokens = torch.stack([
    torch.multinomial(tgt_probs[:, s, :], 1).squeeze(1) for s in range(M)
], dim=1)                                                     # (N, M)

# 实验组：走投机路径。
# ★ 关键：单步验证的正确构造是——
#   target 位置 0  = 本该decode 的那个位置的分布 p[step]
#   target 位置 1  = 候选 0 的真值分布
# 而候选 0 之所以可能正确，恰恰是因为它是从 p[step-1] 后面捞出来的。
# 所以最干净的检验是【单步】：候选的真值分布就是 p[step]，参照组直接采它。
N_STEP = 50000
p_step = torch.randn(N_STEP, V2) + 0.5
probs_ref = torch.softmax(p_step, dim=-1)
ref = torch.multinomial(probs_ref, 1).squeeze(1)          # 参照：直接采目标分布

q_logits1 = torch.randn(N_STEP, 1, V2)
draft_tok = torch.multinomial(
    torch.softmax(q_logits1, dim=-1).reshape(-1, V2), 1
).squeeze(1).view(N_STEP, 1)

# target 传 K+1=2 个位置：位置 0 = 本该 decode 的，位置 1 = 候选验证位
plg = torch.stack([p_step, torch.zeros_like(p_step)], dim=1)   # (N, 2, V)
res1 = verify_batch(q_logits1, plg, draft_tok)
spec1 = torch.where(res1.bonus >= 0, res1.bonus, draft_tok[:, 0])

c_spec = torch.bincount(spec1, minlength=V2).float(); c_spec /= c_spec.sum()
c_ref = torch.bincount(ref, minlength=V2).float(); c_ref /= c_ref.sum()
tv = 0.5 * (c_spec - c_ref).abs().sum().item()

# TV 的蒙特卡洛标准误：每个 bin 的标准差 ≈ sqrt(p(1-p)/N)，V 个 bin 求和
p_unif = 1.0 / V2
sd_bin = (p_unif * (1 - p_unif) / N_STEP) ** 0.5
tv_se = 0.5 * sd_bin * (2 / (3.141592653589793 ** 0.5))

print(f"  单步验证: TV = {tv:.5f}   4σ 阈值 = {4 * tv_se:.5f}")
print(f"  接受率 = {(res1.accepted == 1).float().mean().item():.3f}   "
      f"bonus 使用次数 = {int((res1.bonus >= 0).sum())}/{N_STEP}")
print()
check("无损性：投机路径输出分布 == 目标分布（TV < 4σ）", tv < 4 * tv_se,
      f"{tv:.5f} vs {4 * tv_se:.5f}")

# ======================================================================
print()
print("=" * 70)
if FAILED:
    print(f"✗ {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
else:
    print("✓ 全部通过")