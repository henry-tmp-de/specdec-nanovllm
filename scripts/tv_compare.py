"""比较各轮的 token 频率分布，判断投机路径有没有系统性偏差。

设计要点
--------
单看 TV(draft, base) 没有意义 —— 必须和【噪声地板】比。而且地板要分两种：

  floor_base  = TV(base , base2 )   两份基线独立运行
  floor_draft = TV(draft, draft2)   两份【投机】独立运行

★ 为什么投机路径需要自己的地板：
  投机解码一步产出 3~4 个 token，这些 token 是同一次前向、同一个上下文里出来的，
  序列自相关比逐 token 解码强得多 -> 有效样本量更小 -> 经验分布天然更"散"。
  所以 TV(draft, base) 本来就该略大于 TV(base, base2)，不能直接判成有偏。

判据：跨组的 TV 应该和【两个地板里较大的那个】一个量级。
"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ziru/nano-vllm/repo/lossless_results.txt"
runs = {}
for line in open(path, encoding="utf-8"):
    if line.startswith("@@L@@"):
        d = json.loads(line[5:])
        runs.setdefault(d["mode"], d)

need = ["base", "base2", "draft"]
missing = [m for m in need if m not in runs]
if missing:
    print("还缺结果：", missing)
    sys.exit(0)


def tv(a, b):
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def c(m):
    return runs[m]["counts"]


print("样本量：", {m: runs[m]["n"] for m in runs})
print("distinct：", {m: runs[m]["distinct"] for m in runs})
print()

floor_base = tv(c("base"), c("base2"))
print(f"  TV(base , base2 ) = {floor_base:.4f}   <- 基线噪声地板")

if "draft2" in runs:
    floor_draft = tv(c("draft"), c("draft2"))
    print(f"  TV(draft, draft2) = {floor_draft:.4f}   <- 投机自己的噪声地板")
else:
    floor_draft = None
    print("  （还没跑 draft2，地板只有基线那一半）")

print()
print("  跨组（base x draft）：")
for b in ("base", "base2"):
    for d in ("draft", "draft2"):
        if d in runs:
            print(f"    TV({b:<5}, {d:<6}) = {tv(c(b), c(d)):.4f}")

cross_max = max(tv(c(b), c(d)) for b in ("base", "base2")
                for d in ("draft", "draft2") if d in runs)
ref = max([floor_base] + ([floor_draft] if floor_draft else []))
print()
print(f"  跨组最大 TV = {cross_max:.4f}   参考地板 = {ref:.4f}   比值 = {cross_max/ref:.2f}")
if cross_max < 1.3 * ref:
    print("  结论：跨组差异与噪声同量级 —— 没有发现系统性偏差。")
else:
    print("  结论：跨组差异明显大于噪声地板 —— 需要进一步排查"
          "（注意：也有可能是两条路径的浮点/kernel 配置不同造成的，"
          "   CUDA graph 的 varlen 前向会烘死 max_seqlen_k，"
          "   与 eager 走的 flash-attn 分块配置不同）。")
