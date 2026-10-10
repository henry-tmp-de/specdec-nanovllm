"""逐位等价对拍比较器：读两份 p7_equiv.py 的 dump，逐字段比。

用法: python scripts/p7_equiv_cmp.py <a.json> <b.json> [a_label] [b_label] [--ignore-layout]

判据
----
  forward 记录里 bt / slot 是【物理块表】——  全上下文档用 target 的块表、
  滑窗档用 draft 私有的环形块表，两者按构造就不同（不同池子、不同长度）。
  所以比较分两层：

    严格层（strict）：所有字段逐位相同。用于"同一配置两个代码版本"的对拍
                      （spec_draft_window=0 vs 1c1d907）。
    布局无关层（layout-independent）：忽略 bt / slot，只比
                      tokens / positions / ctx / logits_md5
                      —— 这一层判的是"draft 到底看到了什么、算出了什么，
                      与物理块号无关"。用于 W=0 vs W=2048 的对拍。

  `--ignore-layout` 时以布局无关层作为退出码判据（默认严格层）。

退出码 0 = 等价；1 = 不等价（并打印第一处差异）。
"""
import json
import sys

import torch

LAYOUT = {"bt", "slot"}


def norm(x, ignore_layout):
    if not ignore_layout:
        return x
    if isinstance(x, dict):
        return {k: norm(v, ignore_layout) for k, v in x.items() if k not in LAYOUT}
    if isinstance(x, list):
        return [norm(v, ignore_layout) for v in x]
    return x


def first_diff(a, b):
    if type(a) is not type(b):
        return ("type", type(a), type(b))
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return ("key", k, k in a, k in b)
            d = first_diff(a[k], b[k])
            if d is not None:
                return (k,) + d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return ("len", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y)
            if d is not None:
                return (i,) + d
        return None
    return None if a == b else ("value", a, b)


def cmp_section(A, B, sect, ignore_layout):
    na, nb = len(A[sect]), len(B[sect])
    if na != nb:
        return na, na, f"条数不同 {na} vs {nb}"
    ndiff = 0
    first = None
    for i in range(na):
        d = first_diff(norm(A[sect][i], ignore_layout), norm(B[sect][i], ignore_layout))
        if d is not None:
            ndiff += 1
            if first is None:
                first = f"{sect}[{i}] {d}"
    return na, ndiff, first


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ignore_layout = "--ignore-layout" in sys.argv
    pa, pb = args[0], args[1]
    la = args[2] if len(args) > 2 else "A"
    lb = args[3] if len(args) > 3 else "B"
    A = json.load(open(pa))
    B = json.load(open(pb))
    print(f"比较 {la} ({A['W']=} {A['L']=} {A['OUT']=}) vs {lb} ({B['W']=} {B['L']=} {B['OUT']=})"
          .replace("=", "="))
    ok = True
    for key in ("out_md5", "out_token_ids", "n_forwards", "n_proposes", "n_verifies"):
        same = A[key] == B[key]
        ok &= same
        s = f"{A[key]!r}"
        if not same:
            s += f"  vs  {B[key]!r}"
        print(f"  [{'OK ' if same else 'DIFF'}] {key}: {s[:120]}")

    for sect in ("forwards", "proposes", "verifies"):
        n, ndiff, first = cmp_section(A, B, sect, ignore_layout)
        ok &= (ndiff == 0)
        tag = "布局无关" if ignore_layout else "严格"
        print(f"  [{'OK ' if ndiff == 0 else 'DIFF'}] {sect}（{tag}）: {n} 条，{ndiff} 条不同"
              + (f"   第一处: {first}" if first else ""))

    for suf, tag in ((".logits.pt", "draft"),):
        try:
            ta = torch.load(pa + suf)
            tb = torch.load(pb + suf)
            if ta.shape != tb.shape:
                print(f"  [DIFF] {tag} logits 形状 {tuple(ta.shape)} vs {tuple(tb.shape)}")
                ok = False
            else:
                mx = float((ta - tb).abs().max())
                ag = float((ta.argmax(-1) == tb.argmax(-1)).float().mean())
                print(f"  [{'OK ' if mx == 0.0 else 'DIFF'}] 前 {ta.shape[0]} 次 {tag} 前向 "
                      f"logits max|Δ| = {mx:.8f}，argmax 一致率 = {ag:.6f}")
                ok &= (mx == 0.0)
        except FileNotFoundError:
            print(f"  [SKIP] {tag} logits.pt 缺失")

    print()
    print(("✓ 等价（%s == %s，%s层）" % (la, lb, "布局无关" if ignore_layout else "严格"))
          if ok else ("✗ 不等价（%s vs %s）" % (la, lb)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
