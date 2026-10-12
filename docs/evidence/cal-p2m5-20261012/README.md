# 证据：`(P=2, M=5)` 收尾定标（2026-10-12）

配套文档：`docs/优化台账.md` §2.7.1 / §2.7.2、`docs/优化过程复盘.md` `[17]`。
服务器工作区：`/home/ziru/nano-vllm/cal-work/`（代码 = 主分支 `75729b0` 的原样拷贝）。
复算命令：`scripts/run_cal.sh`。**生产代码零改动**（`git diff 75729b0 -- nanovllm/` 为空）。

## 文件

| 文件 | 是什么 |
|---|---|
| `out/eng/engine_matrix.txt` / `engine_alpha_raw.jsonl` | ★ **主口径**：引擎内确定性仪器（`scripts/engine_alpha.py`，挂 `verify_batch`，用引擎自己的 `p`/`q` 算 `1 − TV`）。25 次运行，含两次独立重跑。 |
| `out/eng/harness_*.json` | 项目原有自由生成口径（`scripts/acceptance_matrix.py --group lc_w0 / mw_m5 / mw_p2m5 --ctxs 8192 --bs 4 --runs 3`）。**它的 `accept_rate` 就是 per-position α**，与主口径差 1–5%。 |
| `out/det/det_matrix.txt` + `out/det/*.json` | 离线【固定上下文】口径（`scripts/det_alpha.py`，wikitext 16 窗口 × 2048 位置）。**对 `w0` 一致、对 `P=0` 偏乐观 0.25、对 16K 的 `p2m5` 偏低 0.046**；`selftest_8k.json` 记录了"窗口盖住全序列 ⇒ 与全上下文逐位相同"的自检。 |
| `logs/dbg_{p0m5,p2m5}.log` | 引擎每次 draft 前向**真实用的** `(positions, context_len, block_table)` |
| `logs/cmpkv_{p0m5,p2m5}.log` | ★ 排除证据①：窗口 prefill 写进环里的 KV 与离线同段前向**逐元素一致**（`max\|ΔV\| = 0.0078` = bf16 舍入；换一个位置比是 0.83–1.39） |
| `logs/cmp_{p0m5,p2m5}.log` | 排除证据②：引擎第一次投机轮的 draft logits 与十几种"候选上下文"的 TV（引擎的分布离任何一种离线重构都 ≥0.21，说明差异不在"窗口取多长"上） |

## 一句话结论

`(P=2, M=5)`（draft 上下文 = 开头 2 块常驻 + 最近 5 块滑窗 = 1792 token）与普通投机**接受率持平**
（8K −1%~−4%、16K +0.5%），换来 draft 显存 −84%、target 容量 +66%、16K 并发 2 条 → 4 条、
TTFT 只贵 8 ms；**16K 下不需要放大 M**。`P=0`（纯滑窗）掉 44%，**不要用**。
