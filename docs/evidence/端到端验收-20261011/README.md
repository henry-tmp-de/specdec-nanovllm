# 原始日志：端到端验收（2026-10-11）

配套文档：`docs/端到端验收.md`。所有数字都能从这里的日志复算。

## 目录

| 目录 | 是什么 | 条件 |
|---|---|---|
| **`serial-1gpu/`** | ★ **主表用的串行数据**（`core` 组） | 串行、GPU 0、单进程 |
| **`serial-1gpu-v2/`** | ★ **主表用的串行数据**（三个滑窗组，带 in-group `S2` 对照） | 串行、GPU 0、单进程 |
| `parallel-4gpu/` | 第一版：4 组**并行**跑在 GPU 0-3（只作并行/串行对照，**不进主表**） | 并行、4 进程 |
| `probe.jsonl` | `scripts/acceptance_prefill_probe.py` 的输出：`U` 的 TTFT 归因（§6） | 串行 |
| `eq_check.log` | `scripts/check_quant_equiv.py`：**还原服务器代码**的数值指纹（§2.3） | 串行 |
| `final/` | 主表用的合并结果（`merged_final.json`）+ 渲染出的 markdown（`table_final.md`） | 由上面两份拼出 |

## 每个 *.log 里能看到什么

- `[load] ...blocks=...`：KV 池划分（target 块数 / draft 池块数 / 窗口块数）——容量结论在这里
- `[spec] draft 滑窗开启: ... target 池 N 块`：滑窗档的池划分
- `[int8] draft 预量化直载 ...`：P9 直载生效
- `@@<group>@@{...}`：**逐 cell 的结构化结果**（每个变体的中位数、原始 5 组、噪声地板、
  `rounds_by_B`、`verify_graph_fallbacks`、`catchup_forwards`、`out_md5`）——
  表里每一行都是从这里取的
- `@@WROTE@@`：该组写出 json

## 三个"每轮"口径

`ms_per_round`（墙钟÷轮数）、`ms_per_round_steady`（剔除 prefill）、
`ms_per_round_maxB`（★ 只在 active B = 最大 batch 的轮上取中位数，剔除收尾尾巴）。
主表跨档比较一律读 `ms_per_round_maxB`，理由见文档 §4.2。

## 复算

```bash
cd /home/ziru/nano-vllm/p1-work
# 主表（串行）：core 用 serial 口径，三个滑窗组用 v2 口径
for g in core win win_int8 win_int8_direct; do
  CUDA_VISIBLE_DEVICES=0 NANOVLLM_PORT=2399 /home/ziru/nano-vllm/venv/bin/python -u \
    scripts/acceptance_matrix.py --group $g --out /tmp/acc_$g.json --runs 5
done
```
