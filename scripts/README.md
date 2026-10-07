# 实验脚本

| 脚本 | 用途 | 输出标记 |
|---|---|---|
| `bench_spec.py` | **主基准**：加速比 / 每步 token / 接受率，`graph 0/1` 切换 CUDA graph | `@@B@@` JSON |
| `bench_batch.py` | batch 门控实测（一个进程里跑「开投机/关投机」两遍，省一次加载） | `@@BATCH@@` JSON |
| `check_draft_quality.py` | draft 固有质量：用 transformers 原生跑两个模型，算 TV 距离与 α 的理论上界 | `@@D@@` JSON |
| `check_graph_and_verify.py` | 图路径 vs eager 路径逐元素对拍 + 拒绝采样内部数值 | `@@G@@` JSON |
| `check_lossless_engine.py` | 引擎级无损性：跑 base / base2 / draft 三轮，比 token 频率分布 | `@@L@@` JSON |
| `check_stat.py` | 多 prompt 统计检验（draft 模式很慢，早期用） | `@@STAT@@` JSON |
| `stat2.py` | 单条长序列的 token 分布对比（早期用，样本量偏小） | `@@S@@` JSON |
| `bench_code.py` / `bench.py` | 早期 benchmark（保留供对照） | — |
| `bench_final.py` | 上一轮的基准。⚠️ **不要再用它下结论**：`enforce_eager=True`， 而且 `accept_rate` 的算式 `(landed − verify)/proposed` 在「k 个候选全被接受」时只有 (k−1)/k，**永远到不了 1** | `@@B@@` JSON |

配套（在服务器 repo 根目录，没进 git）：

```
run_final.sh + final_results.txt        256 token 的最终数据（含 eager 对照）
run_bench2.sh + bench_results.txt       64 token 的 k 扫描
run_lossless.sh + lossless_results.txt  引擎级无损性三轮
tv_compare.py                           比较三轮的 token 频率分布，用 base/base2 当噪声地板
```

⚠️ nano-vllm 硬编码 `tcp://localhost:2333`，**这些脚本必须串行跑**。
⚠️ 跑之前先看空卡：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`，
   本机通常只有 7 号卡空着，用 `CUDA_VISIBLE_DEVICES=7`。
