# 实验脚本

| 脚本 | 用途 | 输出标记 |
|---|---|---|
| `bench_final.py` | 性能基准（base / draft k） | `@@B@@` JSON |
| `stat2.py` | 统计无损性检验 | `@@S@@` JSON |
| `check_stat.py` | 多 prompt 统计检验（draft 模式很慢） | `@@STAT@@` JSON |
| `bench_code.py` | 早期 benchmark（保留供对照） | — |

⚠️ nano-vllm 硬编码 `tcp://localhost:2333`，**这些脚本必须串行跑**。
