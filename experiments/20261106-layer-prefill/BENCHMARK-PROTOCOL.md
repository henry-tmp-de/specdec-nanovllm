# 层间 prefill：实验与 CPU 合同

本轮只比较 Qwen3-4B、BF16、TP1、单卡、无 draft/spec/mixed/quant、关闭 prefix lookup 的固定批请求。用户随后将范围收缩：所有阶段共用本目录 `gpu_control.py` 的一份新 ledger，总阶段墙钟上限 **600 秒**，其中 smoke 上限 **300 秒**，包含模型加载、编译、失败、重试和进程清理；至少保留 45 秒清理时间。具体 GPU 在启动前由主协调现场确认空闲和 UUID。此前 1800 秒、五次重复及矩阵方案已被替代，仅保留在 Git 历史 `b0af4ec`；CPU/mock 测试不是 GPU 结果。

## 控制条件

| 方案 | 内部分块 | 层间流水 | 外部分块 |
|---|---|---|---|
| A 原完整 prefill | 关闭 | 关闭 | 0 |
| B 同 DAG 单流串行 | 128 tokens | `serial`，1 个 worker stream | 0 |
| C 相同内部块流水 | 128 tokens | `pipeline`，最多 2 个 in-flight | 0 |

参数与实现者约定为 `enable_layer_prefill`、`layer_prefill_chunk_tokens`、`layer_prefill_mode`、`layer_prefill_max_inflight`。A 关闭开关；B/C 开关打开且使用相同 chunk、对角 DAG 发射顺序与 in-flight 设置，B 用一个 worker stream，C 用两个。B 不是 chunk-major 的经典逐块执行，也不是 scheduler 外部分块；A/B 差异同时包括内部块、发射顺序、event 和元数据，不能只归因分块。正常 decode Graph 策略、各个实际 batch 的桶、`max_num_seqs`、`max_model_len`、输入哈希、token cap、seed 和**同一固定显存预算**须一致；实际 KV 块数和所有 buffer/显存也分别记录。B/C 任一结构性差异或 fallback 必须显式列出，不能把未执行的流水冒充成功。

本轮不增加外部 scheduler 分块对照。旧 scheduler 在无 running 请求时还会抑制此分块，不能仅凭参数声称它生效。

## 正确性：先跑小 smoke

- 相同真实 prompt 及同一固定续写历史比较 A、B、C；只做一次混长 B2（257、513），覆盖尾块与跨页位置。从已有 workload 选 token IDs，不下载新模型或数据。上下文边界使用实际 block table 映射，按逻辑位置 127/128/255/256 和每条末尾采样**全部层** K/V，不能假设不同引擎的物理块号相等。
- A/B/C 各自分配或重置完整 KV 池，避免一臂漏写后读到另一臂数据；每个已分配 page 先整体置零，再仅对该请求的有效 prompt 逻辑位置填 NaN，避免合法 padding 读取导致无关 NaN，同时让未来块的逻辑位置保持 NaN 以检查错误可见范围。已有 prefix 在 decode 续写时不重置。取样前主 stream 必须等待所有任务完成。K 与 V 各自在全部层、上述各有效边界和末 partial 块上分别检查。
- 捕获原生 prefill 末 logits；以参考分布在前计算 `KL(reference || candidate)`：A 为 A/B 和 A/C 参考，B 为 B/C 参考。各请求末位置及随后 16 步每行分别过门槛，不能只用平均值掩盖单点；全部 arm 的续写都使用同一组固定 token。记录 finite、argmax、top1-top2 margin 及数值差。固定历史是数值诊断，不是用强制输出测速度。
- 首次 GPU 前固定且经独立审核的工程容差：B/C 每行 KL `<1e-4`、每层/采样位置分别对 K/V 计算 relative RMS `<0.01`；A/B、A/C 可容许跨形状 BF16 变动，KL `<0.02`、KV relative RMS `<0.05`。若参考张量 RMS 为零，则检查该张量 `max_abs ≤1e-6`；两级均需 finite 与 argmax 一致。其余 max absolute 误差只作诊断；argmax 不一致先停并核 top1-top2 margin，不自动推断为 race，也不事后放宽门槛。这是预声明的工程门槛，不是 BF16 理论误差保证。
- 真实 `runner.layer_prefill_stats` 要按运行前后差量记录 mode、chunk、in-flight、prefill/completed steps、tasks/planned/enqueued/completed、依赖等待和每个 stream 任务数。warmup 无 KV 的 fallback 独立列出，不作成功；异常 KV 元数据、非有限值、缺块、错误 stream/context 生命周期或证据不足均保留并停止。
- 引擎的 `SamplingParams` 禁止温度接近零，sampler 是正温随机采样。性能使用原生 T=1；正确性用实际 logits 离线计算 argmax，不能以极小温度冒称 greedy。

## 性能与时间口径

只测 B4、4K prompt、output 128、chunk128、in-flight2，A/B/C 各一条 primary 计时记录；默认取已有 4K workload 的 code 面板，三臂使用相同输入、seed、预算和 decode Graph。每臂恰做一次不计入 primary 的实际请求热身，用于编译和 stream setup；加载、内置初始化与显式热身均计入 GPU 守护账。记录完整请求真实输出 token / 入队至全部完成墙钟、TTFT/TPOT、decode wall、实际 batch、pool 与计数。一次测量没有重复范围或置信区间，不能推广成普遍吞吐结论。

正式墙钟不插逐任务 event 或多余同步。资源不足、实际 B 不达标、意外 capture、错误与未运行都保留，不自动减负载、换预算或追加矩阵。本轮不测 output1、B1、1K、8K、chunk512 或 in-flight4；仅看到 CUDA overlap 也不构成吞吐提升。

本次不自动运行 NSYS；仅当数值或性能异常需要定位时，由主协调另行决定。现场只读确认 `/usr/local/cuda/bin/nsys` 为 2024.6.2；`layer_prefill_nvtx=False` 默认关闭。将来若确需 trace，CPU NVTX 时间跨度不能当 GPU overlap，应以 CUDA kernel 区间分析。

## CPU 检查

本机已有 Anaconda Python，不安装依赖：

```powershell
& 'D:\Anaconda\python.exe' -B -c "import numpy; import pytest; raise SystemExit(pytest.main(['-q','experiments/20261106-layer-prefill/tests_layer_prefill_guard.py','experiments/20261105-dspark-window-resume/tests_gpu_control.py']))"
```

新增守护合同验证总预算、smoke 子预算和清理预留在 GPU 查询之前生效；旧测试验证 UUID、PID/start ticks、pidfd、TID→Tgid、foreign process 与 ledger 计费。后续引擎就绪后还需 CPU 模拟 task DAG 同层前块、同块前层、source/join 和 stream 资源归还，检查默认关闭的路径不创建流水 stream、不改 context；不能把这些 mock 计作 GPU 正确性或性能通过。
