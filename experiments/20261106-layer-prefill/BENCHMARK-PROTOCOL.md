# 层间 prefill：实验与 CPU 合同

本轮只比较 Qwen3-4B、BF16、TP1、单卡、无 draft/spec/mixed/quant、关闭 prefix lookup 的固定批请求。所有阶段共用本目录 `gpu_control.py` 的一份新 ledger：总阶段墙钟上限 **1800 秒**，其中 smoke 上限 **300 秒**，包含模型加载、编译、失败、重试和进程清理；至少保留 45 秒清理时间。具体 GPU 在启动前由主协调现场确认空闲和 UUID。CPU/mock 测试不是 GPU 结果。

## 控制条件

| 方案 | 内部分块 | 层间流水 | 外部分块 |
|---|---|---|---|
| A 原完整 prefill | 关闭 | 关闭 | 0 |
| B 相同内部块串行 | 128 tokens | `serial` | 0 |
| C 相同内部块流水 | 128 tokens | `pipeline`，最多 2 个 in-flight | 0 |

参数与实现者约定为 `enable_layer_prefill`、`layer_prefill_chunk_tokens`、`layer_prefill_mode`、`layer_prefill_max_inflight`。A 关闭开关；B/C 开关打开且使用相同 chunk 与 in-flight 设置。正常 decode Graph 策略、各个实际 batch 的桶、`max_num_seqs`、`max_model_len`、输入哈希、token cap、seed 和**同一固定显存预算**须一致；实际 KV 块数和所有 buffer/显存也分别记录。B/C 任一结构性差异或 fallback 必须显式列出，不能把未执行的流水冒充成功。

旧 scheduler 对 `prefill_chunk_tokens>0` 且无 running 请求时会将该轮 chunk 设为 0。因此外部 scheduler 分块只能作为独立参考，且必须凭实际块与调度计数证明生效；它额外包含 scheduler/sampler 成本，不能把差异全归于层间 overlap。

## 正确性：先跑小 smoke

- 相同真实 prompt 及同一固定续写历史比较 A、B、C；B1 和混长 B2（257、513）覆盖尾块与跨页位置。必要时从已有 workload 选 token IDs，不下载新模型或数据。上下文边界使用实际 block table 映射，按逻辑位置 127/128/255/256 和每条末尾采样**全部层** K/V，不能假设不同引擎的物理块号相等。
- A/B/C 各自分配或重置完整 KV 池，避免一臂漏写后读到另一臂数据；取样前主 stream 必须等待所有任务完成。K 与 V 各自在全部层、上述各有效边界和末 partial 块上分别检查。
- 捕获原生 prefill 末 logits；以参考分布在前计算 `KL(reference || candidate)`：A 为 A/B 参考，B 为 B/C 参考。各请求末位置及随后 16 步每行分别过门槛，不能只用平均值掩盖单点；全部 arm 的续写都使用同一组固定 token。记录 finite、argmax、top1-top2 margin 及数值差。固定历史是数值诊断，不是用强制输出测速度。
- 首次 GPU 前固定且经独立审核的工程容差：B/C 每行 KL `<1e-4`、每层/采样位置分别对 K/V 计算 relative RMS `<0.01`；A/B 可容许跨形状 BF16 变动，KL `<0.02`、KV relative RMS `<0.05`。若参考张量 RMS 为零，则检查该张量 `max_abs ≤1e-6`；两级均需 finite 与 argmax 一致。其余 max absolute 误差只作诊断；argmax 不一致先停并核 top1-top2 margin，不自动推断为 race，也不事后放宽门槛。这是预声明的工程门槛，不是 BF16 理论误差保证。
- 真实 `runner.layer_prefill_stats` 要按运行前后差量记录 mode、chunk、in-flight、prefill/completed steps、tasks/planned/enqueued/completed、依赖等待和每个 stream 任务数。warmup 无 KV 的 fallback 独立列出，不作成功；异常 KV 元数据、非有限值、缺块、错误 stream/context 生命周期或证据不足均保留并停止。
- 引擎的 `SamplingParams` 禁止温度接近零，sampler 是正温随机采样。性能使用原生 T=1；正确性用实际 logits 离线计算 argmax，不能以极小温度冒称 greedy。

## 性能与时间口径

1. 先以 `max_output_tokens=1` 测真实 prefill：从请求入队至全部完成的墙钟、首 token 延迟及 `实际输入 token 数 / 墙钟`。输出 1 时 TPOT 不适用。旧 `bench_formal.options_for` 限制输出至少 9，不能直接用它冒充本用例；要用真实引擎新入口。
2. 另测固定批 B1/4、1K/4K prompt、output 128 的完整请求输出 token TPS、TTFT/TPOT、decode wall 和实际批大小；8K 仅在预算允许且由主协调决定时追加。每有效 cell 同 seed 计时 5 次，报告全部值及 median/min–max，范围不是置信区间。不同 arm 的模型加载/编译/热身分开记账并排除出主计时。
3. 单进程固定模式、shape 可复用模型并跑多次；外层旋转 A/B/C 顺序。运行前对要测形状和 decode 图显式预热，记录 capture 与运行期 fallback；正式墙钟不插逐任务 event 或多余同步。资源不足、实际 B 不达标、意外 capture、错误与未运行都保留，不自动减负载或换预算。
4. chunk512 可在 chunk128 小 pilot 有希望后单列；in-flight4 仅作为单独点且需主协调决定，不自动扩成大矩阵。prefill input TPS 与完整请求 output TPS 分开报告；仅看到 CUDA overlap 不构成吞吐提升。

现场只读确认 NSYS 路径为 `/usr/local/cuda/bin/nsys`（2024.6.2，不在 PATH）；可由主协调选一个代表形状对 serial 和 pipeline 各跑一次 `--trace=cuda,nvtx --sample=none --cpuctxsw=none` 的诊断，独立保存无 profiler 的性能记录；无需 metrics 权限、安装或 sudo。`layer_prefill_nvtx=False` 默认关闭，诊断时才显式启用逐 task/layer range。GPU 跨 stream 交并须由 CUDA kernel 区间计算，CPU NVTX 的时间跨度只能用于标识任务，不能当 GPU overlap。

## CPU 检查

本机已有 Anaconda Python，不安装依赖：

```powershell
& 'D:\Anaconda\python.exe' -B -c "import numpy; import pytest; raise SystemExit(pytest.main(['-q','experiments/20261106-layer-prefill/tests_layer_prefill_guard.py','experiments/20261105-dspark-window-resume/tests_gpu_control.py']))"
```

新增守护合同验证总预算、smoke 子预算和清理预留在 GPU 查询之前生效；旧测试验证 UUID、PID/start ticks、pidfd、TID→Tgid、foreign process 与 ledger 计费。后续引擎就绪后还需 CPU 模拟 task DAG 同层前块、同块前层、source/join 和 stream 资源归还，检查默认关闭的路径不创建流水 stream、不改 context；不能把这些 mock 计作 GPU 正确性或性能通过。
