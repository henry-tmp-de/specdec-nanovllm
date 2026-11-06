# 使用实验性分层prefill

此分支已接入正常`LLM`配置，开关在创建引擎时决定，不承诺运行中热切换。默认`enable_layer_prefill=False`，普通路径保留。当前小测显示吞吐比整段prefill慢，建议默认保持关闭，数据见[报告](FINAL_REPORT.md)。

以下是服务器上本轮测过的配置；示例本身没有再执行GPU：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/home/ziru/nano-vllm/models/Qwen3-4B",
    max_num_seqs=4,
    max_model_len=4232,                 # 4096输入 + 128输出 + 8预留
    max_num_batched_tokens=16384,
    gpu_memory_utilization=0.70,
    tensor_parallel_size=1,
    decode_graph_bs=(1, 2, 3, 4),
    enable_prefix_cache=False,
    spec_k=0,
    enable_layer_prefill=True,         # False：回原prefill路径
    layer_prefill_chunk_tokens=128,
    layer_prefill_max_inflight=2,       # 当前仅支持1或2
    layer_prefill_mode="pipeline",     # "serial"：同DAG单流对照
    layer_prefill_nvtx=False,           # 单次性能不采集NVTX范围
)
```

在现有`llm.generate(prompts, SamplingParams(...))`流程使用即可。prefill在各层按task交错，普通decode仍沿原CUDA Graph路径。`serial`保留相同对角DAG但只用一条worker stream，不是旧scheduler的外层分块；maxinflight=1也没有双流收益。

限制：仅Qwen3 BF16、TP1、target-only；prefix必须关闭，draft/spec、mixed overlap和target量化不能组合。若使用旧DSpark配置，应创建独立target-only引擎，不能仅把此开关追加到DSpark参数上。初始化KV尚未分配时的warmup明确走原forward；真实输入缺KV表会报错而非静默回退。

设置为False无需启用任何分层参数；该状态不创建新的worker streams/executor。要重新测试其他长度/批量时按新实验处理，不能把本轮单点扩成普遍加速结论。
