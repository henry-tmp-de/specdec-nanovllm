# 使用已验收的DSpark滑窗方案

当前方案：**target Qwen3-4B保留完整上下文，DSpark draft仅保留最近512个历史位置，采用独立环池和CUDA Graph。** 全历史草稿只是实验对照。

服务器使用现有Python `/home/ziru/nano-vllm/venv/bin/python`；草稿目录 `/home/ziru/nano-vllm/dspark-integration-20261101/models/dspark-4b`。运行前选择现场空闲GPU并设置CUDA_VISIBLE_DEVICES和空闲的NANOVLLM_PORT（本轮用24805）；本轮结束后没有自动运行或占卡任务。

已测的负载配置：

| 输入长度上限 | 最大请求数 | max_model_len | prefill token预算 |
|---:|---:|---:|---:|
| 1,024 | 16 | 1,288 | 16,384 |
| 4,096 | 8 | 4,360 | 32,768 |
| 16,384 | 4 | 16,648 | 4,096 |

三档均按每请求输出上限256配置并预留投机位置。它们是本轮成功测试的配置，不是所有模型/长度的最大并发结论。B16/4K在本轮配置下容量不足。

```python
from nanovllm import LLM, SamplingParams

B = 8  # 此例使用已测4K输入档
graph_batches = tuple(range(1, B + 1))
llm = LLM(
    "/home/ziru/nano-vllm/models/Qwen3-4B",
    draft_model="/home/ziru/nano-vllm/dspark-integration-20261101/models/dspark-4b",
    spec_method="dspark", spec_k=7,
    dspark_window_tokens=512,
    dspark_window_storage="ring",
    enable_prefix_cache=False,
    tensor_parallel_size=1,
    max_num_seqs=B, max_model_len=4360,
    max_num_batched_tokens=32768,
    gpu_memory_utilization=0.85,
    enforce_eager=False, spec_cuda_graph=True,
    decode_graph_bs=graph_batches,
    spec_graph_bs=graph_batches,
    spec_graph_pad=False,
    dspark_split_graph=True, dspark_graph_pad=False,
    spec_overlap_mixed=False,
)

# 可选的启动期建图；W512只需覆盖256/512两种context桶。
# 这部分时间在本轮性能基准中单列，首次懒捕图不属于热身后性能。
llm.model_runner.spec_proposer.prewarm(
    [(b, c) for b in graph_batches for c in (256, 512)],
    range(1, B + 1),
)

text = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "写一个Python二分查找。"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
outputs = llm.generate([text], SamplingParams(temperature=1.0, max_tokens=256))
print(outputs[0]["text"])
```

例子只展示API，没有在文档生成时加载GPU。当前滑窗明确要求TP1、prefix关闭和mixed overlap关闭；使用普通自然EOS停止与本轮固定长度吞吐的ignore_eos=True条件不同。速度数据按METHODS.md和正式报告引用。
