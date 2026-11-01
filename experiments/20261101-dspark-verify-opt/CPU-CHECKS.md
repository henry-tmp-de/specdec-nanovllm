# 实现与 CPU 契约检查

基线 `6f55bb1`。当前实现保留默认的 DSpark 原始单图；新开关 `dspark_split_graph` 和 `dspark_graph_pad` 均默认关闭，且后者必须与前者同时开启。target verify 的 `spec_graph_pad` 与 0.6B AR 原有开关沿用，不把它改成新算法。

拆分路径只把确定性的 7 位 backbone/LM projection 补桶；逐位 Markov 修正和 `softmax(logits/T) / Exp(1)` 仍仅对真实 B 行执行。head CUDA graph 仅按真实 B 编号，不按上下文建副本；backbone 用有限 `(B桶, context桶)`，head 与 backbone 各自共享非并发图池。输出静态缓冲在图外分配并在下一次重放前消费；记录精确命中、补桶命中、运行期捕图的形状/host时间、回退和显存。

CPU 执行：

```text
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 D:\Anaconda\python.exe -m pytest -q tests/test_dspark_pad.py tests/test_dspark_contract.py
16 passed

KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 D:\Anaconda\python.exe tests/test_graph_pad.py
全部通过
```

CPU 覆盖：B=3/11/15 向上补桶的 metadata/真行切分，dummy context=0 且块表=-1，padding 前后真实 q 和 RNG 状态，backbone 不写上下文 KV，原单图的指数竞争采样与拆分 eager 在同 seed 下 q/token/RNG 一致；用 fake graph 逐步重放还验证了实际桶路由、按真实 B 采样、计数和结果切分。现有 P16 的 target pad 和越界/不齐形状已有独立 CPU 回归。

待 GPU 证实：PyTorch CUDA graph 的实际 RNG 状态与显存池复用；真实 target/prefix KV、dummy slot 和 sentinel；B3/11/15 的 eager/精确/补桶 target logits 及同前驱条件 q；真实 p/q 采样的分布；ragged fallback。CPU fake graph 不是 CUDA 正确性或性能证据。

正式阶段前，按主协调确认的方案增加实验层原单图预热 helper，保持原 `propose` 路径不变；新增的有限形状、空 dummy 和错误形状测试通过，完整 pytest **17/17**。逐序列 remaining/accepted 尾批记录及运行顺序控制均在实验脚本中。
