# 阶段一：两处目标路径修复（CPU）

基线 `88f6861a34d7a788901411f2ab41ad609f359879`。以下均为本机 Python 3.13.9 的 CPU 元数据检查；真实调度器、块管理器、序列和相关 ModelRunner 入口运行，tensor/model/sampling 使用桩，不运行模型或 CUDA/GPU。

## 普通 decode 写槽

命令：`D:\Anaconda\python.exe -B experiments\20261102-dspark-window\cpu_decode_slot_repro.py`。另用 `--output experiments\20261105-dspark-window-resume\cpu-decode-slot-before.json` 在修复前保存基线。该次退出码 1：控制组 249→249；接受 0/2 个候选后绝对位置 250/252，实际错误槽 506/508，应为 250/252。修复后退出码 0，三项均与对应绝对位置槽一致。原始数字分别在 `cpu-decode-slot-before.json` 和 `cpu-decode-slot-repro.json`；旧实验 JSON 未覆盖。

原因是预留的 block_table 尾块不一定包含当前 anchor；修复以 `(len(seq)-1) // block_size` 和余数选块与偏移。此修复不能解释任何 DSpark 性能收益。

## 混批 prefill 与抢占

`D:\Anaconda\python.exe -B experiments\20261105-dspark-window-resume\cpu_mixed_preemption_repro.py --baseline` 通过 `git show 88f6861` 装入原始 scheduler，退出码 1，原始输出 `mixed-before.log`：本轮 prefill 序列状态变成 WAITING，块表从 `[2,3]` 变成空，实际 prefill 位置变成 0–3 且 slots 空，而应是位置 4–7 和 slots 12–15。

去掉 `--baseline` 运行当前 scheduler，退出码 0，原始输出 `mixed-after.log`：存在其他 decode 受害者时保护 prefill、抢占该受害者并调度 decode；无其他受害者时保护 prefill、抢占当前 decode，退回纯 prefill；普通非混批的尾部受害者策略不变。成功路径还执行了真实 prefill postprocess。scheduler 只从本轮 prefill 之外选受害者。

## 既有回归

以下命令均使用 `D:\Anaconda\python.exe -B tests\<文件名>`，退出码均为 0，原始输出为对应 `*.py.log`：

- `test_overlap_mixed.py`
- `test_block_alloc.py`
- `test_prefill_chunk_gate.py`
- `test_prefill_multiseq.py`
- `test_spec_gate.py`
- `test_prefix_hash.py`

这些证据不证明 GPU 数值正确性，也不构成吞吐或容量结论。
