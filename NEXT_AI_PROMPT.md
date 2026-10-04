# 给下一个 AI 的 Prompt

下面这段直接复制给新的 AI 会话即可。

---

## 任务

接手一个「在 nano-vllm 上实现投机解码」的项目，继续把它做成能投 AI Infra 实习的简历项目。

## 第一步（必做）

先读这两个文件，它们包含全部上下文：

1. `D:\学习\specdec\HANDOVER.md` ← **完整交接文档，先读这个**
2. `D:\学习\specdec\README.md` ← 面向面试的项目介绍

再读代码：
- `D:\学习\specdec\nanovllm\spec_decode\` （三个核心模块，约 470 行）
- `D:\学习\specdec\nanovllm\engine\model_runner.py` 的 `prepare_verify` / `run_verify`
- `D:\学习\specdec\tests\test_spec_decode.py` （16 项单测，含无损性验证）

## 环境

```bash
ssh lab3090# 10.201.126.245
cd /home/ziru/nano-vllm/repo    # 工作副本（与本地 D:\学习\specdec 对应）
```

**三条铁律**：
1. **用 GPU 前必查 `nvidia-smi`**，不能抢别人的卡（8×3090，通常只有 1-2 张空着）
2. **测试必须串行跑** —— nano-vllm 硬编码 `tcp://localhost:2333`，并发会报 `EADDRINUSE`
3. 服务器在 `~/nano-vllm/models/` 有 Qwen3-4B（target）和 Qwen3-0.6B（draft）

## 当前状态（一句话）

线性链投机解码**实现正确、输出无损、但速度是负的**（17.0 → 11.3 tok/s）。

**根因已量化**：每步 decode 读全部权重，baseline 8.0 GB，draft k=2 是 10.2 GB。
draft 要串行跑 k 次，开销抵消了「每步多产出 token」的收益。

## 你的核心任务：让它真正加速

**推荐方案：实现 EAGLE 式单层 draft head**

```
现在：跑整个 0.6B 模型当 draft → c ≈ 0.14
改成：只跑 target 的最后一层 + 一个轻量 draft head → c ≈ 0.02
```

参考资料：
- EAGLE（arXiv 2401.15077）
- EAGLE-3（arXiv 2503.01840），已进 vLLM `SpeculativeMethod="eagle3"` 和 SGLang
- vLLM 主线 `vllm/model_executor/models/eagle.py`（用 `gh api` 拉源码）

**次要任务（成本低、收益明确）**：
1. **batch 门控实测** —— 扫 batch = 1/4/8/16/32/64，找盈亏平衡点。
   代码已实现（`config.spec_batch_threshold`），只差实测数据。这本身是可写进简历的数据点。
2. **提上游 PR** —— `hash_blocks()` 的调用顺序 bug（上游同源问题，
   `scheduler.py` 里 `< 5 行` 改动，是投机解码必经之路上的真实 bug）。
   相关 issue：#243 / #210 / #233。⚠️ **不要用 #51（已修）和 #99（误报）**。

## 项目的战略定位（别推翻）

调研 1195 条中文 AI Infra 面经的结论：

```
CUDA/算子 14.4% + 系统设计 10% + C++ 6.9% + 显存KV 6.1% ≈ 37%  ← 主战场
投机解码 1.3%（14个主题垫底）← 冷门
```

所以：**投机解码是论据，不是主题**。项目主线应该是「推理引擎优化」，
投机解码是其中一个证明你懂调度 + 显存 + attention mask 的手柄。

## 三个已经站得住的「可 defend 判断」（都��数据）

```
① n-gram 投机无效的根因：没有真实分布 → 接受概率退化为 p[draft]
   实测 0.6% vs draft model 的 31.9%，差 50 倍
② 树状投机是过度设计：论文消融 linear 5.345× > tree 5.175×，
   且 Snowflake 生产代码 use_tree_spec 默认 False
③ draft 开销抵消收益：加速比 = f(权重比, 接受率, k)，
   0.6B→4B 实测净亏损 34%，给出盈亏平衡条件
```

## ★ 两个方法论陷阱（我各浪费了 4 轮，别重犯）

**陷阱 1：不要用低 temperature 做逐 token 一致性对比**
```python
SamplingParams(temperature=0.01)   # ❌
# softmax(logits/0.01) 会把分布压成 one-hot，p 必然 = 1.0000
# baseline 同样如此 → 会误判成「draft 分布退化」→ 白改很多轮
```
正确：`temperature=1.0` + 统计检验。

**陷阱 2：随机采样下逐 token 对比没有意义**
正确：比较 **token 频率分布**，不是逐个 token 是否相同。

**GPU 精度确实影响一致性**（实测 TV=0.033，30 个位置 1 次 argmax 不同 = 3.3%），
所以「逐 token 严格一致」这个标准本身过严，要用统计无偏的判据。

## 代码约定

- 主要语言：Python（nano-vllm 是纯 Python + Triton，无 C++/CUDA 要改）
- 测试：`tests/test_spec_decode.py` 跑起来只要 CPU，`python tests/test_spec_decode.py`
- 改完必须跑一遍单测 + `bench_final.py` 确认没改坏
- 用户是研究生，需要能**讲清整个项目**，所以别引入他理解不了的复杂度

## 交付要求

改动完成后，产出：
1. 更新 `HANDOVER.md`（实测数据 + 结论 + 下一步）
2. 更新 `README.md`（benchmark 表 + 适用边界）
3. 明确回答：「这个改动让加速比从多少变成多少，依据是什么」