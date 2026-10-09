# 入口 Prompt（整段复制给另一个 AI）

> 用途：把项目的**访问方式**和**任务**一次性交给一个新会话。
> 下面的 `====` 之间就是要复制的内容。

====

# 任务：为 nano-vllm 推理引擎优化项目制定「下一步优化方案」

你接手一个**已经做到一半**的项目：在 nano-vllm（1385 行教学版 vLLM）上做推理引擎优化，
目标是**投 AI Infra 实习的简历项目**。

**你的交付物是一份「下一步优化方案」文档 —— 不是代码。**

---

## 一、代码/文档在哪（按你能用的方式挑一个）

### 方式 A：GitHub 仓库（推荐，直接读文件）

```
仓库：https://github.com/henry-tmp-de/specdec-nanovllm
分支：v2-optimizations      ← 注意是这个分支，main 上没有这些内容
```

克隆：
```bash
git clone -b v2-optimizations https://github.com/henry-tmp-de/specdec-nanovllm.git
```

**只想读文件、不想克隆的话，用下面的 raw 直链**（已做 URL 编码，可直接 fetch）：

| 文件 | raw 直链 |
|---|---|
| 所有优化的总览 + 效果 + 证据 | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/docs/%E4%BC%98%E5%8C%96%E6%80%BB%E8%A7%88.md |
| 面试考点分析（172 道真实面经） | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/docs/%E9%9D%A2%E8%AF%95%E8%80%83%E7%82%B9%E5%88%86%E6%9E%90.md |
| 考点逐题分桶（原始数据，每桶全部原题） | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/docs/%E9%9D%A2%E8%AF%95%E8%80%83%E7%82%B9-%E9%80%90%E9%A2%98%E5%88%86%E6%A1%B6.md |
| 生态调研：别人在 nano-vllm 上做了什么 | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/docs/%E7%94%9F%E6%80%81%E8%B0%83%E7%A0%94-%E5%88%AB%E4%BA%BA%E5%9C%A8nanovllm%E4%B8%8A%E5%81%9A%E4%BA%86%E4%BB%80%E4%B9%88.md |
| **你要照着执行的任务说明** | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/docs/%E4%B8%8B%E4%B8%80%E6%AD%A5%E4%BC%98%E5%8C%96%E6%96%B9%E6%A1%88-%E7%BB%99AI%E7%9A%84prompt.md |
| 项目简介（面试向） | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/README.md |
| 交接文档（环境/坑/下一步） | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/HANDOVER.md |
| kernel 那条线的完整记录 | https://raw.githubusercontent.com/henry-tmp-de/specdec-nanovllm/v2-optimizations/nanovllm/kernels/README.md |

### 方式 B：本地路径（如果你和我在同一台机器上）

```
D:\学习\specdec\               项目根目录（git 仓库，分支 v2-optimizations）
D:\学习\AI-Infra-面经整理\      面经原始语料（含 ocr-results/ 的 12 篇 OCR 原文）
D:\学习\AI-Infra-算子项目-从零到实习.md   用户已有的算子项目路线图（避免重复）
```

### 方式 C：实验服务器（只在需要跑实验时用）

```bash
ssh lab3090                      # 10.201.126.245, user=ziru
cd /home/ziru/nano-vllm/repo     # 工作副本（★ 不是 git 仓库）
/home/ziru/nano-vllm/venv/bin/python
```

⚠️ 服务器注意事项（都写在 HANDOVER.md 里，这里先给三条最要紧的）：
1. **8× RTX 3090 是共享的**，用前必须 `nvidia-smi` 查空卡，**不能抢别人的**
2. **`ncu` 被禁**（`RmProfilingAdminOnly: 1`，账号无 sudo），拿不到硬件计数器；`nsys` 和 `torch.profiler` 可用
3. **SSH 会随机掉线**，长任务要 `setsid ... > 文件`，不要直接管道

---

## 二、你要做什么

**打开 `docs/下一步优化方案-给AI的prompt.md`，严格按里面的要求执行。**

那份文件里写清了：
- 必读哪些输入、各自给你什么
- 产出格式（候选方向表必须 8 列、排序、明确「不做什么」、验收标准）
- **硬性要求**（不许用大类占比给具体技术背书、不许和已有工作重复、不许和别人撞车、
  成本要诚实给区间、不许编造）
- 环境约束
- 一份**要避开的外部简历原文**（别人已经做过的方向）

---

## 三、如果你读不到网络

按这个顺序退：方式 B（本地路径）→ 方式 C（服务器）。
**三个都不可用的话，直接告诉我「读不到」，不要凭猜测开始写方案。**

---

## 四、一条最要紧的提醒

那份 prompt 里有一句是留给**你**判断的，没有标准答案：

> **「再找一个新考点来做」的边际收益，是不是已经低于
> 「把已命中的考点从 microbenchmark 升级成端到端数字」？**

**要的是你的论证，不是你的结论。** 请基于你读到的数据说理。

====
