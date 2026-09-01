# 自主机器学习研究智能体 — KuaiRand-Pure

（English version: [README.md](README.md)）

一个由大语言模型驱动的智能体，在一个冻结的推荐系统基准上运行**完整的机器学习研究闭环**：
它复现官方基线，说明自己观察到的瓶颈，提出一个实验，修改真实的模型代码或配置，在训练之前先对该改动
进行验证，随后训练、评估，并通过配对自助法与对照组比较，决定下一步要研究什么，组装一个互补的模型组合，
写出一份经过校验的提交结果——直到触发**主办方自己定义的收敛规则**才会停止。

面向 **TikTok TechJam 2026，赛道二**（面向推荐系统的自主机器学习研究智能体）打造。

```
观察 → 提出假设 → 制定计划 → 编码 → 护栏检查 → 训练 → 评估
  → 对比 → 反思 → 决策 → 下一个实验 → 组合集成 → 收尾定稿
```

---

## 问题定义

对每一位用户，按 `long_view`（一个二元的"看完"信号）对*该用户自己*的曝光记录排序。不涉及任何全局候选池的
检索——这是**用户内排序**，因此纯用户侧特征完全不携带信号。

分数：`主指标 = ½(GAUC + nDCG@5)`，由一个冻结的评估器计算。

两个数字决定了一切：**oracle 上限约为 0.85，而非 1.0**（42% 的验证集用户是全正例或全负例，
任何模型都无法改变他们的 nDCG），以及**配对验证的噪声下限为 σ ≈ 0.0009**。真正的提升是以千分之一为单位度量的，一个分不清
信号与噪声的智能体只会追逐自己的方差。

## 当前结果

来自 `runs/run_20260831_090457`（Gemini 实时运行——4 次已执行实验，450 秒，12,593 个 token，
**0 次人工介入**）：

| | 主指标 |
|---|---|
| FM 基线（由智能体的根节点精确复现） | 0.60147 |
| 发现的最佳单一模型 | 0.60366 |
| 模型组合，在全部验证集上调参 —— *偏乐观，样本内估计* | 0.60463 |
| **模型组合，诚实的 5 折用户级交叉验证** | **0.60409 ± 0.00141**（相对 FM **+0.00259**） |

在 `official_convergence`（ε=0.002，N=3）处停止。提交结果通过了冻结的 `submit.py --check` 校验。

**我们引用的是交叉验证后的数字，而不是调参后的数字。** 系统将较大的那个数值明确标注为样本内估计，
并单独测出了这种乐观偏差的大小（约 +0.0007）。

## 自主性体现在哪里

* **先说明问题，再提出实验** —— `problem_identified` 是一个必填的首要字段。
* **它编写并采纳真实代码**，而不只是超参数：可以是六个代码块主体之一，也可以是整个模型族。
* **它拒绝编造。** 当被要求实现一个位置偏差塔时，它回复说该数据集中根本不存在位置特征——事实也确实如此。
* **它用统计方法解读结果。** 每个节点都会用配对的用户级自助法与对照组比较；驱动结论的是 `P(Δ>0)`，
  而不是一个原始差值。
* **它考虑多样性，而不只是准确率** —— 即使某个模型的独立分数更差，只要它的误差与其他成员不相关，
  仍会保留它。
* **它不轻信不合常理的结果。** 它的"行为感知历史"实验测出了 **+0.0165**（是噪声下限的 18 倍）；
  智能体自己的机制发现了这个异常幅度，诊断出机制是一个评估假象，在诚实策略下该效应变成了 **−0.00167**。
  该特性最终以**禁用**状态提交。详见 [SUBMISSION.md 第10节](docs/CN/SUBMISSION.md#10-自我修正被否决的突破)
  （English: [docs/EN/SUBMISSION.md](docs/EN/SUBMISSION.md#10-self-correction-the-rejected-breakthrough)）。

## 架构一览

一条固定边界隔开的两层——*搜索什么*与*如何搜索*是分开的。

```
冻结的基础设施  data.py · evaluate.py · submit.py · pipeline/run_node.py · pipeline/contracts.py
──────────────────────────── 信任边界 ────────────────────────────
智能体（agent/）                        解空间（pipeline/）
  orchestrator   控制循环                 baseline_blocks/  FM 对照组（根节点）
  roles/         提议者·编码者·反思者     lib/din_blocks/   DeepFM + DIN 模型族
  tree           最佳优先搜索 + ε探索     lib/lgbm_blocks/  LightGBM LambdaRank 模型族
  blockspec      旋钮生效契约             lib/…             损失函数·训练器·缓存
  stats          配对用户自助法
  portfolio      价值评估 + K 折交叉验证  可观测性
  memory/ledger  证据，跨运行             dashboard/research-console.html
```

**冻结的信任边界**是核心设计决策：五个文件经 SHA-256 锁定，并在每次运行开始时校验，因此智能体永远
无法——无论是意外还是通过某种臆想出的"修复"——修改分数所依赖的代码。所有能力都绕过该边界运作，
而不会穿过它。详见 [SYSTEM.md 第3节](docs/CN/SYSTEM.md#3-冻结的信任边界)
（English: [docs/EN/SYSTEM.md](docs/EN/SYSTEM.md#3-frozen-trust-boundary)）。

## 快速开始

```bash
# 1. 安装依赖
pip install numpy torch lightgbm google-genai pydantic pyyaml scipy

# 2. 数据（约 195 MB，无需注册）放入 KuaiRand-Pure/data/ —— 来自 https://kuairand.com

# 3. 运行
python -m agent.run --smoke     # 精确复现 FM 基线并校验 frozen.lock（约 35 秒）
python -m agent.run --mock      # 完整研究闭环的离线版本——无需 API key，无需额度
python -m agent.run --faults    # 注入一次崩溃，验证恢复过程，仍能完成提交
python -m agent.run             # 实时运行（需要 GEMINI_API_KEY 或 .env.local）
```

首次运行会构建 `runs/_cache/`（约 60 秒）；之后每次运行都会复用它。

**测试**（全部确定性执行，无需 API key）。每个测试都是独立模块：

```bash
python -m tests.test_stats          # 自助法与冻结评估器的误差小于 1e-5
python -m tests.test_blockspec      # 旋钮生效契约，含运行时逐位一致性校验
python -m tests.test_leakage        # 隐藏集隔离；恶意代码块被拒绝
python -m tests.test_orchestration  # 收敛语义、去重、证据≠状态
python -m tests.test_sequence       # 时间顺序 + 反馈状态泄漏安全性
```

支持两种解释器；优先使用 `cudaenv/Scripts/python.exe`（Python 3.12 + CUDA torch）——FM 与 LightGBM
是确定性的、与解释器无关，只有 DIN 有所不同，且在 GPU 上跑得快得多。

## 研究控制台

一个展示智能体真实研究闭环的实时/回放视图。它渲染的是智能体本身产出的产物——它从不模拟执行过程。

```bash
python -m agent.console_server        # http://127.0.0.1:8712/
```

**▶ 回放**以 1×–12× 的速度回放一次已完成的运行（无需 API key、网络或 GPU）。
**◉ 实时**则会流式接收正在运行的智能体产生的事件。

## 接下来读什么

文档以英文（`docs/EN/`）维护，并配有简体中文译本（`docs/CN/`），两者是同样的三份文档——
下方仪表盘中的界面文案是实时双语切换的，但这些文档是静态的，请直接选择你想要的语言版本。

| 文档 | 面向 | 内容 | English |
|---|---|---|---|
| **[docs/CN/SUBMISSION.md](docs/CN/SUBMISSION.md)** | 评审与审稿人 | 竞赛叙事：为什么这是一个研究者而不是一个 AutoML 脚本、带诚实不确定性的结果、被否决的突破事件、资源消耗、当前局限 | [docs/EN/SUBMISSION.md](docs/EN/SUBMISSION.md) |
| **[docs/CN/SYSTEM.md](docs/CN/SYSTEM.md)** | 工程师 | 系统现在如何运作：信任边界、六代码块解空间、缓存与泄漏架构、智能体内部机制、配置校验、统计机制、模型组合装配、收敛语义、配置参考、运行产物、扩展规则、已知局限 | [docs/EN/SYSTEM.md](docs/EN/SYSTEM.md) |
| **[docs/CN/RESEARCH.md](docs/CN/RESEARCH.md)** | 评审与研究者 | 科学记录：指标推导、噪声模型、统计方法论、每一项已测量的发现——**包括负面结果**、文献支撑、残余不确定性、未来方向 | [docs/EN/RESEARCH.md](docs/EN/RESEARCH.md) |
| [docs/PROBLEM_STATEMENT.pdf](docs/PROBLEM_STATEMENT.pdf) | 参考 | 主办方的原始说明文件（不可变更，仅英文） | — |

证据标签 **[VERIFIED]（已验证）/ [MEASURED]（已测量）/ [LITERATURE]（文献支撑）/ [PROPOSED]（提议中）**
贯穿 RESEARCH.md 全文（以及其 `docs/CN/` 译本，标签本身未翻译），以确保已实现的事实、实测结果与
未来工作三者不被混淆。

## 诚实状态

相对基线的提升是适度的（诚实估计 +0.0026），而该基准的上限是 0.8484，噪声下限是 0.0009。所有数字
都来自验证集；测试集表现按设计未被测量。若干开放问题——尤其是全部辅助任务分支——因**统计功效不足
而无法下定论**，报告中如实标注，而不是被四舍五入成"获胜"。完整局限说明见
[SUBMISSION.md 第11节](docs/CN/SUBMISSION.md#11-当前局限) 与
[SYSTEM.md 第24节](docs/CN/SYSTEM.md#24-已知的工程局限)。
