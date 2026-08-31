# SYSTEM.md — 工程参考文档

**本文档权威地描述了系统目前的实际状态。** 科学记录（证据、推导过程、负面结果）见
[RESEARCH.md](RESEARCH.md)；竞赛叙事见 [SUBMISSION.md](SUBMISSION.md)；总览与入门见
[README](../../README.md)。

（English version: [docs/EN/SYSTEM.md](../EN/SYSTEM.md)）

若本文档与任何更早的文字记录存在冲突，以本文档与代码为准。

> **文档冻结，2026-08-31。** 八份历史性的设计、集成与规划文档，已被整合进四份权威文档
> （README · SYSTEM · RESEARCH · SUBMISSION），随后被删除。没有任何内容因此丢失：工程内容在本文档中，
> 科学内容与全部测量结果在 [RESEARCH.md](RESEARCH.md) 中，竞赛叙事在
> [SUBMISSION.md](SUBMISSION.md) 中。那些旧文档中的若干表述**已经过时或有误**，
> 现已对照当前代码与运行产物做出更正——特别是收敛窗口、缓存版本、"物理上不可能"这一泄漏相关表述、
> 序列时序安全性保证、集成方法论，以及收尾定稿的行为逻辑。源代码中引用旧文档的注释，已更新为指向本文档。

---

## 目录

1. [系统概览](#1-系统概览) · 2. [设计原则](#2-设计原则) ·
3. [冻结的信任边界](#3-冻结的信任边界) · 4. [代码库架构](#4-代码库架构) ·
5. [六代码块解空间](#5-六代码块解空间) · 6. [节点表示](#6-节点实验表示) ·
7. [数据与缓存架构](#7-数据加载与缓存架构) · 8. [完整性保护](#8-泄漏完整性保护) ·
9. [模型族与干预手段](#9-模型族与已实现的干预手段) · 10. [智能体架构](#10-智能体架构) ·
11. [配置有效性校验](#11-配置有效性校验) · 12. [去重与空操作处理](#12-去重与空操作处理) ·
13. [统计决策机制](#13-统计决策机制) · 14. [研究记忆与台账](#14-研究记忆与跨运行台账) ·
15. [模型组合机制](#15-模型组合集成机制) · 16. [收敛与预算语义](#16-收敛与基准预算语义) ·
17. [故障恢复](#17-故障恢复) · 18. [收尾定稿](#18-提交与收尾定稿) ·
19. [研究控制台](#19-研究控制台架构) · 20. [配置参考](#20-配置参考) ·
21. [运行与测试](#21-运行与测试) · 22. [运行产物与数据结构](#22-运行产物与数据结构) ·
23. [扩展规则](#23-扩展规则) · 24. [已知局限](#24-已知的工程局限)

---

## 1. 系统概览

一个由大语言模型驱动的智能体，在 KuaiRand-Pure 用户内短视频排序基准上运行完整的机器学习研究闭环：
它复现官方的因子分解机（FM）基线，说明自己观察到的瓶颈，提出一个实验，修改真实的模型代码或配置，
在训练之前先对该改动进行验证，随后训练、评估，并通过配对自助法与对照组比较，决定下一步要研究什么，
组装一个模型组合，写出一份经过校验的提交结果——直到触发主办方自己定义的收敛规则才会停止。

**任务。** 对每个用户，按 `long_view` 对*该用户自己*的曝光记录进行排序，不从任何全局候选池中检索。
分数为 `主指标 = ½(GAUC + nDCG@5)`，由冻结的 `evaluate.py` 计算。

**当前实测结果**（`runs/run_20260831_090457`，Gemini 实时运行）：

| 指标 | 数值 |
|---|---|
| FM 基线（根节点） | 0.60147 |
| 最佳单模型 | 0.60366（DIN，节点 `n3`） |
| 模型组合，在全部验证集上调参 —— **偏乐观** | 0.60463 |
| 模型组合，**诚实**的 5 折用户级交叉验证 | **0.60409 ± 0.00141** |
| 停止原因 | `official_convergence` |
| 提交结果 | 通过 `submit.py --check` 校验 |

对外应引用的是诚实的交叉验证估计。关于"调参估计"与"诚实估计"为何不同，见
[RESEARCH.md 第12节](RESEARCH.md#12-集成模型组合发现)。

---

## 2. 设计原则

1. **固定边界之后是两层结构。** 智能体搜索*什么*（`pipeline/`，解空间）与它*如何*搜索
   （`agent/`，策略）相互分离。编排器（orchestrator）驱动 FM、LightGBM 或 DIN 时，
   对这些模型本身一无所知。
2. **被评判的是智能体本身，而不只是模型。** TechJam 评分标准中大约 40% 的权重是智能体的行为
   （自主性、稳健性、可行性），因此这条研究闭环得到了和各干预手段本身同等精心的工程打磨。
3. **信任这个分数。** 分数所依赖的一切都被冻结并做了哈希锁定（见第3节）。
4. **"最佳存档点"不变量。** 提交结果始终是经过校验的最佳对象，独立于搜索路径（该路径可能并非单调）单独追踪。
5. **用低成本分析替代高成本的重新训练。** 自助法、排名相关性、混合评估与记忆综合，
   全部运行在已保存的预测结果上，从不触发一次新的训练（见第13节）。
6. **科学记录是证据，而不是采纳状态。** 一个节点在树中的状态，与关于其效果的证据，是两个独立的字段
   （见第14节）。
7. **算力不是目标。** 基准的迭代次数上限与总耗时上限是限制，不是要去达成的目标（见第16节）。

---

## 3. 冻结的信任边界

五个文件在 `agent/frozen.lock` 中做了 SHA-256 哈希锁定，并由 `agent/guardrails.py::ensure_frozen()`
在每次运行开始时校验。一旦不匹配，运行立即中止：

```
data.py   evaluate.py   submit.py   pipeline/run_node.py   pipeline/contracts.py
```

智能体可以改动的一切都在这条边界的下游：`agent/*`、`pipeline/lib/*`、
`pipeline/*_blocks/*`、新增的 `pipeline/*.py`、`tests/*`、`dashboard/*`。

**如果某项改动看起来需要修改一个冻结文件，那就说明这是一个错误的改动方向——应该绕开它。**
三种已确立的模式：

| 需求 | 应对模式 |
|---|---|
| 冻结的 `Cfg` 装不下的一个旋钮 | 通过 `pipeline/lib/ext.py` 读取的 `cfg_ext.json` 侧车文件（见第6节） |
| 从一次训练中额外产出的第二个推理切分 | 由推理代码块自己写出，`pipeline/lib/extra_infer.py`（见第18节） |
| 运行器没有对应参数的快速失败检查 | 对*缓存*做子采样，调用同一个运行器，`pipeline/debug_cache.py`（见第17节） |
| 对代码块隐藏某些标签 | 在加载阶段处理，`datced.load_bundle`（见第8节） |

第二道静态护栏 `executor.check_imports` 会在每个由智能体编写的代码块*运行之前*解析它，
拒绝不被允许的导入、被禁止的内置函数，以及涉及隐藏集路径的字面量（见第8节）。

---

## 4. 代码库架构

```
冻结的基础设施  data.py · evaluate.py · submit.py · pipeline/run_node.py · pipeline/contracts.py
──────────────────────────── 信任边界 ────────────────────────────
智能体（agent/）                        解空间（pipeline/）
  orchestrator  控制循环                 baseline_blocks/   FM 对照组（根节点）
  tree          最佳优先搜索             lib/din_blocks/    DeepFM+DIN 模型族
  roles/        提议者·编码者·反思者     lib/lgbm_blocks/   LightGBM LambdaRank 模型族
  llm/          Gemini ｜ Mock 驱动器    lib/fm.py          numpy 实现的 FM 主干
  blockspec     旋钮生效契约             lib/losses.py      BCE·BPR·softmax交叉熵·IPS-BCE
  mutate        节点物化                 lib/train_np.py    单点／成对／分组训练器
  provenance    预期 vs 实际执行         lib/din.py         DIN + 辅助头 + 反馈嵌入
  executor      沙箱·准入门·护栏         lib/gbm.py         LightGBM 特征 + 排序器
  datced        缓存构建／加载           lib/seq_build.py   按时间顺序的历史 + 反馈状态
  stats         配对用户自助法           lib/aux_build.py   辅助标签（仅训练切分）
  portfolio     价值评估 + K 折交叉验证  lib/rand_build.py  随机曝光切分
  memory        run_log + 研究状态       lib/ext.py         扩展配置侧车文件
  ledger        跨运行证据               lib/extra_infer.py 由同一次训练额外产出的切分
  events        结构化事件流             lib/debug_cache.py 用于调试准入门的子采样缓存
  reeval        多随机种子重新训练
  champion      跨运行冠军模型
  console_server 研究控制台服务端
可观测性  dashboard/research-console.html · dashboard/hypothesis-ledger.html
```

---

## 5. 六代码块解空间

一个节点的模型，是由冻结的运行器按顺序执行的六个 Python 文件：

```python
build_features(bundle, cfg) -> FeatureSet
build_model(meta, cfg)      -> 暴露 .logits/.apply_grad/.predict 的模型（或其包装器）
build_loss(cfg)             -> lossfn(z, batch) -> (loss, g)      # g = dL/dz，按行
train(model, lossfn, feats, bundle, cfg) -> model                 # 验证集最优，提前停止
infer(model, feats, split)  -> 与 bundle 行顺序对齐的 np.ndarray
combine(base, cfg)          -> np.ndarray                         # 组装阶段钩子
```

`pipeline/baseline_blocks/` 既是 FM+BCE 基线，**也是**消融实验的对照组（根节点）。
`build_loss` 返回 `make_loss(cfg)`，因此 `loss_type` 是一个真正生效的配置旋钮——它此前曾被硬编码为
BCE，这曾静默地使干预手段 A 完全失效（见 [RESEARCH.md 第7节](RESEARCH.md#7-bpr-理论与实测结果)）。

`FeatureSet` 是冻结的，无法新增字段。任何额外信息要么在训练器内部计算，要么搭载在一个已有的可选字段上——
行为感知历史就是以可选的第三元素形式搭载在 `seq` 元组上的。

---

## 6. 节点／实验表示

**一个节点即一次实验**，是六个代码块源码 + 一个 `Cfg` + 一个可选的 `cfg_ext.json` 侧车文件的快照。
完整快照（而非差异补丁）让每个节点都能独立运行，也让两个节点可以并发运行而互不冲突。

磁盘上的结构：`runs/<run_id>/nodes/<id>/{blocks/*.py, cfg.json, cfg_ext.json, provenance.json,
metrics.json, val_scores.npy, test_scores.npy, stdout.log}`。

三种改动类型：

| 类型 | 改变的内容 | 成本 |
|---|---|---|
| **配置** | 仅 `Cfg` / `cfg_ext` 的取值 | token 消耗接近零 |
| **代码块改写** | Coder（编码者）重写恰好一个代码块的主体 | 受 `check_imports` 把关 |
| **整族采纳** | `Hypothesis.adopt_blockset: "din" \| "lgbm"`，从 `pipeline/lib/<name>_blocks/` 整体换入一个预先写好的模型族 | `cfg.model_type` 跟随所挂载的代码块 |

`model_type` **由运行框架本身管理**——一个假设不能直接设置它。它必须跟随所挂载的代码块，
否则每个节点都会坍缩成同一个模型族，模型组合也就无从谈起。

### 扩展侧车文件

`pipeline/contracts.py` 是冻结的，因此 `Cfg` 无法新增字段，`Cfg.from_dict` 会静默丢弃未知的键。
后来添加的旋钮（`use_fb`、`fb_dropout`、`gbm_*` 系列超参数）存放在与 `cfg.json` 同级的
`cfg_ext.json` 中。代码块通过 `pipeline/lib/ext.py::load(__file__)` 读取它——运行器用
`importlib.util.spec_from_file_location` 加载代码块，所以 `__file__` 是
`<node_dir>/blocks/<name>.py`，侧车文件在其上两级目录。该侧车文件同样参与节点的内容签名与溯源哈希计算，
因此无法在去重机制之外偷偷夹带一个不被察觉的改动。

---

## 7. 数据加载与缓存架构

每次实验都重新读取约 106MB 的 CSV 文件，会严重拖累预算，因此所有数据都**只编码一次**，
以内存映射的 `.npy` 文件形式存放在 `runs/_cache/` 下。

**`CACHE_VERSION = 10`。** 每当缓存数组的布局发生变化时递增此版本号，它会强制触发一次重建。
版本历史：6 = 基础+gbm+序列+辅助+随机 · 7 = 隐藏集隔离 · 8 = 按时间顺序的序列 + 反馈状态 ·
9 = 按切分过滤的历史 · 10 = 诚实的 `fb_policy=train_only`。

| 目录 | 逐行数组 | 说明 |
|---|---|---|
| `runs/_cache/` | `{split}_X.npy`（N×5）、`_u.npy`、`_vid.npy`；`_y.npy` **仅限训练/验证集** | 基础编码字段 |
| `runs/_cache/gbm/` | `{split}_X.npy`（N×22）、`_y.npy`、`_u.npy` | LightGBM 特征 |
| `runs/_cache/seq/` | `{split}_{seq,fb,slen,tgt}.npy` | 按时间顺序的历史 + 反馈状态 |
| `runs/_cache/aux/` | `train_aux.npy`、`train_vid.npy` | **仅限训练切分** |
| `runs/_cache/rand/` | `rand_{X,y,u}.npy` | 随机曝光切分（公开标签） |
| `runs/_holdout/` | `test_y.npy`、`aux/{valid,test}_aux.npy` | **绝不传给任何代码块** |

所有数组都按行对齐，这正是调试子采样以及每一次跨缓存关联操作能够保持正确的原因。
`datced._assert_aux_aligned` 会将 `aux/train_vid.npy` 与基础的 `train_vid.npy` 比对，
一旦出现漂移就拒绝运行。

### 时间序列构建（按时间顺序）

`pipeline/lib/seq_build.py` 从原始日志中重新读取 `time_ms`（复用 `data.load()` 相同的文件顺序与日期过滤逻辑，
随后对照 `{split}_vid.npy` 断言行对齐），按真实的 `(user, time_ms, split, row)` 顺序遍历每一个事件，
并在把某一行追加进历史*之前*，先为它快照当时的历史状态。

三项保证，均在构建期断言，并在 `tests/test_sequence.py` 中独立复核：

1. **任何历史中都不会出现未来事件。** 只要排序后仍残留任何用户内部的时间倒挂，构建过程就会报错终止。
   （此前基于行顺序的构建方式，在 30.83% 的训练集行、20.89% 的验证集行与 31.54% 的测试集行上违反了这一点——
   见 [RESEARCH.md 第10节](RESEARCH.md#10-时间序列的时序性一个被修正的保证)。）
2. **训练／验证集的行不会看到测试窗口内的事件。** 同时按切分索引和时间过滤，让这一点成为结构性的保证。
   这一点很重要，因为在切分边界处 `date` 与 `time_ms` 会出现不一致：有 28 行标注为测试集日期的数据，
   其时间戳却早于最后一行验证集数据。
3. **数组按 `data.load()` 的行顺序写回**，因此每一个同级缓存都保持对齐。

`{split}_fb.npy` 记录每个历史事件的反馈状态（`PAD/SKIP/SHORT/NORMAL/LONG/EXPLICIT/UNKNOWN`）。
在默认的 `fb_policy="train_only"` 策略下，**只有训练窗口内的结果才可以成为特征**；验证窗口与测试窗口内的
事件一律为 `UNKNOWN`。这是唯一能让验证集成为测试集无偏代理的策略，它是在实测到替代方案的失败之后才被选定的
（见 [RESEARCH.md 第11节](RESEARCH.md#11-行为感知历史与-00165-假象)）。

---

## 8. 泄漏／完整性保护

与竞赛规则最相关的风险，是一个模型靠看到不该看到的标签拿到高分。相关保护是分层设计的，
没有任何一层被描述为绝对可靠。

| 层级 | 机制 | 所在位置 |
|---|---|---|
| 数据接口层 | `load_bundle` 从不填充 `y["test"]`；`load_aux` 对验证/测试集抛出 `KeyError` | `agent/datced.py`、`pipeline/lib/aux_build.py` |
| 磁盘隔离 | 隐藏测试集标签与 `is_click` 代理标签存放在 `runs/_holdout/`，是缓存目录的**同级目录**，从不传给任何代码块 | `agent/datced.py` |
| 构建期断言 | 一旦某个隐藏集数组出现在代码块可见的缓存中，运行会拒绝启动 | `datced._assert_aux_aligned` |
| 静态护栏 | `check_imports` 拒绝不被允许的导入、`open`/`eval`/`exec`/`compile`/`__import__`，以及匹配
  `_holdout`、`test_y`、`test_aux`、`valid_aux`、`KuaiRand-Pure`、`log_standard` 的路径字面量 | `agent/executor.py` |
| 运行时警报 | 任何得分高于 `leak_tripwire_primary`（0.70）的节点都会被**隔离**，从搜索树与模型组合中排除 | `agent/orchestrator.py` |
| 序列策略 | `fb_policy="train_only"`（见第7节） | `pipeline/lib/seq_build.py` |

**准确的表述是：** *由智能体编写的代码块，既拿不到隐藏测试集标签，也拿不到当前行结果的代理信息；
所有由标签衍生出的隐藏集数据，都被隔离在代码块可见的数据接口之外，并由一道静态读取护栏与一个合理性警报共同兜底。*
早期文档曾称之为"物理上不可能"，这一表述过于绝对，现已弃用——见
[RESEARCH.md 第9节](RESEARCH.md#9-曾经存在的隐藏集泄漏)。

这道警报的阈值为何设在这个位置：仅凭 `is_click` 对验证集排序就能拿到 **0.7466** 分，
相当于 FM 之上全部理论提升空间的 58.8%。这里真正的进步是以千分之一为单位度量的，
出现这么大的跃升，只可能是一个 bug 或一次泄漏，绝不可能是模型本身的功劳。

`tests/test_leakage.py` 对以上全部内容做了断言（共 23 项检查），包括确认六份恶意代码块源码会被拒绝，
而五份合法的源码依然能通过。

---

## 9. 模型族与已实现的干预手段

| 干预手段 | 思路 | 状态 | 所在位置 |
|---|---|---|---|
| **A** | 损失函数对齐：BCE · BPR · softmax交叉熵 · IPS-BCE | 已实现，`loss_type` 是一个真正生效的旋钮 | `lib/losses.py`、`lib/train_np.py` |
| **B** | 序列建模：DeepFM + Deep Interest Network | 已实现 | `lib/din.py`、`lib/din_blocks/` |
| **C** | 多任务辅助头（点击/点赞/关注/评论/转发） | 已实现，仅支持 `mtl_arch="shared"` | `lib/din.py`、`lib/aux_build.py` |
| **D** | 模型族切换：LightGBM LambdaRank | 已实现，**且可通过** `gbm_*` **侧车旋钮调参** | `lib/gbm.py`、`lib/lgbm_blocks/` |
| **E** | 曝光机制：随机曝光切分 + 反热门度加权 | 已实现；随机曝光切分仅覆盖 FM 模型族 | `lib/rand_build.py`、`train_np._ips_weights` |
| **F** | 模型组合：排名空间混合 | 已实现，经 K 折交叉验证 | `agent/portfolio.py` |
| — | 行为感知历史（`use_fb`） | 已实现，**默认关闭**——实测为负面效果 | `lib/seq_build.py`、`lib/din.py` |

两处在科学表述上很重要的命名修正：`train_np._ips_weights` 计算的是
`w ∝ 1/√freq(item)`，这是**反热门度加权，而不是逆倾向得分加权**；随机曝光切分应被理解为
**第二条鲁棒性验证面**，而非竞赛的正式评分目标。

未实现：`mtl_arch ∈ {mmoe, ple}`（会在训练启动前就被 `blockspec` 拒绝）、一个原生的 numpy
LambdaRank 损失函数（`Cfg.lambdarank` 已保留字段名，但目前由 LightGBM 提供 LambdaRank 能力），
以及 DIN／LightGBM 的随机曝光切分（需要各自的配套随机缓存）。

---

## 10. 智能体架构

编排器是确定性的**策略**；大语言模型扮演的角色是**算子**。

| 角色 | 要回答的问题 | 提示词内容 |
|---|---|---|
| **Proposer（提议者）** | 接下来该攻克哪个科学问题？ | 阶段/预算、按证据分级的研究状态、旋钮生效映射表、按干预手段分类的表格、平台期信号、重新提案的反馈 |
| **Coder（编码者）** | 如何在一个代码块内实现它？ | 目标代码块当前的源码、导入白名单、"如实拒绝而非编造"的指令 |
| **Reflector（反思者）** | 执行为什么失败？该如何恢复？ | 失败类别 + 报错堆栈尾部 |

`Hypothesis` 要求 `problem_identified` **必须最先填写**——智能体必须先说明瓶颈，才能提出改动。
所有大语言模型的输出都受 Pydantic 模式约束（`agent/llm/schemas.py`），因此解析从不会失败；
`MockDriver` 用于离线测试时回放预先写好的动作。

### 搜索策略

带 ε 探索阀的最佳优先搜索（`agent/tree.py::select`）。刻意**不**采用蒙特卡洛树搜索（MCTS）：
每个节点都对应一次真实训练，按官方收敛规则，一次运行合理情况下在约 4–6 次实验后就会结束——
样本量太少，不足以支撑滚动模拟与回溯得出可靠的价值估计。

在收敛前出现研究平台期时，`prefer_diverse=True` 会让探索优先扩展*最不相关*的可行节点，而不是随机选择。
平台期升级改变的是**提出什么样的提案**；它绝不能推迟收敛的判定（见第16节）。

### 单次迭代流程

```
选择父节点 → Proposer → 校验配置（第11节） → [Coder → check_imports] → 物化节点
  → 基于内容签名去重（第12节） → [torch 类型的调试准入门] → 训练（+ 测试集推理）
  → 评估 → 与对照组做配对自助法比较（第13节） → 模型组合价值评估（第15节）
  → 判定采纳状态 → 判定是否具有研究信息量 → 写入记忆、事件流、best_series
```

一个未通过校验、与更早节点重复、或被如实拒绝的提案，会被**纠正后重新提出**
（最多 `max_reproposals` 次），既不会触发训练，也不会推进实验计数（见第12、16节）。

---

## 11. 配置有效性校验

`agent/blockspec.py` 针对每个代码块集合，声明了哪些 `Cfg` 字段与 `cfg_ext` 键**能被证明真正影响到某条执行路径**，
以及它们各自允许的取值范围。

```
fm   : batch epochs grad_clip group_filter ips k l2 loss_type lr neg_ratio patience seed tau
din  : aux_tasks aux_weights batch epochs fb_dropout k l2 loss_type lr mtl_arch neg_ratio
       patience seed use_fb
lgbm : seed gbm_learning_rate gbm_num_boost_round gbm_num_leaves gbm_min_data_in_leaf
       gbm_feature_fraction gbm_lambda_l2 gbm_bagging_fraction
```

`validate_delta` 会对一次提案改动中的每一个键做分类：

| 类别 | 含义 | 后果 |
|---|---|---|
| `effective`（生效） | 被真正读取、取值合法、且与当前值不同 | 写入 `cfg.json` |
| `ineffective`（无效） | 被真正读取，但取值与当前一致 | 丢弃，并反馈 |
| `not_honoured`（未生效） | 该代码块集合根本不读取这个字段 | 丢弃，反馈给 Proposer |
| `invalid`（非法） | 未知的键、超出取值范围、或是一个受托管的字段 | 丢弃，反馈 |

一次生效集合为空、且不含任何代码块改写或整族采纳的改动，就是一次**结构性空操作**：
可以被证明根本不会改变执行结果，因此它从不会被训练，也从不会进入科学记录。

`agent/provenance.py` 为每个节点记录：预期的改动、实际生效的改动、每一个被拒绝的键及其原因、
生效配置的哈希、每个代码块的源码哈希、缓存版本、代码状态哈希与随机种子——因此*预期的干预*与
*实际执行的干预*总是可以被相互比对。当实际执行的实验范围比提案更窄时，`intervention_matched` 会是 `False`。

`tests/test_blockspec.py` 会在一份子采样缓存上于运行时校验这些声明：一个生效的旋钮必须改变预测结果，
一个未生效的旋钮则必须让预测结果**逐位完全相同**。

---

## 12. 去重与空操作处理

**节点身份基于内容判定。** `signature = sha256(cfg + cfg_ext + 全部六个代码块的源码)`。
更早的版本是对*相对父节点的统一 diff*做哈希，导致两个配置相同、代码块源码也相同的节点，
如果是从不同父节点到达的，会得到不同的签名——一次对早期节点的重跑，就是这样被误记为一项新的科学发现的。

三种事后分类，彼此严格区分：

| 类别 | 判定方式 | 是否算作证据？ |
|---|---|---|
| `STRUCTURAL_NOOP`（结构性空操作） | 执行前就已判定：没有任何生效的改动 | **否** |
| `EXACT_NOOP`（完全空操作） | 预测结果与父节点逐位相同 | **否** |
| `NEAR_NOOP`（近似空操作） | 排名相关性 > 0.9999，但并非完全相同 | **是**——这是"影响可忽略不计"的合法证据 |

"高度相似"被刻意*不*当作"没有真正的干预"的证明：同一个 DIN 配置训练两次，
相关性也只有 0.926，说明一个随机性模型族确实可能产出看起来相似、但实际上确有不同的模型。

---

## 13. 统计决策机制

存在三种彼此独立、绝不混为一谈的方差：

| 来源 | 量级 | 能否靠重新训练降低？ |
|---|---|---|
| 训练随机性——fm、lgbm | 固定随机种子下为 **0.00000** | 不适用 |
| 训练随机性——din（torch） | σ ≈ 0.00025 | 可以 |
| **验证样本噪声**（配对用户自助法） | **σ ≈ 0.0009** | **不能** |

`agent/stats.py` 提供了一个 `Evaluator`，复现了冻结评估器的语义
（带平均秩次修正的 Mann-Whitney U 检验、仅在有区分度的用户上计算正加权 GAUC、
带稳定降序排序的 nDCG@5），与冻结评估器的一致性误差**小于 1e-5**，
已在 `tests/test_stats.py` 中于合成数据与真实节点预测结果上分别验证。

`paired_bootstrap` 对**用户**做有放回重采样，并在同一批重采样样本上对两个模型重新打分，
从而抵消两个模型在用户维度上的强相关性。它会输出 Δ主指标、ΔGAUC、ΔnDCG、自助法标准误、
95% 置信区间与 `P(Δ>0)`——每次比较约耗时 2 秒，且**无需重新训练**。

证据分类基于 `P(Δ>0)` 划定：`confirmed`（已确认）≥ 0.90 · `promising`（有潜力）0.60–0.90 ·
`inconclusive`（不确定） · `rejected`（已拒绝）≤ 0.10。

多随机种子的**重新训练**（`agent/reeval.py`）专门用于随机性模型族，因为在那种场景下它能度量出
自助法无法度量的信息。对确定性完全固定的 FM 或 LightGBM，它从不会被触发。在收尾定稿阶段，
它会为一个随机性的最终候选模型报告训练方差，**但不会**用它替换该模型（见第18节）。

---

## 14. 研究记忆与跨运行台账

`agent/memory.py` 负责写出只追加的 `run_log.jsonl`，并综合出 Proposer 所看到的状态。

**科学证据与树状态是彼此独立的。** 一个节点携带 `status`（树／采纳记账）、`evidence`
（自助法给出的结论）与 `noop_class`（是否确实发生了某种干预）三个独立字段。
曾经把这几者混为一谈，一度导致系统告诉智能体自己最好的模型"REJECTED——不要重复"。
现在的分类桶为：`confirmed`（已确认）、`promising`（有潜力）、`inconclusive`（不确定）、
`rejected`（已拒绝）、`no_effect`（无影响）、`unsupported_capability`（能力范围之外），
再加上一个明确的 `diverse_portfolio_candidates`（多样化组合候选）列表，以及单独呈现的冠军模型。

`inconclusive`（不确定）的措辞刻意让 Proposer 明白：单纯重复同一个实验解决不了问题——
这个效应的大小低于当前验证集能够分辨的范围，必须改变机制或改变实验设计本身。

`agent/ledger.py` 会把每一次已执行的实验持久化到 `runs/_ledger.jsonl` 中，以一个**臂**
（`family|loss|aux`，刻意不含随机种子，因为随机种子是重复而非不同的臂）为键。汇总操作受
`compatible()` 把关，要求 `cache_version` **与** `code_state` 哈希同时一致。
这道护栏是承重结构：正是它，使一个此前跨越了一次缓存变更所得出的跨运行结论被判定为无效
（见 [RESEARCH.md 第8节](RESEARCH.md#8-辅助任务调查)）。`ResearchInsight` 记录基于台账、按规则生成，
不需要额外调用大语言模型。

---

## 15. 模型组合／集成机制

各个基础学习器的分数处于互不可比的量纲上，而这个指标只关心用户内部的相对顺序，
因此混合发生在**排名空间**中：对每个用户，`r_i = rank_u(s_i)/(|I_u|−1) ∈ [0,1]`——
这是单调的（不会让单个模型变差），也是与量纲无关的。

**价值评估**（成本低廉，基于已保存的 `val_scores.npy`，从不触发训练）：`rank_corr_to_best`
（与最佳模型的排名相关性）、`pair_blend_gain`（成对混合增益），以及 `emc`
（留一法对整体混合池的边际贡献）。这些指标会被呈现在运行日志、研究记忆、Proposer 的上下文，
以及控制台中——因为仅凭独立分数，会把那些真正撑起模型组合价值的模型给淘汰掉。

**组装**是对*整套流程*做用户级的 **K 折交叉验证**：

```
A(S) = 去重（排名相关性 > 0.999） → 贪心前向成员选择 → 权重网格搜索，
       全部只基于用户集合 S 计算

对每一折 k：  members_k, weights_k = A(除第 k 折之外的全部验证集)      # 第 k 折从不参与计算
             score_k               = 在第 k 折上的主指标
诚实估计 = mean(score_k) ± sd/√K
最终制品 = A(全部验证集)      # 用于生成测试集的混合结果；不作为估计值被报告
```

四种数据角色从不混用：选择成员／调整权重／诚实报告／为提交结果重新拟合。
只有第三者可以被引用为无偏估计。正则化方式：粗粒度的权重网格（步长 0.25），最多 4 个成员。

---

## 16. 收敛与基准预算语义

**三个停止相关的概念，严格区分。**

| 概念 | 规则 | 是否可以结束一次健康的运行？ |
|---|---|---|
| **官方收敛** | 对已执行实验的历史最佳分数序列应用 `eps = 0.002, N = 3`；外加 `max_iter = 50` 硬性上限与 6 小时总耗时兜底 | **可以——也只有它应该** |
| **内部研究记账** | `research_stall`（研究停滞）、平台期升级 | 不可以。它改变的是*提出什么样的提案* |
| **存活性护栏** | `proposal_guard_limit`——Proposer 无法产出任何可执行的内容 | 仅在病态情况下才会触发；报告为 `proposal_guard`，绝不会被算作收敛 |

`OFFICIAL_EPS` 与 `OFFICIAL_N` 在导入时从 `baseline_scores.json → convergence_rule` 中读取，
因此代码不可能与主办方给出的原始文件产生偏差。本代码库此前曾使用 `N = 6`，这是朝*放宽*方向的偏差
（N=6 需要连续 7 个没有提升的已评分节点，N=3 只需要 4 个）；`docs/EN/PROBLEM_STATEMENT.pdf`
只是一份纯图片扫描件，无法用来仲裁，因此以 JSON 文件为准。**N = 3。**

**算力被刻意设计为不是约束性条件。** 按官方规则，一次运行大约在 4–6 次实验后就会收敛。
目标*不是*把配额用完——而是让每一次收敛前的实验都货真价实。这里刻意**没有**设置最少实验次数的规则，
也没有出于研究需要而延长运行的机制。

### 计数方式

| 计数器 | 计数对象 | 与什么比较 |
|---|---|---|
| `experiments_executed`（已执行实验数） | 完成训练并产出指标的节点 | 与 `max_iter` 比较；驱动 `best_series` → 收敛判定 |
| `proposal_attempts`（提案尝试数） | 每一次 Proposer 调用，包括被拒绝的 | 不与任何值比较——仅用于可观测性 |
| `wall_clock_used`（已用耗时） | 自运行开始以来的秒数 | 与 6 小时兜底上限比较 |

一个从未训练出模型的提案，不算作一次基准迭代。两个计数器分别可观测，
因此相关论断是可审计的，而不是凭空断言的。`adopt_eps` 只影响搜索树形状，**不是**一条停止规则。

---

## 17. 故障恢复

`executor.run_node` 把每个节点作为一个带超时限制、强制 UTF-8 输入输出的独立子进程启动，
并把结果归类为一个指标字典，或一个带类型的 `Failure(kind ∈ {code, timeout, numerical})`。

| 故障类型 | 恢复方式 |
|---|---|
| `code`（代码错误） | Reflector 提供一个修正后的代码块；重新经过 `check_imports` 把关；重新运行 → 否则放弃 |
| `timeout`（超时） | `degrade`（通过配置改动减小 epoch 数／历史长度）→ 否则放弃 |
| `numerical`（数值问题） | `degrade`（梯度裁剪、降低学习率）→ 否则放弃 |

**优先调试的准入门**（仅限 torch 类模型族）：`pipeline/debug_cache.py` 会为*每一个*逐行缓存数组构建一份行对齐的子采样，
节点先用有限的 epoch 数、通过同一个冻结的运行器在这份子采样上运行一遍。一次崩溃只花几秒钟，而不是一整次训练的时间。
成本低廉的 FM 节点会跳过这一步。

每一次迭代都被包裹保护，因此一步失误不可能拖垮整次运行；即便每一条分支都失败了，
`finalize` 仍然会写出目前为止经过校验的最佳提交结果。已通过 `python -m agent.run --faults`
验证：该命令会注入一次崩溃，验证修复过程，并最终完成一份有效的提交。

---

## 18. 提交与收尾定稿

**模型身份得到保持。** 每次节点运行都会传入 `extra_split="test"`，因此 `test_scores.npy`
是由产出 `val_scores.npy` 的同一次训练写出的，收尾阶段没有重新训练。这一点很重要，
因为重新训练一个 DIN 会产出一个真正意义上不同的模型（同一配置两次运行之间排名相关性仅为 0.926），
这会打破 torch 类模型族的"最佳存档点"不变量。

`run_node` 是冻结的，每次调用只能推理出一个额外切分，因此第二个切分（`rand`，随机曝光切分）
是由推理代码块自己通过 `pipeline/lib/extra_infer.py` 写出的。

收尾阶段依次执行：构建模型组合成员 → 计算价值评估 → 运行 K 折交叉验证组装（第15节）→
混合已保存的测试集数组 → 写出 `best/submission_test.csv` → 用冻结的 `submit.py --check`
校验它 → 写出 `resource_report.json` 与 `results.md` → 追加到跨运行台账 → 可选地保存冠军模型。

---

## 19. 研究控制台架构

这是一套展示基础设施。**它渲染的是智能体实际产出的记录；它从不模拟执行过程，
也从不自行计算任何指标，它不是一项模型性能改进。**

| 组件 | 作用 |
|---|---|
| `agent/events.py` | 只追加写入的 `runs/<id>/events.jsonl`；共 17 种事件类型（`RUN_START OBSERVE HYPOTHESIZE PLAN CODE GUARD DEBUG TRAIN EVALUATE COMPARE REFLECT RECOVER ENSEMBLE DECIDE CONVERGENCE FINALIZE RUN_END`），逐行刷新，让一个实时界面可以持续追踪它 |
| `agent/console_server.py` | 基于标准库的 `http.server`；提供 `/api/runs`、`/api/events?run&since`、`/api/log`、`/api/report`。只读 |
| `dashboard/research-console.html` | 控制台界面：活动流、按证据分级的研究状态、带 EMC 的模型组合表、实验与完整性检查表 |

`run_log.jsonl` 保持其原有的按节点记录的结构；`events.jsonl` 是**增量新增**的，
因此更早的 `dashboard/hypothesis-ledger.html` 仍然可以正常使用。

两种模式共享同一套组件：**回放**（确定性的，来自一次已完成的运行——不需要 API key、GPU 或网络）
与**实时**（轮询 `since=<seq>` 获取新事件）。回放可以压缩时间间隔；但绝不会改变顺序、指标、
假设、决策或结果。

状态栏刻意把两组容易被混淆的概念分开呈现：**官方**收敛状态 与 内部研究记账；
以及**已执行的实验** 与 提案尝试次数。

---

## 20. 配置参考

**`Cfg`**（`pipeline/contracts.py`，冻结，按节点独立）：`seed, use_seq, L, use_vstat, use_aux,
model_type, k, loss_type, alpha, tau, neg_ratio, lambdarank, group_filter, lr, l2, epochs, batch,
patience, grad_clip, aux_tasks, aux_weights, mtl_arch, ips, ensemble_members`。
其中哪些字段真正生效，取决于挂载的是哪个代码块集合——见第11节。
`use_seq, use_vstat, use_aux, lambdarank, ensemble_members, L, alpha` 虽然已声明，但没有任何代码块会读取它们。

**`cfg_ext.json`**（侧车文件）：`use_fb, fb_dropout`（din）· `gbm_learning_rate, gbm_num_boost_round,
gbm_num_leaves, gbm_min_data_in_leaf, gbm_feature_fraction, gbm_lambda_l2, gbm_bagging_fraction`（lgbm）。

**`agent/config.py`**（智能体侧配置，未冻结）：

| 分组 | 字段（默认值） |
|---|---|
| `Config` | `data_dir`、`cache_dir=runs/_cache`、`runs_dir=runs`、`seed=0`、`gpu=auto`、`debug_gate=True`、`debug_train_n=20000`、`debug_other_n=10000`、`debug_epochs=2`、`recheck=True`、`recheck_seeds=(1,2)`、`recheck_top_k=3`、`resume=False`、`champion_dir=runs/_champion`、`ledger_path=runs/_ledger.jsonl`、`use_ledger=True`、`unbiased_eval=False`、`events=True` |
| `Budget`（官方） | `max_iter=50`（硬性上限）、`wall_clock_hours=6.0`（兜底上限）、`per_iter_timeout_s=900`、`eps=0.002`、`N=3`、`adopt_eps=0.001`（仅影响搜索树形状） |
| `Research` | `bootstrap_B=1000`、`adopt_p=0.90`、`promising_p=0.60`、`ens_eps=0.0002`、`diversity_corr=0.90`、`max_reproposals=3`、`proposal_guard_limit=12`、`plateau_after=2`、`explore_p_escalated=0.40`、`cv_folds=5`、`max_members=4`、`weight_step=0.25`、`leak_tripwire_primary=0.70` |
| `Phases` | `breadth_until=12`、`depth_until=40`、`ablation_every=6`（未使用）、`explore_p=0.15` |
| `LLM` | `provider=gemini`、每个角色各自的模型 ID、`temperature=0.4`、`max_retries=5`、`max_llm_usd=0.0`（已声明，**但未强制执行**） |

`Config.load(path)` 会把一个可选的 `agent/config.yaml` 合并到默认配置之上。

---

## 21. 运行与测试

两种解释器：系统自带的 `python` 3.14（CPU 版 torch）与 `cudaenv/` 中的 3.12 + torch 2.6.0+cu124
（**GPU**）。FM 与 LightGBM 是基于 numpy／booster 的确定性实现，与解释器无关；只有 DIN 会有差异，
且在 `cudaenv` 上快得多。所有运行都建议优先使用 `cudaenv/Scripts/python.exe`。

```bash
python -m agent.run --smoke        # M0 准入门：构建缓存，复现 FM（约 0.6015），校验 frozen.lock
python -m agent.run --mock         # 通过 MockDriver 运行完整闭环——无需 API key，无需额度
python -m agent.run --faults       # 鲁棒性测试：注入一次崩溃，验证恢复过程，仍能完成收尾
python -m agent.run                # 实时运行（需要 GEMINI_API_KEY 或 .env.local）；用 --max-iter 设上限
python -m agent.console_server     # 研究控制台，地址为 http://127.0.0.1:8712/
```

针对性测试组（每个都是独立模块，逐项打印 PASS/FAIL）：

```bash
python -m tests.test_stats          # 自助法与冻结评估器的误差 (<1e-5)
python -m tests.test_blockspec      # 旋钮生效契约，含运行时逐位一致性校验
python -m tests.test_leakage        # 隐藏集隔离，恶意代码块被拒绝
python -m tests.test_orchestration  # 收敛语义、去重、空操作分类、"证据≠状态"
python -m tests.test_sequence       # 时间顺序 + 反馈状态泄漏安全性
```

**任何涉及缓存、代码块或冻结边界的改动之后，都应运行 `--smoke`。**

---

## 22. 运行产物与数据结构

`runs/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `run_log.jsonl` | 每个节点一行——研究台账与智能体的记忆 |
| `events.jsonl` | 细粒度的结构化研究事件（见第19节） |
| `resource_report.json` | 基准记账、调参估计 vs 诚实估计、模型组合价值评估、训练方差、停止原因 |
| `results.md` | 简短的、供人阅读的结果表格 |
| `best/submission_test.csv` | 提交结果（通过 `submit.py --check` 校验） |
| `nodes/<id>/` | 代码块快照、`cfg.json`、`cfg_ext.json`、`provenance.json`、`metrics.json`、`val_scores.npy`、`test_scores.npy`、`stdout.log` |

持久化内容：`runs/_cache/`（DataBundle 数据束）、`runs/_holdout/`（标签衍生数据，代码块永不可见）、
`runs/_champion/`（跨运行冠军模型）、`runs/_ledger.jsonl`（跨运行证据）。

**`run_log.jsonl` 的记录字段**：`iter, phase, node_id, parent_id, lever, hypothesis, problem_identified,
config, cfg_ext, code_diff, metrics{GAUC, nDCG@5, primary_valid, primary_unbiased, …},
status, evidence{class, delta_primary, delta_GAUC, delta_nDCG, boot_se, p_gt0, ci_lo, ci_hi,
control_id}, portfolio{rank_corr_to_best, pair_blend_gain, emc, standalone_primary},
provenance{…}, noop_class, informative, events, cost, signature`。
`status ∈ {root, improved, no_gain, abandoned, duplicate, rejected_proposal, quarantined}`。

**`events.jsonl` 的记录字段**：`{seq, ts, type, node_id, phase, summary, data}`。

---

## 23. 扩展规则

1. 永远不要修改冻结文件。 2. 每次改动之后都跑一次 `--smoke`。 3. 所有缓存数组必须保持按行对齐。
4. 任何缓存布局改动都要递增 `CACHE_VERSION`。

**新增一个配置旋钮：** 如果冻结的 `Cfg` 已经有合适的字段，直接使用它，并在相应代码块集合的
`blockspec` 中声明它。否则把它放进 `cfg_ext.json` 侧车文件，并加入该代码块集合的 `ext_honoured`。
**一个未在 `blockspec` 中声明的旋钮，会在训练开始前就被拒绝**——这正是设计的目的。

**新增一个干预手段／模型族：**
1. 在 `pipeline/lib/<name>.py` 中实现该模型。
2. 创建 `pipeline/lib/<name>_blocks/`，遵循这六个函数签名。
3. 如果需要新的缓存数据，新增 `<name>_build.py`，将其接入 `datced.build_or_load`，
   **对照 `{split}_vid.npy` 断言行对齐**，递增 `CACHE_VERSION`，并把它的数组加入
   `debug_cache.PER_ROW`。
4. 在 `agent/blockspec.py` 中新增一个 `BlockSetSpec`，列出它真正读取的旋钮，并声明它是否具有随机性。
5. 在 `tests/mock_moves.py` 中新增一个采纳它的动作，并运行 `--mock`。

**常见陷阱：** 分组与模型族相关逻辑以 `cfg.model_type` 为键，它必须跟随挂载的代码块；
`bundle` 中的数组是 `mmap_mode='r'`（在做大量索引操作前应先用 `np.asarray` 包装）；
给一个 Pydantic 模式新增一个必填字段，会破坏每一个直接调用构造函数的地方，包括 mock 动作。

---

## 23a. 已采纳能力的来源说明

六项能力，均来自对三位队友各自代码归档（`archives/aerin`、`archives/jx`、`archives/jon`）的审阅，
并在本代码库自己的框架规范内重新实现，而非直接照搬。**其中没有任何一项触碰了冻结的运行框架**——
即便是安全护栏本身，也是绕开冻结的运行器工作的。

| 能力 | 思路来源 | 现在的位置 |
|---|---|---|
| 多任务辅助头（干预手段 C） | `aerin/sequence_ranker.py` | `lib/din.py` 的辅助头、`lib/aux_build.py`、`lib/din_blocks/` |
| 实验日志仪表盘（原名"假设台账"） | `jx/hypothesis-ledger.html` | `dashboard/hypothesis-ledger.html`（仍可正常使用；演示场景下已被研究控制台取代，见第19节） |
| 跨运行冠军断点续跑 | `jx/agent/controller.py` | `agent/champion.py`（默认关闭） |
| 多随机种子复评 | `jon/agent/reeval.py` | `agent/reeval.py`——现已限定仅用于随机性模型族（见第13节） |
| 优先调试的采样准入门 | `jon/agent/debug_run.py` | `pipeline/debug_cache.py`、`executor.debug_gate`（见第17节） |
| 测试标签数据护栏 | `jon/agent/data_guard.py` | `datced.load_bundle` + 第8节中描述的分层保护 |

在实际阅读了原始源码之后，我们刻意**没有**采纳的部分：Aerin 的 `IntraUserPairSampler`
（我们的 `train_np._fit_pair` 已经用一个扁平化的负样本池完全向量化了；Aerin 的实现是每个 epoch 对用户做循环）、
Aerin 把 DIN 实现为一套并行的代码块集合（与 `din_blocks` 重复；我们只采纳了其中的多任务思路）、
Aerin 硬编码的线性实验注册表、JX 的整文件重写式控制器（一种单路径改写器，不如基于代码块契约的树搜索）、
以及 Jon 受限的 `action_space` + 贪心线性链（他的智能体只能在一个手写的 FM 上调超参数，且从不分支）。
我们采纳的是 Jon 的*安全机制*，而不是他的循环逻辑。

---

## 24. 已知的工程局限

* **`max_llm_usd` 已声明但从未强制执行。** 目前没有 LLM 花费上限。*（已知问题；文档冻结期间未修复。）*
* **失效的 `Cfg` 字段。** `use_seq, use_vstat, use_aux, lambdarank, ensemble_members, L, alpha`
  没有任何代码块会读取它们。`Cfg.loss_type` 的文件内注释提到了一个 `blend` 取值，但
  `make_loss` 并未实现它——`blockspec` 会在训练启动前拦下它，避免白白浪费一次运行。
* **`Phases.ablation_every` 未被使用。**
* **随机曝光切分目前仅覆盖 FM 模型族。** DIN 与 LightGBM 需要各自的配套随机缓存来支撑其
  `seq`／`gbm` 特征。`unbiased_eval` 默认是 `False`，因为它会给每个节点多加一次推理开销。
* **`AblationRead` 模式已定义并有配套 mock，但目前没有任何角色使用它。**
* **`Cfg` 的侧车文件拆分是一种权宜之计，** 而非一个干净的契约设计。它的存在是因为
  `pipeline/contracts.py` 是冻结的。现在要了解一个节点的完整配置，需要查看两个地方。
* **跨运行汇总在实践中通常是空的。** 兼容性护栏在设计上是严格的，因此任何缓存或代码变更都会使整份台账失效。
  这个判定是对的，但也意味着这套台账机制只有在代码状态保持稳定的情况下才能真正发挥作用。
* **行为感知历史（`use_fb`）已经实现但处于禁用状态**——在诚实策略下实测为负面效果
  （见 [RESEARCH.md 第11节](RESEARCH.md#11-行为感知历史与-00165-假象)）。
* **调试准入门按设计仅限 torch 模型族，** 因此一个有问题的 FM／LightGBM 代码块仍然会耗费一整次运行的成本。
* **`resume: false`（默认值）不会降低可达成的分数上限。** 跨运行冠军机制改变的只是*比较基准*——
  也就是哪个节点会被标记为"提升"——而不会改变任何节点的实际分数。冷启动只是把第一个真正的突破重新标记为一次提升而已。
* **`--mock` 回放的是预先写好脚本的算子动作。** 它"0 次人工干预"这一点对*机制本身*是成立的，
  但其中的研究决策来自 `tests/mock_moves.py`，而不是来自模型。只有**实时**运行才能展示真正自主的研究决策；
  正因如此，[SUBMISSION.md 第8节](SUBMISSION.md#8-结果) 中的头条结果，来自一次实时运行。

---

*[README](../../README.md) · [RESEARCH.md](RESEARCH.md) · [SUBMISSION.md](SUBMISSION.md)*
