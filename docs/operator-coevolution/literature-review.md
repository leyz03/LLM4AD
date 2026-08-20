# 距离、变异退化与信用分配：跨领域调研

面向"LLM 协同进化组合优化启发式算子"的研究备忘 · v2 · 2026-08

> **v2 变更说明（重要）**
>
> v1 把交互项 γ 当成了主题。那是个错误：α+β+γ 分解本来是**解决信用分配的工具**，不是研究对象。
> Stage 0.5 测出 γ ≤ 2%（有功效支撑）之后，v1 的主线顺势滑向"给 G-LNS 做机制归因"——
> 那是一篇批评别人的论文，不解决我们任何一个原始问题。
>
> v2 回到三个原始问题：**距离度量、变异退化为相似、信用分配**，并论证它们是同一个问题的三个面。
> γ 降级为 §3.7 的一个技术细节；G-LNS 机制归因降级为 §3.1 的一个可选副产物。
> 保留 v1 的全部文献材料，重新组织。

---

## 0. 三个问题是一个问题

### 0.1 因果链

```
LLM 变异算子本身是个收缩映射
        │
        ▼
  变异退化为相似 (#2)  ──→  种群有效维度坍缩
        │                        │
        │                        ├──→ 距离度量(#3) 若用句法/嵌入层，看不见坍缩
        │                        │     （代码看起来很不一样，行为上是同一个算子）
        │                        │             │
        │                        │             ▼
        │                        │        多样性维持机制全部失效
        │                        │        （它们都以距离为输入）
        │                        │
        │                        └──→ 信用分配(#1) 近似重复算子瓜分信用
        │                                        → 各自都显得平庸 → 被剪枝或被平均掉
        │                                        → 幸存的是"被选得最多"的那个
        ▼                                        → 正反馈，坍缩加剧
   下一代继续从坍缩的种群里采样
```

### 0.2 统一论断

> **正确的信用分配本身就是一种多样性维持机制。**

这不是我编的。Fréchette 等人用 Shapley 值分析 SAT 竞赛求解器组合时的原话是：

> "报告单机性能会**错误地奖励近似克隆**，同时惩罚那些有小而独特优势的算法；
> 只测对完整 portfolio 的边际贡献又会惩罚强相关的算法组。"
> —— [Using the Shapley Value to Analyze Algorithm Portfolios, AAAI 2016](https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf)

这段话就是 #1 和 #2 的接口。**边际贡献式的记账会自动给近似克隆打低分**（因为把它删掉，portfolio 几乎不损失），而累加式记账（G-LNS 的 `F += σ`）会把克隆的分数加满。

现有 AHD 领域把"多样性维持"和"信用分配"当两件事：前者交给 niching / MAP-Elites，后者交给 ALNS 式打分。**没有人把信用分配当作多样性机制来设计。** 这是主线。

### 0.3 三个问题各自的角色

| # | 问题 | 角色 | 我们已有的证据 |
|---|---|---|---|
| 2 | 变异退化为相似 | **现象** | Stage 0.5：人手设计的 6 个修复算子有效自由度 **1.29 / 6** |
| 3 | 距离度量 | **工具**（另两个都以它为输入） | 尚缺；但指标已实现（参与比） |
| 1 | 信用分配 | **机制**（也是坍缩的成因之一） | Stage 0 E3：85 分位新算子 **94%** 被误杀 |

---

## 第一部分：变异退化为相似（现象）

### 1.1 现象已经被精确测量了

**[Mutation Without Variation: Convergence Dynamics in LLM-Driven Program Evolution](https://arxiv.org/abs/2606.05408)**（2026-06）是这条线上最重要的一篇。它把 LLM 变异当作**程序空间上的动力系统**来研究，脱离优化目标单独分析变异链。核心发现：

- LLM 变异持续收敛到程序空间中受限的**吸引子区域**；
- 结构层面尤其严重：**87% 的变异链中，超过 93% 的变异回到了之前出现过的结构形式**，大部分变化被限制在重复模板内的**终结符替换**；
- 转移结构被**短环和自环**主导；
- 收敛速度随 prompt 措辞与模型选择而变，但现象**在各种条件下都稳健**；
- **经典 GP 的子树变异不表现出可比的收敛** → 这不是进化算法的通病，是 **LLM 变异管线特有的**。

最后一条是关键。它意味着：**你不能靠"这是进化算法的老问题，用 niching 就行"来打发它。**

论文自己的总结值得引用：

> 让 LLM 能做语义感知程序变换的那些能力，同时也带来了朝结构同质化的系统性偏置。

### 1.2 上游成因：typicality bias → mode collapse

**[Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity](https://arxiv.org/abs/2510.01171)** 给出了更上游的解释，而且不是"算法局限"这种含糊说法，是**数据层面的成因**：

> 偏好数据里存在 **typicality bias**——标注者系统性地偏好熟悉、易读、可预测的文本（这是认知心理学的既有结论）。
> RLHF 把这个偏置放大成 mode collapse。

它同时给了一个几乎零成本的缓解手段：**让模型"口头化"一个分布**（"生成 5 个方案及其对应概率"）而不是直接生成一个，创意写作任务上多样性提升 1.6–2.1 倍。

→ **直接可迁移**：与其让 LLM 一次生成一个算子，不如让它一次生成 k 个候选并自报概率，再从中按多样性挑。这是 prompt 层的一行改动，应该作为基线实验做掉。

其它相关证据：
- [NoveltyBench](https://arxiv.org/html/2504.05228v2)：评估 LLM 产出的"类人多样性"
- [Annotations Mitigate Post-Training Mode Collapse](https://arxiv.org/pdf/2605.09995)
- [创意评测的批判性分析](https://arxiv.org/pdf/2508.05470)：基于嵌入的 Semantic Novelty Score **无法**跨领域稳定区分新颖性 ← 与 §2.1 的结论互相印证

### 1.3 经典 EC 的答案：niching 家族

在 LLM 之前，EC 已经积累了完整的多样性维持工具箱（[多模态优化 GA 综述](https://arxiv.org/pdf/1508.05342)）：

| 方法 | 机制 | 迁移到算子演化的难点 |
|---|---|---|
| **Fitness sharing**（Goldberg & Richardson） | 用共享适应度替代绝对适应度，相似个体互相压低分数 | 需要 σ_share 半径 → **需要距离** |
| **Clearing** | 每个小生境只保留若干优胜者，其余清零 | 同上 |
| **Deterministic / probabilistic crowding** | 子代只与最相似的个体竞争 | 同上 |
| **Restricted tournament selection** | 在最近邻窗口内做锦标赛（[运行时分析](https://arxiv.org/pdf/1803.09766)） | 同上 |
| **Island model / cellular GA** | 拓扑隔离，靠迁移交换 | 不需要距离，但收敛慢 |
| **Speciation** | 按相似度聚类成物种，物种内竞争 | 需要距离 |

**注意一个共同点：除了 island model，全部需要一个距离函数。** 这就是为什么 #3 是 #2 的前置条件——不是我们方便这么讲，是这些方法本身的结构要求。

AHD 领域已有的应用：
- **PartEvo**（[NeurIPS 2025](https://openreview.net/pdf?id=OEawM2coNT)，已集成进 LLM4AD）：feature-assisted niche construction，个体 = 代码 + "thoughts"，用 LLM 的特征投影来构造小生境
- **MEoH**（[AAAI 2025](https://arxiv.org/abs/2409.16867)）：dominance-dissimilarity，搜索空间的代码相异度 + 目标空间的支配关系。论文明确报告 "**EoH 的多样性急剧恶化**，而 MEoH 能维持"——这是同行对现象 #2 的独立确认
- **HSEvo**（[AAAI 2025](https://arxiv.org/abs/2412.14995)）：Shannon-Wiener 多样性指数（SWDI）与累积多样性指数（CDI）+ harmony search
- **AlphaEvolve / OpenEvolve**：MAP-Elites + island，网格维度为"代码长度 × 与种群的编辑距离"（[实现分析](https://deepwiki.com/algorithmicsuperintelligence/openevolve/3.1-map-elites-algorithm)）

### 1.4 我们自己的证据：连人手设计的算子都会退化

Stage 0.5 的复核里，我们对 6 个经典修复算子算了行为相关矩阵的**参与比（有效自由度）**：

| 问题 | 因子 | 两两相关均值 | 最大 | 有效自由度 / 名义 |
|---|---|---:|---:|---:|
| CVRP | repair | +0.855 | +0.896 | **1.29 / 6** |
| CVRP | destroy | +0.245 | +0.416 | 4.54 / 6 |
| VRPTW | repair | +0.774 | **+0.946** | **1.49 / 6** |
| VRPTW | destroy | +0.265 | +0.421 | 4.37 / 6 |

这六个修复算子**有各自的名字、各自的文献出处、各自的代码实现**，行为上却等价于一个多一点。`regret_2` 与 `regret_3` 相关 0.946；`greedy_insertion` 字面上就是打乱顺序后调用 `sequential_insertion`。

**这是现象 #2 的一个强化版本**：退化不只发生在 LLM 变异链里，人类文献几十年积累的"不同"算子同样如此。它也顺带给出一个可发表的观察：

> 文献层面的算子多样性（不同名字、不同论文）与行为层面的多样性之间存在巨大落差。

### 1.5 三个干预点 —— 以及为什么第三个是空白

**Mutation Without Variation 的发现有一个直接推论：问题出在变异层，那么选择层的修复是不够的。** fitness sharing 只能在已经生成的候选里做取舍；如果变异算子本身就是收缩映射，候选池里根本没有远处的点可选。

| 层 | 时机 | 手段 | 现状 |
|---|---|---|---|
| **变异层** | 生成时 | Verbalized Sampling；温度/采样策略；"远离档案、靠近父代"双约束 prompt（[语义距离 GP](https://openreview.net/forum?id=DwOaHJJKy9)）；强制结构性编辑；OMNI 式 interestingness gate | LLM4AD 里基本空白 |
| **选择层** | 存活时 | fitness sharing / clearing / RTS / MAP-Elites / DNS / PartEvo / MEoH | **已被占领**，不好切入 |
| **记账层** | 信用分配时 | 边际贡献式记账自动惩罚近似克隆 | **完全空白** ← 主线 |

变异层里最值得抄的一条，来自 GP 语义学派：**"语义上远离已评估程序、但语义上接近其父代"的双重约束变异**。翻译成 prompt 约束就是——"改动要小，但产生的行为必须与档案里所有算子都不同"。这个约束目前没人在 AHD 里用过，而它恰好正面对抗 Mutation Without Variation 描述的"终结符替换 + 结构重复"。

---

## 第二部分：距离（工具）

### 2.1 四层分类学

| 层 | 名称 | 定义在什么上 | 成本 | 与行为差异的相关性 |
|---|---|---|---|---|
| L0 | **句法距离** | 代码本身：AST 编辑距离、token、代码长度 | 极低 | 弱 |
| L1 | **表征距离** | 代码嵌入：CodeBERT / GraphCodeBERT / LLM embedding | 低 | 中等偏弱 |
| L2 | **行为距离** | 算子作用于状态时的输入-输出映射，或搜索中的动态轨迹 | 中 | **强** |
| L3 | **互补性距离** | "对哪些实例/哪些搜索阶段有用"的足迹向量 | 高 | 定义上最强 |

**四个独立领域都得出"语法近 ≠ 行为近"**：

- **遗传编程**：整个语义学派的动机就是这个（[语义距离方法综述](https://arxiv.org/pdf/2009.12401)）
- **神经进化 / QD**：behaviour characterization 把个体映射到行为向量，novelty = 到档案 k 近邻的平均距离（[QD 综述](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)）
- **强化学习**：bisimulation metric 用 Wasserstein 距离衡量状态行为相似性；策略层面用轨迹分布间的 Wasserstein 距离（[多策略学习](https://arxiv.org/pdf/1802.03976)）
- **集成学习**：Kuncheva & Whitaker 系统比较 10 种多样性统计量，结论是**它们与集成精度的关系比想象中弱得多**（[ML 2003](http://machine-learning.martinsewell.com/ensembles/KunchevaWhitaker2003.pdf)）→ **多样性本身不是目标，"有用的多样性"才是**

**对我们的直接含义**：AHD 领域现在用的多样性度量——HSEvo 的 SWDI/CDI、MEoH 的 code dissimilarity、AlphaEvolve 的编辑距离——**全都在 L0/L1**。如果 §1.4 的现象（代码很不同、行为相同）普遍成立，那么这些方法在测的是**假多样性**。

这是我们最直接的切入点，而且有一个半天就能做完的判决性实验（§2.5）。

### 2.2 关键度量：有效自由度（参与比）

这是我们在复核 Stage 0.5 时构造的指标，也是回答"种群还剩多少有效维度"最直接的答案。

给定 n 个算子在同一批探针状态上的响应矩阵，算其相关矩阵 `C` 的特征值 `λ_i`，则

```
有效自由度 = (Σ λ_i)² / Σ λ_i²        # participation ratio
```

取值在 `[1, n]`：等于 n 表示完全独立，等于 1 表示全部塌成一个方向。

它的优点：
- **单个标量，可跨代追踪**——直接画"有效自由度 vs 代数"曲线，坍缩一目了然
- 不需要预设生态位边界（对比 MAP-Elites 的网格）
- 与 HSEvo 的 SWDI/CDI 是同类物，但**定义在行为响应上而非代码上**
- 已实现于 `experiments/stage05_robustness_audit.py` 的 `redundancy()`

**它是我们全部三个问题的公共仪表盘**：#2 的现象由它测量，#3 的有效性由它验证，#1 的机制好坏由"它是否阻止了曲线下滑"来评判。

### 2.3 各领域的行为描述子方法

| 方法 | 做法 | 迁移价值 |
|---|---|---|
| **AURORA**（[arXiv:2106.05648](https://arxiv.org/pdf/2106.05648)） | 用 VAE/PCA 从轨迹数据**自动学出**行为描述子，QD 阶段与编码器更新交替 | 高：省掉手工设计 BD，描述子随探索自适应细化 |
| **QDHF**（[ICML 2024](https://arxiv.org/abs/2310.12103)） | 从人类相似性判断中**推断**多样性度量 | 中 |
| **QDAIF**（[ICLR 2024](https://arxiv.org/abs/2310.13032)） | 用 LM 同时评估质量与多样性 | 高：让 LLM 回答"这两个破坏算子的破坏逻辑本质不同吗"，比 embedding 余弦可靠，因为它在回答具体问题而非做通用表征 |
| **OMNI**（[arXiv:2306.01711](https://arxiv.org/abs/2306.01711)） | 基础模型作为"有趣性模型" | 高：**"新颖"≠"有趣"**。把移除比例 0.15 改成 0.16 在嵌入空间可能很远但毫无意义。interestingness gate 比距离阈值更适合去重 |
| **Dominated Novelty Search**（[GECCO 2025](https://arxiv.org/pdf/2502.00593)） | 用**动态适应度变换**实现局部竞争，不需要网格或预定义边界 | 高：算子行为空间的维度和范围事先未知，固定网格 MAP-Elites 很难设 |
| **算法足迹 / 实例空间分析**（[Smith-Miles](https://www.researchgate.net/profile/Kate-Smith-Miles/publication/261342864_Measuring_algorithm_footprints_in_instance_space/links/57e1f80708ae1f0b4d93fa6f/Measuring-algorithm-footprints-in-instance-space.pdf)） | 算法在实例空间中表现好的区域 | L3 距离最成熟的形式化；EoH-S 已用类似思路，且证明目标函数**单调超模**（贪心有近似保证） |
| **DPP**（[贪心 MAP 推断](https://arxiv.org/pdf/1703.03389)） | 核矩阵行列式 ≈ 向量张成的体积，同时编码质量与多样性 | `L = diag(q)·K·diag(q)` 后 k-DPP 采样出算子 portfolio。**AHD 领域几乎没人用过** |
| **元启发式行为刻画** | Search Trajectory Networks；[LLM 元启发式行为空间分析](https://arxiv.org/abs/2507.03605) 记录 exploration/exploitation/convergence/stagnation 四类动态指标 | 最直接对口的方法学参考 |
| **Operator Telemetry**（[OPAL](https://arxiv.org/pdf/2512.12809)） | `Operator Precision`（子代严格优于父代的比例）与 `Operator Impact`（成功更新的平均增益） | 两个量正交，天然的 2D 行为描述子 |

代码表征距离（L1）的现状：GraphCodeBERT 用数据流图补充结构信息，优于 CodeBERT（[arXiv:2408.08903](https://arxiv.org/html/2408.08903)），但 Type-IV 语义克隆（功能相同、实现完全不同）仍是硬骨头。**结论：L1 适合做"是不是换皮"的快速过滤器，不适合当核心多样性度量。**

### 2.4 探针语义签名（L2 的具体构造）

从 GP 语义距离最直接的迁移，性价比最高：

固定 K 个探针状态（K ≈ 50–200，覆盖不同规模、结构、搜索阶段），然后

- **破坏算子**：记录残解特征——被移除元素的指示向量、移除数量、空间半径 / 时间跨度 / 路由归属熵、可行性破坏程度、残解与原解的结构距离
- **修复算子**：在一组**标准残解**（由固定参考破坏算子生成）上，记录插入决策序列——插入顺序、贪心程度（所选位置在候选中的排名分布）、后悔值使用痕迹、是否触发重排

四个优点：

1. **不跑完整搜索**就能得到距离，比 L3 便宜两个数量级
2. 是算子的**内在属性**，不受配对伙伴污染
3. 破坏与修复的签名**处在同一个残解空间**里，可直接定义跨种群兼容性距离
4. 可同时用作：MAP-Elites 描述子、参与比的输入、新算子的冷启动先验

### 2.5 判决性实验：现有 AHD 多样性度量是不是在测假多样性

**这个实验半天能做完，零 LLM 成本，而且同时是 #2 的存在性证据和 #3 的必要性论证。**

拿 Stage 0.5 已有的 6 个修复算子（行为相关矩阵已知，是真值），计算：

| 距离 | 层 | 对应现有方法 |
|---|---|---|
| AST 编辑距离 | L0 | AlphaEvolve 的网格维度 |
| 代码嵌入余弦（CodeBERT / LLM embedding） | L1 | MEoH 的 code dissimilarity |
| LLM 直接判断"这两个算子逻辑上本质不同吗" | L1+ | QDAIF 式 |
| 探针语义签名距离 | L2 | 我们提的 |

然后与行为相关矩阵比一致性（Spearman / Mantel 检验）。

**预测**：L0 和 L1 会显示这六个算子"很不一样"（它们引用六篇不同文献、代码结构差异明显），而行为上它们是同一个算子。若成立，结论是：

> 现有 AHD 方法的多样性维持机制，其输入距离与真实行为差异几乎不相关，
> 因此它们**维持的是代码表面的多样性，不是行为的多样性**。

这一条足以支撑一篇独立的短文，且风险极低——无论结果正反都有信息。

---

## 第三部分：信用分配（机制）

### 3.1 现有做法的诊断

G-LNS（[arXiv:2602.08253](https://arxiv.org/abs/2602.08253)）公式 (5)–(8)：选中的算子对应用后得到分层奖励 σ，然后

```
w_i^d ← λ w_i^d + (1−λ)σ        # 自适应权重
F(d_i) += σ,  F(r_j) += σ        # 全局适应度（用于剪枝）
S_ij   += σ                      # 协同矩阵（用于联合交叉）
```

CoEvo-AHD（[arXiv:2606.00718](https://arxiv.org/abs/2606.00718)）同构：`r_R` 给路由算子、`r_B` 给打包算子、`r_RB` 给 pair。

**按与主线的相关性重排后的缺陷清单：**

**(1) 奖励近似克隆 ← 与 #2 的直接接口**
`F` 是与"被选中次数"几乎同义的累加量。两个近似克隆的算子，其中被 roulette 选中更多的那个 F 更高、更不容易被剪枝，于是被选中更多——**正反馈直接驱动坍缩**。而边际贡献式记账会给克隆打低分（删掉它，portfolio 几乎不损失）。这就是 §0.2 的论断在具体方法上的落点。

**(2) 冷启动误杀 ← 我们已有实测**
Stage 0 E3：真实质量在 85 分位的新算子，被朴素累加误杀的比例 **94%**；改均值 44%；方差分解 23%；再叠一个 ρ=0.7 的先验降到 4%。
方差分解取胜的机制不是去偏，而是**跨伙伴维度借统计强度**——新破坏算子的少量样本会按"它碰到的是哪些修复算子"做校正。这是单种群方法结构上给不出的。
G-LNS 用"每代把 F 和 S 清零"打补丁，代价是丢掉全部跨代信息。

**(3) 伙伴混淆**
`F(d_i)` 取决于它碰巧被配到哪些修复算子。一个优秀的破坏算子若早期被反复配到糟糕的修复算子，会被判死刑。这是协同进化的经典难题（Wiegand 对 collaborator selection 的分析）。

**(4) 时序短视 + 尺度未归一化**
四级 σ 只看单步。把解暂时打坏以跳出局部最优的算子会被持续惩罚。且 σ 在搜索早期与后期语义完全不同，跨实例、跨阶段没有归一化。

**(5) 协同矩阵没有减去主效应**（← v1 的主题，现降级）
`S_ij += σ` 意味着 `S_ij ≈ 选中次数 × E[σ|i,j]`，没有减主效应，所以 `argmax S` ≈ `argmax(α_i + β_j)`。Stage 0 实测：`S` 与 α+β 相关 +0.79~+0.87，与真实 γ 只有 +0.23~+0.43；`argmax S` 命中真正最协同那一对的概率只有 3%–10%。
**但 Stage 0.5 已证明经典算子上 γ ≤ 2%（功效 ≥ 80%）**，所以这个缺陷虽然统计上真实，实践影响有限。它可以作为一个副产物写进论文（"该组件的收益并非来自它声称的机制"），但不该是主线。

### 3.2 反事实强度谱系

| 阶 | 方法 | 核心思想 | 代表工作 |
|---|---|---|---|
| 0 | 均分 / 共享奖励 | 每个参与者拿相同的 σ | ALNS, G-LNS, CoEvo-AHD |
| 0.5 | **极值统计** | 用奖励的极值而非均值——罕见的大跳跃比频繁的小改进更重要 | [Extreme Value Based AOS (Fialho)](https://inria.hal.science/inria-00287355v3/document) |
| 1 | 方差分解 | 拆成 μ + 主效应 + 交互 | [fANOVA (Hutter, Hoos, Leyton-Brown, ICML 2014)](https://automl.github.io/fanova/includeme.html) |
| 1.5 | 消融路径 | 从默认到目标配置贪心走一条路径逐步归因 | [Ablation Analysis (Fawcett & Hoos, JoH 2016)](https://www.cs.ubc.ca/labs/algorithms/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf) |
| 2 | 差分奖励 | `D_i = G(z) − G(z_{−i} + c_i)`：换成默认动作，其余不动 | Wolpert & Tumer；[COMA (AAAI 2018)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf) |
| 2.5 | 值分解 | 学可分解的联合值函数 | VDN（线性）, QMIX（单调） |
| **3** | **Shapley** | 所有子集上的平均边际贡献 | [Shapley Q-value](https://arxiv.org/abs/1907.05707)；[**算法组合 Shapley**](https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf)；Data Shapley |

**第 3 阶那一行是主线的落点。** Fréchette 等人的工作不只是"另一个信用分配方法"——它是**唯一一个明确把记账方式与近似克隆问题联系起来**的先例，而且是在算法组合这个与我们同构的场景里。

关于 VDN/QMIX 的一个理论论据：VDN 的可加分解无法表示依赖联合动作协同的值函数，QMIX 的单调性约束也表示不了某些值函数。翻译过来——**"把 pair 的收益可加地拆给两方"在原理上有表达能力上限**。但既然 Stage 0.5 表明 γ ≤ 2%，这个上限在我们的场景里暂时不咬人。留作备注。

### 3.3 时序维度

| 方法 | 思想 | 对我们的意义 |
|---|---|---|
| ALNS 四级打分 | 单步、离散 | 基线 |
| EVB-AOS + Dynamic MAB | 极值统计 + bandit 跟踪非平稳最优 | 廉价升级，一行代码 |
| [RUDDER](http://papers.neurips.cc/paper/9509-rudder-return-decomposition-for-delayed-rewards.pdf) | 用回报预测器的相邻时刻差分把 episodic reward 分解到每步 | **迁移价值最高**：训一个轻量 `V(搜索状态特征)`，第 t 步信用 = `V(s_{t+1}) − V(s_t)`，自动解决短视 |
| [Hindsight Credit Assignment](http://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf) | 后见之明的逆向模型 | 适合长搜索链 |
| [DR-ALNS](https://arxiv.org/pdf/2211.00759)、[GRLOS 图 RL](https://arxiv.org/pdf/2302.14678) | 把算子选择当序贯决策 | 已有工作，作对比基线 |

### 3.4 LLM 特有的线索

- **[Which Agent Causes Task Failures and When?](https://arxiv.org/abs/2505.00212)**（ICML 2025 Spotlight）：Who&When 数据集上，最好的自动归因方法识别责任 agent 准确率 **53.5%**，定位决定性错误步骤 **14.2%**，o1/R1 也达不到实用水平。
  → **不要让 LLM 直接当信用分配的裁判。** 正确用法是让它**生成假设**，再用可执行的反事实实验验证。

- **[Exact Is Easier: Credit Assignment for Cooperative LLM Agents](https://arxiv.org/abs/2603.06859)**：LLM agent 的交互历史是可观测文本的确定性函数、无隐状态，所以任何决策点都能被精确还原，从而做直接因果测量而非参数化近似。
  → **这个洞见迁移到 LNS 上比在 LLM agent 上更成立**：LNS 状态完全可序列化（当前解 + RNG 状态 + 温度 + 迭代计数），重放成本是一次算子调用而非一次 LLM 生成，便宜好几个数量级且完全确定。**反事实重放消耗 0 次 LLM 调用，只花 CPU**——在 AHD 里稀缺资源是 LLM sample 不是 CPU，这个方案在正确的维度上花钱。

- **[Who Gets the Reward & Who Gets the Blame?](https://arxiv.org/abs/2511.10687)**：把 Shapley 归因与过程奖励模型统一，产生**局部的、带符号的、信用守恒的**信号。
  → "信用守恒"（所有分配之和 = 系统级收益）是个廉价且有力的正确性断言。G-LNS 的 `F(d)+=σ, F(r)+=σ` 恰恰违反它（总信用是实际收益的 2 倍）。

- **Cook & Tumer, *Learning Aligned Local Evaluations for Better Credit Assignment in Cooperative Coevolution*, GECCO 2024**（[Tumer 组主页](https://web.engr.oregonstate.edu/~ktumer/publications/)）：
  - **Fitness critic + GALE**：学一个局部模型近似单个 agent 的贡献，用 **Global Aligned Local Error** 损失显式最大化局部评估与全局评估的**一致性（alignment）**
  - **Sequential Aristocrat Utility (SeqAU)**：针对"agent 按**固定顺序**行动、共享单一团队奖励"的场景，给出**唯一**能最大化每个 agent 动作可学习性的信号，把 Wolpert & Tumer (2002) 扩展到序贯设定

  destroy → repair 就是"两个 agent 按固定顺序行动、共享单一团队奖励"的教科书实例。SeqAU 给的是**唯一最优**分配信号，不是又一个启发式打分规则。GALE 的 alignment 还给了一个**独立于最终性能**的评价指标——可以直接用来论证"我们的信用信号比 G-LNS 的 σ 均分有更高的 global-local alignment"。

### 3.5 记账层作为多样性机制（主线的技术核心）

把 §0.2 的论断落成可实现的东西。三个具体机制：

**(a) 边际贡献替代绝对贡献**
把 `F(d_i)` 从"累加的 σ"换成"从当前算子池里删掉 d_i 后，池子性能的损失"。近似克隆的边际贡献天然接近零。
实现：截断蒙特卡洛 Shapley，或更便宜的 leave-one-out difference reward（基线算子固定为 `random_removal` / `greedy_insert`，跨实验可比）。

**(b) 共享适应度的记账版本**
Fitness sharing 在**选择层**压低相似个体的分数；把同一思想搬到**记账层**——用行为相似度核 `K` 对信用做去重加权：

```
F_shared(d_i) = F(d_i) / Σ_j sh(dist(d_i, d_j))
```

差别在于：selection-level sharing 需要在每一代重新计算并影响存活；accounting-level sharing 直接改变**记录下来的历史信用**，因而也影响跨代的先验、亲本选择与编辑归因。这是个小改动，但据调研没人做过。

**(c) DPP 记账**
用 `L = diag(q)·K·diag(q)` 把质量与多样性统一在一个核里，算子池的"价值"定义为 `log det(L_S)`，单个算子的信用 = 它对该行列式的边际增量。这个量**在数学上就是"它给种群增加了多少体积"**，克隆的增量为零。
DPP 在 AHD 领域几乎没被用过，而它恰好是"信用即多样性"这个论断最优雅的形式化。

**验证方式**：三种机制各跑一遍，主指标不是最终 best fitness，而是 **§2.2 的有效自由度曲线**——记账方式的改变能否阻止种群有效维度下滑。这是一个直接检验 §0.2 论断的实验设计。

### 3.6 跨代：编辑级信用库

给 LLM 的每次编辑打结构化标签（"引入 regret 项"、"随机移除→相关性移除"、"加入自适应移除比例"），维护血缘树，统计每种编辑类型在所有血缘中的平均 fitness 增量。

对应 Fawcett & Hoos 的 ablation analysis：他们发现两个配置间的性能差异通常可由**极少数几个参数改动**解释（Spear 求解器 473 倍加速的 99.7% 来自单个参数改动）。同样现象很可能在算子编辑上成立——**大部分 LLM 编辑是噪音，少数几种贡献了绝大部分提升**。

**与 #2 的联系**：Mutation Without Variation 说 87% 的变异链里 93% 的变异是结构重复。编辑级记账正好能把"哪些编辑真的改变了行为"和"哪些只是终结符替换"分开——**它既是信用分配机制，也是变异退化的诊断工具**。

### 3.7 技术附录：α+β+γ 分解的定位

方差分解（[fANOVA](https://automl.github.io/fanova/includeme.html) 的思路）在 v1 里被当成核心贡献，v2 降级为**估计 α/β 的正确工具**：

```
Δ_ij = μ + α_i + β_j + γ_ij
```

- **保留的理由**：Stage 0 E3 证明它在冷启动上显著优于均值（误杀 23% vs 44%），机制是跨伙伴维度借统计强度。这个优势与 γ 的大小**无关**。
- **降级的理由**：Stage 0.5 证明 γ ≤ 2%（功效 ≥ 80%），所以 γ 项本身不承载研究价值。
- **IPS 逆倾向加权**：Stage 0 实测无收益（+0.912 vs +0.928），诚实报告，作为一个 ablation 项而非贡献。
- **保留的一个开放问题**：Stage 0.5 测的是**人手设计的通用型**算子。若 §1.4 的"通才不产生交互"成立，则 γ_经典 是 γ_LLM演化 的下界，可能很松。等 Stage 1 环境就绪后，把 G-LNS 报告的演化算子对（ACSR × DAPI）与经典算子混在一起重跑同一套审计，是个便宜的复核。

---

## 第四部分：贡献点（按新主线重排）

| # | 贡献点 | 对应问题 | 新颖性 | 难度 |
|---|---|---|---|---|
| **1** | **记账层多样性机制**：边际贡献 / sharing 记账 / DPP 记账，以有效自由度曲线为主指标 | #1 × #2 | **高**（完全空白） | 中 |
| **2** | **AHD 多样性度量的有效性审计**：证明现有 L0/L1 度量与行为差异不相关 | #3 | 中高 | **低**（半天） |
| **3** | **有效自由度作为坍缩仪表盘** + 探针语义签名作为它的输入 | #3 | 中高 | 低 |
| **4** | **变异层反坍缩**：Verbalized Sampling + "远离档案、靠近父代"双约束 prompt | #2 | 中 | 低 |
| **5** | **SeqAU / GALE 迁移**：destroy→repair 是"固定顺序 + 单一团队奖励"的标准设定；用 global-local alignment 当独立评价指标 | #1 | 高 | 中 |
| **6** | **精确重放反事实**：LNS 可完全序列化 → 真因果 difference reward，零 LLM 成本 | #1 | 高 | 中 |
| **7** | **编辑级信用库**：既是跨代信用分配，也是变异退化的诊断工具 | #1 × #2 | 中高 | 中 |
| 8 | G-LNS 机制归因（J0–J5 消融） | 副产物 | 中 | 低 |

**如果只做一条**：#1 + #3 的组合。#3 提供仪表盘，#1 是被测的干预。论证链条完整：现象（有效自由度下滑）→ 归因（记账方式驱动坍缩）→ 干预（换记账方式）→ 验证（曲线不再下滑）。而且全程不依赖 γ。

**最便宜的先手**：#2。半天，零 LLM 成本，正反都有信息，且它是 #1 和 #3 的共同前提。

---

## 第五部分：实验陷阱清单

1. **零结果必须配功效分析。** Stage 0.5 的教训：没有功效分析，"没测到"不可解释。补上之后才能说"排除 X 以上"。
2. **换统计量口径必须配零分布。** 复核 Stage 0.5 时，标准差口径的 γ 占比 14.6% 看起来像"抬头"，但 H0 下的中位数是 73.7%——噪声统计量的分解本来就被噪声主导。差点报出一个假阳性。
3. **有效自由度诊断应作为门禁。** 用一个有效自由度 1.3 的因子去测任何东西，结论都不可信。
4. **奖励尺度归一化不做，一切白搭。** 按 (实例, 搜索阶段) 分桶做秩归一化。
5. **LLM 不当裁判。** Who&When 的 53.5% / 14.2% 是明确警告。LLM 提假设，实验做验证。
6. **多样性 ≠ 有用的多样性。** Kuncheva 的负面结论。最终判据要落在 L3（实例足迹 / 互补性）。
7. **别在没有 grounded 搜索状态特征时做动态适应。** DyACE（[arXiv:2603.13344](https://arxiv.org/html/2603.13344)）的消融显示，缺少真实搜索轨迹反馈时，动态适应**反而不如静态算法**。
8. **held-out 实例 + racing。** 训练用轮转子集，验证 held-out，测试用不同规模/分布；用 successive halving 早杀。

---

## 参考文献分组索引

**变异退化 / 多样性坍缩（#2）**
- [Mutation Without Variation: Convergence Dynamics in LLM-Driven Program Evolution](https://arxiv.org/abs/2606.05408) ← **本主题最重要的一篇**
- [Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity](https://arxiv.org/abs/2510.01171)
- [NoveltyBench](https://arxiv.org/html/2504.05228v2)｜[Annotations Mitigate Post-Training Mode Collapse](https://arxiv.org/pdf/2605.09995)
- [Genetic Algorithms for multimodal optimization: a review](https://arxiv.org/pdf/1508.05342)（niching 家族综述）
- [Runtime Analysis of Probabilistic Crowding and RTS](https://arxiv.org/pdf/1803.09766)
- [PartEvo (NeurIPS 2025)](https://openreview.net/pdf?id=OEawM2coNT)｜[MEoH (AAAI 2025)](https://arxiv.org/abs/2409.16867)｜[HSEvo (AAAI 2025)](https://arxiv.org/abs/2412.14995)

**距离 / 多样性度量（#3）**
- [Kuncheva & Whitaker, Measures of Diversity in Classifier Ensembles, ML 2003](http://machine-learning.martinsewell.com/ensembles/KunchevaWhitaker2003.pdf)
- [Semantic-based Distance Approaches in GP](https://arxiv.org/pdf/2009.12401)｜[Using semantic distance for diverse and sample efficient GP](https://openreview.net/forum?id=DwOaHJJKy9)
- [QD 综述](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)｜[AURORA](https://arxiv.org/pdf/2106.05648)｜[Dominated Novelty Search (GECCO 2025)](https://arxiv.org/pdf/2502.00593)
- [QDHF (ICML 2024)](https://arxiv.org/abs/2310.12103)｜[QDAIF (ICLR 2024)](https://arxiv.org/abs/2310.13032)｜[OMNI](https://arxiv.org/abs/2306.01711)
- [Smith-Miles, Measuring algorithm footprints in instance space](https://www.researchgate.net/profile/Kate-Smith-Miles/publication/261342864_Measuring_algorithm_footprints_in_instance_space/links/57e1f80708ae1f0b4d93fa6f/Measuring-algorithm-footprints-in-instance-space.pdf)｜[EoH-S (AAAI 2026)](https://arxiv.org/abs/2508.03082)
- [Faster Greedy MAP Inference for DPPs](https://arxiv.org/pdf/1703.03389)
- [Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery](https://arxiv.org/abs/2507.03605)｜[Metaheuristic search behavior characterization 综述](https://link.springer.com/article/10.1007/s11721-025-00254-1)
- [GraphCodeBERT 代码相似性](https://arxiv.org/html/2408.08903)

**信用分配（#1）**
- [Using the Shapley Value to Analyze Algorithm Portfolios (AAAI 2016)](https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf) ← **#1 与 #2 的接口**
- Cook & Tumer, *Learning Aligned Local Evaluations...* (GECCO 2024) — [Tumer 组](https://web.engr.oregonstate.edu/~ktumer/publications/)
- [COMA (AAAI 2018)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf)｜[Shapley Q-value](https://arxiv.org/abs/1907.05707)｜[VDN/QMIX 局限](https://arxiv.org/pdf/2112.04454)
- [RUDDER](http://papers.neurips.cc/paper/9509-rudder-return-decomposition-for-delayed-rewards.pdf)｜[Hindsight Credit Assignment](http://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf)
- [Extreme Value Based AOS (Fialho)](https://inria.hal.science/inria-00287355v3/document)｜[Dynamic MAB for AOS](https://inria.hal.science/inria-00377401/)
- [fANOVA (ICML 2014)](https://automl.github.io/fanova/includeme.html)｜[Ablation Analysis (Fawcett & Hoos, JoH 2016)](https://www.cs.ubc.ca/labs/algorithms/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf)
- [Which Agent Causes Task Failures and When? (ICML 2025)](https://arxiv.org/abs/2505.00212)｜[Exact Is Easier](https://arxiv.org/abs/2603.06859)｜[Who Gets the Reward & Who Gets the Blame?](https://arxiv.org/abs/2511.10687)

**LLM 启发式进化（直接相关工作）**
- [G-LNS](https://arxiv.org/abs/2602.08253)｜[CoEvo-AHD](https://arxiv.org/abs/2606.00718)｜[VRPAgent](https://arxiv.org/pdf/2510.07073)
- [ReEvo (NeurIPS 2024)](https://arxiv.org/abs/2402.01145)｜[DyACE](https://arxiv.org/html/2603.13344)
- [DR-ALNS](https://arxiv.org/pdf/2211.00759)｜[GRLOS 图 RL](https://arxiv.org/pdf/2302.14678)｜[ALNS 算子综述与排名](https://www.sciencedirect.com/science/article/pii/S0377221724003928)

**探索信号**
- [CURIOUS](https://arxiv.org/pdf/1810.06284)｜[Intrinsically Motivated Goal-Conditioned RL 短综述](https://inria.hal.science/hal-03099891v1/document)
