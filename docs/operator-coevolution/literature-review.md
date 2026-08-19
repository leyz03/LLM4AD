# 算子距离与信用分配：跨领域调研与迁移设计

面向“LLM Agent 协同进化组合优化启发式算子”的研究备忘 · 2026-08

---

## 0. 先说结论

**关于领域现状（重要）**：你构想的"LLM 协同进化 destroy-repair 算子对"已经有两篇非常接近的工作落地了：

- **G-LNS**（[arXiv:2602.08253](https://arxiv.org/abs/2602.08253)）：LLM 协同进化紧耦合的 destroy-repair 算子对，引入"synergy matrix"捕捉耦合，用 Synergistic Joint Crossover 把高协同对作为整体演化。
- **CoEvo-AHD**（[arXiv:2606.00718](https://arxiv.org/abs/2606.00718)）：双组件耦合问题（TTP/TPP）上的协同进化 AHD，组件级 Thompson sampling 选算子，记录 pair-level collaboration score。

好消息是：**这两篇的信用分配都极其朴素**——同一个标量奖励 σ 被同时加到 destroy、repair 和 pair 三个账户上（G-LNS 公式 (7)(8)：`F(d_i) += σ, F(r_j) += σ, S_ij += σ`）。这里有至少四个可证明的统计缺陷（§2.1），修好它们本身就是一篇论文的体量。你的两个问题正好命中当前方法的薄弱环节。

**关于距离**：不要用单一距离。至少要分层，且**不同用途必须用不同层**：防止重复生成用句法/嵌入距离，维持生态位用行为距离，选择组合与配对用互补性距离。跨领域最一致的负面结论是——**句法/表征距离与性能差异的相关性很弱**（GP 语义学派、集成学习 Kuncheva、LLM 创意评测的 SNS 都独立得出这个结论）。

**关于信用**：所有方法可以放在一条"反事实强度"的谱系上，从 0 阶（平均分配）到 3 阶（Shapley）。而 LNS 有一个别的领域羡慕不来的性质：**运行可精确重放**（固定种子 + 固定初始解 + 确定性接受准则），所以你能做**真·反事实**而不是估计的反事实。这是最强的一个切入点。

---

## 第一部分：距离与新颖性

### 1.1 一个分类学：距离的四个层次

| 层 | 名称 | 定义在什么上 | 计算成本 | 与性能的相关性 |
|---|---|---|---|---|
| L0 | **句法距离** | 代码本身：AST 编辑距离、token、代码长度 | 极低 | 弱 |
| L1 | **语义/嵌入距离** | 代码的表征：CodeBERT/GraphCodeBERT/LLM embedding | 低 | 中等偏弱 |
| L2 | **行为距离** | 算子作用于状态时的输入-输出映射，或它在搜索中的动态轨迹 | 中 | 强 |
| L3 | **互补性距离** | "对哪些实例/哪些搜索阶段有用"的足迹向量 | 高（需评估） | 定义上最强 |

这个分层不是我杜撰的，它在四个独立领域反复出现：

- **遗传编程**：语义（个体在数据集上的输出向量）距离 vs. 语法距离，语义学派的整个动机就是"语法近 ≠ 行为近"。[Semantic-based Distance Approaches in GP](https://arxiv.org/pdf/2009.12401) 综述了这一族；[Using semantic distance for diverse and sample efficient GP](https://openreview.net/forum?id=DwOaHJJKy9) 提出了"语义上远离已评估程序、但语义上接近其父代"的变异算子——**这个"远离档案、靠近父代"的双重约束值得直接抄到算子生成的 prompt 约束里**。
- **神经进化/QD**：behaviour characterization (BC) 是把个体映射到低维行为向量，novelty = 到档案中 k 近邻的平均距离（[QD 综述](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)）。BC 的选择被公认为是 novelty search 成败的关键设计。
- **强化学习**：bisimulation metric 用 Wasserstein 距离衡量状态的行为相似性；策略层面则用轨迹分布之间的 Wasserstein 距离衡量策略差异（[Wasserstein 正则化多策略学习](https://arxiv.org/pdf/1802.03976)）。
- **集成学习**：Kuncheva & Whitaker 的经典研究系统比较了 10 种多样性统计量（Q 统计量、disagreement、double-fault 等），结论是**这些多样性度量与集成精度的关系比想象中弱得多**（[Kuncheva & Whitaker 2003](http://machine-learning.martinsewell.com/ensembles/KunchevaWhitaker2003.pdf)）。这是个重要的警告：**多样性本身不是目标，"有用的多样性"才是**。

LLM 生成内容的创意评测领域也给了同样的负面证据：基于嵌入的 Semantic Novelty Score **无法**跨领域稳定区分新颖性（[创意评测综述](https://arxiv.org/pdf/2511.07448)）。所以指望"LLM embedding 的余弦距离"当算子新颖性的主度量，大概率会失望。

### 1.2 各领域的具体方法与可迁移性

#### (a) QD 的行为描述子：从手工 → 无监督 → 学习

- **手工 BD**：AlphaEvolve 用的是"代码长度 + 与种群的编辑距离"作为 MAP-Elites 的两个维度（[实现分析](https://deepwiki.com/algorithmicsuperintelligence/openevolve/3.1-map-elites-algorithm)）。这是纯 L0，很粗糙但能用。
- **AURORA**（[arXiv:2106.05648](https://arxiv.org/pdf/2106.05648)）：用 VAE/PCA 从轨迹数据里**自动学出**行为描述子，QD 阶段和编码器更新阶段交替进行。→ 迁移：把算子的"探针轨迹"喂给自编码器，自动学出算子的行为潜空间。省掉手工设计 BD 的痛苦，而且描述子会随着种群探索而自适应细化。
- **QDHF**（[arXiv:2310.12103](https://arxiv.org/abs/2310.12103)）：从人类的相似性判断中**推断**多样性度量。
- **QDAIF**（[arXiv:2310.13032](https://arxiv.org/abs/2310.13032)）：用 LM 同时评估质量和多样性，把 QD 扩展到难以形式化度量的定性领域。→ 迁移：让 LLM 判断"这两个 destroy 算子在破坏逻辑上是否本质不同"，作为 L1 距离的一种，比 embedding 余弦更可靠（因为它是在回答一个具体问题而不是做通用表征）。
- **OMNI**（[arXiv:2306.01711](https://arxiv.org/abs/2306.01711)）：用基础模型作为"有趣性模型"筛选任务——因为 FM 从人类语料里内化了"什么算有趣"。→ 迁移：**"新颖"和"有趣"不是一回事**。一个把移除比例从 0.15 改成 0.16 的算子在嵌入空间可能很"远"，但毫无意思。用 LLM 做 interestingness gate 过滤掉平凡变体，是比距离阈值更靠谱的去重手段。
- **Dominated Novelty Search**（[arXiv:2502.00593](https://arxiv.org/pdf/2502.00593)，GECCO 2025）：不用固定网格或档案，通过**动态适应度变换**实现局部竞争，免去预定义边界和难调参数。→ 迁移价值很高：算子的行为空间维度和范围你根本预先不知道，固定网格的 MAP-Elites 很难设。DNS 正好绕开这个问题。

#### (b) 程序/代码表征距离

GraphCodeBERT 用数据流图补充结构信息，在代码相似性任务上优于 CodeBERT（[arXiv:2408.08903](https://arxiv.org/html/2408.08903)）。但 Type-IV 语义克隆（功能相同、实现完全不同）依然是硬骨头。

**结论**：代码嵌入距离适合做"这个新算子是不是已有算子的换皮"的快速过滤器，不适合做核心的多样性度量。

#### (c) 算法足迹 / 实例空间分析

Smith-Miles 的 Instance Space Analysis：把实例投影到 2D 平面，算法的 **footprint** 是它表现好的区域（[Measuring algorithm footprints in instance space](https://www.researchgate.net/profile/Kate-Smith-Miles/publication/261342864_Measuring_algorithm_footprints_in_instance_space/links/57e1f80708ae1f0b4d93fa6f/Measuring-algorithm-footprints-in-instance-space.pdf)）。两个算法的距离 = 足迹的非重叠度。

→ 这是 L3 距离最成熟的形式化。迁移到算子：给每个算子（或算子对）算一个实例空间足迹，**互补 = 足迹互补而非重叠**。EoH-S（[arXiv:2508.03082](https://arxiv.org/abs/2508.03082)，AAAI）已经把这个思路用到了 LLM 启发式集合设计上：目标是生成小规模互补启发式集合，使每个实例至少被集合中一个启发式很好地服务——并且**证明了该目标函数是单调超模的**，因此贪心选择有近似保证。这个超模性结论对你直接可用：如果你要从算子池里选一个 portfolio，贪心 + 超模保证是现成的理论支撑。

#### (d) DPP：把质量和多样性统一在一个核里

DPP 用核矩阵行列式建模子集概率，行列式 ≈ 向量张成的体积，天然同时编码质量（向量模长）和多样性（向量夹角）。→ 迁移：构造 `L = diag(q) · K · diag(q)`，其中 q 是算子质量、K 是行为相似度核，然后 k-DPP 采样出算子 portfolio。这比"先按质量排序再按距离去重"优雅得多，且有现成的贪心 MAP 推断算法（[arXiv:1703.03389](https://arxiv.org/pdf/1703.03389)）。**这是一个几乎没人在 AHD 领域用过的工具。**

#### (e) 元启发式的搜索行为刻画（最直接对口）

- [Survey and analysis of metaheuristic search behavior characterization](https://link.springer.com/article/10.1007/s11721-025-00254-1)：系统整理了如何刻画元启发式的搜索行为。
- **Search Trajectory Networks**：把搜索轨迹编码成图来可视化和比较元启发式行为。
- [Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery](https://arxiv.org/abs/2507.03605)：直接对 LLM 生成的元启发式做行为空间分析，记录 exploration / exploitation / convergence / stagnation 四类动态指标，并结合 Code Evolution Graph（静态代码特征）和 behaviour-based STN。**这篇是你做"算子行为距离"最直接的方法学参考。**
- OPAL（[arXiv:2512.12809](https://arxiv.org/pdf/2512.12809)）提出的 **Operator Telemetry** 概念很好用：`Operator Precision`（子代严格优于父代的比例）衡量与搜索方向的一致性，`Operator Impact`（成功更新的平均适应度增益）区分小的局部改进和大的结构性改动。这两个量正交，构成一个天然的 2D 行为描述子。

### 1.3 迁移设计：三层"算子指纹"

我建议为每个算子维护一个三段式指纹，三段服务于三个不同目的：

#### 指纹 A：探针语义签名（Probe Semantics）—— 静态、廉价、可冷启动

这是从 GP 语义距离最直接的迁移，也是我认为**性价比最高的贡献点**。

固定一组 K 个"探针状态"（K ≈ 50–200，覆盖不同规模、不同结构、不同搜索阶段的解），然后：

- **对破坏算子**：在每个探针解上运行一次，记录输出的残解特征向量——被移除元素的指示向量、移除数量、被移除元素的空间半径 / 时间跨度 / 路由归属熵、可行性破坏程度、残解与原解的结构距离。
- **对修复算子**：准备 K 个**标准残解**（由一组固定的参考破坏算子生成），记录插入决策序列——插入顺序、贪心程度（选择的位置在候选中的排名分布）、后悔值使用痕迹、是否触发重排。

两个算子的距离 = 签名向量的距离（或直接用"移除指示向量"的 Jaccard 距离，这是最纯粹的语义定义）。

**为什么这个好**：
1. **不需要跑完整搜索**就能得到距离，成本比 L3 低两个数量级。
2. 它是算子的**内在属性**，不受配对伙伴污染——这一点对协同进化至关重要（对比：G-LNS 的 F(d_i) 被伙伴严重污染）。
3. 破坏算子的签名和修复算子的签名**处在同一个"残解空间"里**，所以你可以直接定义**跨种群的兼容性距离**：`compat(d, r) = -‖ removal_profile(d) − preferred_input_profile(r) ‖`。这就直接给了配对问题一个零成本的先验，解决新生成算子的冷启动。
4. 它可以做**MAP-Elites 的行为描述子**（取签名的前几个主成分），也可以做 **contextual bandit 的 context 向量**。

#### 指纹 B：动态行为签名（Runtime Behaviour）—— 需要跑，但可以复用评估数据

从 Behaviour Space Analysis 和 Operator Telemetry 迁移：接受率、Operator Precision、Operator Impact、被调用时的搜索阶段分布、导致的解结构变化幅度、诱导的停滞长度、运行时开销。

用途：MAP-Elites 生态位划分、"探索型 vs. 利用型"算子的角色标注（这个标注反过来能指导配对——探索型破坏配鲁棒修复、利用型破坏配精细修复）。

#### 指纹 C：实例足迹（Instance Footprint）—— 昂贵，但定义互补性只能靠它

在验证实例集上的表现向量（归一化后）。距离用 **double-fault 型**度量（Kuncheva 那套里最有用的一个）：两算子在同一实例上**同时失败**的比例。这个比 Q 统计量更贴合"我要的是互补而不是形式上的不同"。

配 EoH-S 的超模贪心或 k-DPP 做 portfolio 选择。

#### 一条实用规则

| 你想干什么 | 用哪个指纹 |
|---|---|
| 过滤掉 LLM 生成的换皮算子 | L0/L1（AST + embedding）+ LLM interestingness gate |
| 维持种群多样性 / MAP-Elites 生态位 | 指纹 A 或 B |
| 冷启动预测配对兼容性 | 指纹 A 的跨种群兼容性距离 + LLM 语义先验 |
| 选择最终的算子 portfolio | 指纹 C + 超模贪心 / k-DPP |

### 1.4 一个能单独成文的验证问题

**如何证明你的距离度量是"有效"的？** 这个问题在整个 AHD 领域几乎没人认真回答过（HSEvo 提出了 Shannon-Wiener 和 Cumulative Diversity Index 来测种群多样性，但没验证这些指数与最终性能的因果关系，[arXiv:2412.14995](https://arxiv.org/abs/2412.14995)）。

可操作的验证协议：
1. **预测性**：距离能否预测性能差异？算 `corr(dist(o_i,o_j), |perf(o_i) − perf(o_j)|)`，以及用距离核做 GP 回归预测算子性能的 R²。哪一层指纹的 R² 最高，哪一层就最有用。
2. **配对预测性**：跨种群兼容性距离能否预测 pair 的交互项 γ_ij？这是直接检验"距离能不能指导配对"。
3. **干预性**：用距离 A 驱动进化 vs. 用距离 B 驱动，比较最终 QD-score 和 best fitness。

这套协议本身可以做成一篇"AHD 中多样性度量的实证研究"，风险低、审稿友好。

---

## 第二部分：信用分配

### 2.1 先诊断现有做法的病灶

G-LNS 的机制（公式 5–8）：选中的 pair `(d_i, r_j)` 应用后得到分层奖励 σ ∈ {σ₁ 新最优, σ₂ 改进当前, σ₃ SA 接受, σ₄ 拒绝}，然后

```
w_i^d ← λ w_i^d + (1−λ)σ      # 自适应权重
F(d_i) += σ,  F(r_j) += σ      # 全局适应度（用于剪枝）
S_ij   += σ                    # 协同矩阵（用于联合交叉）
```

CoEvo-AHD 类似：`r_R` 给路由算子、`r_B` 给打包算子、`r_RB` 给 pair，三者由同一个 raw gain 用固定系数拆分。

这里有**五个可指出的统计缺陷**：

**(1) 频率混淆（frequency confounding）**
`F` 是奖励的**累加和**而非均值。被 roulette wheel 选中更多次的算子累积更多 F，于是排名更高，于是被选中更多——**正反馈失控**。剪枝按 F 排序会系统性淘汰"被选得少但每次都很好"的算子。这是一个可以直接用实验暴露的 bug。

**(2) 伙伴混淆（partner confounding）**
`F(d_i)` 完全取决于它碰巧被配到哪些修复算子。一个优秀的破坏算子如果在早期被反复配到糟糕的修复算子，会被判死刑。这是协同进化的核心难题（Wiegand 等人对 collaborator selection 的分析），G-LNS 完全没有处理。

**(3) 协同矩阵没有减去主效应**
`S_ij += σ` 意味着 S_ij ≈ (选中次数) × E[σ | i,j]。一个好破坏 × 一个好修复即使**毫无交互**，S_ij 也会很高。所以 "Synergistic Joint Crossover 选 S 最大的对" 实际上大概率只是在选 "最好的破坏 + 最好的修复"，而不是在选**真正协同**的对。**这是论文声称的核心创新点的一个逻辑漏洞**，也是你最容易做出对比实验的地方。

正确的量是**交互项**：
```
γ_ij = E[Δ | i,j] − E[Δ | i,·] − E[Δ | ·,j] + E[Δ]
```

**(4) 时序短视 + 尺度未归一化**
四级 σ 是 Ropke & Pisinger 原始 ALNS 的方案，只看单步结果。一个"把解暂时打坏以跳出局部最优"的破坏算子会被持续惩罚。而且 σ 在搜索早期（随便改都有改进）和后期的语义完全不同，跨实例、跨阶段没有归一化。

**(5) 状态重置丢弃血缘信息**
G-LNS 每轮把 F 和 S 清零"以防历史偏见"。这确实解决了新老算子不公平的问题，但代价是**跨代的知识全部丢失**——你无法回答"哪一类 LLM 编辑是有效的"。

### 2.2 跨领域方法谱系：按"反事实强度"排序

| 阶 | 方法 | 核心思想 | 代表工作 | 成本 |
|---|---|---|---|---|
| 0 | 均分 / 共享奖励 | 每个参与者拿相同的 σ | ALNS, G-LNS, CoEvo-AHD | 免费 |
| 0.5 | 极值统计 | 用奖励的**极值**而非均值——罕见的大跳跃比频繁的小改进更重要 | [Extreme Value Based AOS (Fialho et al.)](https://inria.hal.science/inria-00287355v3/document) | 免费 |
| 1 | 方差分解 / 主效应+交互 | 把性能分解成 μ + 主效应 + 交互项 | [fANOVA (Hutter, Hoos, Leyton-Brown, ICML 2014)](https://automl.github.io/fanova/includeme.html) | 极低 |
| 1.5 | 消融路径 | 从默认配置到目标配置贪心走一条路径，每步归因 | [Ablation Analysis (Fawcett & Hoos, JoH 2016)](https://www.cs.ubc.ca/labs/algorithms/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf) | 中 |
| 2 | 差分奖励 / 反事实基线 | `D_i = G(z) − G(z_{−i} + c_i)`：把自己换成默认动作，其余不动 | Wolpert & Tumer; [COMA (Foerster et al., AAAI 2018)](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf) | 中 |
| 2.5 | 值分解 | 学一个可分解的联合值函数（VDN 线性 / QMIX 单调） | VDN, QMIX | 需要学习 |
| 3 | Shapley | 在所有子集上求平均边际贡献 | [Shapley Q-value (AAAI 2020)](https://arxiv.org/abs/1907.05707); [Shapley for algorithm portfolios (Fréchette et al., AAAI 2016)](https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf); Data Shapley | 高 |

**几个关键洞察：**

- **fANOVA 是最被低估的工具**。它做的正是"把性能方差分解到单个超参和超参交互上"——把"超参"换成"算子"，你就得到了 §2.1(3) 里那个正确的 γ_ij。它有成熟实现（`automl/fanova`），基于随机森林代理模型，能在稀疏采样下工作。**直接拿来做算子交互分析，几乎是白捡的。** 局限：假设可加模型，且不告诉你该往哪个方向调。
- **Fialho 的极值信用分配**是 AOS 领域的一个重要但常被忽略的结论：**用滑动窗口内的最佳改进而非平均改进作为算子奖励**，"罕见但巨大的跳跃"和"频繁但微小的改进"同等重要甚至更重要。G-LNS/ALNS 的指数平滑本质上是取均值，系统性低估了那些偶尔制造大突破的高方差算子。这几乎是一行代码的改动，但有明确的文献支撑。
- **Shapley 用于算法组合有直接先例**：Fréchette 等人用 Shapley 值分析 SAT 竞赛求解器的贡献，并明确指出——"报告单机性能会错误地奖励近似克隆并惩罚有小而独特优势的算法；只测对完整 portfolio 的边际贡献又会惩罚强相关的算法组"。这段论述**可以原封不动搬到算子池的评价上**，是你论文 motivation 段落的现成弹药。
- **VDN/QMIX 的表达能力限制反过来是个警示**：VDN 的可加分解无法表示依赖联合动作协同的值函数；QMIX 的单调性约束也表示不了某些值函数。翻译过来就是：**如果 destroy-repair 之间存在强交互，任何"把 pair 的收益可加地拆给两方"的方案在原理上就是不够的**——你必须显式保留交互项 γ，不能只留 α 和 β。这是一个理论层面的论据。

### 2.3 时序维度：从 ALNS 打分到现代 RL

| 方法 | 思想 | 对你的意义 |
|---|---|---|
| ALNS 四级打分 | 单步、离散 | 基线 |
| EVB-AOS + Dynamic MAB | 极值统计 + bandit 跟踪非平稳最优 | 廉价升级 |
| [RUDDER](http://papers.neurips.cc/paper/9509-rudder-return-decomposition-for-delayed-rewards.pdf) | 用回报预测器的**相邻时刻差分**把 episodic reward 分解到每步 | 直接对应"这次算子调用对最终解质量贡献多少" |
| [Hindsight Credit Assignment](http://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf) | 用后见之明的逆向模型分配信用 | 适合长搜索链 |
| Deep RL for ALNS operator selection（[DR-ALNS](https://arxiv.org/pdf/2211.00759)、[GRLOS 图 RL](https://arxiv.org/pdf/2302.14678)） | 把算子选择当序贯决策，state=搜索状态特征 | 已有工作，可作对比基线 |

**RUDDER 的迁移最有价值**：训一个轻量回报预测器 `V(搜索状态特征)`，然后第 t 步算子调用的信用 = `V(s_{t+1}) − V(s_t)`。这自动解决了短视问题——一个把解打坏但提升了未来潜力的破坏算子，`V` 会给它正的差分。而且这个预测器的输入特征（当前解质量、停滞计数、多样性、剩余预算）都是现成的。

### 2.4 LLM 特有的信用分配线索

这条线 2025–2026 发展很快，且和你的场景高度相关：

- **失败归因的困难度基准**：[Which Agent Causes Task Failures and When?](https://arxiv.org/abs/2505.00212)（ICML 2025 Spotlight）构建了 Who&When 数据集，最好的自动归因方法识别"责任 agent"准确率只有 **53.5%**，定位"决定性错误步骤"只有 **14.2%**，o1/R1 这类推理模型也达不到实用水平。
  → **强结论：不要让 LLM 直接当信用分配的裁判。** LLM 的正确用法是生成假设，然后用可执行的反事实实验去验证，而不是当法官。

- **[Exact Is Easier: Credit Assignment for Cooperative LLM Agents](https://arxiv.org/abs/2603.06859)**：核心洞见是——LLM agent 的交互历史是可观测文本的确定性函数、无隐状态，所以**任何决策点都可以被精确还原**，从而能做直接的因果测量而不需要参数化近似。方法 C3：在每个决策点固定完整历史，在冻结的行为策略下采样替代动作，用 leave-one-out 基线算无偏的 per-decision advantage。

  → **这个洞见迁移到 LNS 上比在 LLM agent 上更成立**：LNS 的状态是完全可序列化的（当前解 + RNG 状态 + 温度 + 迭代计数），重放成本是**一次算子调用**而非一次 LLM 生成，便宜好几个数量级，而且完全确定。这是我认为你最值得押注的技术路线，详见 §2.5 层 3。

- **[Who Gets the Reward & Who Gets the Blame?](https://arxiv.org/abs/2511.10687)**：把博弈论归因（Shapley）和过程奖励模型（PRM）统一起来，产生**局部的、带符号的、信用守恒的**信号——成功时用 Shapley 公平分配并细化到 message 级，失败时用**首错定位**产生修复导向的偏好。"信用守恒"（所有分配之和 = 系统级收益）是个很好的设计约束，G-LNS 的 `F(d)+=σ, F(r)+=σ` 恰恰违反了它（总信用是实际收益的 2 倍）。

### 2.4b 一篇被埋没但和你的问题几乎完全同构的工作

**Cook & Tumer, *Learning Aligned Local Evaluations for Better Credit Assignment in Cooperative Coevolution*, GECCO 2024**（[Tumer 组主页](https://web.engr.oregonstate.edu/~ktumer/publications/)）

它的问题设定是：协同进化在**高耦合**任务（fitness 严重依赖特定的联合动作）上，会被信用分配问题卡死。两个贡献：

- **Fitness critic + GALE**：用一个学到的局部模型近似单个 agent 的贡献，把它当作 fitness 函数。问题是局部近似质量下降时会失效，所以提出 **Global Aligned Local Error (GALE)** 损失函数——**显式最大化局部评估与全局评估的一致性（alignment）**。
- **Sequential Aristocrat Utility (SeqAU)**：针对"agent 按**固定顺序**行动、共享单一团队奖励"的场景，给出**唯一**能最大化每个 agent 动作可学习性（learnability）的 per-agent 学习信号，把 Wolpert & Tumer (2002) 的经典框架扩展到序贯设定。

**为什么这是你最该读的一篇**：destroy → repair 就是"两个 agent 按固定顺序行动、共享单一团队奖励"的教科书实例。SeqAU 几乎是为你的场景量身定制的理论结果——它给出的是**唯一最优**的分配信号，而不是又一个启发式打分规则。而 GALE 的 alignment 思想给了你一个评价自己信用分配方案好坏的**独立指标**：局部信用与全局收益的一致性，这可以直接做成实验章节的一个度量（"我们的分解比 G-LNS 的 σ 均分有更高的 global-local alignment"）。

→ 同时这也支持我上次提的**"模块合并"**：当 γ_ij 持续高且稳定，说明该对已共适应，把它合并成单一复合算子，让信用单元与因果单元重合，从根上消除分配问题。

### 2.5 迁移设计：四层信用分配架构

#### 层 1（每次调用 · 零成本）：归一化的时序奖励

替换掉四级 σ：

```
raw_t   = V(s_{t+1}) − V(s_t)                    # RUDDER 式回报差分
norm_t  = rank_normalize(raw_t | 同一 (实例, 搜索阶段) 桶)   # 尺度归一化
r_t     = max over sliding window                # Fialho 极值统计
```

三个改动各自都有独立文献支撑，加起来是一个干净的消融矩阵（2³ = 8 组）。

#### 层 2（每 episode · 在线回归）：去偏的方差分解

拟合
```
Δ_ij = μ + α_i + β_j + γ_ij + ε
```
用带 L2 正则的贝叶斯线性模型在线更新，得到每个系数的**后验均值和方差**。

**关键技术点：逆倾向加权（IPS）**。你的奖励数据是自适应策略（roulette wheel / Thompson sampling）采集的，存在严重的选择偏差——这正是 §2.1(1) 频率混淆的根源。用 `1/P_t(i,j)` 加权每个样本，就能得到无偏的 α、β、γ 估计。**据我调研，AOS / AHD 领域没有人做过这件事**，而这在离线策略评估（off-policy evaluation）里是标准操作。这是一个技术含量高、容易证明有效、且审稿人容易认可的贡献。

用途分工：
- `α_i, β_j` → 个体存活与亲本选择（终于是无偏的了）
- `γ_ij` 的后验均值 → 配对价值，指导 Joint Crossover
- `γ_ij` 的后验方差 → 探索信号，指导下一步评估哪个 pair

#### 层 3（每代 · 精确反事实）：LNS 版的 difference reward

这是最强的一层，也是最有论文价值的部分。

**核心机制**：LNS 运行完全可重放。记录 `(初始解, RNG 种子, 接受准则参数, 算子调用序列)`，就能确定性地复现整条轨迹。于是可以做真正的干预实验：

- **调用级反事实**（对应 C3 的 per-decision advantage）：在第 t 步把算子对 `(d_i, r_j)` 替换成基线对 `(random_removal, greedy_insert)`，其余全部保持不变，重放到结束，比较最终 best。这给出**无偏的 per-call difference reward**。
- **交互检验**：把 `(d_i, r_j)` 替换成 `(d_i, r_baseline)` 和 `(d_baseline, r_j)`，两个单独效应之和 vs. 联合效应之差，就是**因果意义上的交互项**（不是回归拟合出来的，是干预测出来的）。这是对层 2 里 γ 的黄金标准验证。
- **Portfolio 级 Shapley**：在小规模算子子集上做蒙特卡洛 Shapley，衡量算子对整个池子的边际贡献。参考 Fréchette 等人在 SAT portfolio 上的做法。

**预算控制**：不要对所有调用做反事实。用层 2 的后验方差挑选**信息量最大的少数决策点**（比如那些 γ 后验方差最大、或者那些造成了最大 |Δ| 的调用），做 successive halving 式的预算分配。这本身是个 active learning 问题，也是个可以写的技术点。

#### 层 4（跨代 · 血缘）：编辑级信用库

给 LLM 的每次编辑打结构化标签（"引入 regret 项"、"随机移除→相关性移除"、"加入自适应移除比例"、"加入禁忌表"……），维护血缘树，统计每种编辑类型在所有出现过的血缘中的平均 fitness 增量。

这对应 Fawcett & Hoos 的 ablation analysis 思想：他们发现两个配置之间的性能差异通常可以用**极少数几个参数改动**解释（Spear 求解器 473 倍加速的 99.7% 来自单个参数改动）。同样的现象很可能在算子编辑上成立——**大部分 LLM 编辑是噪音，少数几种编辑贡献了绝大部分提升**。把它们识别出来并沉淀成可跨问题复用的先验，是让 agent"越用越强"的关键，也是叙事上最有吸引力的部分。

注意：这一层要求**不能像 G-LNS 那样把统计量清零**。正确做法是保留历史但用贝叶斯先验处理新老不平等（新算子从先验开始，先验由 LLM embedding 和编辑类型的历史效果给出）。

### 2.6 把两个问题闭环：配对选择 = 信用模型驱动的主动实验设计

现在两个问题合并了：

```
        ┌─────────────── 信用模型 (层2: α, β, γ 的后验) ───────────────┐
        │                                                              │
        ↓                                                              │
  配对选择 = argmax over (i,j) of  UCB(α_i + β_j + γ_ij)               │
             或 Thompson sampling                                      │
             或 信息增益最大（选 γ 后验方差下降最快的 pair）             │
        │                                                              │
        ↓                                                              │
   评估该 pair ──→ 层1 奖励 ──→ 层3 反事实（选择性）─────────────────────┘
        │
        ↓
   指纹 A 的跨种群兼容性距离 + LLM 语义先验 ──→ 新算子的 α/γ 先验（冷启动）
```

再叠一个来自发展机器人学的探索信号：**learning progress**（[Oudeyer 那条线](https://inria.hal.science/hal-03099891v1/document)，CURIOUS 把模块选择建模成以绝对学习进度为奖励的非平稳 MAB）。翻译过来：优先评估那些**估计值正在快速变化**的算子对——它们要么在快速变好（值得跟进），要么在快速变差（需要确认淘汰）。这比纯 UCB 更适合非平稳的进化过程，因为算子池本身在变。

---

## 第三部分：可落地的贡献点（按推荐度排序）

| # | 贡献点 | 新颖性 | 实现难度 | 对标 |
|---|---|---|---|---|
| 1 | **精确重放反事实做算子级 difference reward / 交互项因果测量** | 高 | 中 | G-LNS 的 synergy matrix；理论根基 = C3 + Wolpert-Tumer |
| 2 | **IPS 去偏的 α/β/γ 方差分解**，修复频率混淆与主效应污染 | 高 | 低 | fANOVA + off-policy evaluation；AOS 领域空白 |
| 3 | **探针语义签名**：不跑搜索就能算的算子内在指纹，用于冷启动配对先验 | 中高 | 低 | GP 语义距离的迁移 |
| 4 | **编辑级信用库**：跨问题可迁移的 LLM 编辑先验 | 中高 | 中 | ablation analysis 的迁移；ReEvo 的 reflection 缺乏系统记账 |
| 5 | **SeqAU / GALE 迁移**：destroy→repair 是"固定顺序 + 单一团队奖励"的标准设定，直接套用唯一最优信号；用 global-local alignment 当评价指标 | 高 | 中 | Cook & Tumer GECCO 2024 |
| 5b | **高 γ 对的模块化合并**：让信用单元与因果单元重合 | 高（叙事强） | 中高 | 协同进化中的模块涌现 |
| 6 | **k-DPP / 超模贪心做算子 portfolio 选择** | 中 | 低 | EoH-S 的超模性结论可直接引用 |
| 7 | **AHD 中距离度量有效性的实证基准** | 中 | 低 | HSEvo 提了指标但没验证；可独立成文 |

**如果只能做一件事**：做 #1 + #2 的组合。它们直击 G-LNS/CoEvo-AHD 的核心缺陷，有明确的对照实验（直接跑 G-LNS 原方法 vs. 替换信用分配模块），故事清晰（"现有协同进化 AHD 的 synergy matrix 其实测的不是 synergy"），且有跨领域的理论支撑（fANOVA + difference reward + off-policy evaluation 三条腿）。

---

## 第四部分：实验设计与陷阱清单

**必须做的**
- **奖励归一化**：按 (实例, 搜索阶段) 分桶做秩归一化。不做的话所有信用分配都被实例难度主导。
- **Held-out 实例集**：训练用轮转子集，验证用 held-out，测试用不同规模/分布（EoH-S 的实验设置可参考）。否则进化出的是实例特定的 hack。
- **Racing / successive halving**：irace 那套，早杀明显差的 pair，把预算留给有前途的。
- **多次独立运行 + 统计检验**：LLM 生成有巨大方差，单次运行的结论不可信。
- **信用守恒检查**：分配给 destroy、repair、interaction 的信用之和应该等于系统实际收益。这是个廉价的正确性断言。

**必须防的**
- **Relative overgeneralization**：协同进化会收敛到"和谁搭都还行"的平庸算子。用 MAP-Elites / Dominated Novelty Search 强制生态位多样性来对抗。
- **LLM 当裁判**：Who&When 的 53.5% / 14.2% 是明确警告。LLM 提假设，实验做验证。
- **多样性 ≠ 有用的多样性**：Kuncheva 的负面结论。始终用指纹 C（实例足迹）做最终的互补性判据。
- **别在没有 grounded perception 的情况下做动态适应**：DyACE（[arXiv:2603.13344](https://arxiv.org/html/2603.13344)）的消融显示，**没有 grounded 的搜索轨迹特征反馈时，动态适应反而不如静态算法**。这意味着你的 state 特征设计（喂给 bandit / LLM 的搜索状态描述）不是可选项，是成败关键。

**消融矩阵建议**

| 维度 | 取值 |
|---|---|
| 奖励形式 | 四级 σ / RUDDER 差分 / + 极值统计 |
| 归一化 | 无 / 秩归一化 |
| 信用分解 | 均分（G-LNS） / α+β / α+β+γ / +IPS 去偏 |
| 反事实 | 无 / 采样式调用级 difference reward |
| 配对策略 | roulette / Thompson / γ-UCB / +语义先验 |
| 多样性维持 | 无 / MAP-Elites / DNS |

---

## 参考文献分组索引

**协同进化与信用分配（经典）**
- Wolpert & Tumer, difference rewards / COIN
- Wiegand et al., collaborator selection 与 relative overgeneralization
- Cook & Tumer, *Learning Aligned Local Evaluations for Better Credit Assignment in Cooperative Coevolution*, GECCO 2024 — https://web.engr.oregonstate.edu/~ktumer/publications/

**多智能体 RL 信用分配**
- Foerster et al., *Counterfactual Multi-Agent Policy Gradients* (COMA), AAAI 2018 — https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf
- Wang et al., *Shapley Q-value*, AAAI 2020 — https://arxiv.org/abs/1907.05707
- VDN / QMIX 及其单调性限制 — https://arxiv.org/pdf/2112.04454

**时序信用分配**
- Arjona-Medina et al., *RUDDER*, NeurIPS 2019 — http://papers.neurips.cc/paper/9509-rudder-return-decomposition-for-delayed-rewards.pdf
- Harutyunyan et al., *Hindsight Credit Assignment*, NeurIPS 2019 — http://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf

**LLM Agent 信用分配（2025–2026）**
- Zhang et al., *Which Agent Causes Task Failures and When?*, ICML 2025 Spotlight — https://arxiv.org/abs/2505.00212
- *Exact Is Easier: Credit Assignment for Cooperative LLM Agents* — https://arxiv.org/abs/2603.06859
- Yang et al., *Who Gets the Reward & Who Gets the Blame?* — https://arxiv.org/abs/2511.10687
- *Shapley-Coop* — https://arxiv.org/abs/2506.07388

**算法配置与组件归因**
- Hutter, Hoos & Leyton-Brown, *An Efficient Approach for Assessing Hyperparameter Importance* (fANOVA), ICML 2014 — https://automl.github.io/fanova/includeme.html
- Fawcett & Hoos, *Analysing differences between algorithm configurations through ablation*, JoH 2016 — https://www.cs.ubc.ca/labs/algorithms/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf
- Fréchette et al., *Using the Shapley Value to Analyze Algorithm Portfolios*, AAAI 2016 — https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf

**自适应算子选择**
- Fialho et al., *Extreme Value Based Adaptive Operator Selection* — https://inria.hal.science/inria-00287355v3/document
- Fialho et al., *Dynamic Multi-Armed Bandits and Extreme Value-Based Rewards for AOS* — https://inria.hal.science/inria-00377401/
- *Online Control of ALNS using Deep RL* (DR-ALNS) — https://arxiv.org/pdf/2211.00759
- *Graph RL for Operator Selection in ALNS* — https://arxiv.org/pdf/2302.14678
- *A review and ranking of operators in ALNS for VRP* — https://www.sciencedirect.com/science/article/pii/S0377221724003928

**距离 / 多样性 / QD**
- Kuncheva & Whitaker, *Measures of Diversity in Classifier Ensembles*, ML 2003 — http://machine-learning.martinsewell.com/ensembles/KunchevaWhitaker2003.pdf
- Pugh et al., *Quality Diversity: A New Frontier for Evolutionary Computation*, 2016 — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full
- Grillotti & Cully, *AURORA: Unsupervised Behaviour Discovery with QD* — https://arxiv.org/pdf/2106.05648
- Bahlous-Boldi et al., *Dominated Novelty Search*, GECCO 2025 — https://arxiv.org/pdf/2502.00593
- Ding et al., *Quality Diversity through Human Feedback*, ICML 2024 — https://arxiv.org/abs/2310.12103
- Bradley et al., *Quality-Diversity through AI Feedback*, ICLR 2024 — https://arxiv.org/abs/2310.13032
- Zhang et al., *OMNI: Open-endedness via Models of human Notions of Interestingness* — https://arxiv.org/abs/2306.01711
- *Semantic-based Distance Approaches in Genetic Programming* — https://arxiv.org/pdf/2009.12401
- *Using semantic distance for diverse and sample efficient GP* — https://openreview.net/forum?id=DwOaHJJKy9
- Smith-Miles, *Measuring algorithm footprints in instance space* — https://www.researchgate.net/profile/Kate-Smith-Miles/publication/261342864_Measuring_algorithm_footprints_in_instance_space/links/57e1f80708ae1f0b4d93fa6f/Measuring-algorithm-footprints-in-instance-space.pdf
- *Faster Greedy MAP Inference for DPPs* — https://arxiv.org/pdf/1703.03389
- *Survey and analysis of metaheuristic search behavior characterization*, Swarm Intelligence 2025 — https://link.springer.com/article/10.1007/s11721-025-00254-1

**LLM 启发式进化（直接竞品）**
- *G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design* — https://arxiv.org/abs/2602.08253
- *LLM-Driven Co-Evolutionary AHD for Bi-Component Coupled CO* (CoEvo-AHD) — https://arxiv.org/abs/2606.00718
- *VRPAgent: LLM-Driven Discovery of Heuristic Operators for VRP* — https://arxiv.org/pdf/2510.07073
- Ye et al., *ReEvo*, NeurIPS 2024 — https://arxiv.org/abs/2402.01145
- *EoH-S: Evolution of Heuristic Set*, AAAI — https://arxiv.org/abs/2508.03082
- *MEoH: Multi-objective Evolution of Heuristic*, AAAI 2025 — https://arxiv.org/abs/2409.16867
- Dat et al., *HSEvo*, AAAI 2025 — https://arxiv.org/abs/2412.14995
- *Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery* — https://arxiv.org/abs/2507.03605
- *DyACE: Dynamic Algorithm Co-evolution for Online AHD* — https://arxiv.org/html/2603.13344

**探索信号**
- Colas et al., *CURIOUS: Intrinsically Motivated Modular Multi-Goal RL* — https://arxiv.org/pdf/1810.06284
- *Intrinsically Motivated Goal-Conditioned RL: a Short Survey* — https://inria.hal.science/hal-03099891v1/document
