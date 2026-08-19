# 在 LLM4AD 上做算子协同进化：落地实施方案

配套材料：[跨领域调研](./literature-review.md)、[Stage 0 验证脚本](./experiments/stage0_credit_estimator_validation.py)与[实验结果](./experiments/results/)。

---

## 0. Stage 0 已经跑完了，先看结论

我把合成验证写好并跑了。**结果比我原来的假设更好，而且改变了优先级排序。**

### 结果一：协同矩阵 S 测的不是协同（E1）

8×8 算子、3000 次评估、30 次重复，Spearman 相关：

| 采样策略 | S vs 真实 γ（声称测的） | S vs α+β（实际测的） | ANOVA γ̂ vs 真实 γ |
|---|---|---|---|
| uniform | **+0.410** | **+0.873** | **+0.972** |
| roulette（G-LNS） | +0.433 | +0.787 | +0.928 |
| thompson（CoEvo-AHD） | +0.233 | +0.571 | +0.398 |

下游决策——**joint crossover 是否选中真正最协同的那一对**：

| | argmax S（G-LNS 做法） | argmax γ̂（方差分解） |
|---|---|---|
| uniform | 10% | **70%** |
| roulette | 3% | **53%** |

`argmax S` 的命中率 3%–10%，也就是**基本等同于随机猜**（1/64 ≈ 1.6% 是纯随机，10% 只比随机好一点，且这点优势来自主效应而非交互）。

这就是论文的第一张图，而且它是在**已知 ground truth** 上得到的，无可辩驳。

### 结果二：自适应选择会饿死交互项的估计（E2）—— 这是最重要的发现

我原本以为主要病灶是选择偏差，需要 IPS 逆倾向加权来修。**跑下来 IPS 基本没用**（+0.912 vs +0.928，反而略差）。

真正的病灶是**覆盖率**。Thompson sampling 在 600 次预算下只覆盖了 payoff 矩阵的 38%，γ 根本无从估计。

引入一个"定向探索"项（以概率 ε 转为采样最稀疏的配对格）后：

| ε | 平均奖励 | 矩阵覆盖 | γ 恢复 | 最优对命中 |
|---|---|---|---|---|
| 0.00（现有做法） | **+2.767** | 38% | +0.282 | 7% |
| 0.10 | +2.537 | 86% | +0.572 | 30% |
| 0.20 | +2.208 | 99% | +0.686 | 23% |
| **0.35** | +1.810 | 100% | +0.776 | **57%** |
| 0.50 | +1.365 | 100% | +0.817 | 33% |
| 1.00 | +0.012 | 100% | +0.884 | 37% |

**这是一个干净的、可命名的矛盾：利用—估计冲突（exploitation–estimation conflict）。**

> 协同进化 AHD 的自适应算子选择层，为了最大化评估期内的即时收益而集中采样少数配对，
> 恰恰摧毁了它自己估计"协同"所需要的数据。
> 而协同进化的目标是**产出好算子对**，不是最大化评估期奖励——这个奖励是被丢弃的中间产物。
> 所以现有方法在优化一个与目标不一致的量。

这个论断比"他们的统计量有偏"要强得多，也更好写。**建议把它作为论文的核心 claim。**

### 结果三：累加式 F 让新算子几乎必被误杀（E3）

场景：跑满一轮后把 3 个最差算子换成真实质量在 85 分位的新算子，新算子只拿到约 8 次评估机会，然后按分数剪掉最差 3 个。误杀率（40–60 次重复）：

| 估计器 | 误杀率 |
|---|---|
| F 累加（G-LNS 原方案） | **94%** |
| F 均值（频率去偏） | 44% |
| ANOVA α̂ | **23%** |
| ANOVA α̂ + 先验收缩（ρ=0.5） | 9% |
| ANOVA α̂ + 先验收缩（ρ=0.7） | 4% |
| ANOVA α̂ + 先验收缩（ρ=0.9） | 3% |

两点值得注意：

1. **方差分解在冷启动上也赢，而且原因不是去偏，是"跨伙伴维度借统计强度"**——新破坏算子的少量样本会按"它碰到的是哪些修复算子"做校正。这是支持 α+β+γ 分解的**第三个独立论据**（E1 交互识别、E2 覆盖、E3 冷启动）。三个角度都指向同一个改动，论文结构会非常紧。

2. **先验收缩的价值曲线给了你一个可量化的决策依据**：先验与真实质量的相关性达到 ρ≈0.5 时误杀率减半，ρ≥0.7 时几乎归零。所以"探针语义指纹要不要做"这个问题，现在变成"我能不能造出 ρ≥0.5 的先验"——这是可以先小规模试验回答的。

### 优先级调整

| 原计划 | 调整后 |
|---|---|
| ~~IPS 去偏为核心贡献~~ | 降级为一个 ablation 项（诚实报告"在我们的设定下无显著收益"） |
| 方差分解 | **升为核心**，三个独立论据支撑 |
| γ 导向的主动探索 | **升为核心**，E2 是它的存在理由 |
| 探针语义指纹 | 保留，但先做"能否达到 ρ≥0.5"的可行性验证再投入 |

复现命令：

```bash
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment all --selection thompson --n-repeat 40
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment coverage --selection roulette
```

---

## 1. LLM4AD 的架构约束（决定了你怎么改）

我读了 LLM4AD 的文档和结构。关键事实：

```
llm4ad/
├── base/
│   ├── code.py          Function / Program / Converter  —— 演化的个体表示
│   ├── evaluate.py      Evaluation (抽象基类) + SecureEvaluator
│   ├── modify_code.py   基于 ast 的代码改写
│   └── sample.py        LLM 采样接口 + SampleTrimmer
├── method/              EoH, MEoH, FunSearch, ReEvo, LLaMEA, MCTS-AHD,
│                        PartEvo, EoH-S, LHNS, MLES, NSGA2, MOEAD, HillClimb
├── task/                optimization / machine_learning / science_discovery
└── tools/llm, tools/profiler
```

**约束 1：个体 = 单个函数。** 用户接口是

```python
class MyEvaluation(Evaluation):
    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        ...
```

一个 `program_str`、一个 `callable_func`、返回一个可比较的标量（不合法返回 `None`）。**没有原生的"两个种群"概念。**

**约束 2：评估在子进程里跑**（`SecureEvaluator` 做超时保护 + 多进程）。这意味着你的信用账本不能只是个普通的实例属性——子进程写的东西主进程看不到。

**约束 3：LLM4AD 没有 LNS destroy/repair 任务。** 现有 optimization 任务全是 constructive heuristic 或 guided local search。这个环境要你自己建。

**约束 4（好消息）**：文档明确说 `program_str` 传给你是为了让你"计算代码的'新颖性'或检查是否已评估过"。所以指纹/去重的钩子官方就是留在这里的。

**约束 5（好消息）**：MEoH / NSGA2 / MOEAD 是多目标的，说明框架支持返回多个值。看一眼 MEoH 的实现就知道富返回值的正确写法。

### 三个绕过"单函数"约束的方案

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 对作为一个个体** | template 里放两个函数，LLM 一次生成一对 | 简单，几乎零改动，但**没有两个种群 = 没有配对问题 = 没有你的研究课题**。只能当 baseline |
| **B. 有状态的 Evaluation + 跑两次 EoH** | `Evaluation` 对象持有共享的算子档案和信用账本；跑两个 EoH 实例（一个演化 destroy、一个演化 repair），共用同一个 Evaluation | **最快能跑起来的路径**，用现成 method 不改 method 层。缺点：两个 EoH 的代际不同步，配对策略只能塞在 Evaluation 里 |
| **C. 自定义 method** | 抄 `method/eoh/` 写 `method/coevo/`，双种群 + 配对控制器 + 信用账本 | 正确答案，但要等 B 验证完再投入 |

**建议：B → C。** 先用 B 把管线跑通拿到第一批真实数据，确认 γ 在你的问题上确实显著（这是整个课题成立的前提），再花时间写 C。

### 关于跨进程的信用账本

三个可选实现，按推荐度：

1. **SQLite（推荐）**：子进程直接 append 到一个 `records` 表（`run_id, gen, i, j, step, delta, p_sel, seed`）。WAL 模式下并发 append 安全，主进程随时查询。天然支持断点续跑（LLM4AD 本来就支持 resume）。
2. **返回可比较对象**：`evaluate_program` 返回一个实现了 `__gt__` 的对象，里面挂着完整记录。文档明确允许 "comparable" 返回值。优雅但脆——排序逻辑散落在 method 里。
3. `multiprocessing.Manager().list()`：最简单，但和 `SecureEvaluator` 的进程池怎么配合要试。

---

## 2. 测试床推荐

你问了这个。我的建议：

**主战场 CVRP，高耦合对照组 VRPTW。**

理由：

- **CVRP** 给你和 G-LNS / CoEvo-AHD 的直接可比性，而且 LLM4AD 已有 CVRP 实例生成代码可复用。
- **VRPTW** 时间窗约束让 destroy 和 repair 的耦合明显更强——破坏算子留下的时间窗结构直接决定修复算子能不能插进去。**γ 在这里应该显著更大。** 而你整篇论文的前提就是"γ 重要且现有方法测不准它"，所以你需要一个 γ 确实大的测试床来支撑。
- LLM4AD 已有 VRPTW 的 constructive 任务，实例生成器可复用。

### Stage 0.5：经典算子的 γ 审计（本周就能做，零 LLM 成本）

这一步我强烈建议不要跳过。取 6 个经典破坏算子 × 6 个经典修复算子：

```
destroy: random_removal, shaw_removal(相关性), worst_removal,
         route_removal, cluster_removal, time_oriented_removal
repair:  greedy_insertion, regret_2, regret_3,
         greedy_with_noise, best_position_first, sequential_insertion
```

用 **uniform 配对**（关键：均匀采样才能无偏估计 γ）跑满 6×6 网格，每格若干次，然后拟合 α+β+γ，看：

**γ 占 payoff 总方差的比例是多少？**

这一个数字决定你整个课题的成色：

| γ 方差占比 | 含义 | 行动 |
|---|---|---|
| > 25% | 交互确实重要 | 全力推进，这就是你的核心发现 |
| 10–25% | 中等 | 可做，需要在多个问题上做才有说服力 |
| < 10% | 交互其实不重要 | **换问题或换角度**——可加分解就够了，那 G-LNS 的做法虽然统计上错但实践上无害 |

在 CVRP 和 VRPTW 上各做一次，比较 γ 占比。如果 VRPTW 显著更高，这本身就是论文的一个有价值的观察："算子交互的重要性随问题约束强度上升"，并且给了你选测试床的正当理由。

**这一步的成本：写 12 个经典算子 + 一个 LNS driver，1–2 天，零 LLM 调用。产出：决定课题走向的关键数字 + 论文的第二张图。**

---

## 3. 分阶段实施路线

### Stage 1：可重放的 LNS 环境（地基）

新建 `llm4ad/task/optimization/cvrp_lns/`：

```python
# evaluation.py
class CVRPLNSEvaluation(Evaluation):
    """有状态的 Evaluation：持有伙伴档案 + 信用账本 + 配对策略。"""

    def __init__(self, role: str, partner_archive, ledger, pairing_policy, ...):
        self.role = role                      # 'destroy' | 'repair'
        self.partner_archive = partner_archive
        self.ledger = ledger                  # SQLite 句柄
        self.pairing_policy = pairing_policy

    def evaluate_program(self, program_str, callable_func):
        partners = self.pairing_policy.select(self.role, callable_func, k=...)
        records = []
        for p, p_prob in partners:
            for inst, seed in self.eval_plan:
                traj = run_lns(inst, destroy, repair, seed=seed, record=True)
                records.extend(traj.step_records)   # (i, j, step, delta, p_sel)
        self.ledger.write(records)
        return self.ledger.fitness_for(self.role, program_str)   # <- 信用模块说了算
```

**这里的关键设计**：`evaluate_program` 返回的 fitness **不是**直接的性能，而是**信用模块的输出**。这样你换信用分配方案就只改一行，method 层完全不用动。这是整套实验能廉价做 ablation 的原因。

`run_lns` 的硬性要求：

1. **可精确重放**：`(初始解, RNG 种子, 温度序列, 算子调用序列)` 完全确定轨迹。用独立的 `np.random.Generator`，不要用全局 random。
2. **逐步记录**：每次算子调用记 `(i, j, step, f_before, f_after, accepted, best_so_far)`。
3. **算子接口契约**：破坏算子签名 `destroy(solution, rng, n_remove) -> (partial_solution, removed)`；修复算子 `repair(partial_solution, removed, rng) -> solution`。类型统一，否则 LLM 生成的代码没法互换。
4. **探针状态导出**：跑的时候顺便把 K 个探针解 dump 出来，供后面算指纹。

验收：跑 stock EoH 在"方案 A（对作为一个个体）"上，能得到合理的启发式 → 环境没问题。

### Stage 2：信用分配 ablation（第一个可发表单元）

保持 EoH 不动，只换 `ledger.fitness_for` 的实现：

| arm | 信用方案 |
|---|---|
| C0 | σ 四级累加（G-LNS 复现） |
| C1 | σ 均值（频率去偏） |
| C2 | ANOVA α/β/γ 分解 |
| C3 | C2 + 极值统计（Fialho：滑窗内最佳改进而非均值） |
| C4 | C3 + 按 (实例, 搜索阶段) 秩归一化 |
| C5 | C4 + IPS 加权（诚实报告，Stage 0 显示可能无收益） |

奖励形式正交维度：四级 σ / RUDDER 式 `V(s_{t+1}) − V(s_t)` 差分。

指标：最终 best fitness、QD-score、**以及 Cook & Tumer 的 global-local alignment**（局部信用与全局收益的一致性）——这个指标可以独立于最终性能来证明你的信用信号更好。

### Stage 3：γ 导向的配对策略（核心贡献）

现在必须写自定义 method 了（方案 C）。arm：

| arm | 配对策略 |
|---|---|
| P0 | roulette（ALNS / G-LNS） |
| P1 | 组件级 Thompson（CoEvo-AHD） |
| P2 | γ-UCB：`argmax UCB(α_i + β_j + γ_ij)` |
| **P3** | **P2 + 定向覆盖探索**（Stage 0 的 ε，建议起点 0.2–0.35） |
| P4 | P3 + 低秩矩阵补全（填未采样格子） |
| P5 | P4 + 探针语义先验冷启动 |

**P3 是最重要的一个 arm**，因为 E2 已经在合成数据上证明它有效。P4 是 P3 的自然延伸——当算子池大到无法覆盖时，低秩假设 + LLM embedding 上下文是唯一出路。

### Stage 4：精确重放反事实（把回归的 γ 升级成因果的 γ）

在 Stage 1 的可重放基础上加：

```python
def counterfactual_delta(traj, t, baseline_pair=(random_removal, greedy_insert)):
    """把第 t 步的算子对换成基线对，其余全部不变，重放到结束。"""
    replay = replay_from(traj, until=t)
    replay.apply(baseline_pair)
    replay.continue_with(traj.decisions[t+1:])   # 后续决策序列保持一致
    return traj.final_best - replay.final_best
```

三种粒度：单次调用替换（difference reward）、拆对检验（因果交互项）、portfolio 级蒙特卡洛 Shapley。

**预算控制**：只对 γ 后验方差最大的少数决策点做。这是主动学习问题，本身也是一个技术点。

**杀手级实验**：把因果 γ 当 ground truth，画 `S_ij`（G-LNS）vs `γ_causal` 的散点图，以及 `γ̂_ANOVA` vs `γ_causal`。这把 Stage 0 的合成结论**在真实 LNS 上复现一遍**。这一张图基本就锁定了论文。

**成本论据（写进论文）**：反事实重放**消耗 0 次 LLM 调用**，只花 CPU。在 AHD 里稀缺资源是 LLM sample 而不是 CPU，所以这个方案在正确的维度上花钱。

### Stage 5：多样性与指纹

注意**这块地已经被占了一部分**——LLM4AD 里已经集成了：

- **PartEvo**（NeurIPS 2025）：niching-enhanced evolution
- **MEoH**（AAAI 2025）：dominance-dissimilarity，代码相异度 + 目标空间支配
- **EoH-S**（AAAI 2026）：互补启发式集合，目标函数单调超模
- **HSEvo**（AAAI 2025）：Shannon-Wiener / Cumulative Diversity Index

所以纯"多样性"角度不好切。**你的差异化在于：这些都是单种群的多样性，没人做过"跨种群的配对兼容性距离"。** 探针语义指纹的独特价值恰恰在这里——破坏算子的"移除画像"和修复算子的"偏好输入画像"落在同一个残解空间里，可以直接算兼容性。这是单种群方法结构上做不到的事。

做之前先做可行性验证：**用 Stage 0.5 的 12 个经典算子，算它们的探针指纹兼容性距离，看它能不能预测已经测出来的 γ_ij。** 如果 Spearman ≥ 0.5，值得投入；如果 ≈ 0，说明这条路不通，及时止损。

---

## 4. 建议的时间线

| 周 | 任务 | 产出 |
|---|---|---|
| 0（已完成） | Stage 0 合成验证 | 论文图 1，且已修正研究优先级 |
| 1 | Stage 0.5 经典算子 γ 审计（CVRP + VRPTW） | **决定课题成色的关键数字** + 图 2 |
| 2–3 | Stage 1 可重放 LNS 环境 + LLM4AD 接入 | 地基，stock EoH 能跑通 |
| 4–5 | Stage 2 信用 ablation（方案 B，不改 method） | 第一批真实数据 |
| 6–8 | Stage 3 自定义 coevo method + 配对策略 | 核心贡献 |
| 9–10 | Stage 4 反事实重放 + 图 1 在真实环境复现 | 论文的决定性证据 |
| 11+ | Stage 5 指纹 / 与 PartEvo·MEoH·EoH-S 对比 | 扩展 |

**最大的风险点在第 1 周**：如果 γ 方差占比 < 10%，整个"交互项"叙事就站不住，需要及时换角度（比如转向时序信用分配 + RUDDER，那条线不依赖 γ 大）。所以务必先做 Stage 0.5，不要先花 3 周搭环境。

---

## 5. 用现成的，别重造

LLM4AD 里可以直接当 baseline 的：

| 你要比的 | 用哪个 |
|---|---|
| 单启发式 SOTA | EoH, MCTS-AHD |
| 反思式进化 | ReEvo |
| 多样性维持 | PartEvo, MEoH |
| 互补集合 | EoH-S |
| 邻域搜索式演化 | LHNS |
| 协同进化算子对 | **G-LNS / CoEvo-AHD（框架里没有，需自己复现）** |

最后一行是必须自己实现的——而这正好，因为你需要精确复现它的 `F += σ, S += σ` 才能做对照消融。复现它的代价很低（就是几行累加），而且你已经从论文里拿到了确切公式。

## 6. 三个容易踩的坑

1. **别在没有 grounded 搜索状态特征的情况下做动态适应。** DyACE 的消融显示，缺少真实的搜索轨迹反馈时，动态适应反而**不如静态算法**。你喂给 bandit / LLM 的状态描述（当前解质量、停滞计数、多样性、剩余预算、上一步接受与否）不是可选项。

2. **奖励尺度归一化不做，一切白搭。** 按 (实例, 搜索阶段) 分桶做秩归一化。搜索早期随便改都有大改进，不归一化的话信用完全被"这个算子碰巧在早期被调用"主导。

3. **LLM 不要当信用分配的裁判。** Who&When 基准上最好的自动归因方法识别责任 agent 准确率 53.5%、定位错误步骤 14.2%，o1/R1 也达不到实用水平。LLM 的正确用法是**生成假设**（"这个修复算子可能没利用破坏留下的空间聚簇"），然后用可执行的反事实实验去验证。
