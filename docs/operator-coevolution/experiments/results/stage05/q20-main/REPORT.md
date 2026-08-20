# Stage 0.5：经典算子 γ 审计结果

## 实验决定

采用 **6 个破坏算子 × 6 个修复算子的完整均衡因子设计**。每个 `(实例, 重复)` 是一个区组：36 个算子对共享同一初始可行解，并使用确定性派生随机种子。这样实例难度和初始解质量不会混入算子对效应。配对为 uniform（每格样本数完全相同），全程零 LLM 调用。

- 问题：CVRP、VRPTW；每类 12 个独立实例，每实例 8 个初始状态
- 每格样本：96；每问题总观测：3456
- 客户数：50；每次移除比例：20%
- payoff：一次 destroy–repair 后的相对距离改善百分比（允许负值）
- 主指标：`Var(γ) / Var(M)`，与 Stage 0 脚本定义一致；95% CI 对实例做聚类 bootstrap
- 辅助检验：随机区组二因素 ANOVA 的交互 F 检验及 partial η²（包含区组×处理噪声）

## 核心结果

| 问题 | γ 方差占比 | 95% 聚类 CI | partial η² | F(df) | p | 判定 |
|---|---:|---:|---:|---:|---:|---|
| CVRP | **0.9%** | [0.7%, 2.5%] | 0.7% | 0.89 (25, 3325) | 0.614 | 弱（<10%） |
| VRPTW | **3.0%** | [2.3%, 7.4%] | 1.0% | 1.37 (25, 3325) | 0.104 | 弱（<10%） |

系统性 cell-mean 方差的来源（3 项按构造合计为 100%）：

| 问题 | destroy 主效应 α | repair 主效应 β | 交互 γ |
|---|---:|---:|---:|
| CVRP | 93.9% | 5.2% | 0.9% |
| VRPTW | 52.7% | 44.3% | 3.0% |

## 决策

- **CVRP**：交互叙事证据不足，建议转向时序信用分配或更强耦合测试床。
- **VRPTW**：交互叙事证据不足，建议转向时序信用分配或更强耦合测试床。
- **跨问题比较**：两者仅相差 2.1%，没有实质证据表明时间窗提高了 γ。

### 邻域大小敏感性复核

为排除 20% 移除比例的偶然性，另以 8 个实例 × 6 个状态（每格 48 次）复核了 10% 和 30%：

| 移除比例 | CVRP γ 占比 | CVRP p | VRPTW γ 占比 | VRPTW p |
|---:|---:|---:|---:|---:|
| 10% | 1.9% | 0.996 | 9.0% | 0.965 |
| 20%（主实验） | 0.9% | 0.614 | 3.0% | 0.104 |
| 30% | 1.9% | 0.121 | 3.7% | 0.282 |

三个邻域尺度的点估计均未超过 10%，交互检验也均不显著；因此“γ 弱”的决策不依赖单一移除比例。10% VRPTW 的比例估计接近阈值但极不稳定（95% CI [3.8%, 26.1%]），应视为后续标准实例复核的候选，而不是正面证据。

判定应以占比及其区间为主，而不是只看 p 值：样本多时很小的交互也可能显著。

## 最强与最弱的交互格

### CVRP

最正协同：

- `route_removal × best_position_first`：γ = +0.648 个百分点
- `route_removal × greedy_with_noise`：γ = +0.502 个百分点
- `shaw_removal × greedy_insertion`：γ = +0.422 个百分点

最负协同：

- `route_removal × regret_2`：γ = -0.543 个百分点
- `shaw_removal × greedy_with_noise`：γ = -0.513 个百分点
- `route_removal × greedy_insertion`：γ = -0.493 个百分点

### VRPTW

最正协同：

- `random_removal × best_position_first`：γ = +0.672 个百分点
- `time_oriented_removal × regret_3`：γ = +0.513 个百分点
- `time_oriented_removal × regret_2`：γ = +0.478 个百分点

最负协同：

- `time_oriented_removal × sequential_insertion`：γ = -0.802 个百分点
- `random_removal × regret_2`：γ = -0.785 个百分点
- `random_removal × regret_3`：γ = -0.623 个百分点

## 解释边界

1. 这是**经典算子的直接一步 payoff 审计**，回答“配对本身是否产生不可加的即时效果”；它不等同于完整 ALNS 长轨迹的最终性能。
2. CVRP 没有时间窗，因此 `time_oriented_removal` 在 CVRP 中按路线归一化进度定义；VRPTW 使用实际服务开始时间。
3. 主指标只分解 36 个 cell mean 的系统性差异，不把重复间噪声放进分母；报告的 partial η² 则把该噪声纳入，二者回答不同问题。
4. 合成实例沿用 LLM4AD constructive task 的分布而非 Solomon/CVRPLIB 标准实例，因此当前结论是 Stage 1 投资决策证据，不应直接作为最终 benchmark 结论。

## 复现

```powershell
python docs/operator-coevolution/experiments/stage05_gamma_audit.py --instances 12 --repeats 8 --customers 50 --remove-fraction 0.2 --bootstrap 2000 --seed 20240820
```

本次运行耗时 67.7 秒。原始数据见 `raw_observations.csv`，完整数值见 `summary.json`，热力图见 `gamma_heatmaps.png`。
