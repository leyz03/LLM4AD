# Stage 0.5：经典算子 γ 审计结果

## 实验决定

采用 **6 个破坏算子 × 6 个修复算子的完整均衡因子设计**。每个 `(实例, 重复)` 是一个区组：36 个算子对共享同一初始可行解，并使用确定性派生随机种子。这样实例难度和初始解质量不会混入算子对效应。配对为 uniform（每格样本数完全相同），全程零 LLM 调用。

- 问题：CVRP、VRPTW；每类 2 个独立实例，每实例 2 个初始状态
- 每格样本：4；每问题总观测：144
- 客户数：15；每次移除比例：20%
- payoff：一次 destroy–repair 后的相对距离改善百分比（允许负值）
- 主指标：`Var(γ) / Var(M)`，与 Stage 0 脚本定义一致；95% CI 对实例做聚类 bootstrap
- 辅助检验：随机区组二因素 ANOVA 的交互 F 检验及 partial η²（包含区组×处理噪声）

## 核心结果

| 问题 | γ 方差占比 | 95% 聚类 CI | partial η² | F(df) | p | 判定 |
|---|---:|---:|---:|---:|---:|---|
| CVRP | **66.4%** | [53.5%, 77.2%] | 18.6% | 0.96 (25, 105) | 0.526 | 强（>25%） |
| VRPTW | **17.0%** | [17.0%, 62.9%] | 6.2% | 0.28 (25, 105) | 1 | 中等（10%–25%） |

## 决策

- **CVRP**：交互是主要信号，建议全力推进以 γ 为核心的研究路线。
- **VRPTW**：课题可做，但应保留多问题验证并控制结论边界。
- **跨问题比较**：VRPTW 比 CVRP 低 49.4%，不支持原先关于时间窗必然增强 γ 的预期。

判定应以占比及其区间为主，而不是只看 p 值：样本多时很小的交互也可能显著。

## 最强与最弱的交互格

### CVRP

最正协同：

- `cluster_removal × regret_2`：γ = +4.597 个百分点
- `time_oriented_removal × best_position_first`：γ = +2.832 个百分点
- `worst_removal × regret_3`：γ = +1.939 个百分点

最负协同：

- `cluster_removal × best_position_first`：γ = -4.932 个百分点
- `time_oriented_removal × greedy_with_noise`：γ = -4.463 个百分点
- `worst_removal × sequential_insertion`：γ = -3.672 个百分点

### VRPTW

最正协同：

- `time_oriented_removal × regret_3`：γ = +2.504 个百分点
- `random_removal × regret_2`：γ = +2.042 个百分点
- `shaw_removal × regret_2`：γ = +1.860 个百分点

最负协同：

- `time_oriented_removal × sequential_insertion`：γ = -4.791 个百分点
- `route_removal × regret_2`：γ = -3.432 个百分点
- `cluster_removal × regret_3`：γ = -2.370 个百分点

## 解释边界

1. 这是**经典算子的直接一步 payoff 审计**，回答“配对本身是否产生不可加的即时效果”；它不等同于完整 ALNS 长轨迹的最终性能。
2. CVRP 没有时间窗，因此 `time_oriented_removal` 在 CVRP 中按路线归一化进度定义；VRPTW 使用实际服务开始时间。
3. 主指标只分解 36 个 cell mean 的系统性差异，不把重复间噪声放进分母；报告的 partial η² 则把该噪声纳入，二者回答不同问题。
4. 合成实例沿用 LLM4AD constructive task 的分布而非 Solomon/CVRPLIB 标准实例，因此当前结论是 Stage 1 投资决策证据，不应直接作为最终 benchmark 结论。

## 复现

```powershell
python stage05_gamma_audit.py --instances 2 --repeats 2 --customers 15 --remove-fraction 0.2 --bootstrap 100 --seed 20240820
```

本次运行耗时 1.0 秒。原始数据见 `raw_observations.csv`，完整数值见 `summary.json`，热力图见 `gamma_heatmaps.png`。
