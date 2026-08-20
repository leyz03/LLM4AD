# Stage 0.5：经典算子 γ 审计结果

## 实验决定

采用 **6 个破坏算子 × 6 个修复算子的完整均衡因子设计**。每个 `(实例, 重复)` 是一个区组：36 个算子对共享同一初始可行解，并使用确定性派生随机种子。这样实例难度和初始解质量不会混入算子对效应。配对为 uniform（每格样本数完全相同），全程零 LLM 调用。

- 问题：CVRP、VRPTW；每类 8 个独立实例，每实例 6 个初始状态
- 每格样本：48；每问题总观测：1728
- 客户数：50；每次移除比例：30%
- payoff：一次 destroy–repair 后的相对距离改善百分比（允许负值）
- 主指标：`Var(γ) / Var(M)`，与 Stage 0 脚本定义一致；95% CI 对实例做聚类 bootstrap
- 辅助检验：随机区组二因素 ANOVA 的交互 F 检验及 partial η²（包含区组×处理噪声）

## 核心结果

| 问题 | γ 方差占比 | 95% 聚类 CI | partial η² | F(df) | p | 判定 |
|---|---:|---:|---:|---:|---:|---|
| CVRP | **1.9%** | [1.4%, 4.8%] | 2.0% | 1.34 (25, 1645) | 0.121 | 弱（<10%） |
| VRPTW | **3.7%** | [3.2%, 9.8%] | 1.7% | 1.14 (25, 1645) | 0.282 | 弱（<10%） |

## 决策

- **CVRP**：交互叙事证据不足，建议转向时序信用分配或更强耦合测试床。
- **VRPTW**：交互叙事证据不足，建议转向时序信用分配或更强耦合测试床。
- **跨问题比较**：两者仅相差 1.8%，没有实质证据表明时间窗提高了 γ。

判定应以占比及其区间为主，而不是只看 p 值：样本多时很小的交互也可能显著。

## 最强与最弱的交互格

### CVRP

最正协同：

- `route_removal × best_position_first`：γ = +1.391 个百分点
- `route_removal × greedy_with_noise`：γ = +1.344 个百分点
- `random_removal × regret_2`：γ = +0.834 个百分点

最负协同：

- `route_removal × regret_2`：γ = -1.253 个百分点
- `random_removal × best_position_first`：γ = -0.877 个百分点
- `worst_removal × best_position_first`：γ = -0.733 个百分点

### VRPTW

最正协同：

- `time_oriented_removal × regret_3`：γ = +1.128 个百分点
- `random_removal × greedy_with_noise`：γ = +1.074 个百分点
- `random_removal × best_position_first`：γ = +0.848 个百分点

最负协同：

- `time_oriented_removal × best_position_first`：γ = -0.913 个百分点
- `random_removal × regret_3`：γ = -0.877 个百分点
- `random_removal × regret_2`：γ = -0.653 个百分点

## 解释边界

1. 这是**经典算子的直接一步 payoff 审计**，回答“配对本身是否产生不可加的即时效果”；它不等同于完整 ALNS 长轨迹的最终性能。
2. CVRP 没有时间窗，因此 `time_oriented_removal` 在 CVRP 中按路线归一化进度定义；VRPTW 使用实际服务开始时间。
3. 主指标只分解 36 个 cell mean 的系统性差异，不把重复间噪声放进分母；报告的 partial η² 则把该噪声纳入，二者回答不同问题。
4. 合成实例沿用 LLM4AD constructive task 的分布而非 Solomon/CVRPLIB 标准实例，因此当前结论是 Stage 1 投资决策证据，不应直接作为最终 benchmark 结论。

## 复现

```powershell
python stage05_gamma_audit.py --instances 8 --repeats 6 --customers 50 --remove-fraction 0.3 --bootstrap 1000 --seed 20240820
```

本次运行耗时 67.1 秒。原始数据见 `raw_observations.csv`，完整数值见 `summary.json`，热力图见 `gamma_heatmaps.png`。
