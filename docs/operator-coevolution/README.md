# 算子协同进化研究材料

本目录汇总在 LLM4AD 上开展 destroy–repair 算子协同进化研究所需的调研、实施方案与实验材料。所有实验代码和结果都在 [`experiments/`](./experiments/) 下，仓库根目录不再存放研究产物。

## 文档导航

| 文档 | 内容 |
|---|---|
| [跨领域调研与迁移设计](./literature-review.md) | 梳理算子距离、行为新颖性和信用分配方法，给出可迁移设计 |
| [落地实施方案](./implementation-plan.md) | Stage 0 结论、LLM4AD 架构约束、分阶段路线与实验优先级 |
| [**实验结果汇总**](./experiments/RESULTS.md) | **Stage 0 与 Stage 0.5 的全部关键数字、结论及其对课题走向的影响** |
| [实验运行说明](./experiments/README.md) | 两个实验的环境、命令与产物说明 |

## 推荐阅读顺序

1. 先读[实验结果汇总](./experiments/RESULTS.md)，一页看完已经确定的事实和当前的决策状态。
2. 再读[跨领域调研](./literature-review.md)，理解距离度量与信用分配的设计空间。
3. 最后读[落地实施方案](./implementation-plan.md)，了解阶段安排；按[运行说明](./experiments/README.md)复现即可核对全部数字。

## 当前进度

| 阶段 | 状态 | 产物 |
|---|---|---|
| Stage 0 合成验证 | 已完成 | [`experiments/results/stage0/`](./experiments/results/stage0/) |
| Stage 0.5 经典算子 γ 审计 | 已完成 | [`experiments/results/stage05/`](./experiments/results/stage05/) |
| Stage 1 可重放 LNS 环境 | 未开始 | — |

**Stage 0.5 的结果触发了实施方案预先规定的止损条件**（γ 方差占比 < 10%），Stage 3 的 γ-UCB 配对策略不应按原计划直接投入。下一步的三条候选路径见[实验结果汇总](./experiments/RESULTS.md)末尾。
