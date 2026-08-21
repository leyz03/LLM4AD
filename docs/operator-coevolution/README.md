# 算子协同进化研究材料

本目录汇总在 LLM4AD 上开展 destroy–repair 算子协同进化研究所需的调研、实施方案与实验材料。所有实验代码和结果都在 [`experiments/`](./experiments/) 下，仓库根目录不再存放研究产物。

## 文档导航

| 文档 | 内容 |
|---|---|
| [跨领域调研与迁移设计](./literature-review.md) | 梳理算子距离、行为新颖性和信用分配方法，给出可迁移设计 |
| [**行为感知的组合库协同进化**](./behavior-aware-portfolio-coevolution.md) | **当前设计基线：分别给出距离、变异坍缩、信用分配的操作方案，再说明三者如何进入同一协同进化闭环** |
| [落地实施方案](./implementation-plan.md) | Stage 0 结论、LLM4AD 架构约束、分阶段路线与实验优先级 |
| [Stage 0.5 复核与下一步](./stage05-review-and-next-steps.md) | 统计复核、算子退化诊断、构念效度问题与预注册规则 |
| [**实验结果汇总**](./experiments/RESULTS.md) | **Stage 0 与 Stage 0.5 的全部关键数字、结论及其对课题走向的影响** |
| [**第一轮变异/多样性综合报告**](./experiments/results/2026-08-21-mutation-diversity-report.md) | **Qwen 变异链、L0/L1+/L2/L3 度量审计、主线决策与下一实验** |
| [实验运行说明](./experiments/README.md) | 两个实验的环境、命令与产物说明 |

## 推荐阅读顺序

1. 先读[实验结果汇总](./experiments/RESULTS.md)，一页看完已经确定的事实和当前的决策状态。
2. 再读[跨领域调研](./literature-review.md)，理解距离度量与信用分配的设计空间。
3. 然后读[行为感知的组合库协同进化](./behavior-aware-portfolio-coevolution.md)，了解调研与协同进化汇合后的当前设计。
4. 最后用[落地实施方案](./implementation-plan.md)核对原始架构约束，并按[运行说明](./experiments/README.md)复现实验。

## 当前进度

| 阶段 | 状态 | 产物 |
|---|---|---|
| Stage 0 合成验证 | 已完成 | [`experiments/results/stage0/`](./experiments/results/stage0/) |
| Stage 0.5 经典算子 γ 审计 | 已完成 | [`experiments/results/stage05/`](./experiments/results/stage05/) |
| Qwen repair 变异链 pilot | 已完成，证据不足 | [`experiments/results/mutation-collapse/pilot/`](./experiments/results/mutation-collapse/pilot/) |
| 多样性度量效度 pilot | 已完成，L2 通过 | [`experiments/results/diversity-metrics/pilot/`](./experiments/results/diversity-metrics/pilot/) |
| Stage 1 可重放 LNS 环境 | 未开始 | — |

Stage 0.5 在当前经典算子与单步 payoff 下没有发现足以支撑 `γ-UCB` 的交互信号；复核同时发现 repair 算子有效自由度严重退化，因此该结果不能直接外推到专才或 LLM 演化算子。当前主线已调整为[行为感知的组合库协同进化](./behavior-aware-portfolio-coevolution.md)，`γ` 保留为通过构念门禁后再决定是否启用的条件分支。

2026-08-21 的第一轮主线探索进一步确定：当前优先研究 **LLM 变异退化为相似**，以通过 held-out 效度审计的 L2 probe 输出距离作为核心仪表。具体证据、边界与下一步两臂实验见[综合报告](./experiments/results/2026-08-21-mutation-diversity-report.md)。
