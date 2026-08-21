# 多样性度量综合审计

本报告比较 9 个 repair 算子的代码、LLM 机制判断、probe 输出行为与性能足迹。
calibration 实例用于构造候选距离；validation 实例只用于 held-out 效度检验。

## 结论速览

按 held-out L2 与 L3 相关之和排序，当前最佳候选度量是 **`probe_output_l2`**。
这只是本 pilot 算子集上的排序；是否进入选择机制还要同时满足重复测量信度和独立实例效度。

| 度量 | 重测信度 | 有效自由度 | held-out L2 ρ / p | L2 最近邻一致 | held-out L3 ρ / p | L3 最近邻一致 |
|---|---:|---:|---:|---:|---:|---:|
| `probe_output_l2` | 0.983 | 2.068 | 0.727 / 0.010 | 44% | 0.571 / 0.050 | 33% |
| `performance_footprint_l3` | 0.900 | 1.945 | 0.539 / 0.055 | 33% | 0.548 / 0.045 | 44% |
| `token_l0` | 1.000 | 2.145 | 0.224 / 0.119 | 56% | 0.417 / 0.025 | 33% |
| `ast_l0` | 1.000 | 2.132 | 0.108 / 0.264 | 67% | 0.235 / 0.129 | 33% |

说明：Mantel p 通过置换算子标签得到；最近邻一致表示该度量与 held-out 目标为同一算子找到相同最近邻的比例。
有效自由度来自各距离的 median-bandwidth RBF kernel，只适合在同一批算子内比较其是否夸大/压缩维度。

## 假多样性：AST 很远但 held-out 行为很近

| 算子对 | AST | held-out output | 差值 |
|---|---:|---:|---:|
| `control:sequential_rewrite` × `specialist:block_reinsert` | 0.860 | 0.026 | +0.835 |
| `classic:sequential_insertion` × `control:sequential_rewrite` | 0.772 | 0.000 | +0.772 |
| `classic:sequential_insertion` × `specialist:block_reinsert` | 0.785 | 0.026 | +0.760 |
| `classic:regret_2` × `classic:best_position_first` | 0.923 | 0.175 | +0.748 |
| `classic:greedy_insertion` × `classic:best_position_first` | 0.897 | 0.156 | +0.741 |

## 隐藏多样性：AST 很近但 held-out 行为很远

| 算子对 | AST | held-out output | 差值 |
|---|---:|---:|---:|
| `classic:regret_2` × `classic:regret_3` | 0.000 | 0.016 | +0.016 |
| `classic:sequential_insertion` × `specialist:new_route` | 0.422 | 0.352 | -0.070 |
| `classic:greedy_insertion` × `specialist:new_route` | 0.809 | 0.368 | -0.441 |
| `classic:best_position_first` × `specialist:new_route` | 0.822 | 0.361 | -0.461 |
| `specialist:block_reinsert` × `specialist:new_route` | 0.821 | 0.352 | -0.469 |

## 推荐的度量栈

1. AST/token 只做免费预过滤和精确/近换皮报警，不直接代表行为多样性。
2. Qwen direct judgment 若跨重复稳定，可作低成本机制 gate；它不能代替可执行行为测量。
3. calibration L2 probe output 是当前选择、聚类和变异审计的核心候选，前提是其 held-out 效度通过。
4. L3 performance footprint 表示“在哪里有用”，成本更高，适合组合库边际信用和最终验证。
5. 有效自由度是任一表示上的群体汇总仪表，不是独立的个体距离，也不是性能目标。

## 边界与未覆盖项

- 当前 L1 使用 Qwen 机制描述 + TF-IDF proxy 和直接成对判断；运行环境没有 torch/transformers，尚未加入 CodeBERT/GraphCodeBERT。报告不会把 proxy 冒充代码 embedding。
- Qwen 同时生成部分 mutation 候选并担任语义裁判，可能有同模型偏置；经典算子和手工对照可缓解但不能消除。
- L2 目前比较单步 repair 后的边集合，尚未包含完整搜索轨迹、插入决策序列或多搜索阶段。
- L3 使用固定残解上的收益秩足迹，不等价于长程 LNS portfolio value。
- 小样本矩阵中的 pair 不是独立观测，因此以 Mantel 置换而不是普通相关 p 值为主。

## 排除的算子

无。

## 复现

```powershell
python docs/operator-coevolution/experiments/diversity_metric_audit.py --output D:/Study/LLM4AD/docs/operator-coevolution/experiments/results/diversity-metrics/self-test --mutation-results D:/Study/LLM4AD/docs/operator-coevolution/experiments/results/mutation-collapse/pilot
```

运行耗时 19.2 秒。完整距离矩阵见 `matrices/`，Qwen 原始判断见 `llm_judgments/`。
