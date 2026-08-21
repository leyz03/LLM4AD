# 多样性度量综合审计

本报告比较 13 个 repair 算子的代码、LLM 机制判断、probe 输出行为与性能足迹。
calibration 实例用于构造候选距离；validation 实例只用于 held-out 效度检验。

## 结论速览

按 held-out L2 与 L3 相关之和排序，当前最佳候选度量是 **`probe_output_l2`**。
这只是本 pilot 算子集上的排序；是否进入选择机制还要同时满足重复测量信度和独立实例效度。

| 度量 | 重测信度 | 有效自由度 | held-out L2 ρ / p | L2 最近邻一致 | held-out L3 ρ / p | L3 最近邻一致 |
|---|---:|---:|---:|---:|---:|---:|
| `probe_output_l2` | 0.981 | 2.357 | 0.958 / 0.000 | 92% | 0.861 / 0.000 | 77% |
| `performance_footprint_l3` | 0.899 | 2.334 | 0.744 / 0.001 | 54% | 0.816 / 0.000 | 46% |
| `llm_direct_l1plus` | 0.203 | 2.551 | 0.262 / 0.153 | 31% | 0.452 / 0.038 | 31% |
| `llm_descriptor_tfidf_l1proxy` | — | 2.112 | 0.036 / 0.379 | 54% | 0.171 / 0.134 | 46% |
| `ast_l0` | 1.000 | 2.312 | -0.133 / 0.838 | 31% | -0.091 / 0.732 | 23% |
| `token_l0` | 1.000 | 2.279 | -0.193 / 0.921 | 15% | -0.112 / 0.754 | 8% |

说明：Mantel p 通过置换算子标签得到；最近邻一致表示该度量与 held-out 目标为同一算子找到相同最近邻的比例。
有效自由度来自各距离的 median-bandwidth RBF kernel，只适合在同一批算子内比较其是否夸大/压缩维度。

## 假多样性：AST 很远但 held-out 行为很近

| 算子对 | AST | held-out output | 差值 |
|---|---:|---:|---:|
| `control:sequential_rewrite` × `specialist:block_reinsert` | 0.860 | 0.008 | +0.852 |
| `classic:sequential_insertion` × `specialist:block_reinsert` | 0.785 | 0.008 | +0.777 |
| `classic:sequential_insertion` × `control:sequential_rewrite` | 0.772 | 0.000 | +0.772 |
| `classic:regret_2` × `qwen:chain_02:g08` | 0.961 | 0.192 | +0.769 |
| `classic:regret_3` × `qwen:chain_02:g08` | 0.961 | 0.213 | +0.748 |

## 隐藏多样性：AST 很近但 held-out 行为很远

| 算子对 | AST | held-out output | 差值 |
|---|---:|---:|---:|
| `classic:regret_2` × `classic:regret_3` | 0.000 | 0.190 | +0.190 |

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
python docs/operator-coevolution/experiments/diversity_metric_audit.py --output D:/Study/LLM4AD/docs/operator-coevolution/experiments/results/diversity-metrics/pilot --mutation-results D:/Study/LLM4AD/docs/operator-coevolution/experiments/results/mutation-collapse/pilot
```

运行耗时 229.9 秒。完整距离矩阵见 `matrices/`，Qwen 原始判断见 `llm_judgments/`。
