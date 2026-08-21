# LLM repair 变异链坍缩审计

本次是管线自检，不含 LLM 采样，不能用于研究结论。

## 判定

**INSUFFICIENT_FOR_SCIENTIFIC_DECISION**

预注册的强证据要求：calibration 与 validation 的有效自由度都下降至少 20%，且后半程行为重访率至少 50%。

| 指标 | 数值 |
|---|---:|
| 有效子代 | 4 / 4 |
| 无效/失败率 | 0.0% |
| validation 行为噪声 95% 分位 | 0.0000 |
| 后半程行为重访率 | 0.0% |
| calibration 有效自由度 | 1.46 → 1.46 |
| validation 有效自由度 | 1.46 → 1.46 |

## 设计

- 2 条独立链 × 2 代；起点循环使用 sequential、regret-2、block-reinsert、new-route 四种 repair。
- probe：CVRP/VRPTW 各 2 个实例 × 1 个状态 × 5 种残解来源。
- 前半实例只用于 calibration，后半实例只用于 validation。
- L0：AST node-type 3-gram Jaccard；L2：固定残解上输出边集合的平均 Jaccard。
- 每个算子用两套 repair RNG 重复测量，validation 自距离的 95% 分位定义行为重访噪声线。
- 没有 fitness 选择；有效子代总是接替父代，从而把生成退化与选择退化分开。

## 解释规则

- 强证据：优先进入 verbalized sampling、双约束和缺口驱动变异实验。
- 只有部分证据：增加链数/代数，并检查 L0 变远而 L2 不变的换皮现象。
- 未检出：只能说明当前模型、prompt、预算和 repair 任务下未检出，不能证明 LLM 变异普遍不会坍缩。
- 初始有效自由度不足 2：先换更分化的种子，不解释收缩曲线。

## 复现

```powershell
python docs/operator-coevolution/experiments/mutation_collapse_audit.py --chains 2 --generations 2 --probe-instances 2 --probe-repeats 1 --customers 30 --seed 20260821 --output D:/Study/LLM4AD/docs/operator-coevolution/experiments/results/mutation-collapse/self-test
```

运行耗时 11.6 秒。原始回复和候选代码见 `chains/`，逐代指标见 `observations.csv`。
