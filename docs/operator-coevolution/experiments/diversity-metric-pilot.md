# 多样性度量综合审计：实验说明

> 主线：用可信的多样性度量诊断 LLM 变异坍缩  
> 脚本：[`diversity_metric_audit.py`](./diversity_metric_audit.py)  
> 默认结果：`results/diversity-metrics/pilot/`

## 要比较什么

| 度量 | 层 | 实现 | 成本 |
|---|---|---|---|
| token 3-gram Jaccard | L0 | 归一化局部变量/常数后的代码 token | 极低 |
| AST node-type 3-gram Jaccard | L0 | 忽略变量命名的结构 shingles | 极低 |
| Qwen mechanism descriptor + TF-IDF | L1 proxy | 三次机制描述拼接后向量化 | 低 |
| Qwen direct pair judgment | L1+ | 三次打乱顺序的 0–100 机制差异矩阵 | 低 |
| probe output edge Jaccard | L2 | calibration 残解上输出边集合差异 | 中 |
| performance footprint | L3 pilot | calibration 残解上收益秩相关距离 | 中 |

运行环境当前没有 torch/transformers，因此本轮不把 CodeBERT/GraphCodeBERT 纳入，也不把 descriptor TF-IDF 冒充 embedding。若 L0 与 L2 的结论值得继续，再单独安装和缓存代码模型完成 L1 embedding arm。

## 如何避免循环论证

- 前半实例为 calibration：构造 L2/L3 候选距离；
- 后半实例为 validation：构造 held-out output behavior 和 held-out performance footprint；
- 同一算子使用第二套 RNG 重复测量，检验目标和候选距离的稳定性；
- Qwen 判断只看代码，不看 probe 结果；
- 统计量使用 Mantel 标签置换，避免把算子对误当独立样本。

## 压力测试算子集

- Stage 0.5 的 6 个经典 repair；
- 与 sequential 行为等价但代码重写的阴性对照；
- block-reinsert 和 new-route 两个行为阳性对照；
- mutation pilot 每条链最后一个合法 Qwen 候选（若存在）。

这套构造能同时检验：代码很远但行为相同的“假多样性”、代码较近但行为不同的“隐藏多样性”，以及指标对 LLM 代码的泛化。

## 判读顺序

1. 先看 held-out target 的重复测量信度；目标本身不稳定时不排名度量。
2. 再看候选度量与 held-out L2/L3 的 Spearman 和 Mantel p。
3. 再看最近邻一致率，判断它是否足以用于去重/局部竞争。
4. 最后看有效自由度；它只说明某表示把种群展开成多少维，不说明这些维度有用。

建议进入协同进化的门槛：重测信度 `≥ 0.8`、held-out 相关 `≥ 0.5` 且 Mantel `p < 0.05`。未过门槛的指标只能用于分析图，不能直接改变存活。

## 运行

先做零 LLM CPU 检查：

```powershell
python docs/operator-coevolution/experiments/diversity_metric_audit.py `
  --skip-llm --exclude-mutations --probe-instances 2 --probe-repeats 1 --customers 30 `
  --permutations 200 --output docs/operator-coevolution/experiments/results/diversity-metrics/self-test
```

正式运行（默认从 `.env` 读取 Qwen 3.5 Flash，3 次判断）：

```powershell
python docs/operator-coevolution/experiments/diversity_metric_audit.py
```

中断后加 `--resume`，保存过的 Qwen 响应会被复用。

