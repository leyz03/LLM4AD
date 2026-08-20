# Stage 0 结果：信用估计器合成验证

在已知 α / β / γ 真值的合成环境里，对比协同矩阵 `S`、频率去偏的均值和 ANOVA 方差分解三类估计器。全部为控制台原始输出，未做后期编辑。

## 文件与对应的命令

脚本的随机种子是 `range(n_repeat)`，所以**给定 `--n-repeat` 和 `--selection` 结果完全确定**；不同的 `--n-repeat` 会给出不同的平均值。文件名中的 `nXX` 即该次运行的重复次数。

| 文件 | 命令 |
|---|---|
| [`n30-thompson.txt`](./n30-thompson.txt) | `--experiment all --selection thompson --n-repeat 30` |
| [`n30-roulette.txt`](./n30-roulette.txt) | `--experiment all --selection roulette --n-repeat 30` |
| [`n40-thompson.txt`](./n40-thompson.txt) | `--experiment all --selection thompson --n-repeat 40` |
| [`n40-roulette.txt`](./n40-roulette.txt) | `--experiment all --selection roulette --n-repeat 40` |
| [`n60-coldstart-roulette.txt`](./n60-coldstart-roulette.txt) | `--experiment coldstart --selection roulette --n-repeat 60` |

命令前缀统一为（在仓库根目录运行）：

```bash
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py
```

## 落地实施方案中的表格出自哪一次运行

`implementation-plan.md` 第 0 节的三张表来自不同配置，此处一一对应，便于核对：

| 方案中的表 | 出处文件 |
|---|---|
| 结果一（E1，S vs γ、argmax 命中率） | `n30-thompson.txt` 或 `n30-roulette.txt`（E1 不受 `--selection` 影响，两份相同） |
| 结果二（E2，ε–覆盖–γ 恢复权衡） | `n30-thompson.txt` |
| 结果三（E3，冷启动误杀率） | `n60-coldstart-roulette.txt` |

`n40-*.txt` 是更早保存的一批运行，数字与方案中的表相差 1–2 个百分点。保留它们是因为这个差异本身有信息量：**结论的方向在各配置下都稳定，但具体数值对 `--n-repeat` 和 `--selection` 敏感，引用时必须连同配置一起写。** 尤其是 E3 的 `F 累加` 误杀率，在 roulette 下为 93%–94%，在 thompson 下只有 63%——因为 thompson 本身的采样集中度就不同。

## 三个结论

1. **E1**：`S_ij` 与真实 γ 的 Spearman 只有 +0.41（uniform），与 α+β 却有 +0.87——它测的是主效应而非交互。下游 joint crossover 选中真正最协同配对的命中率，`argmax S` 为 10%，`argmax γ̂` 为 70%。
2. **E2**：Thompson 采样在 600 次预算下只覆盖 38% 的 payoff 矩阵，γ 无从估计；加入概率 ε 的定向探索后覆盖率与 γ 恢复同步上升（ε=0.35 时覆盖 100%、γ 恢复 +0.776、最优对命中 57%），代价是评估期平均奖励从 +2.767 降到 +1.810。**这是"利用—估计冲突"的直接证据**，也是把 IPS 从核心贡献降级为 ablation 项的原因（IPS 在本设定下无收益：+0.912 vs +0.928）。
3. **E3**：累加式 F 让 85 分位的新算子有 94% 概率被剪枝误杀，改均值降到 44%，ANOVA α̂ 降到 23%，叠加 ρ≥0.7 的先验收缩后接近归零。

完整解读见[落地实施方案](../../../implementation-plan.md)第 0 节，跨阶段汇总见[实验结果汇总](../../RESULTS.md)。
