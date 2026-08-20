# Stage 0 / 0.5 实验：运行说明

两个实验都是零 LLM 调用的纯 CPU 实验，可独立复现。**结果与结论请看 [实验结果汇总](./RESULTS.md)**；本页只讲怎么跑。

## 目录结构

```
experiments/
├── RESULTS.md                              两阶段结果汇总（先读这个）
├── stage0_credit_estimator_validation.py   Stage 0：合成环境信用估计器验证
├── stage05_gamma_audit.py                  Stage 0.5：经典算子 γ 审计
└── results/
    ├── stage0/     README + 5 份原始控制台输出（按 n-repeat 与 selection 命名）
    └── stage05/    README + q20-main / q10 / q30 三次运行的完整产物
```

## 运行环境

- Python 3.9+
- NumPy、SciPy（Stage 0.5 另需 matplotlib）

以下命令均**在仓库根目录**运行。

## Stage 0：信用估计器合成验证

在已知 α / β / γ 真值的合成环境中，对比协同矩阵、方差分解、覆盖率及冷启动策略。

```bash
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment all --selection thompson --n-repeat 30
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment coldstart --selection roulette --n-repeat 60
```

可用实验：`estimators`（E1）、`coverage`（E2）、`coldstart`（E3）、`all`。完整参数见 `--help`。

**复现口径**：脚本种子为 `range(n_repeat)`，所以给定 `--n-repeat` 和 `--selection` 后结果完全确定，但换了这两个参数数值会变（方向不变）。上面两条命令分别复现落地实施方案中的 E1/E2 表和 E3 表。已保存的原始输出及其对应命令见 [`results/stage0/README.md`](./results/stage0/README.md)。

## Stage 0.5：经典 destroy–repair 算子 γ 审计

6 个破坏算子 × 6 个修复算子的完整均衡因子设计，在 CVRP 与 VRPTW 上测量交互项占 payoff 系统性方差的比例。

主实验（默认参数，12 实例 × 8 初始状态 × 36 配对 × 2 问题 = 6,912 次评估，约 68 秒）：

```powershell
python docs/operator-coevolution/experiments/stage05_gamma_audit.py
```

邻域大小敏感性复核：

```powershell
python docs/operator-coevolution/experiments/stage05_gamma_audit.py --instances 8 --repeats 6 --remove-fraction 0.1 --bootstrap 1000 --output docs/operator-coevolution/experiments/results/stage05/q10
python docs/operator-coevolution/experiments/stage05_gamma_audit.py --instances 8 --repeats 6 --remove-fraction 0.3 --bootstrap 1000 --output docs/operator-coevolution/experiments/results/stage05/q30
```

`--output` 默认为脚本旁的 `results/stage05/q20-main`，与运行时的工作目录无关。每次运行产出四个文件：

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 该次运行的完整报告：决策、最强/最弱交互格、解释边界 |
| `summary.json` | cell mean、α/β/γ 分解、聚类 bootstrap CI、检验统计量 |
| `raw_observations.csv` | 逐次 destroy–repair 的原始观测 |
| `gamma_heatmaps.png` | payoff 与 γ 热力图 |

已保存的三次运行见 [`results/stage05/README.md`](./results/stage05/README.md)。
