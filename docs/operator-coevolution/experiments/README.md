# Stage 0–0.5：信用估计器合成验证与真实算子审计

该实验在已知主效应和交互效应真值的合成环境中，验证协同矩阵、方差分解、覆盖率及冷启动策略。

## 运行环境

- Python 3.9+
- NumPy
- SciPy

在仓库根目录运行：

```bash
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment all --selection thompson --n-repeat 40
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment coverage --selection roulette
```

可用实验包括 `estimators`、`coverage`、`coldstart` 和 `all`。完整参数请运行：

```bash
python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --help
```

## 已保存结果

- [`results/stage0_results_thompson.txt`](./results/stage0_results_thompson.txt)
- [`results/stage0_results_roulette.txt`](./results/stage0_results_roulette.txt)

结果文件是原始控制台输出，保留用于核对实施方案中的汇总表格。

## Stage 0.5：经典 destroy–repair 算子 γ 审计

在仓库根目录运行：

```bash
python stage05_gamma_audit.py
```

默认对 CVRP 和 VRPTW 各生成 12 个实例、每实例 8 个共同初始状态，并完整评估 6×6 算子网格。产出位于：

- `stage05_results/REPORT.md`：主实验报告与决策
- `stage05_results/raw_observations.csv`：6,912 条原始观测
- `stage05_results/summary.json`：方差分解、置信区间与检验统计量
- `stage05_results/gamma_heatmaps.png`：payoff 与 γ 热力图
- `stage05_sensitivity/`：10% / 30% 移除比例复核
