# Stage 0：信用估计器合成验证

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
