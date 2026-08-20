"""Stage 0.5 复核：对 γ 审计结果做四项独立的稳健性检验。

直接读取 ``stage05_gamma_audit.py`` 产出的 ``raw_observations.csv``，不重跑算子。

四项检验
--------
A. **噪声偏差修正**  ——  从 cell mean 估出的 γ̂ 内含抽样噪声，故 Var(γ̂) 系统性偏大。
   随机区组 ANOVA 的无偏方差分量为 ``(MS_int - MS_err) / n_blocks``；当 F < 1 时该分量
   为负，意味着真实交互与零无法区分。回答"报告的占比是不是上界"。

B. **算子集退化诊断**  ——  γ 要求两个因子**各自有差异化响应**。若 6 个修复算子行为高度
   雷同，γ 在构造上就没有容身之处。用两两相关矩阵量化各因子的有效自由度。
   回答"零结果是问题的性质，还是算子集的性质"。

C. **置换检验**  ——  在"保留区组效应与 α/β、仅打乱残差"的零假设下重采样，得到 γ 占比的
   零分布。比 F 检验更稳健（不依赖正态与方差齐性）。同时对多种统计量口径（均值 / 尾部 /
   离散度）检验，防止"换个口径 γ 就抬头"的假阳性。

D. **功效分析**  ——  向数据注入已知大小的 γ，看当前设计能否检出。
   **这是把"我们没测到"升级成"我们能排除 γ > X%"的唯一途径。**
   没有这一步，零结果不可解释。

用法
----
    python docs/operator-coevolution/experiments/stage05_robustness_audit.py
    python docs/operator-coevolution/experiments/stage05_robustness_audit.py \
        --input docs/operator-coevolution/experiments/results/stage05/q10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "results" / "stage05" / "q20-main"

DESTROY_NAMES = (
    "random_removal", "shaw_removal", "worst_removal",
    "route_removal", "cluster_removal", "time_oriented_removal",
)
REPAIR_NAMES = (
    "greedy_insertion", "regret_2", "regret_3",
    "greedy_with_noise", "best_position_first", "sequential_insertion",
)
ND, NR = len(DESTROY_NAMES), len(REPAIR_NAMES)


# --------------------------------------------------------------------------- utils

def load_cubes(path: Path, metric: str = "relative_gain_pct") -> dict[str, np.ndarray]:
    """-> {problem: cube[block, destroy, repair]}"""
    di = {n: i for i, n in enumerate(DESTROY_NAMES)}
    ri = {n: i for i, n in enumerate(REPAIR_NAMES)}
    with (path / "raw_observations.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cubes: dict[str, np.ndarray] = {}
    for problem in sorted({r["problem"] for r in rows}):
        sub = [r for r in rows if r["problem"] == problem]
        n_blocks = 1 + max(int(r["block"]) for r in sub)
        cube = np.full((n_blocks, ND, NR), np.nan)
        for r in sub:
            cube[int(r["block"]), di[r["destroy"]], ri[r["repair"]]] = float(r[metric])
        if np.isnan(cube).any():
            raise AssertionError(f"{problem}: incomplete factorial grid")
        cubes[problem] = cube
    return cubes


def decompose(cell_mean: np.ndarray):
    mu = cell_mean.mean()
    alpha = cell_mean.mean(axis=1) - mu
    beta = cell_mean.mean(axis=0) - mu
    gamma = cell_mean - mu - alpha[:, None] - beta[None, :]
    return mu, alpha, beta, gamma


def gamma_share(cell_mean: np.ndarray) -> float:
    _, _, _, gamma = decompose(cell_mean)
    total = np.var(cell_mean)
    return float(np.var(gamma) / total) if total > 1e-15 else 0.0


def additive_fit(cube: np.ndarray) -> np.ndarray:
    """grand + block + alpha + beta，**不含** gamma —— 即 H0 下的拟合值。"""
    grand = cube.mean()
    block = cube.mean(axis=(1, 2)) - grand
    cell_mean = cube.mean(axis=0)
    mu, alpha, beta, _ = decompose(cell_mean)
    return grand + block[:, None, None] + alpha[None, :, None] + beta[None, None, :]


# ------------------------------------------------------------------- A. bias correction

def bias_correction(cube: np.ndarray) -> dict[str, float]:
    n_blocks = cube.shape[0]
    cell_mean = cube.mean(axis=0)
    _, _, _, gamma = decompose(cell_mean)

    grand = cube.mean()
    fitted_full = cube.mean(axis=(1, 2))[:, None, None] + cell_mean[None, :, :] - grand
    ss_error = float(np.square(cube - fitted_full).sum())
    ss_interaction = float(n_blocks * np.square(gamma).sum())
    df_int = (ND - 1) * (NR - 1)
    df_err = (n_blocks - 1) * (ND * NR - 1)

    ms_int = ss_interaction / df_int
    ms_err = ss_error / df_err
    var_component = (ms_int - ms_err) / n_blocks       # 无偏方差分量，可为负

    naive_var_gamma = float(np.var(gamma))
    corrected_var_gamma = max(var_component * df_int / (ND * NR), 0.0)
    denominator = float(np.var(cell_mean)) - (naive_var_gamma - corrected_var_gamma)
    corrected = corrected_var_gamma / denominator if denominator > 1e-15 else 0.0

    return {
        "f_statistic": ms_int / ms_err if ms_err > 0 else float("inf"),
        "naive_share": gamma_share(cell_mean),
        "variance_component": float(var_component),
        "corrected_share": float(corrected),
    }


# ------------------------------------------------------------ B. operator-set degeneracy

def redundancy(cube: np.ndarray) -> dict[str, dict[str, float]]:
    """两两相关越高 => 该因子的算子越雷同 => γ 越没有容身之处。"""
    out = {}
    for axis_name, matrix in (
        ("repair", cube.reshape(-1, NR)),
        ("destroy", cube.transpose(0, 2, 1).reshape(-1, ND)),
    ):
        corr = np.corrcoef(matrix.T)
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        # 有效自由度：相关矩阵特征值的参与比 (participation ratio)
        eig = np.linalg.eigvalsh(corr)
        eig = np.clip(eig, 0, None)
        eff_dof = float(eig.sum() ** 2 / np.square(eig).sum())
        out[axis_name] = {
            "mean_pairwise_corr": float(off.mean()),
            "max_pairwise_corr": float(off.max()),
            "effective_dof": eff_dof,
            "nominal_dof": float(corr.shape[0]),
        }
    return out


# ------------------------------------------------------------------ C. permutation test

STATISTICS = {
    "均值 (原口径)": lambda c: c.mean(axis=0),
    "P90 尾部": lambda c: np.percentile(c, 90, axis=0),
    "最大值": lambda c: c.max(axis=0),
    "标准差 (离散度)": lambda c: c.std(axis=0),
}


def permutation_test(cube: np.ndarray, statistic, n_perm: int, rng) -> dict[str, float]:
    """H0：无交互。保留区组效应与 α/β，只在块内打乱残差。"""
    n_blocks = cube.shape[0]
    fitted = additive_fit(cube)
    resid_flat = (cube - fitted).reshape(n_blocks, ND * NR)
    observed = gamma_share(statistic(cube))
    null = np.empty(n_perm)
    for t in range(n_perm):
        order = np.argsort(rng.random((n_blocks, ND * NR)), axis=1)
        shuffled = np.take_along_axis(resid_flat, order, axis=1).reshape(cube.shape)
        null[t] = gamma_share(statistic(fitted + shuffled))
    return {
        "observed": observed,
        "null_median": float(np.median(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((null >= observed).mean()),
    }


# --------------------------------------------------------------------- D. power analysis

def power_analysis(cube: np.ndarray, targets, n_sim: int, n_perm: int, rng) -> dict[float, float]:
    """注入已知占比的 γ，报告检出率。回答'这个设计能排除多大的 γ'。"""
    n_blocks = cube.shape[0]
    base = additive_fit(cube)
    resid_flat = (cube - base).reshape(n_blocks, ND * NR)
    cell_mean = cube.mean(axis=0)
    _, alpha, beta, _ = decompose(cell_mean)
    var_main = float(np.var(alpha[:, None] + beta[None, :]))

    power = {}
    for target in targets:
        needed = target / (1.0 - target) * var_main
        hits = 0
        for _ in range(n_sim):
            gamma = rng.normal(0, 1, (ND, NR))
            gamma -= gamma.mean(axis=0, keepdims=True)
            gamma -= gamma.mean(axis=1, keepdims=True)
            gamma *= np.sqrt(needed / np.var(gamma))
            order = np.argsort(rng.random((n_blocks, ND * NR)), axis=1)
            noise = np.take_along_axis(resid_flat, order, axis=1).reshape(cube.shape)
            sim = base + gamma[None, :, :] + noise
            result = permutation_test(sim, STATISTICS["均值 (原口径)"], n_perm, rng)
            hits += result["p_value"] < 0.05
        power[target] = hits / n_sim
    return power


# ------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n-perm", type=int, default=400)
    parser.add_argument("--power-sims", type=int, default=20)
    parser.add_argument("--power-perm", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--skip-power", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cubes = load_cubes(args.input)
    report: dict[str, dict] = {}

    for problem, cube in cubes.items():
        print("=" * 78)
        print(f"  {problem.upper()}   blocks={cube.shape[0]}  mean gain={cube.mean():.2f}%")
        print("=" * 78)
        entry: dict[str, object] = {}

        bc = bias_correction(cube)
        entry["bias_correction"] = bc
        verdict = "负 → 真实交互与零无法区分" if bc["variance_component"] < 0 else f"{bc['variance_component']:.4f}"
        print("\n  [A] 噪声偏差修正")
        print(f"      F = {bc['f_statistic']:.3f}"
              f"{'   (F<1：交互均方低于误差均方)' if bc['f_statistic'] < 1 else ''}")
        print(f"      朴素 γ 占比      {bc['naive_share']:.2%}   <- 这是**上界**")
        print(f"      无偏方差分量      {verdict}")
        print(f"      偏差修正后占比    {bc['corrected_share']:.2%}")

        red = redundancy(cube)
        entry["redundancy"] = red
        print("\n  [B] 算子集退化诊断")
        for axis_name, stats_ in red.items():
            print(f"      {axis_name:<8s} 两两相关 均值 {stats_['mean_pairwise_corr']:+.3f} "
                  f"最大 {stats_['max_pairwise_corr']:+.3f}   "
                  f"有效自由度 {stats_['effective_dof']:.2f} / {stats_['nominal_dof']:.0f}")

        entry["permutation"] = {}
        print(f"\n  [C] 置换检验  (H0=无交互, B={args.n_perm})")
        print(f"      {'统计量':<18s} {'实测':>8s} {'H0中位':>8s} {'H0 95%':>8s} {'p':>7s}")
        for name, fn in STATISTICS.items():
            res = permutation_test(cube, fn, args.n_perm, rng)
            entry["permutation"][name] = res
            mark = "  **显著" if res["p_value"] < 0.05 else ""
            print(f"      {name:<18s} {res['observed']:>7.1%} {res['null_median']:>8.1%} "
                  f"{res['null_q95']:>8.1%} {res['p_value']:>7.3f}{mark}")

        if not args.skip_power:
            targets = (0.02, 0.05, 0.10, 0.20)
            pw = power_analysis(cube, targets, args.power_sims, args.power_perm, rng)
            entry["power"] = {str(k): v for k, v in pw.items()}
            print(f"\n  [D] 功效分析  (注入已知 γ, alpha=0.05, {args.power_sims} 次模拟/档)")
            for target, value in pw.items():
                print(f"      注入 γ 占比 {target:>5.0%}  ->  检出率 {value:>5.0%}")
            detectable = [t for t, v in pw.items() if v >= 0.8]
            if detectable:
                print(f"\n      => 该设计对 γ >= {min(detectable):.0%} 的检出功效 >= 80%，")
                print(f"         因此零结果可以表述为'排除 γ > {min(detectable):.0%}'，而非'未能测到'。")

        report[problem] = entry
        print()

    out = args.input / "robustness_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
