"""Stage 0.5: audit destroy/repair interaction effects on CVRP and VRPTW.

The experiment is deliberately LLM-free.  Every (destroy, repair) pair is
applied to exactly the same collection of feasible base solutions (complete
block design), so differences between cells cannot be caused by an easier
instance or a luckier starting solution.  The primary statistic follows the
definition used by ``stage0_credit_estimator_validation.py``::

    M[d, r] = mu + alpha[d] + beta[r] + gamma[d, r]
    gamma_share = Var(gamma) / Var(M)

Raw observations, summaries, figures, and a Markdown report are written to
``stage05_results`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DESTROY_NAMES = (
    "random_removal",
    "shaw_removal",
    "worst_removal",
    "route_removal",
    "cluster_removal",
    "time_oriented_removal",
)
REPAIR_NAMES = (
    "greedy_insertion",
    "regret_2",
    "regret_3",
    "greedy_with_noise",
    "best_position_first",
    "sequential_insertion",
)


@dataclass(frozen=True)
class Problem:
    name: str
    coordinates: np.ndarray
    distance: np.ndarray
    demand: np.ndarray
    capacity: int
    service: np.ndarray | None = None
    windows: np.ndarray | None = None

    @property
    def n_customers(self) -> int:
        return len(self.demand) - 1


Routes = list[list[int]]
DestroyFn = Callable[[Problem, Routes, np.random.Generator, int], tuple[Routes, list[int]]]
RepairFn = Callable[[Problem, Routes, list[int], np.random.Generator], Routes]


def clone_routes(routes: Routes) -> Routes:
    return [route.copy() for route in routes]


def route_distance(problem: Problem, route: Sequence[int]) -> float:
    if not route:
        return 0.0
    nodes = (0, *route, 0)
    return float(sum(problem.distance[a, b] for a, b in zip(nodes, nodes[1:])))


def solution_distance(problem: Problem, routes: Routes) -> float:
    return float(sum(route_distance(problem, route) for route in routes))


def route_feasible(problem: Problem, route: Sequence[int]) -> bool:
    if sum(int(problem.demand[c]) for c in route) > problem.capacity:
        return False
    if problem.windows is None:
        return True
    assert problem.service is not None
    clock = 0.0
    previous = 0
    for customer in route:
        clock += float(problem.distance[previous, customer])
        clock = max(clock, float(problem.windows[customer, 0]))
        if clock > float(problem.windows[customer, 1]) + 1e-12:
            return False
        clock += float(problem.service[customer])
        previous = customer
    return clock + float(problem.distance[previous, 0]) <= float(problem.windows[0, 1]) + 1e-12


def validate_solution(problem: Problem, routes: Routes) -> None:
    customers = [c for route in routes for c in route]
    expected = list(range(1, problem.n_customers + 1))
    if sorted(customers) != expected:
        raise AssertionError("solution must contain every customer exactly once")
    if not all(route and route_feasible(problem, route) for route in routes):
        raise AssertionError("solution contains an empty or infeasible route")


def insertion_candidates(problem: Problem, routes: Routes, customer: int) -> list[tuple[float, int, int]]:
    """Return (distance delta, route index, position); route index -1 means new route."""
    candidates: list[tuple[float, int, int]] = []
    for route_index, route in enumerate(routes):
        for position in range(len(route) + 1):
            candidate_route = route[:position] + [customer] + route[position:]
            if route_feasible(problem, candidate_route):
                before = route_distance(problem, route)
                after = route_distance(problem, candidate_route)
                candidates.append((after - before, route_index, position))
    singleton = [customer]
    if route_feasible(problem, singleton):
        candidates.append((route_distance(problem, singleton), -1, 0))
    if not candidates:
        raise RuntimeError(f"customer {customer} has no feasible insertion")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates


def apply_insertion(routes: Routes, customer: int, candidate: tuple[float, int, int]) -> None:
    _, route_index, position = candidate
    if route_index == -1:
        routes.append([customer])
    else:
        routes[route_index].insert(position, customer)


def remove_customers(routes: Routes, selected: Iterable[int]) -> Routes:
    removed = set(selected)
    return [[c for c in route if c not in removed] for route in routes if any(c not in removed for c in route)]


def customer_locations(routes: Routes) -> dict[int, tuple[int, int]]:
    return {customer: (ri, pi) for ri, route in enumerate(routes) for pi, customer in enumerate(route)}


def random_removal(problem: Problem, routes: Routes, rng: np.random.Generator, q: int):
    del problem
    customers = np.asarray([c for route in routes for c in route], dtype=int)
    removed = rng.choice(customers, size=min(q, len(customers)), replace=False).tolist()
    return remove_customers(routes, removed), removed


def shaw_removal(problem: Problem, routes: Routes, rng: np.random.Generator, q: int):
    customers = [c for route in routes for c in route]
    locations = customer_locations(routes)
    seed = int(rng.choice(customers))
    max_distance = max(float(problem.distance.max()), 1e-12)
    max_demand = max(float(problem.demand.max()), 1.0)
    if problem.windows is not None:
        mid = problem.windows.mean(axis=1)
        max_time = max(float(problem.windows[0, 1]), 1e-12)
    else:
        mid = np.zeros(len(problem.demand))
        max_time = 1.0
    related = []
    for customer in customers:
        same_route = locations[customer][0] == locations[seed][0]
        value = (
            1.0 * problem.distance[seed, customer] / max_distance
            + 0.25 * abs(problem.demand[seed] - problem.demand[customer]) / max_demand
            + 0.75 * abs(mid[seed] - mid[customer]) / max_time
            + (0.0 if same_route else 0.35)
        )
        related.append((float(value), customer))
    related.sort()
    pool = related[: max(q * 2, q)]
    # Biased sampling prevents Shaw from being a deterministic clone of clustering.
    removed = [seed]
    remaining = [c for _, c in pool if c != seed]
    while remaining and len(removed) < q:
        rank = min(int((rng.random() ** 2.5) * len(remaining)), len(remaining) - 1)
        removed.append(remaining.pop(rank))
    if len(removed) < q:
        extras = [c for _, c in related if c not in removed]
        removed.extend(extras[: q - len(removed)])
    return remove_customers(routes, removed), removed


def worst_removal(problem: Problem, routes: Routes, rng: np.random.Generator, q: int):
    savings: list[tuple[float, int]] = []
    for route in routes:
        nodes = [0, *route, 0]
        for position, customer in enumerate(route, start=1):
            before = problem.distance[nodes[position - 1], customer] + problem.distance[customer, nodes[position + 1]]
            after = problem.distance[nodes[position - 1], nodes[position + 1]]
            savings.append((float(before - after), customer))
    savings.sort(reverse=True)
    remaining = savings.copy()
    removed: list[int] = []
    while remaining and len(removed) < q:
        rank = min(int((rng.random() ** 3.0) * len(remaining)), len(remaining) - 1)
        removed.append(remaining.pop(rank)[1])
    return remove_customers(routes, removed), removed


def route_removal(problem: Problem, routes: Routes, rng: np.random.Generator, q: int):
    del problem
    order = rng.permutation(len(routes)).tolist()
    removed: list[int] = []
    for route_index in order:
        candidates = routes[route_index].copy()
        rng.shuffle(candidates)
        removed.extend(candidates[: q - len(removed)])
        if len(removed) >= q:
            break
    return remove_customers(routes, removed), removed


def cluster_removal(problem: Problem, routes: Routes, rng: np.random.Generator, q: int):
    customers = [c for route in routes for c in route]
    seed = int(rng.choice(customers))
    ranked = sorted(customers, key=lambda c: (problem.distance[seed, c], c))
    removed = ranked[:q]
    return remove_customers(routes, removed), removed


def service_start_times(problem: Problem, routes: Routes) -> dict[int, float]:
    values: dict[int, float] = {}
    for route in routes:
        clock = 0.0
        previous = 0
        scale = max(route_distance(problem, route), 1e-12)
        for position, customer in enumerate(route):
            clock += float(problem.distance[previous, customer])
            if problem.windows is not None:
                clock = max(clock, float(problem.windows[customer, 0]))
                values[customer] = clock / max(float(problem.windows[0, 1]), 1e-12)
                assert problem.service is not None
                clock += float(problem.service[customer])
            else:
                # Route-progress is the CVRP analogue; there is no physical time window.
                values[customer] = (clock / scale + position / max(len(route), 1)) / 2.0
            previous = customer
    return values


def time_oriented_removal(problem: Problem, routes: Routes, rng: np.random.Generator, q: int):
    starts = service_start_times(problem, routes)
    seed = int(rng.choice(list(starts)))
    ranked = sorted(starts, key=lambda c: (abs(starts[c] - starts[seed]), problem.distance[seed, c], c))
    removed = ranked[:q]
    return remove_customers(routes, removed), removed


DESTROYERS: dict[str, DestroyFn] = {
    "random_removal": random_removal,
    "shaw_removal": shaw_removal,
    "worst_removal": worst_removal,
    "route_removal": route_removal,
    "cluster_removal": cluster_removal,
    "time_oriented_removal": time_oriented_removal,
}


def sequential_insertion(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator):
    del rng
    result = clone_routes(routes)
    for customer in removed:
        apply_insertion(result, customer, insertion_candidates(problem, result, customer)[0])
    return result


def greedy_insertion(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator):
    order = removed.copy()
    rng.shuffle(order)
    return sequential_insertion(problem, routes, order, rng)


def regret_insertion(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator, k: int):
    del rng
    result = clone_routes(routes)
    pending = removed.copy()
    while pending:
        choices = []
        for customer in pending:
            candidates = insertion_candidates(problem, result, customer)
            kth = candidates[min(k - 1, len(candidates) - 1)][0]
            # A unique feasible location is urgent, hence the small-candidate bonus.
            regret = kth - candidates[0][0] + (k - len(candidates)) * 1e3 if len(candidates) < k else kth - candidates[0][0]
            choices.append((float(regret), -candidates[0][0], -customer, customer, candidates[0]))
        *_, customer, candidate = max(choices)
        apply_insertion(result, customer, candidate)
        pending.remove(customer)
    return result


def regret_2(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator):
    return regret_insertion(problem, routes, removed, rng, 2)


def regret_3(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator):
    return regret_insertion(problem, routes, removed, rng, 3)


def greedy_with_noise(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator):
    result = clone_routes(routes)
    pending = removed.copy()
    noise_scale = 0.20 * float(np.mean(problem.distance))
    while pending:
        choices = []
        for customer in pending:
            for candidate in insertion_candidates(problem, result, customer):
                noisy_delta = candidate[0] + rng.uniform(-noise_scale, noise_scale)
                choices.append((float(noisy_delta), customer, candidate))
        _, customer, candidate = min(choices, key=lambda item: item[0])
        apply_insertion(result, customer, candidate)
        pending.remove(customer)
    return result


def best_position_first(problem: Problem, routes: Routes, removed: list[int], rng: np.random.Generator):
    del rng
    result = clone_routes(routes)
    pending = removed.copy()
    while pending:
        choices = [(insertion_candidates(problem, result, c)[0][0], c, insertion_candidates(problem, result, c)[0]) for c in pending]
        _, customer, candidate = min(choices, key=lambda item: (item[0], item[1]))
        apply_insertion(result, customer, candidate)
        pending.remove(customer)
    return result


REPAIRERS: dict[str, RepairFn] = {
    "greedy_insertion": greedy_insertion,
    "regret_2": regret_2,
    "regret_3": regret_3,
    "greedy_with_noise": greedy_with_noise,
    "best_position_first": best_position_first,
    "sequential_insertion": sequential_insertion,
}


def generate_cvrp_instances(n_instances: int, n_customers: int, seed: int) -> list[Problem]:
    rng = np.random.default_rng(seed)
    result = []
    for index in range(n_instances):
        coordinates = rng.random((n_customers + 1, 2))
        demand = np.r_[0, rng.integers(1, 10, n_customers)]
        distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        result.append(Problem(f"cvrp_{index:02d}", coordinates, distance, demand, 40))
    return result


def generate_vrptw_instances(n_instances: int, n_customers: int, seed: int) -> list[Problem]:
    """Use the same distributions as LLM4AD's vrptw_construct generator."""
    rng = np.random.default_rng(seed)
    result = []
    horizon = 4.6
    for index in range(n_instances):
        coordinates = rng.random((n_customers + 1, 2))
        demand = np.r_[0, rng.integers(1, 10, n_customers)]
        distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        service = np.r_[0.0, rng.random(n_customers) * 0.05 + 0.15]
        lengths = rng.random(n_customers) * 0.05 + 0.15
        depot_distance = distance[0, 1:]
        upper_factor = np.maximum((horizon - service[1:] - lengths) / np.maximum(depot_distance, 1e-12) - 1.0, 1.0)
        factors = rng.random(n_customers) * (upper_factor - 1.0) + 1.0
        early = factors * depot_distance
        windows = np.vstack(([0.0, horizon], np.column_stack((early, early + lengths))))
        result.append(Problem(f"vrptw_{index:02d}", coordinates, distance, demand, 40, service, windows))
    return result


def construct_base_solution(problem: Problem, rng: np.random.Generator) -> Routes:
    """Randomized feasible insertion construction, shared by all 36 cells in a block."""
    customers = list(range(1, problem.n_customers + 1))
    if problem.windows is None:
        rng.shuffle(customers)
    else:
        # Mostly respect time-window order while retaining state-to-state diversity.
        assert problem.windows is not None
        jitter = rng.normal(0.0, 0.20, len(customers))
        customers.sort(key=lambda c: float(problem.windows[c, 0] + jitter[c - 1]))
    routes: Routes = []
    for customer in customers:
        candidates = insertion_candidates(problem, routes, customer)
        restricted = candidates[: min(3, len(candidates))]
        weights = np.exp(-np.arange(len(restricted), dtype=float))
        choice = restricted[int(rng.choice(len(restricted), p=weights / weights.sum()))]
        apply_insertion(routes, customer, choice)
    validate_solution(problem, routes)
    return routes


def seed_for(base_seed: int, problem_index: int, repeat: int, destroy_index: int, repair_index: int) -> int:
    seq = np.random.SeedSequence([base_seed, problem_index, repeat, destroy_index, repair_index])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def run_problem(
    problem_type: str,
    problems: list[Problem],
    repeats: int,
    remove_fraction: float,
    base_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for problem_index, problem in enumerate(problems):
        for repeat in range(repeats):
            base_rng = np.random.default_rng(seed_for(base_seed + 17, problem_index, repeat, 99, 99))
            base = construct_base_solution(problem, base_rng)
            before = solution_distance(problem, base)
            q = max(2, min(problem.n_customers - 1, int(round(problem.n_customers * remove_fraction))))
            block = problem_index * repeats + repeat
            for destroy_index, destroy_name in enumerate(DESTROY_NAMES):
                for repair_index, repair_name in enumerate(REPAIR_NAMES):
                    pair_seed = seed_for(base_seed, problem_index, repeat, destroy_index, repair_index)
                    # Common random numbers: for a given block and destroyer, every
                    # repairer receives exactly the same partial solution.  Repair
                    # randomness is then isolated in a separate, reproducible stream.
                    destroy_seed = seed_for(base_seed, problem_index, repeat, destroy_index, 10_000)
                    repair_seed = seed_for(base_seed, problem_index, repeat, 20_000, repair_index)
                    destroy_rng = np.random.default_rng(destroy_seed)
                    repair_rng = np.random.default_rng(repair_seed)
                    partial, removed = DESTROYERS[destroy_name](problem, clone_routes(base), destroy_rng, q)
                    candidate = REPAIRERS[repair_name](problem, partial, removed, repair_rng)
                    validate_solution(problem, candidate)
                    after = solution_distance(problem, candidate)
                    rows.append(
                        {
                            "problem": problem_type,
                            "instance": problem_index,
                            "repeat": repeat,
                            "block": block,
                            "destroy": destroy_name,
                            "repair": repair_name,
                            "seed": pair_seed,
                            "n_remove": q,
                            "routes_before": len(base),
                            "routes_after": len(candidate),
                            "cost_before": before,
                            "cost_after": after,
                            "gain": before - after,
                            "relative_gain_pct": 100.0 * (before - after) / before,
                        }
                    )
    return rows


def rows_to_cube(rows: list[dict[str, object]], metric: str = "relative_gain_pct") -> np.ndarray:
    n_blocks = 1 + max(int(row["block"]) for row in rows)
    cube = np.full((n_blocks, len(DESTROY_NAMES), len(REPAIR_NAMES)), np.nan)
    di = {name: i for i, name in enumerate(DESTROY_NAMES)}
    ri = {name: i for i, name in enumerate(REPAIR_NAMES)}
    for row in rows:
        cube[int(row["block"]), di[str(row["destroy"])], ri[str(row["repair"])]] = float(row[metric])
    if np.isnan(cube).any():
        raise AssertionError("incomplete factorial grid")
    return cube


def decompose(cell_mean: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    mu = float(cell_mean.mean())
    alpha = cell_mean.mean(axis=1) - mu
    beta = cell_mean.mean(axis=0) - mu
    gamma = cell_mean - mu - alpha[:, None] - beta[None, :]
    return mu, alpha, beta, gamma


def interaction_statistics(cube: np.ndarray) -> dict[str, object]:
    blocks, nd, nr = cube.shape
    cell_mean = cube.mean(axis=0)
    mu, alpha, beta, gamma = decompose(cell_mean)
    model_variance = float(np.var(cell_mean))
    gamma_share = float(np.var(gamma) / model_variance) if model_variance > 1e-15 else 0.0

    grand = float(cube.mean())
    block_mean = cube.mean(axis=(1, 2))
    fitted_full = block_mean[:, None, None] + cell_mean[None, :, :] - grand
    ss_error = float(np.square(cube - fitted_full).sum())
    ss_interaction = float(blocks * np.square(gamma).sum())
    df_interaction = (nd - 1) * (nr - 1)
    df_error = (blocks - 1) * (nd * nr - 1)
    f_stat = (ss_interaction / df_interaction) / (ss_error / df_error) if ss_error > 0 else math.inf
    p_value = float(stats.f.sf(f_stat, df_interaction, df_error))
    partial_eta2 = ss_interaction / (ss_interaction + ss_error) if ss_interaction + ss_error > 0 else 0.0
    return {
        "n_blocks": blocks,
        "cell_mean": cell_mean,
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "gamma_variance_share": gamma_share,
        "partial_eta_squared": float(partial_eta2),
        "f_statistic": float(f_stat),
        "df_interaction": df_interaction,
        "df_error": df_error,
        "p_value": p_value,
    }


def cluster_bootstrap_ci(cube: np.ndarray, repeats: int, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_instances = cube.shape[0] // repeats
    grouped = cube.reshape(n_instances, repeats, len(DESTROY_NAMES), len(REPAIR_NAMES))
    estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        instance_indices = rng.integers(0, n_instances, n_instances)
        sampled_groups = []
        for index in instance_indices:
            repeat_indices = rng.integers(0, repeats, repeats)
            sampled_groups.append(grouped[index, repeat_indices])
        sampled = np.concatenate(sampled_groups, axis=0)
        estimates[b] = float(interaction_statistics(sampled)["gamma_variance_share"])
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def classification(share: float) -> tuple[str, str]:
    if share > 0.25:
        return "强（>25%）", "交互是主要信号，建议全力推进以 γ 为核心的研究路线。"
    if share >= 0.10:
        return "中等（10%–25%）", "课题可做，但应保留多问题验证并控制结论边界。"
    return "弱（<10%）", "交互叙事证据不足，建议转向时序信用分配或更强耦合测试床。"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(path: Path, summaries: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, len(summaries), figsize=(6.4 * len(summaries), 10.5), constrained_layout=True)
    if len(summaries) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (problem, summary) in enumerate(summaries.items()):
        for row_index, (key, title, cmap) in enumerate(
            (("cell_mean", "Mean relative gain (%)", "RdYlGn"), ("gamma", "Interaction effect gamma (pp)", "coolwarm"))
        ):
            matrix = np.asarray(summary[key])
            vmax = float(np.max(np.abs(matrix))) if key == "gamma" else None
            vmin = -vmax if key == "gamma" else None
            image = axes[row_index, column].imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axes[row_index, column].set_title(f"{problem.upper()}: {title}")
            axes[row_index, column].set_xticks(range(len(REPAIR_NAMES)), [name.replace("_", "\n") for name in REPAIR_NAMES], fontsize=8)
            axes[row_index, column].set_yticks(range(len(DESTROY_NAMES)), [name.replace("_", " ") for name in DESTROY_NAMES], fontsize=8)
            axes[row_index, column].set_xlabel("repair")
            axes[row_index, column].set_ylabel("destroy")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axes[row_index, column].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=axes[row_index, column], shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def serializable_summary(summary: dict[str, object], ci: tuple[float, float]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in summary.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    result["gamma_share_ci95"] = list(ci)
    return result


def write_report(
    path: Path,
    args: argparse.Namespace,
    summaries: dict[str, dict[str, object]],
    cis: dict[str, tuple[float, float]],
    runtime: float,
) -> None:
    decisions = {problem: classification(float(summary["gamma_variance_share"])) for problem, summary in summaries.items()}
    lines = [
        "# Stage 0.5：经典算子 γ 审计结果",
        "",
        "## 实验决定",
        "",
        "采用 **6 个破坏算子 × 6 个修复算子的完整均衡因子设计**。每个 `(实例, 重复)` 是一个区组：36 个算子对共享同一初始可行解，并使用确定性派生随机种子。这样实例难度和初始解质量不会混入算子对效应。配对为 uniform（每格样本数完全相同），全程零 LLM 调用。",
        "",
        f"- 问题：CVRP、VRPTW；每类 {args.instances} 个独立实例，每实例 {args.repeats} 个初始状态",
        f"- 每格样本：{args.instances * args.repeats}；每问题总观测：{args.instances * args.repeats * 36}",
        f"- 客户数：{args.customers}；每次移除比例：{args.remove_fraction:.0%}",
        "- payoff：一次 destroy–repair 后的相对距离改善百分比（允许负值）",
        "- 主指标：`Var(γ) / Var(M)`，与 Stage 0 脚本定义一致；95% CI 对实例做聚类 bootstrap",
        "- 辅助检验：随机区组二因素 ANOVA 的交互 F 检验及 partial η²（包含区组×处理噪声）",
        "",
        "## 核心结果",
        "",
        "| 问题 | γ 方差占比 | 95% 聚类 CI | partial η² | F(df) | p | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for problem, summary in summaries.items():
        ci = cis[problem]
        label, _ = decisions[problem]
        p = float(summary["p_value"])
        p_text = f"{p:.3g}" if p >= 1e-4 else "<1e-4"
        lines.append(
            f"| {problem.upper()} | **{float(summary['gamma_variance_share']):.1%}** | "
            f"[{ci[0]:.1%}, {ci[1]:.1%}] | {float(summary['partial_eta_squared']):.1%} | "
            f"{float(summary['f_statistic']):.2f} ({summary['df_interaction']}, {summary['df_error']}) | {p_text} | {label} |"
        )
    lines.extend(
        [
            "",
            "系统性 cell-mean 方差的来源（3 项按构造合计为 100%）：",
            "",
            "| 问题 | destroy 主效应 α | repair 主效应 β | 交互 γ |",
            "|---|---:|---:|---:|",
        ]
    )
    for problem, summary in summaries.items():
        cell_mean = np.asarray(summary["cell_mean"])
        denominator = float(np.var(cell_mean))
        alpha_share = float(np.var(np.asarray(summary["alpha"])) / denominator)
        beta_share = float(np.var(np.asarray(summary["beta"])) / denominator)
        gamma_share = float(summary["gamma_variance_share"])
        lines.append(f"| {problem.upper()} | {alpha_share:.1%} | {beta_share:.1%} | {gamma_share:.1%} |")
    lines.extend(["", "## 决策", ""])
    for problem, (_, action) in decisions.items():
        lines.append(f"- **{problem.upper()}**：{action}")
    cvrp = float(summaries.get("cvrp", {}).get("gamma_variance_share", 0.0))
    vrptw = float(summaries.get("vrptw", {}).get("gamma_variance_share", 0.0))
    if "cvrp" in summaries and "vrptw" in summaries:
        delta = vrptw - cvrp
        if delta > 0.03:
            comparison = f"VRPTW 比 CVRP 高 {delta:.1%}，支持“约束增强会提高算子耦合”的方向性假设。"
        elif delta < -0.03:
            comparison = f"VRPTW 比 CVRP 低 {-delta:.1%}，不支持原先关于时间窗必然增强 γ 的预期。"
        else:
            comparison = f"两者仅相差 {abs(delta):.1%}，没有实质证据表明时间窗提高了 γ。"
        lines.append(f"- **跨问题比较**：{comparison}")
    lines.extend(
        [
            "",
            "判定应以占比及其区间为主，而不是只看 p 值：样本多时很小的交互也可能显著。",
            "",
            "## 最强与最弱的交互格",
            "",
        ]
    )
    for problem, summary in summaries.items():
        gamma = np.asarray(summary["gamma"])
        flat_order = np.argsort(gamma, axis=None)
        lines.append(f"### {problem.upper()}")
        lines.append("")
        lines.append("最正协同：")
        lines.append("")
        for flat in flat_order[-3:][::-1]:
            i, j = np.unravel_index(flat, gamma.shape)
            lines.append(f"- `{DESTROY_NAMES[i]} × {REPAIR_NAMES[j]}`：γ = {gamma[i, j]:+.3f} 个百分点")
        lines.append("")
        lines.append("最负协同：")
        lines.append("")
        for flat in flat_order[:3]:
            i, j = np.unravel_index(flat, gamma.shape)
            lines.append(f"- `{DESTROY_NAMES[i]} × {REPAIR_NAMES[j]}`：γ = {gamma[i, j]:+.3f} 个百分点")
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "1. 这是**经典算子的直接一步 payoff 审计**，回答“配对本身是否产生不可加的即时效果”；它不等同于完整 ALNS 长轨迹的最终性能。",
            "2. CVRP 没有时间窗，因此 `time_oriented_removal` 在 CVRP 中按路线归一化进度定义；VRPTW 使用实际服务开始时间。",
            "3. 主指标只分解 36 个 cell mean 的系统性差异，不把重复间噪声放进分母；报告的 partial η² 则把该噪声纳入，二者回答不同问题。",
            "4. 合成实例沿用 LLM4AD constructive task 的分布而非 Solomon/CVRPLIB 标准实例，因此当前结论是 Stage 1 投资决策证据，不应直接作为最终 benchmark 结论。",
            "",
            "## 复现",
            "",
            "```powershell",
            f"python stage05_gamma_audit.py --instances {args.instances} --repeats {args.repeats} --customers {args.customers} --remove-fraction {args.remove_fraction} --bootstrap {args.bootstrap} --seed {args.seed}",
            "```",
            "",
            f"本次运行耗时 {runtime:.1f} 秒。原始数据见 `raw_observations.csv`，完整数值见 `summary.json`，热力图见 `gamma_heatmaps.png`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("all", "cvrp", "vrptw"), default="all")
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--remove-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240820)
    parser.add_argument("--output", type=Path, default=Path("stage05_results"))
    args = parser.parse_args()
    if args.instances < 2 or args.repeats < 2 or args.customers < 10 or not 0 < args.remove_fraction < 1:
        parser.error("need instances>=2, repeats>=2, customers>=10, and 0<remove-fraction<1")
    if args.bootstrap < 100:
        parser.error("bootstrap must be at least 100")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = ("cvrp", "vrptw") if args.problem == "all" else (args.problem,)
    all_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    cis: dict[str, tuple[float, float]] = {}

    for offset, problem_type in enumerate(selected):
        if problem_type == "cvrp":
            problems = generate_cvrp_instances(args.instances, args.customers, args.seed + 1000)
        else:
            problems = generate_vrptw_instances(args.instances, args.customers, args.seed + 2000)
        print(f"Running {problem_type.upper()}: {args.instances} instances x {args.repeats} states x 36 pairs", flush=True)
        rows = run_problem(problem_type, problems, args.repeats, args.remove_fraction, args.seed + offset * 100_000)
        cube = rows_to_cube(rows)
        summary = interaction_statistics(cube)
        ci = cluster_bootstrap_ci(cube, args.repeats, args.bootstrap, args.seed + 9000 + offset)
        all_rows.extend(rows)
        summaries[problem_type] = summary
        cis[problem_type] = ci
        print(
            f"  gamma share={float(summary['gamma_variance_share']):.1%} "
            f"CI=[{ci[0]:.1%}, {ci[1]:.1%}], p={float(summary['p_value']):.3g}",
            flush=True,
        )

    write_csv(args.output / "raw_observations.csv", all_rows)
    serial = {problem: serializable_summary(summary, cis[problem]) for problem, summary in summaries.items()}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "summary.json").write_text(
        json.dumps({"config": config, "results": serial}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_heatmaps(args.output / "gamma_heatmaps.png", summaries)
    runtime = time.perf_counter() - started
    write_report(args.output / "REPORT.md", args, summaries, cis, runtime)
    print(f"Wrote results to {args.output.resolve()} ({runtime:.1f}s)")


if __name__ == "__main__":
    main()
