"""Compare diversity metrics for repair operators on disjoint probe sets.

The audit compares cheap code-space distances, Qwen semantic judgments, probe
output behavior, and performance footprints.  Calibration instances construct
the candidate metrics; validation instances provide independent held-out
targets.  It can include final valid endpoints from mutation_collapse_audit.py.

Run from the repository root after (or during) the mutation pilot::

    python docs/operator-coevolution/experiments/diversity_metric_audit.py

Use --skip-llm for a zero-API-call CPU-only run.  Raw Qwen judgments are saved
and reused by --resume.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import keyword
import math
import re
import time
import tokenize
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import mutation_collapse_audit as mutation
import stage05_gamma_audit as stage05


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "diversity-metrics" / "pilot"
DEFAULT_MUTATION_RESULTS = SCRIPT_DIR / "results" / "mutation-collapse" / "pilot"
SCRIPT_INVOCATION = "docs/operator-coevolution/experiments/diversity_metric_audit.py"


SEQUENTIAL_REWRITE = """def repair_operator(problem, routes, removed, rng):
    del rng
    result = clone_routes(routes)
    pending = tuple(removed)
    index = 0
    while index < len(pending):
        customer = pending[index]
        options = insertion_candidates(problem, result, customer)
        apply_insertion(result, customer, min(options, key=lambda item: item[0]))
        index += 1
    return result
"""


def normalized_source(function: Any) -> str:
    return mutation.extract_and_validate_code(mutation._source_as_repair(function))


def collect_operators(mutation_results: Path, include_mutations: bool) -> dict[str, str]:
    operators: dict[str, str] = {}
    for name in stage05.REPAIR_NAMES:
        operators[f"classic:{name}"] = normalized_source(stage05.REPAIRERS[name])
    operators["control:sequential_rewrite"] = mutation.extract_and_validate_code(SEQUENTIAL_REWRITE)
    operators["specialist:block_reinsert"] = mutation.extract_and_validate_code(mutation.BLOCK_REPAIR_SOURCE)
    operators["specialist:new_route"] = mutation.extract_and_validate_code(mutation.NEW_ROUTE_REPAIR_SOURCE)

    if include_mutations and mutation_results.exists():
        chains_dir = mutation_results / "chains"
        for chain_dir in sorted(chains_dir.glob("chain_*")):
            valid: list[tuple[int, Path]] = []
            for generation_dir in chain_dir.glob("generation_*"):
                metrics_path = generation_dir / "metrics.json"
                candidate_path = generation_dir / "candidate.py"
                if metrics_path.exists() and candidate_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics.get("status") == "valid":
                        generation = int(metrics["generation"])
                        valid.append((generation, candidate_path))
            if valid:
                generation, candidate_path = max(valid)
                name = f"qwen:{chain_dir.name}:g{generation:02d}"
                operators[name] = mutation.extract_and_validate_code(candidate_path.read_text(encoding="utf-8"))
    return operators


def token_shingles(code: str, width: int = 3) -> frozenset[tuple[str, ...]]:
    values: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
            continue
        value = token.string
        if token.type == tokenize.NAME and not keyword.iskeyword(value):
            # Keep helper/API names but erase arbitrary local variable spelling.
            if value not in {
                "repair_operator", "clone_routes", "insertion_candidates", "apply_insertion",
                "route_feasible", "route_distance", "solution_distance", "regret_insertion",
                "sequential_insertion", "greedy_insertion", "regret_2", "regret_3",
                "greedy_with_noise", "best_position_first", "np", "rng",
            }:
                value = "NAME"
        elif token.type == tokenize.NUMBER:
            value = "NUMBER"
        elif token.type == tokenize.STRING:
            value = "STRING"
        values.append(value)
    if len(values) < width:
        return frozenset({tuple(values)})
    return frozenset(tuple(values[index : index + width]) for index in range(len(values) - width + 1))


def jaccard_sets(left: frozenset[Any], right: frozenset[Any]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def pairwise_from_items(items: list[Any], distance: Callable[[Any, Any], float]) -> np.ndarray:
    matrix = np.zeros((len(items), len(items)), dtype=float)
    for i in range(len(items)):
        for j in range(i):
            matrix[i, j] = matrix[j, i] = float(distance(items[i], items[j]))
    return matrix


def edge_distance(
    left: mutation.CandidateEvaluation,
    right: mutation.CandidateEvaluation,
    split: str,
    replicate: bool = False,
) -> float:
    indices = mutation._indices_for_split(left, split)
    left_edges = left.edges_replicate if replicate else left.edges_primary
    right_edges = right.edges_replicate if replicate else right.edges_primary
    return mutation._mean_jaccard_distance(left_edges, right_edges, indices)


def footprint_distance(
    left: mutation.CandidateEvaluation,
    right: mutation.CandidateEvaluation,
    split: str,
    replicate: bool = False,
) -> float:
    indices = mutation._indices_for_split(left, split)
    left_values = left.gains_replicate if replicate else left.gains_primary
    right_values = right.gains_replicate if replicate else right.gains_primary
    a, b = left_values[indices], right_values[indices]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0 if np.allclose(a, b) else 1.0
    correlation = float(stats.spearmanr(a, b).statistic)
    return float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def matrix_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = upper_triangle(left), upper_triangle(right)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def mantel_test(left: np.ndarray, right: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, float]:
    observed = matrix_spearman(left, right)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        order = rng.permutation(left.shape[0])
        null[index] = matrix_spearman(left[np.ix_(order, order)], right)
    p_value = (1.0 + float(np.sum(null >= observed))) / (permutations + 1.0)
    return observed, p_value


def nearest_neighbor_agreement(metric: np.ndarray, target: np.ndarray) -> float:
    matches = []
    for index in range(metric.shape[0]):
        metric_row = metric[index].copy()
        target_row = target[index].copy()
        metric_row[index] = target_row[index] = math.inf
        matches.append(int(np.argmin(metric_row)) == int(np.argmin(target_row)))
    return float(np.mean(matches))


def distance_effective_dimension(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return float(matrix.shape[0])
    positive = upper_triangle(matrix)
    positive = positive[positive > 1e-12]
    if not len(positive):
        return 1.0
    bandwidth = max(float(np.median(positive)), 1e-6)
    kernel = np.exp(-np.square(matrix) / (2.0 * bandwidth * bandwidth))
    eigenvalues = np.maximum(np.linalg.eigvalsh(kernel), 0.0)
    denominator = float(np.square(eigenvalues).sum())
    return float(eigenvalues.sum() ** 2 / denominator) if denominator > 1e-15 else 1.0


def parse_llm_judgment(response: str, expected_ids: list[str]) -> tuple[np.ndarray, dict[str, str]]:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced or [response]
    payload = None
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            payload = json.loads(candidate[start : end + 1])
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        raise ValueError("no valid JSON object in LLM response")
    ids = [str(value) for value in payload["ids"]]
    matrix = np.asarray(payload["matrix"], dtype=float)
    descriptors = [str(value) for value in payload["descriptors"]]
    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError("LLM response ids do not match requested operator ids")
    if matrix.shape != (len(ids), len(ids)) or len(descriptors) != len(ids):
        raise ValueError("LLM matrix/descriptors have invalid dimensions")
    matrix = np.clip((matrix + matrix.T) / 200.0, 0.0, 1.0)
    np.fill_diagonal(matrix, 0.0)
    order = [ids.index(value) for value in expected_ids]
    canonical = matrix[np.ix_(order, order)]
    descriptor_map = {operator_id: descriptors[ids.index(operator_id)] for operator_id in expected_ids}
    return canonical, descriptor_map


def llm_judgment_prompt(order: list[str], operators: dict[str, str]) -> str:
    blocks = []
    for operator_id in order:
        blocks.append(f"ID: {operator_id}\n```python\n{operators[operator_id].rstrip()}\n```")
    return """Judge the algorithmic-mechanism diversity of the repair operators below.

Ignore variable names, formatting, comments, and semantically equivalent refactoring. Focus on how
customers are ordered, how insertion positions are chosen, whether customers are treated jointly,
how randomness is used, and what residual structures the operator specializes in.

Return JSON only with exactly these fields:
- "ids": the IDs in the supplied order;
- "descriptors": one concise mechanism description per ID in the same order;
- "matrix": a symmetric numeric matrix in the same order, diagonal 0, where 0 means behaviorally
  equivalent mechanism and 100 means fundamentally different mechanism.

Do not add Markdown or explanations outside JSON.

""" + "\n\n".join(blocks)


def collect_llm_judgments(
    args: argparse.Namespace,
    operators: dict[str, str],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if args.skip_llm:
        return None, None, {"status": "skipped"}
    llm_dir = args.output / "llm_judgments"
    llm_dir.mkdir(parents=True, exist_ok=True)
    canonical_ids = list(operators)
    matrices: list[np.ndarray] = []
    descriptor_repeats: list[dict[str, str]] = []
    errors: list[str] = []
    rng = np.random.default_rng(args.seed + 50_000)

    for repeat in range(args.llm_repeats):
        order = list(np.asarray(canonical_ids)[rng.permutation(len(canonical_ids))])
        prompt = llm_judgment_prompt(order, operators)
        prompt_path = llm_dir / f"prompt_{repeat:02d}.txt"
        response_path = llm_dir / f"response_{repeat:02d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        response = response_path.read_text(encoding="utf-8") if args.resume and response_path.exists() else None
        for attempt in range(args.llm_attempts):
            if response is None:
                try:
                    response = mutation.call_openai_compatible(prompt, args)
                    response_path.write_text(response, encoding="utf-8")
                except Exception as exc:
                    errors.append(f"repeat {repeat} API attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                    response = None
                    continue
            try:
                matrix, descriptors = parse_llm_judgment(response, canonical_ids)
                matrices.append(matrix)
                descriptor_repeats.append(descriptors)
                break
            except Exception as exc:
                errors.append(f"repeat {repeat} parse attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                response = None

    if not matrices:
        return None, None, {"status": "failed", "errors": errors}
    direct = np.mean(matrices, axis=0)
    descriptor_documents = [
        " ".join(descriptors[operator_id] for descriptors in descriptor_repeats)
        for operator_id in canonical_ids
    ]
    descriptor_matrix = tfidf_distance(descriptor_documents)
    repeat_reliability = (
        float(np.mean([matrix_spearman(matrices[i], matrices[j]) for i in range(len(matrices)) for j in range(i)]))
        if len(matrices) > 1
        else math.nan
    )
    metadata = {
        "status": "ok",
        "model": args.model or mutation._env_value(mutation.MODEL_ENV) or "UNSPECIFIED",
        "successful_repeats": len(matrices),
        "requested_repeats": args.llm_repeats,
        "direct_judgment_repeat_reliability": repeat_reliability,
        "errors": errors,
        "descriptors": {operator_id: [values[operator_id] for values in descriptor_repeats] for operator_id in canonical_ids},
    }
    return direct, descriptor_matrix, metadata


def descriptor_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[\u4e00-\u9fff]", text)]


def tfidf_distance(documents: list[str]) -> np.ndarray:
    tokenized = [descriptor_tokens(document) for document in documents]
    vocabulary = sorted({token for document in tokenized for token in document})
    if not vocabulary:
        return np.zeros((len(documents), len(documents)))
    index = {token: position for position, token in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)
    document_frequency = np.zeros(len(vocabulary), dtype=float)
    for row, tokens in enumerate(tokenized):
        for token in tokens:
            matrix[row, index[token]] += 1.0
        for token in set(tokens):
            document_frequency[index[token]] += 1.0
    matrix *= np.log((1.0 + len(documents)) / (1.0 + document_frequency))[None, :] + 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    return np.clip(1.0 - normalized @ normalized.T, 0.0, 1.0)


def evaluate_operators(
    operators: dict[str, str],
    probes: list[mutation.Probe],
    timeout: float,
    output: Path,
) -> tuple[dict[str, str], dict[str, mutation.CandidateEvaluation], dict[str, str]]:
    valid_codes: dict[str, str] = {}
    evaluations: dict[str, mutation.CandidateEvaluation] = {}
    errors: dict[str, str] = {}
    code_dir = output / "operators"
    code_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, code) in enumerate(operators.items(), start=1):
        evaluation, error = mutation.evaluate_candidate(code, probes, timeout)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        (code_dir / f"{index:02d}_{safe_name}.py").write_text(code, encoding="utf-8")
        if evaluation is None:
            errors[name] = error or "unknown evaluation error"
            print(f"  [{index}/{len(operators)}] {name}: excluded ({errors[name]})", flush=True)
            continue
        valid_codes[name] = code
        evaluations[name] = evaluation
        print(f"  [{index}/{len(operators)}] {name}: valid", flush=True)
    return valid_codes, evaluations, errors


def analyze_metrics(
    args: argparse.Namespace,
    names: list[str],
    codes: list[str],
    evaluations: list[mutation.CandidateEvaluation],
    llm_direct: np.ndarray | None,
    llm_descriptor: np.ndarray | None,
    llm_metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    token_matrix = pairwise_from_items([token_shingles(code) for code in codes], jaccard_sets)
    ast_matrix = pairwise_from_items([mutation.ast_shingles(code) for code in codes], jaccard_sets)
    calibration_l2 = pairwise_from_items(evaluations, lambda a, b: edge_distance(a, b, "calibration"))
    calibration_l3 = pairwise_from_items(evaluations, lambda a, b: footprint_distance(a, b, "calibration"))
    validation_l2 = pairwise_from_items(evaluations, lambda a, b: edge_distance(a, b, "validation"))
    validation_l3 = pairwise_from_items(evaluations, lambda a, b: footprint_distance(a, b, "validation"))
    validation_l2_replicate = pairwise_from_items(
        evaluations, lambda a, b: edge_distance(a, b, "validation", replicate=True)
    )
    validation_l3_replicate = pairwise_from_items(
        evaluations, lambda a, b: footprint_distance(a, b, "validation", replicate=True)
    )

    matrices: dict[str, np.ndarray] = {
        "token_l0": token_matrix,
        "ast_l0": ast_matrix,
        "probe_output_l2": calibration_l2,
        "performance_footprint_l3": calibration_l3,
        "heldout_output_target": validation_l2,
        "heldout_performance_target": validation_l3,
    }
    reliability = {
        "token_l0": 1.0,
        "ast_l0": 1.0,
        "probe_output_l2": matrix_spearman(validation_l2, validation_l2_replicate),
        "performance_footprint_l3": matrix_spearman(validation_l3, validation_l3_replicate),
    }
    if llm_direct is not None:
        matrices["llm_direct_l1plus"] = llm_direct
        reliability["llm_direct_l1plus"] = float(llm_metadata["direct_judgment_repeat_reliability"])
    if llm_descriptor is not None:
        matrices["llm_descriptor_tfidf_l1proxy"] = llm_descriptor
        reliability["llm_descriptor_tfidf_l1proxy"] = math.nan

    metric_names = [name for name in matrices if not name.startswith("heldout_")]
    rng = np.random.default_rng(args.seed + 70_000)
    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        matrix = matrices[metric_name]
        corr_l2, p_l2 = mantel_test(matrix, validation_l2, args.permutations, rng)
        corr_l3, p_l3 = mantel_test(matrix, validation_l3, args.permutations, rng)
        rows.append(
            {
                "metric": metric_name,
                "reliability": reliability.get(metric_name, math.nan),
                "effective_dimension": distance_effective_dimension(matrix),
                "heldout_l2_spearman": corr_l2,
                "heldout_l2_mantel_p": p_l2,
                "heldout_l2_nn_agreement": nearest_neighbor_agreement(matrix, validation_l2),
                "heldout_l3_spearman": corr_l3,
                "heldout_l3_mantel_p": p_l3,
                "heldout_l3_nn_agreement": nearest_neighbor_agreement(matrix, validation_l3),
            }
        )
    rows.sort(key=lambda row: (row["heldout_l2_spearman"] + row["heldout_l3_spearman"]), reverse=True)

    pairs = []
    for i in range(len(names)):
        for j in range(i):
            pairs.append(
                {
                    "left": names[j],
                    "right": names[i],
                    "token_l0": token_matrix[i, j],
                    "ast_l0": ast_matrix[i, j],
                    "probe_output_l2": calibration_l2[i, j],
                    "heldout_output": validation_l2[i, j],
                    "heldout_performance": validation_l3[i, j],
                }
            )
    false_diversity = sorted(
        [pair for pair in pairs if pair["ast_l0"] > pair["heldout_output"]],
        key=lambda pair: (pair["ast_l0"] - pair["heldout_output"]),
        reverse=True,
    )[:5]
    hidden_diversity = sorted(
        [pair for pair in pairs if pair["heldout_output"] > pair["ast_l0"]],
        key=lambda pair: (pair["heldout_output"] - pair["ast_l0"]),
        reverse=True,
    )[:5]
    summary = {
        "n_operators": len(names),
        "operator_names": names,
        "ranking": [row["metric"] for row in rows],
        "llm": llm_metadata,
        "heldout_target_reliability": {
            "output_behavior": matrix_spearman(validation_l2, validation_l2_replicate),
            "performance_footprint": matrix_spearman(validation_l3, validation_l3_replicate),
        },
        "false_diversity_examples": false_diversity,
        "hidden_diversity_examples": hidden_diversity,
    }
    return matrices, rows, summary


def write_matrix_csv(path: Path, names: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operator", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[f"{float(value):.8f}" for value in row]])


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_heatmaps(path: Path, names: list[str], matrices: dict[str, np.ndarray]) -> None:
    selected = [
        name
        for name in (
            "ast_l0",
            "llm_direct_l1plus",
            "probe_output_l2",
            "heldout_output_target",
            "heldout_performance_target",
        )
        if name in matrices
    ]
    figure, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4.5), squeeze=False)
    labels = [name.split(":")[-1][:14] for name in names]
    for axis, metric_name in zip(axes[0], selected):
        image = axis.imshow(matrices[metric_name], vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(metric_name)
        axis.set_xticks(range(len(names)), labels, rotation=90, fontsize=6)
        axis.set_yticks(range(len(names)), labels, fontsize=6)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_value(value: Any, percent: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "—"
    return f"{numeric:.0%}" if percent else f"{numeric:.3f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    errors: dict[str, str],
    runtime: float,
) -> None:
    lines = [
        "# 多样性度量综合审计",
        "",
        f"本报告比较 {summary['n_operators']} 个 repair 算子的代码、LLM 机制判断、probe 输出行为与性能足迹。",
        "calibration 实例用于构造候选距离；validation 实例只用于 held-out 效度检验。",
        "",
        "## 结论速览",
        "",
    ]
    if rows:
        best = rows[0]
        lines.extend(
            [
                f"按 held-out L2 与 L3 相关之和排序，当前最佳候选度量是 **`{best['metric']}`**。",
                "这只是本 pilot 算子集上的排序；是否进入选择机制还要同时满足重复测量信度和独立实例效度。",
                "",
            ]
        )
    lines.extend(
        [
            "| 度量 | 重测信度 | 有效自由度 | held-out L2 ρ / p | L2 最近邻一致 | held-out L3 ρ / p | L3 最近邻一致 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['metric']}` | {_format_value(row['reliability'])} | {_format_value(row['effective_dimension'])} | "
            f"{_format_value(row['heldout_l2_spearman'])} / {_format_value(row['heldout_l2_mantel_p'])} | "
            f"{_format_value(row['heldout_l2_nn_agreement'], True)} | "
            f"{_format_value(row['heldout_l3_spearman'])} / {_format_value(row['heldout_l3_mantel_p'])} | "
            f"{_format_value(row['heldout_l3_nn_agreement'], True)} |"
        )
    lines.extend(
        [
            "",
            "说明：Mantel p 通过置换算子标签得到；最近邻一致表示该度量与 held-out 目标为同一算子找到相同最近邻的比例。",
            "有效自由度来自各距离的 median-bandwidth RBF kernel，只适合在同一批算子内比较其是否夸大/压缩维度。",
            "",
            "## 假多样性：AST 很远但 held-out 行为很近",
            "",
            "| 算子对 | AST | held-out output | 差值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in summary["false_diversity_examples"]:
        lines.append(
            f"| `{pair['left']}` × `{pair['right']}` | {pair['ast_l0']:.3f} | {pair['heldout_output']:.3f} | "
            f"{pair['ast_l0'] - pair['heldout_output']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 隐藏多样性：AST 很近但 held-out 行为很远",
            "",
            "| 算子对 | AST | held-out output | 差值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in summary["hidden_diversity_examples"]:
        lines.append(
            f"| `{pair['left']}` × `{pair['right']}` | {pair['ast_l0']:.3f} | {pair['heldout_output']:.3f} | "
            f"{pair['heldout_output'] - pair['ast_l0']:+.3f} |"
        )
    if not summary["hidden_diversity_examples"]:
        lines.append("| 本次没有 AST 明显低估 held-out 输出差异的算子对 | — | — | — |")
    lines.extend(
        [
            "",
            "## 推荐的度量栈",
            "",
            "1. AST/token 只做免费预过滤和精确/近换皮报警，不直接代表行为多样性。",
            "2. Qwen direct judgment 若跨重复稳定，可作低成本机制 gate；它不能代替可执行行为测量。",
            "3. calibration L2 probe output 是当前选择、聚类和变异审计的核心候选，前提是其 held-out 效度通过。",
            "4. L3 performance footprint 表示“在哪里有用”，成本更高，适合组合库边际信用和最终验证。",
            "5. 有效自由度是任一表示上的群体汇总仪表，不是独立的个体距离，也不是性能目标。",
            "",
            "## 边界与未覆盖项",
            "",
            "- 当前 L1 使用 Qwen 机制描述 + TF-IDF proxy 和直接成对判断；运行环境没有 torch/transformers，尚未加入 CodeBERT/GraphCodeBERT。报告不会把 proxy 冒充代码 embedding。",
            "- Qwen 同时生成部分 mutation 候选并担任语义裁判，可能有同模型偏置；经典算子和手工对照可缓解但不能消除。",
            "- L2 目前比较单步 repair 后的边集合，尚未包含完整搜索轨迹、插入决策序列或多搜索阶段。",
            "- L3 使用固定残解上的收益秩足迹，不等价于长程 LNS portfolio value。",
            "- 小样本矩阵中的 pair 不是独立观测，因此以 Mantel 置换而不是普通相关 p 值为主。",
            "",
            "## 排除的算子",
            "",
        ]
    )
    if errors:
        lines.extend(f"- `{name}`：{error}" for name, error in errors.items())
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "```powershell",
            f"python {SCRIPT_INVOCATION} --output {args.output.as_posix()} --mutation-results {args.mutation_results.as_posix()}",
            "```",
            "",
            f"运行耗时 {runtime:.1f} 秒。完整距离矩阵见 `matrices/`，Qwen 原始判断见 `llm_judgments/`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mutation-results", type=Path, default=DEFAULT_MUTATION_RESULTS)
    parser.add_argument("--exclude-mutations", action="store_true")
    parser.add_argument("--probe-instances", type=int, default=4)
    parser.add_argument("--probe-repeats", type=int, default=2)
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--remove-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--evaluation-timeout", type=float, default=25.0)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-repeats", type=int, default=3)
    parser.add_argument("--llm-attempts", type=int, default=2)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=mutation.API_KEY_ENV)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--api-timeout", type=float, default=240.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.probe_instances < 2 or args.probe_instances % 2:
        parser.error("probe-instances must be even and >= 2")
    if args.probe_repeats < 1 or args.customers < 10 or not 0 < args.remove_fraction < 1:
        parser.error("invalid probe configuration")
    if args.permutations < 100 or args.llm_repeats < 1 or args.llm_attempts < 1:
        parser.error("permutations>=100, llm-repeats>=1, and llm-attempts>=1 are required")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    mutation.load_dotenv(args.env_file.resolve())
    args.output = args.output.resolve()
    args.mutation_results = args.mutation_results.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    operators = collect_operators(args.mutation_results, not args.exclude_mutations)
    probe_args = argparse.Namespace(
        probe_instances=args.probe_instances,
        customers=args.customers,
        seed=args.seed,
        probe_repeats=args.probe_repeats,
        remove_fraction=args.remove_fraction,
    )
    probes = mutation.build_probes(probe_args)
    print(f"Evaluating {len(operators)} operators on {len(probes)} disjoint probes", flush=True)
    codes, evaluation_map, errors = evaluate_operators(operators, probes, args.evaluation_timeout, args.output)
    names = list(codes)
    if len(names) < 4:
        raise RuntimeError("fewer than four valid operators; diversity audit is not meaningful")

    llm_direct, llm_descriptor, llm_metadata = collect_llm_judgments(args, codes)
    evaluations = [evaluation_map[name] for name in names]
    matrices, rows, summary = analyze_metrics(
        args,
        names,
        [codes[name] for name in names],
        evaluations,
        llm_direct,
        llm_descriptor,
        llm_metadata,
    )
    matrix_dir = args.output / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    for name, matrix in matrices.items():
        write_matrix_csv(matrix_dir / f"{name}.csv", names, matrix)
    write_metric_csv(args.output / "metric_comparison.csv", rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_heatmaps(args.output / "distance_heatmaps.png", names, matrices)
    runtime = time.perf_counter() - started
    write_report(args.output / "REPORT.md", args, rows, summary, errors, runtime)
    print(f"Top metric: {rows[0]['metric']}", flush=True)
    print(f"Wrote results to {args.output} ({runtime:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
