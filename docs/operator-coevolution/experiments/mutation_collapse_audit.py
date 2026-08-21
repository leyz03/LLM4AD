"""Pilot audit for behavioral collapse in LLM mutation chains.

The experiment deliberately freezes evolutionary selection.  Each chain starts
from a behaviorally distinct repair operator and repeatedly asks an LLM for one
modified child.  Every valid child becomes the next parent regardless of its
fitness.  This separates mutation collapse (candidate generation) from collapse
caused by survival selection or credit assignment.

The script supports any OpenAI-compatible chat-completions endpoint through
environment variables and saves every raw response before executing generated
code.  Generated functions are AST-filtered and evaluated in a timeout-limited
child process.  This is defense in depth, not a strong security sandbox.

Examples (from the repository root)::

    # No network or LLM: validate the complete measurement pipeline.
    python docs/operator-coevolution/experiments/mutation_collapse_audit.py --self-test

    # One-call API smoke test after setting the three environment variables.
    python docs/operator-coevolution/experiments/mutation_collapse_audit.py \
        --chains 1 --generations 1 --output .../results/mutation-collapse/smoke

    # Preregistered pilot: 4 chains x 8 generations = 32 calls (before retries).
    python docs/operator-coevolution/experiments/mutation_collapse_audit.py

Environment variables::

    LLM_BASE_URL   e.g. https://api.openai.com/v1
    LLM_API_KEY
    LLM_MODEL
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import math
import multiprocessing as mp
import os
import queue
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import stage05_gamma_audit as stage05


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "mutation-collapse" / "pilot"
SCRIPT_INVOCATION = "docs/operator-coevolution/experiments/mutation_collapse_audit.py"

BASE_URL_ENV = "LLM_BASE_URL"
API_KEY_ENV = "LLM_API_KEY"
MODEL_ENV = "LLM_MODEL"
LEGACY_ENV_NAMES = {
    BASE_URL_ENV: "LLM4AD_LLM_BASE_URL",
    API_KEY_ENV: "LLM4AD_LLM_API_KEY",
    MODEL_ENV: "LLM4AD_LLM_MODEL",
}

PROBE_DESTROYERS = (
    "random_removal",
    "shaw_removal",
    "route_removal",
    "cluster_removal",
    "time_oriented_removal",
)

TASK_DESCRIPTION = """We are evolving a repair operator for CVRP and VRPTW large-neighborhood search.
The operator receives a feasible partial solution and the removed customers, and must return a
complete feasible solution. Improve the parent while changing its algorithmic mechanism, not just
renaming variables or tuning a numeric constant.

You may use only these existing helpers (do not import anything):
- clone_routes(routes) -> deep copy of the route list
- insertion_candidates(problem, routes, customer) -> sorted (delta, route_index, position) tuples
- apply_insertion(routes, customer, candidate) -> mutates routes with one insertion
- route_feasible(problem, route) -> bool
- route_distance(problem, route) -> float
- solution_distance(problem, routes) -> float
- np and the supplied rng (numpy Generator)

Contract:
def repair_operator(problem, routes, removed, rng):
    # do not mutate routes or removed; return list[list[int]]

Return one short algorithm description inside braces, followed by exactly one complete Python
function named repair_operator. Do not emit imports, classes, files, network calls, global state,
or additional helper functions."""


BLOCK_REPAIR_SOURCE = """def repair_operator(problem, routes, removed, rng):
    del rng
    result = clone_routes(routes)
    if not removed:
        return result
    best = None
    for order in (removed.copy(), list(reversed(removed))):
        for route_index, route in enumerate(result):
            old_cost = route_distance(problem, route)
            for position in range(len(route) + 1):
                candidate_route = route[:position] + order + route[position:]
                if route_feasible(problem, candidate_route):
                    delta = route_distance(problem, candidate_route) - old_cost
                    item = (float(delta), route_index, position, order)
                    if best is None or item[:3] < best[:3]:
                        best = item
    if best is not None:
        _, route_index, position, order = best
        result[route_index][position:position] = order
        return result
    for customer in removed:
        apply_insertion(result, customer, insertion_candidates(problem, result, customer)[0])
    return result
"""


NEW_ROUTE_REPAIR_SOURCE = """def repair_operator(problem, routes, removed, rng):
    del problem, rng
    result = clone_routes(routes)
    for customer in removed:
        result.append([customer])
    return result
"""


@dataclass
class Probe:
    probe_id: str
    split: str
    problem: stage05.Problem
    partial: stage05.Routes
    removed: list[int]
    cost_before: float
    seed_primary: int
    seed_replicate: int


@dataclass
class CandidateEvaluation:
    edges_primary: list[frozenset[int]]
    edges_replicate: list[frozenset[int]]
    gains_primary: np.ndarray
    gains_replicate: np.ndarray
    route_counts_primary: np.ndarray
    route_counts_replicate: np.ndarray
    splits: tuple[str, ...]


def _source_as_repair(function: Any) -> str:
    tree = ast.parse(inspect.getsource(function))
    function_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function_node.name = "repair_operator"
    function_node.decorator_list = []
    function_node.returns = None
    for arg in (*function_node.args.posonlyargs, *function_node.args.args, *function_node.args.kwonlyargs):
        arg.annotation = None
    return ast.unparse(ast.Module(body=[function_node], type_ignores=[])) + "\n"


SEED_OPERATORS: tuple[tuple[str, str], ...] = (
    ("sequential", _source_as_repair(stage05.sequential_insertion)),
    ("regret_2", _source_as_repair(stage05.regret_2)),
    ("block_reinsert", BLOCK_REPAIR_SOURCE),
    ("new_route", NEW_ROUTE_REPAIR_SOURCE),
)


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.With,
    ast.AsyncWith,
    ast.Global,
    ast.Nonlocal,
)
FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "help",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
}
ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def extract_and_validate_code(response: str) -> str:
    """Extract one function, normalize its name, and reject unsafe constructs."""
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced or [response]
    tree = None
    for text in candidates:
        start = text.find("def ")
        if start < 0:
            continue
        try:
            parsed = ast.parse(text[start:])
        except SyntaxError:
            continue
        functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        if len(functions) == 1:
            tree = ast.Module(body=[functions[0]], type_ignores=[])
            break
    if tree is None:
        raise ValueError("response does not contain exactly one parseable function")

    function_node = tree.body[0]
    assert isinstance(function_node, ast.FunctionDef)
    function_node.name = "repair_operator"
    function_node.decorator_list = []
    function_node.returns = None
    if len(function_node.args.args) != 4:
        raise ValueError("repair_operator must have exactly four positional arguments")
    for arg, expected in zip(function_node.args.args, ("problem", "routes", "removed", "rng")):
        arg.arg = expected
        arg.annotation = None

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Delete) and any(not isinstance(target, ast.Name) for target in node.targets):
            raise ValueError("only deletion of local names is allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"private attribute access is forbidden: {node.attr}")

    ast.fix_missing_locations(tree)
    normalized = ast.unparse(tree) + "\n"
    compile(normalized, "<generated-repair>", "exec")
    return normalized


def _execution_namespace() -> dict[str, Any]:
    return {
        "__builtins__": ALLOWED_BUILTINS,
        "np": np,
        "Problem": stage05.Problem,
        "Routes": stage05.Routes,
        "clone_routes": stage05.clone_routes,
        "insertion_candidates": stage05.insertion_candidates,
        "apply_insertion": stage05.apply_insertion,
        "regret_insertion": stage05.regret_insertion,
        "sequential_insertion": stage05.sequential_insertion,
        "greedy_insertion": stage05.greedy_insertion,
        "regret_2": stage05.regret_2,
        "regret_3": stage05.regret_3,
        "greedy_with_noise": stage05.greedy_with_noise,
        "best_position_first": stage05.best_position_first,
        "route_feasible": stage05.route_feasible,
        "route_distance": stage05.route_distance,
        "solution_distance": stage05.solution_distance,
    }


def _edge_ids(routes: stage05.Routes, n_customers: int) -> frozenset[int]:
    width = n_customers + 1
    edges: set[int] = set()
    for route in routes:
        nodes = [0, *route, 0]
        for left, right in zip(nodes, nodes[1:]):
            a, b = sorted((int(left), int(right)))
            edges.add(a * width + b)
    return frozenset(edges)


def _evaluation_worker(code: str, probes: list[Probe], result_queue: mp.Queue) -> None:
    try:
        namespace = _execution_namespace()
        exec(compile(code, "<generated-repair>", "exec"), namespace)
        function = namespace["repair_operator"]
        edges_primary: list[frozenset[int]] = []
        edges_replicate: list[frozenset[int]] = []
        gains_primary: list[float] = []
        gains_replicate: list[float] = []
        routes_primary: list[int] = []
        routes_replicate: list[int] = []
        for probe in probes:
            outputs = []
            for seed in (probe.seed_primary, probe.seed_replicate):
                rng = np.random.default_rng(seed)
                candidate = function(
                    probe.problem,
                    stage05.clone_routes(probe.partial),
                    probe.removed.copy(),
                    rng,
                )
                stage05.validate_solution(probe.problem, candidate)
                outputs.append(candidate)
            primary, replicate = outputs
            after_primary = stage05.solution_distance(probe.problem, primary)
            after_replicate = stage05.solution_distance(probe.problem, replicate)
            edges_primary.append(_edge_ids(primary, probe.problem.n_customers))
            edges_replicate.append(_edge_ids(replicate, probe.problem.n_customers))
            gains_primary.append(100.0 * (probe.cost_before - after_primary) / probe.cost_before)
            gains_replicate.append(100.0 * (probe.cost_before - after_replicate) / probe.cost_before)
            routes_primary.append(len(primary))
            routes_replicate.append(len(replicate))
        result_queue.put(
            {
                "ok": True,
                "edges_primary": edges_primary,
                "edges_replicate": edges_replicate,
                "gains_primary": gains_primary,
                "gains_replicate": gains_replicate,
                "routes_primary": routes_primary,
                "routes_replicate": routes_replicate,
            }
        )
    except BaseException as exc:  # child process must report generated-code failures
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def evaluate_candidate(code: str, probes: list[Probe], timeout: float) -> tuple[CandidateEvaluation | None, str | None]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_evaluation_worker, args=(code, probes, result_queue), daemon=True)
    process.start()
    try:
        payload = result_queue.get(timeout=timeout)
    except queue.Empty:
        payload = {"ok": False, "error": f"evaluation timed out after {timeout:.1f}s"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        result_queue.close()
    if not payload.get("ok"):
        return None, str(payload.get("error", "unknown evaluation failure"))
    return (
        CandidateEvaluation(
            edges_primary=list(payload["edges_primary"]),
            edges_replicate=list(payload["edges_replicate"]),
            gains_primary=np.asarray(payload["gains_primary"], dtype=float),
            gains_replicate=np.asarray(payload["gains_replicate"], dtype=float),
            route_counts_primary=np.asarray(payload["routes_primary"], dtype=float),
            route_counts_replicate=np.asarray(payload["routes_replicate"], dtype=float),
            splits=tuple(probe.split for probe in probes),
        ),
        None,
    )


def build_probes(args: argparse.Namespace) -> list[Probe]:
    probes: list[Probe] = []
    for problem_offset, problem_type in enumerate(("cvrp", "vrptw")):
        if problem_type == "cvrp":
            problems = stage05.generate_cvrp_instances(args.probe_instances, args.customers, args.seed + 1000)
        else:
            problems = stage05.generate_vrptw_instances(args.probe_instances, args.customers, args.seed + 2000)
        calibration_count = args.probe_instances // 2
        for problem_index, problem in enumerate(problems):
            split = "calibration" if problem_index < calibration_count else "validation"
            for repeat in range(args.probe_repeats):
                base_seed = stage05.seed_for(args.seed + 17 + problem_offset, problem_index, repeat, 99, 99)
                base = stage05.construct_base_solution(problem, np.random.default_rng(base_seed))
                before = stage05.solution_distance(problem, base)
                q = max(2, min(problem.n_customers - 1, int(round(problem.n_customers * args.remove_fraction))))
                for destroy_index, destroy_name in enumerate(PROBE_DESTROYERS):
                    destroy_seed = stage05.seed_for(
                        args.seed + problem_offset * 100_000,
                        problem_index,
                        repeat,
                        destroy_index,
                        10_000,
                    )
                    partial, removed = stage05.DESTROYERS[destroy_name](
                        problem,
                        stage05.clone_routes(base),
                        np.random.default_rng(destroy_seed),
                        q,
                    )
                    probe_key = f"{problem_type}-{problem_index}-{repeat}-{destroy_name}"
                    probes.append(
                        Probe(
                            probe_id=probe_key,
                            split=split,
                            problem=problem,
                            partial=partial,
                            removed=removed,
                            cost_before=before,
                            seed_primary=stage05.seed_for(args.seed, problem_index, repeat, destroy_index, 20_000),
                            seed_replicate=stage05.seed_for(args.seed, problem_index, repeat, destroy_index, 30_000),
                        )
                    )
    return probes


def _indices_for_split(evaluation: CandidateEvaluation, split: str) -> list[int]:
    return [index for index, value in enumerate(evaluation.splits) if value == split]


def _mean_jaccard_distance(left: list[frozenset[int]], right: list[frozenset[int]], indices: list[int]) -> float:
    values = []
    for index in indices:
        union = left[index] | right[index]
        values.append(0.0 if not union else 1.0 - len(left[index] & right[index]) / len(union))
    return float(np.mean(values)) if values else math.nan


def behavior_distance(left: CandidateEvaluation, right: CandidateEvaluation, split: str) -> float:
    indices = _indices_for_split(left, split)
    return _mean_jaccard_distance(left.edges_primary, right.edges_primary, indices)


def noise_adjusted_behavior_distance(left: CandidateEvaluation, right: CandidateEvaluation, split: str) -> float:
    """Exploratory local correction for heterogeneous stochastic behavior."""
    raw = behavior_distance(left, right, split)
    local_noise = max(self_noise(left, split), self_noise(right, split))
    return max(0.0, raw - local_noise)


def self_noise(evaluation: CandidateEvaluation, split: str) -> float:
    indices = _indices_for_split(evaluation, split)
    return _mean_jaccard_distance(evaluation.edges_primary, evaluation.edges_replicate, indices)


def mean_gain(evaluation: CandidateEvaluation, split: str) -> float:
    indices = _indices_for_split(evaluation, split)
    return float(np.mean(evaluation.gains_primary[indices]))


def ast_shingles(code: str, width: int = 3) -> frozenset[tuple[str, ...]]:
    sequence = [type(node).__name__ for node in ast.walk(ast.parse(code))]
    if len(sequence) < width:
        return frozenset({tuple(sequence)})
    return frozenset(tuple(sequence[index : index + width]) for index in range(len(sequence) - width + 1))


def code_distance(left: str, right: str) -> float:
    a, b = ast_shingles(left), ast_shingles(right)
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)


def effective_dimension(evaluations: list[CandidateEvaluation], split: str, noise_adjusted: bool = False) -> float:
    if len(evaluations) <= 1:
        return float(len(evaluations))
    distances = np.zeros((len(evaluations), len(evaluations)), dtype=float)
    distance_function = noise_adjusted_behavior_distance if noise_adjusted else behavior_distance
    for i in range(len(evaluations)):
        for j in range(i):
            distances[i, j] = distances[j, i] = distance_function(evaluations[i], evaluations[j], split)
    positive = distances[distances > 1e-12]
    if not len(positive):
        return 1.0
    bandwidth = max(float(np.median(positive)), 1e-6)
    kernel = np.exp(-np.square(distances) / (2.0 * bandwidth * bandwidth))
    eigenvalues = np.maximum(np.linalg.eigvalsh(kernel), 0.0)
    denominator = float(np.square(eigenvalues).sum())
    return float(eigenvalues.sum() ** 2 / denominator) if denominator > 1e-15 else 1.0


def make_prompt(parent_code: str, seed_name: str, chain: int, generation: int) -> str:
    return f"""{TASK_DESCRIPTION}

This is mutation chain {chain}, generation {generation}. The chain started from the
{seed_name!r} repair. Here is the current parent:

```python
{parent_code.rstrip()}
```

Create exactly one modified child. The experiment will accept every valid child regardless of
fitness, so do not discuss selection or compare multiple alternatives."""


def _chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def load_dotenv(path: Path) -> None:
    """Load a minimal KEY=VALUE .env without logging names or secret values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match:
            continue
        name, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _env_value(primary_name: str) -> str:
    return os.environ.get(primary_name, "") or os.environ.get(LEGACY_ENV_NAMES[primary_name], "")


def call_openai_compatible(prompt: str, args: argparse.Namespace) -> str:
    base_url = args.base_url or _env_value(BASE_URL_ENV)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_env == API_KEY_ENV:
        api_key = _env_value(API_KEY_ENV)
    model = args.model or _env_value(MODEL_ENV)
    missing = [name for name, value in ((BASE_URL_ENV, base_url), (args.api_key_env, api_key), (MODEL_ENV, model)) if not value]
    if missing:
        raise RuntimeError("missing LLM configuration: " + ", ".join(missing))
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful algorithm engineer. Return executable Python exactly as requested."},
                {"role": "user", "content": prompt},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _chat_completions_url(base_url),
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.api_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
    return str(result["choices"][0]["message"]["content"])


def self_test_response(chain: int, generation: int) -> str:
    # Deterministic responses exercise parsing, execution, resume, and collapse metrics.
    source = SEED_OPERATORS[(chain + generation) % len(SEED_OPERATORS)][1]
    return "{Deterministic pipeline self-test candidate.}\n```python\n" + source + "```\n"


def obtain_candidate(
    prompt: str,
    chain: int,
    generation: int,
    generation_dir: Path,
    args: argparse.Namespace,
) -> tuple[str | None, int, str | None]:
    generation_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = generation_dir / "candidate.py"
    if args.resume and candidate_path.exists():
        return candidate_path.read_text(encoding="utf-8"), 0, None

    last_error = None
    for attempt in range(1, args.attempts_per_generation + 1):
        response_path = generation_dir / f"response_attempt_{attempt}.txt"
        if args.resume and response_path.exists():
            response = response_path.read_text(encoding="utf-8")
        else:
            try:
                response = self_test_response(chain, generation) if args.self_test else call_openai_compatible(prompt, args)
            except Exception as exc:  # persist API failures and let the run remain resumable
                last_error = f"{type(exc).__name__}: {exc}"
                (generation_dir / f"api_error_attempt_{attempt}.txt").write_text(last_error, encoding="utf-8")
                continue
            response_path.write_text(response, encoding="utf-8")
        try:
            code = extract_and_validate_code(response)
        except ValueError as exc:
            last_error = str(exc)
            (generation_dir / f"parse_error_attempt_{attempt}.txt").write_text(last_error, encoding="utf-8")
            continue
        candidate_path.write_text(code, encoding="utf-8")
        return code, attempt, None
    return None, args.attempts_per_generation, last_error


def _blank_observation(chain: int, generation: int, seed_name: str, status: str, error: str | None) -> dict[str, Any]:
    return {
        "chain": chain,
        "generation": generation,
        "seed": seed_name,
        "status": status,
        "attempts": 0,
        "error": error or "",
        "l0_parent": math.nan,
        "l0_ancestor": math.nan,
        "l2_parent_calibration": math.nan,
        "l2_parent_validation": math.nan,
        "l2_ancestor_calibration": math.nan,
        "l2_ancestor_validation": math.nan,
        "l2_nearest_history_calibration": math.nan,
        "l2_nearest_history_validation": math.nan,
        "l2_nearest_history_calibration_adjusted": math.nan,
        "l2_nearest_history_validation_adjusted": math.nan,
        "self_noise_calibration": math.nan,
        "self_noise_validation": math.nan,
        "mean_gain_calibration": math.nan,
        "mean_gain_validation": math.nan,
        "behavior_revisit": "",
    }


def run_chains(args: argparse.Namespace, probes: list[Probe]) -> tuple[list[dict[str, Any]], list[list[tuple[str, CandidateEvaluation]]]]:
    observations: list[dict[str, Any]] = []
    histories: list[list[tuple[str, CandidateEvaluation]]] = []
    for chain in range(args.chains):
        seed_name, seed_code = SEED_OPERATORS[chain % len(SEED_OPERATORS)]
        seed_code = extract_and_validate_code(seed_code)
        seed_eval, error = evaluate_candidate(seed_code, probes, args.evaluation_timeout)
        if seed_eval is None:
            raise RuntimeError(f"internal seed {seed_name} failed evaluation: {error}")
        chain_history = [(seed_code, seed_eval)]
        histories.append(chain_history)
        seed_row = _blank_observation(chain, 0, seed_name, "seed", None)
        seed_row.update(
            self_noise_calibration=self_noise(seed_eval, "calibration"),
            self_noise_validation=self_noise(seed_eval, "validation"),
            mean_gain_calibration=mean_gain(seed_eval, "calibration"),
            mean_gain_validation=mean_gain(seed_eval, "validation"),
        )
        observations.append(seed_row)
        print(f"Chain {chain}: seed={seed_name}", flush=True)

        for generation in range(1, args.generations + 1):
            parent_code, parent_eval = chain_history[-1]
            prompt = make_prompt(parent_code, seed_name, chain, generation)
            generation_dir = args.output / "chains" / f"chain_{chain:02d}" / f"generation_{generation:02d}"
            (generation_dir / "prompt.txt").parent.mkdir(parents=True, exist_ok=True)
            (generation_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            code, attempts, obtain_error = obtain_candidate(prompt, chain, generation, generation_dir, args)
            if code is None:
                row = _blank_observation(chain, generation, seed_name, "generation_failed", obtain_error)
                row["attempts"] = attempts
                observations.append(row)
                chain_history.append((parent_code, parent_eval))
                print(f"  generation {generation}: generation failed ({obtain_error})", flush=True)
                continue

            evaluation, eval_error = evaluate_candidate(code, probes, args.evaluation_timeout)
            if evaluation is None:
                row = _blank_observation(chain, generation, seed_name, "evaluation_failed", eval_error)
                row["attempts"] = attempts
                observations.append(row)
                chain_history.append((parent_code, parent_eval))
                (generation_dir / "evaluation_error.txt").write_text(eval_error or "unknown", encoding="utf-8")
                print(f"  generation {generation}: evaluation failed ({eval_error})", flush=True)
                continue

            ancestor_code, ancestor_eval = chain_history[0]
            previous = chain_history.copy()
            row = _blank_observation(chain, generation, seed_name, "valid", None)
            row.update(
                attempts=attempts,
                l0_parent=code_distance(code, parent_code),
                l0_ancestor=code_distance(code, ancestor_code),
                l2_parent_calibration=behavior_distance(evaluation, parent_eval, "calibration"),
                l2_parent_validation=behavior_distance(evaluation, parent_eval, "validation"),
                l2_ancestor_calibration=behavior_distance(evaluation, ancestor_eval, "calibration"),
                l2_ancestor_validation=behavior_distance(evaluation, ancestor_eval, "validation"),
                l2_nearest_history_calibration=min(
                    behavior_distance(evaluation, item[1], "calibration") for item in previous
                ),
                l2_nearest_history_validation=min(
                    behavior_distance(evaluation, item[1], "validation") for item in previous
                ),
                l2_nearest_history_calibration_adjusted=min(
                    noise_adjusted_behavior_distance(evaluation, item[1], "calibration") for item in previous
                ),
                l2_nearest_history_validation_adjusted=min(
                    noise_adjusted_behavior_distance(evaluation, item[1], "validation") for item in previous
                ),
                self_noise_calibration=self_noise(evaluation, "calibration"),
                self_noise_validation=self_noise(evaluation, "validation"),
                mean_gain_calibration=mean_gain(evaluation, "calibration"),
                mean_gain_validation=mean_gain(evaluation, "validation"),
            )
            observations.append(row)
            chain_history.append((code, evaluation))  # no fitness selection: every valid child becomes parent
            (generation_dir / "metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
            )
            print(
                f"  generation {generation}: valid, "
                f"L0(parent)={row['l0_parent']:.3f}, "
                f"L2(parent,val)={row['l2_parent_validation']:.3f}, "
                f"gain(val)={row['mean_gain_validation']:+.2f}%",
                flush=True,
            )
    return observations, histories


def summarize(
    args: argparse.Namespace,
    observations: list[dict[str, Any]],
    histories: list[list[tuple[str, CandidateEvaluation]]],
) -> dict[str, Any]:
    valid_rows = [row for row in observations if row["status"] in ("seed", "valid")]
    validation_noises = [float(row["self_noise_validation"]) for row in valid_rows]
    noise95 = float(np.quantile(validation_noises, 0.95)) if validation_noises else math.nan
    for row in observations:
        if row["status"] == "valid":
            row["behavior_revisit"] = bool(float(row["l2_nearest_history_validation"]) <= noise95 + 1e-12)

    curves: dict[str, list[float]] = {"calibration": [], "validation": []}
    adjusted_curves: dict[str, list[float]] = {"calibration": [], "validation": []}
    for generation in range(args.generations + 1):
        endpoints = [history[min(generation, len(history) - 1)][1] for history in histories]
        for split in curves:
            curves[split].append(effective_dimension(endpoints, split))
            adjusted_curves[split].append(effective_dimension(endpoints, split, noise_adjusted=True))

    late_rows = [
        row
        for row in observations
        if row["status"] == "valid" and int(row["generation"]) > args.generations / 2
    ]
    late_revisit_rate = (
        float(np.mean([bool(row["behavior_revisit"]) for row in late_rows])) if late_rows else math.nan
    )
    local_adjusted_revisit_rate = (
        float(np.mean([float(row["l2_nearest_history_validation_adjusted"]) <= 1e-12 for row in late_rows]))
        if late_rows
        else math.nan
    )
    attempted = [row for row in observations if int(row["generation"]) > 0]
    invalid_rate = float(np.mean([row["status"] != "valid" for row in attempted])) if attempted else math.nan
    initial_cal, final_cal = curves["calibration"][0], curves["calibration"][-1]
    initial_val, final_val = curves["validation"][0], curves["validation"][-1]
    cal_drop = 1.0 - final_cal / initial_cal if initial_cal > 1e-12 else 0.0
    val_drop = 1.0 - final_val / initial_val if initial_val > 1e-12 else 0.0

    sufficient = args.chains >= 3 and args.generations >= 4 and initial_val >= 2.0 and not args.self_test
    if not sufficient:
        decision = "INSUFFICIENT_FOR_SCIENTIFIC_DECISION"
    elif cal_drop >= 0.20 and val_drop >= 0.20 and late_revisit_rate >= 0.50:
        decision = "STRONG_EVIDENCE_OF_MUTATION_COLLAPSE"
    elif cal_drop >= 0.20 or val_drop >= 0.20 or late_revisit_rate >= 0.50:
        decision = "PARTIAL_EVIDENCE_REQUIRES_FOLLOWUP"
    else:
        decision = "NO_COLLAPSE_DETECTED_AT_THIS_BUDGET"

    return {
        "mode": "self-test" if args.self_test else "llm",
        "decision": decision,
        "preregistered_rule": {
            "minimum_design": "at least 3 chains, 4 generations, and validation initial effective dimension >= 2",
            "strong_evidence": "effective dimension drop >=20% on calibration and validation, and late revisit rate >=50%",
        },
        "n_attempted_generations": len(attempted),
        "n_valid_children": sum(row["status"] == "valid" for row in attempted),
        "invalid_rate": invalid_rate,
        "behavior_noise_95_validation": noise95,
        "late_behavior_revisit_rate": late_revisit_rate,
        "effective_dimension": curves,
        "effective_dimension_drop": {"calibration": cal_drop, "validation": val_drop},
        "sensitivity_noise_adjusted": {
            "note": "exploratory, not the preregistered decision statistic",
            "late_behavior_revisit_rate": local_adjusted_revisit_rate,
            "effective_dimension": adjusted_curves,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, observations: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for chain in sorted({int(row["chain"]) for row in observations}):
        rows = [row for row in observations if int(row["chain"]) == chain and row["status"] == "valid"]
        axes[0].plot(
            [int(row["generation"]) for row in rows],
            [float(row["l2_ancestor_validation"]) for row in rows],
            marker="o",
            label=f"chain {chain}",
        )
    axes[0].set(title="Held-out behavior distance to ancestor", xlabel="Generation", ylabel="Mean edge Jaccard distance")
    axes[0].legend(fontsize=8)
    for split, values in summary["effective_dimension"].items():
        axes[1].plot(range(len(values)), values, marker="o", label=split)
    axes[1].set(title="Population effective dimension", xlabel="Generation", ylabel="Participation ratio")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any], runtime: float) -> None:
    mode_note = (
        "本次是管线自检，不含 LLM 采样，不能用于研究结论。"
        if args.self_test
        else "本次为冻结选择的 LLM 变异链；所有有效子代无论性能都成为下一代父代。"
    )
    curves = summary["effective_dimension"]
    lines = [
        "# LLM repair 变异链坍缩审计",
        "",
        mode_note,
        "",
        "## 判定",
        "",
        f"**{summary['decision']}**",
        "",
        "预注册的强证据要求：calibration 与 validation 的有效自由度都下降至少 20%，且后半程行为重访率至少 50%。",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 有效子代 | {summary['n_valid_children']} / {summary['n_attempted_generations']} |",
        f"| 无效/失败率 | {summary['invalid_rate']:.1%} |",
        f"| validation 行为噪声 95% 分位 | {summary['behavior_noise_95_validation']:.4f} |",
        f"| 后半程行为重访率 | {summary['late_behavior_revisit_rate']:.1%} |",
        f"| 后半程局部噪声校正重访率（敏感性） | {summary['sensitivity_noise_adjusted']['late_behavior_revisit_rate']:.1%} |",
        f"| calibration 有效自由度 | {curves['calibration'][0]:.2f} → {curves['calibration'][-1]:.2f} |",
        f"| validation 有效自由度 | {curves['validation'][0]:.2f} → {curves['validation'][-1]:.2f} |",
        f"| validation 局部噪声校正有效自由度（敏感性） | "
        f"{summary['sensitivity_noise_adjusted']['effective_dimension']['validation'][0]:.2f} → "
        f"{summary['sensitivity_noise_adjusted']['effective_dimension']['validation'][-1]:.2f} |",
        "",
        "## 设计",
        "",
        f"- 模型：`{args.resolved_model}`。",
        f"- {args.chains} 条独立链 × {args.generations} 代；起点循环使用 sequential、regret-2、block-reinsert、new-route 四种 repair。",
        f"- probe：CVRP/VRPTW 各 {args.probe_instances} 个实例 × {args.probe_repeats} 个状态 × {len(PROBE_DESTROYERS)} 种残解来源。",
        "- 前半实例只用于 calibration，后半实例只用于 validation。",
        "- L0：AST node-type 3-gram Jaccard；L2：固定残解上输出边集合的平均 Jaccard。",
        "- 每个算子用两套 repair RNG 重复测量，validation 自距离的 95% 分位定义行为重访噪声线。",
        "- 另报告逐算子局部噪声扣除的探索性敏感性分析；它不替换预注册判定。",
        "- 没有 fitness 选择；有效子代总是接替父代，从而把生成退化与选择退化分开。",
        "",
        "## 解释规则",
        "",
        "- 强证据：优先进入 verbalized sampling、双约束和缺口驱动变异实验。",
        "- 只有部分证据：增加链数/代数，并检查 L0 变远而 L2 不变的换皮现象。",
        "- 未检出：只能说明当前模型、prompt、预算和 repair 任务下未检出，不能证明 LLM 变异普遍不会坍缩。",
        "- 初始有效自由度不足 2：先换更分化的种子，不解释收缩曲线。",
        "",
        "## 复现",
        "",
        "```powershell",
        f"python {SCRIPT_INVOCATION} --chains {args.chains} --generations {args.generations} --probe-instances {args.probe_instances} --probe-repeats {args.probe_repeats} --customers {args.customers} --seed {args.seed} --output {args.output.as_posix()}",
        "```",
        "",
        f"运行耗时 {runtime:.1f} 秒。原始回复和候选代码见 `chains/`，逐代指标见 `observations.csv`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--probe-instances", type=int, default=4)
    parser.add_argument("--probe-repeats", type=int, default=2)
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--remove-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=None, help=f"OpenAI-compatible base URL; defaults to ${BASE_URL_ENV}")
    parser.add_argument("--api-key-env", default=API_KEY_ENV, help="environment variable containing the API key")
    parser.add_argument("--model", default=None, help=f"model name; defaults to ${MODEL_ENV}")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="dotenv file loaded without printing values")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--evaluation-timeout", type=float, default=25.0)
    parser.add_argument("--attempts-per-generation", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="reuse saved candidate/response files")
    parser.add_argument("--self-test", action="store_true", help="use deterministic built-in responses; never draw conclusions")
    args = parser.parse_args()
    if args.chains < 1 or args.generations < 1:
        parser.error("chains and generations must be positive")
    if args.probe_instances < 2 or args.probe_instances % 2:
        parser.error("probe-instances must be an even integer >= 2 for calibration/validation splitting")
    if args.probe_repeats < 1 or args.customers < 10:
        parser.error("probe-repeats must be positive and customers must be >= 10")
    if not 0 < args.remove_fraction < 1:
        parser.error("remove-fraction must be in (0, 1)")
    if args.attempts_per_generation < 1:
        parser.error("attempts-per-generation must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    load_dotenv(args.env_file.resolve())
    args.resolved_model = "self-test" if args.self_test else (args.model or _env_value(MODEL_ENV) or "UNSPECIFIED")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.pop("api_key_env", None)  # do not persist even the custom secret-variable name unnecessarily
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    probes = build_probes(args)
    print(f"Built {len(probes)} probes ({sum(p.split == 'calibration' for p in probes)} calibration, "
          f"{sum(p.split == 'validation' for p in probes)} validation)", flush=True)
    observations, histories = run_chains(args, probes)
    summary = summarize(args, observations, histories)
    write_csv(args.output / "observations.csv", observations)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_plot(args.output / "trajectories.png", observations, summary)
    runtime = time.perf_counter() - started
    write_report(args.output / "REPORT.md", args, summary, runtime)
    print(f"Decision: {summary['decision']}", flush=True)
    print(f"Wrote results to {args.output} ({runtime:.1f}s)", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
