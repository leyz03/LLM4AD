"""Paired TSP/CVRP constructive benchmarks with independent search/test data.

Uses LLM4AD's published task interfaces and uniform random instance family,
not TSPLIB/CVRPLIB or a claim of known-optimum gaps. All methods share this
stricter evaluator. Evaluation is process-isolated by SecureEvaluator; it is
not an OS security sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.method.hypoevo.hypoevo import validate_program


def integer_action(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('action must be an integer node ID')
    return int(value)


class RoutingEvaluation(Evaluation):
    def __init__(self, task='tsp', size=50, instances=16, seed=2024,
                 capacity=40, timeout=30, split='search', detailed=True):
        if task not in ('tsp', 'cvrp') or size < 3 or instances < 1 or capacity < 9:
            raise ValueError('invalid routing configuration')
        # Read the upstream literal task definitions without importing its
        # package __init__, which eagerly loads plotting/evaluation dependencies.
        path = Path(__file__).resolve().parents[3] / 'llm4ad/task/optimization' / f'{task}_construct/template.py'
        definitions = {}
        for statement in ast.parse(path.read_text()).body:
            if isinstance(statement, ast.Assign):
                for name in statement.targets:
                    if isinstance(name, ast.Name) and name.id in {'template_program', 'task_description'}:
                        definitions[name.id] = ast.literal_eval(statement.value)
        template_program = definitions['template_program']
        task_description = definitions['task_description']
        super().__init__(template_program=template_program,
                         task_description=task_description +
                         '\nReturn an integer from the provided feasible unvisited nodes. '
                         'For CVRP, depot 0 is also allowed when not already at the depot. '
                         'Do not modify input arrays. Evaluation score is negative mean tour length.',
                         timeout_seconds=timeout, safe_evaluate=True, fork_proc=False,
                         exec_code=False)
        self.task, self.size, self.instances = task, size, instances
        self.seed, self.capacity, self.split, self.detailed = seed, capacity, split, detailed
        config = {'version': 1, 'task': task, 'size': size, 'instances': instances,
                  'seed': seed, 'capacity': capacity, 'split': split}
        self.protocol_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        rng = np.random.default_rng(seed)
        self.data = []
        for i in range(instances):
            n = size + (task == 'cvrp')
            coordinates = rng.random((n, 2))
            distance = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
            demands = rng.integers(1, 10, size=n)
            demands[0] = 0
            self.data.append((distance, demands))

    def baseline_program(self):
        tree = ast.parse(self.template_program)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        fn.body = ast.parse('return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])').body
        return ast.unparse(tree) + '\n'

    @staticmethod
    def readonly(a):
        result = a.copy()
        result.flags.writeable = False
        return result

    def run_instance(self, fn, distance, demands):
        remaining = set(range(1, len(distance)))
        current, capacity = 0, self.capacity
        route = [0]
        # A valid CVRP construction needs at most one depot return per customer.
        for _ in range(2 * len(distance)):
            if not remaining:
                route.append(0)
                break
            feasible = sorted(n for n in remaining if self.task == 'tsp' or demands[n] <= capacity)
            if not feasible:
                if current == 0:
                    raise ValueError('infeasible demands')
                route.append(0)
                current, capacity = 0, self.capacity
                continue
            nodes = np.array(feasible, dtype=int)
            # Match TSP's nearest-first input convention in the upstream task.
            if self.task == 'tsp':
                nodes = nodes[np.argsort(distance[current, nodes], kind='stable')]
                node = fn(current, 0, self.readonly(nodes), self.readonly(distance))
            else:
                node = fn(current, 0, self.readonly(nodes), capacity,
                          self.readonly(demands), self.readonly(distance))
            node = integer_action(node)
            if self.task == 'cvrp' and node == 0:
                if current == 0:
                    raise ValueError('repeated depot without progress')
                current, capacity = 0, self.capacity
                route.append(0)
                continue
            if node not in feasible:
                raise ValueError('infeasible, duplicate, or out-of-range node')
            remaining.remove(node)
            if self.task == 'cvrp':
                capacity -= int(demands[node])
                if capacity < 0:
                    raise ValueError('capacity violation')
            route.append(node)
            current = node
        if remaining or route[-1] != 0:
            raise ValueError('incomplete route')
        length = float(sum(distance[a, b] for a, b in zip(route, route[1:])))
        if not np.isfinite(length) or length <= 0:
            raise ValueError('nonfinite or nonpositive tour cost')
        return -length

    def evaluate_program(self, program_str, callable_func=None, **kwargs):
        scores = {}
        try:
            code = validate_program(program_str, self.template_program)
            # Fresh globals and seeds per instance prevent cross-instance state leakage.
            entry = next(n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
            for i, (distance, demands) in enumerate(self.data):
                np.random.seed(self.seed + i)
                random.seed(self.seed + i)
                scope = {}
                exec(code, scope)
                scores[f'{self.protocol_id}:{i}'] = self.run_instance(scope[entry], distance, demands)
            result = {'score': float(np.mean(list(scores.values()))),
                      'instance_scores': scores, 'protocol_id': self.protocol_id}
        except Exception as exc:
            result = {'score': None, 'instance_scores': scores, 'protocol_id': self.protocol_id,
                      'error': f'{type(exc).__name__}: {exc}'}
        return result if self.detailed else result['score']
