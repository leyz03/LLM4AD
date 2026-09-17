"""Portable, serial, resumable-at-run-boundaries HypoEvo experiment matrix.

Default is a dry run. --smoke never reads credentials or invokes a remote LLM.
--run explicitly enables real calls using local environment configuration.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.environ['LLM4AD_MINIMAL_IMPORTS'] = '1'

import numpy as np
from llm4ad.base import LLM, SecureEvaluator
from llm4ad.method.hypoevo import HypoEvo
from llm4ad.method.hypoevo.hypoevo import normalized_result
from llm4ad.method.hypoevo.population import Candidate
from llm4ad.method.hypoevo.runtime import BudgetedLLM
from evaluation import RoutingEvaluation

VARIANTS = {
    'hypoevo': {},
    'no_memory': {'memory_enabled': False},
    'no_hypothesis': {'hypothesis_enabled': False},
    'no_failures': {'include_failures': False},
    'no_partition': {'partitions': 1},
    'no_reflection': {'operators': ('summary', 'cross')},
    'no_summary': {'operators': ('reflection', 'cross')},
    'no_cross': {'operators': ('reflection', 'summary')},
}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_env(path):
    """Read only LLM configuration assignments; never execute shell content."""
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix('export ')
            key, sep, raw = line.partition('=')
            key = key.strip()
            if sep and key.startswith(('LLM_', 'LLM4AD_')):
                values = shlex.split(raw, comments=True)
                if len(values) == 1:
                    os.environ.setdefault(key, values[0])
    for source, dest in [('LLM_API_KEY', 'LLM4AD_API_KEY'), ('LLM_MODEL', 'LLM4AD_API_MODEL'),
                         ('LLM_BASE_URL', 'LLM4AD_API_BASE_URL')]:
        if source in os.environ:
            os.environ.setdefault(dest, os.environ[source])


class FakeLLM(LLM):
    """Deterministic varied code for plumbing checks, never research evidence."""
    def __init__(self, task):
        super().__init__(do_auto_trim=False)
        self.task, self.count = task, 0
        self.last_usage = {}

    def draw_sample(self, prompt='', *args, **kwargs):
        self.count += 1
        messages = kwargs.get('messages')
        text = json.dumps(messages) if messages else prompt
        if messages and 'Summarize empirical' in messages[0]['content']:
            return '<summary>Offline fixture only: paired improvements do not prove a mechanism.</summary>'
        hypothesis = {'observation': 'Code-based conjecture, not a measured failure.',
                      'mechanism': 'Distance lookahead may change route order.',
                      'intervention': 'Modify one distance score term.',
                      'prediction': 'mean_score_increases', 'risk': 'May increase route length.'}
        h = '<hypothesis>' + json.dumps(hypothesis) + '</hypothesis>'
        if messages and 'You propose' in messages[0]['content']:
            return h if 'mean_score_increases' in text else '<reflection>Try a distance lookahead term.</reflection>'
        tree = ast.parse(self.task.template_program)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        bodies = [
            'return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])',
            'scores = []\nfor n in unvisited_nodes:\n    scores.append(distance_matrix[current_node, n] - 0.2 * distance_matrix[n, 0])\nreturn int(unvisited_nodes[np.argmin(scores)])',
            'scores = distance_matrix[current_node, unvisited_nodes] + 0.15 * np.mean(distance_matrix[unvisited_nodes], axis=1)\nreturn int(unvisited_nodes[np.argmin(scores)])',
            'n = min(unvisited_nodes, key=lambda n: distance_matrix[current_node, n] + 0.1 * distance_matrix[n, 0])\nreturn int(n)',
            'scores = np.square(distance_matrix[current_node, unvisited_nodes]) - 0.05 * distance_matrix[unvisited_nodes, 0]\nreturn int(unvisited_nodes[np.argmin(scores)])',
        ]
        fn.body = ast.parse(bodies[self.count % len(bodies)]).body
        code = ast.unparse(tree)
        if messages:
            return h + '\n<concept>Offline test heuristic.</concept>\n```python\n' + code + '\n```'
        return '{Offline test heuristic.}\n```python\n' + code + '\n```'

    def close(self):
        pass


class BaselineRecorder:
    def __init__(self, baseline, path, llm):
        self.best, self.path, self.llm = baseline, path, llm
        self.attempts = self.valid = 0

    def record_parameters(self, *args):
        pass

    def register_function(self, func, program='', **kwargs):
        self.attempts += 1
        score = func.score
        if score is not None and np.isfinite(score):
            self.valid += 1
            if score > self.best.score:
                self.best = Candidate(self.attempts, program, func.algorithm, {'score': float(score)})
        else:
            score = None
        with self.path.open('a') as f:
            f.write(json.dumps({'id': self.attempts, 'score': score, 'code': program,
                                'calls': self.llm.calls, 'best_score': self.best.score}) + '\n')

    def finish(self):
        pass


def task_for(config, name, split):
    t = config['task']
    return RoutingEvaluation(task=name, size=t['size'],
                             instances=t['train_instances' if split == 'search' else 'test_instances'],
                             seed=t['train_seed' if split == 'search' else 'test_seed'],
                             capacity=t['capacity'], timeout=t['timeout'], split=split)


def run_worker(config, task_name, method, seed, out):
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    train = task_for(config, task_name, 'search')
    evaluator = SecureEvaluator(train)
    base_code = train.baseline_program()
    base_result = normalized_result(evaluator.evaluate_program(base_code))
    if base_result['score'] is None:
        raise RuntimeError('common nearest-neighbor seed failed evaluation')
    baseline = Candidate(0, base_code, 'Nearest feasible neighbor', base_result)
    if config['backend'] == 'fake':
        delegate = FakeLLM(train)
    else:
        from llm4ad.tools.llm.llm_api_https import HttpsApi
        load_env(config['dotenv'])
        delegate = HttpsApi(host=config['endpoint'], model=config['model'],
                            key=os.environ['LLM4AD_API_KEY'], max_retries=1,
                            do_auto_trim=False, **config['llm'])
    llm = BudgetedLLM(delegate, config['max_calls'], out / 'llm_trace.jsonl')
    if method == 'eoh':
        from llm4ad.method.eoh import EoH
        scalar_task = copy.copy(train)
        scalar_task.detailed = False
        recorder = BaselineRecorder(baseline, out / 'candidates.jsonl', llm)
        search = EoH(llm=llm, evaluation=scalar_task, profiler=recorder,
                     max_generations=None, max_sample_nums=config['max_calls'],
                     pop_size=config['population_size'], num_samplers=1, num_evaluators=1)
        search.run()
        best, attempts, valid = recorder.best, recorder.attempts, recorder.valid
    else:
        options = {'pop_size': config['population_size'], 'partitions': config['partitions'],
                   'seed': seed, 'summary_interval': config['summary_interval'],
                   'operators': tuple(config['operators']), **VARIANTS[method]}
        search = HypoEvo(llm, train, out, initial_candidate=baseline, **options)
        best = search.run()
        attempts, valid = search.attempts, search.valid
    # Test data is constructed only after search; it never enters prompts or memory.
    test = task_for(config, task_name, 'test')
    test_evaluator = SecureEvaluator(test)
    tested = normalized_result(test_evaluator.evaluate_program(best.code))
    base_test = normalized_result(test_evaluator.evaluate_program(base_code))
    (out / 'best.py').write_text(best.code)
    success = llm.stop_reason != 'api_error' and tested['score'] is not None and base_test['score'] is not None
    status = 'completed' if success and llm.calls == config['max_calls'] else 'incomplete'
    result = {'status': status, 'protocol_fingerprint': config['fingerprint'],
              'backend': config['backend'], 'task': task_name, 'method': method, 'seed': seed,
              'llm_calls': llm.calls, 'failed_calls': llm.failures, 'usage': llm.usage,
              'stop_reason': llm.stop_reason or 'budget_exhausted', 'attempts': attempts, 'valid': valid,
              'search_score': best.score, 'test': tested, 'baseline_test': base_test,
              'elapsed_seconds': time.monotonic() - started}
    if success:
        result['test_mean_cost'] = -tested['score']
        result['improvement_over_nn_percent'] = 100 * (tested['score'] - base_test['score']) / abs(base_test['score'])
    dump(out / 'result.json', result)
    return 0 if status == 'completed' else 1


def source_hash():
    paths = list((ROOT / 'llm4ad/method/hypoevo').glob('*.py')) + list(HERE.glob('*.py'))
    paths += list((ROOT / 'llm4ad/method/eoh').glob('*.py'))
    paths += list((ROOT / 'llm4ad/base').glob('*.py'))
    paths += [ROOT / 'llm4ad/tools/llm/llm_api_https.py']
    paths += [ROOT / p for p in ('llm4ad/__init__.py', 'llm4ad/method/__init__.py', 'llm4ad/task/__init__.py')]
    paths += [ROOT / f'llm4ad/task/optimization/{t}_construct/template.py' for t in ('tsp', 'cvrp')]
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(str(p.relative_to(ROOT)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs/pilot')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--suite', choices=['pilot', 'ablations'], default='pilot')
    parser.add_argument('--methods', nargs='+', choices=[*VARIANTS, 'eoh'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--tasks', nargs='+', choices=['tsp', 'cvrp'])
    parser.add_argument('--max-calls', type=int)
    parser.add_argument('--dotenv', type=Path, default=ROOT / '.env')
    parser.add_argument('--worker', nargs=3, metavar=('TASK', 'METHOD', 'SEED'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        task_name, method, seed = args.worker
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            return run_worker(config, task_name, method, int(seed), args.output_dir)
        except Exception as exc:
            dump(args.output_dir / 'result.json', {'status': 'failed', 'task': task_name,
                 'method': method, 'seed': int(seed), 'error_type': type(exc).__name__,
                 'protocol_fingerprint': config['fingerprint']})
            raise
    if args.suite == 'ablations':
        config['methods'] = [*VARIANTS, 'eoh']
    for attr in ('methods', 'seeds', 'tasks', 'max_calls'):
        if getattr(args, attr) is not None:
            config[attr] = getattr(args, attr)
    config['backend'] = 'fake' if args.smoke else 'https'
    if args.smoke:
        config['seeds'] = [2024]
        config['max_calls'] = args.max_calls or 12
        config['population_size'], config['partitions'] = 4, 2
        config['task'].update(size=8, train_instances=3, test_instances=4)
    if (config['max_calls'] < 1 or not 1 <= config['partitions'] <= config['population_size'] or
            config['task']['train_seed'] == config['task']['test_seed']):
        raise ValueError('invalid budget, population, or non-disjoint data seeds')
    if any(m not in VARIANTS and m != 'eoh' for m in config['methods']):
        raise ValueError('unknown method')
    config['source_hash'] = source_hash()
    config['python_version'] = sys.version.split()[0]
    config['package_versions'] = {name: importlib.metadata.version(name) for name in
                                  ('numpy', 'scipy', 'scikit-learn', 'codebleu', 'tree-sitter', 'tree-sitter-python')}
    if args.run:
        load_env(args.dotenv)
        if not os.environ.get('LLM4AD_API_KEY'):
            raise ValueError('Set LLM_API_KEY or LLM4AD_API_KEY; do not put keys in config files')
        config['endpoint'] = os.environ.get('LLM4AD_API_BASE_URL') or os.environ.get('LLM4AD_API_HOST')
        config['model'] = os.environ.get('LLM4AD_API_MODEL')
        if not config['endpoint'] or not config['model']:
            raise ValueError('Set LLM_BASE_URL and LLM_MODEL')
    config['dotenv'] = str(args.dotenv.resolve())
    fingerprinted = {k: v for k, v in config.items() if k != 'dotenv'}
    config['fingerprint'] = hashlib.sha256(json.dumps(fingerprinted, sort_keys=True).encode()).hexdigest()
    runs = [(t, m, s) for t in config['tasks'] for m in config['methods'] for s in config['seeds']]
    print(json.dumps({'runs': len(runs), 'maximum_calls': len(runs) * config['max_calls'],
                      'backend': config['backend'], 'methods': config['methods'],
                      'task': config['task']}, indent=2), flush=True)
    if not (args.smoke or args.run):
        print('Dry run only. Use --smoke for offline verification or --run for real LLM calls.')
        return 0
    out = args.output_dir.resolve()
    resolved = out / 'resolved_config.json'
    if resolved.exists() and json.loads(resolved.read_text())['fingerprint'] != config['fingerprint']:
        raise ValueError('output directory belongs to a different protocol/source; choose a new one')
    dump(resolved, config)
    for task_name, method, seed in runs:
        run_dir = out / task_name / method / str(seed)
        result_path = run_dir / 'result.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'completed':
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f'Incomplete run at {run_dir}; preserve it and choose a new output directory')
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--config', str(resolved),
                   '--output-dir', str(run_dir), '--worker', task_name, method, str(seed)]
        print(f'Running {task_name}/{method}/{seed}', flush=True)
        with (run_dir / 'console.log').open('w') as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=(os.name == 'posix'))
                returncode = process.wait(timeout=config['worker_timeout'])
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                dump(result_path, {'status': 'timeout', 'task': task_name, 'method': method,
                                   'seed': seed, 'protocol_fingerprint': config['fingerprint']})
                returncode = 1
        if returncode:
            print(f'Stopped on incomplete run. Inspect {run_dir / "console.log"}', flush=True)
            return 1
    from analyze_results import analyze
    analyze(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
