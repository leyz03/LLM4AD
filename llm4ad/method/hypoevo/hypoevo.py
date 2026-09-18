"""Hypothesis-guided evolution with explicitly updated, empirical memory.

This is a new method, not an in-place change to published PartEvo or EoH.
The supplied evaluator must return per-instance scores and a protocol ID to
support paired evidence. Scalar-only evaluators still work without that claim.
"""
from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path

from ...base import SecureEvaluator
from .memory import ExperimentMemory, Hypothesis
from .population import Candidate, NichePopulation
from .prompt import context, reflection_prompt, generation_prompt, summary_prompt
from .runtime import BudgetedLLM, BudgetExhausted


def extract_tag(text, tag):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.S)
    if m is None or not m.group(1).strip():
        raise ValueError(f'missing {tag}')
    return m.group(1).strip()


def parse_hypothesis(text):
    """Accept tagged, JSON-fenced or bare JSON without changing its contents.

Providers often omit XML wrappers around an otherwise valid JSON response.
Schema and prediction checks remain identical for all transport formats.
"""
    tagged = re.search(r'<hypothesis>(.*?)</hypothesis>', text, flags=re.S)
    fenced = re.search(r'```json\s*\n(.*?)```', text, flags=re.S)
    raw = tagged.group(1) if tagged else fenced.group(1) if fenced else text
    start = raw.find('{')
    if start < 0:
        raise ValueError('missing hypothesis JSON object')
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    return Hypothesis.from_dict(value)


def validate_program(code, template):
    """Strict interface/AST gate; deliberately not advertised as a security sandbox."""
    tree, reference = ast.parse(code), ast.parse(template)
    expected = next(n for n in reference.body if isinstance(n, ast.FunctionDef))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != expected.name:
        raise ValueError('expected exactly the template entry function')
    entry = functions[0]
    if ast.dump(entry.args, include_attributes=False) != ast.dump(expected.args, include_attributes=False):
        raise ValueError('function signature changed')
    if entry.decorator_list:
        raise ValueError('entry decorators not allowed')
    allowed = {'numpy', 'math', 'random', 'statistics', 'collections', 'itertools', 'functools'}
    blocked_calls = {'open', 'exec', 'eval', 'compile', '__import__', 'getattr', 'setattr',
                     'globals', 'locals', 'vars', 'input', 'breakpoint'}
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            raise ValueError('only imports and the entry function may appear at top level')
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split('.')[0] not in allowed for a in n.names):
            raise ValueError('unsupported import')
        if isinstance(n, ast.ImportFrom) and (n.level or (n.module or '').split('.')[0] not in allowed):
            raise ValueError('unsupported import')
        if isinstance(n, ast.Attribute) and n.attr.startswith('__'):
            raise ValueError('dunder access not allowed')
        if isinstance(n, ast.Name) and n.id in blocked_calls:
            raise ValueError('unsupported builtin')
    compile(tree, '<candidate>', 'exec')
    return ast.unparse(tree) + '\n'


def normalized_result(value):
    if isinstance(value, (int, float)):
        value = {'score': float(value)}
    if not isinstance(value, dict):
        return {'score': None, 'error': 'evaluation_failed_or_timed_out'}
    result = dict(value)
    score = result.get('score')
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        return {'score': None, 'error': result.get('error', 'nonfinite_or_missing_score')}
    result['score'] = float(score)
    return result


class HypoEvo:
    def __init__(self, llm, evaluation, log_dir, *, max_calls=30, pop_size=16,
                 partitions=4, seed=2024, operators=('reflection', 'summary', 'cross'),
                 hypothesis_enabled=True, memory_enabled=True, include_failures=True,
                 summary_interval=6, memory_capacity=128, initial_candidate=None):
        if not operators or set(operators) - {'reflection', 'summary', 'cross'}:
            raise ValueError('unknown or empty operator set')
        if summary_interval < 1:
            raise ValueError('summary_interval must be positive')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if any((self.log_dir / f).exists() for f in ('trials.jsonl', 'candidates.jsonl')):
            raise ValueError('use a fresh run directory; in-run resume is not supported')
        self.llm = llm if isinstance(llm, BudgetedLLM) else BudgetedLLM(llm, max_calls, self.log_dir / 'llm_trace.jsonl')
        self.evaluation = evaluation
        self.evaluator = SecureEvaluator(evaluation)
        self.population = NichePopulation(pop_size, partitions, seed)
        self.memory = ExperimentMemory(self.log_dir / 'trials.jsonl', memory_capacity)
        self.rng = random.Random(seed)
        self.operators = operators
        self.hypothesis_enabled = hypothesis_enabled
        self.memory_enabled = memory_enabled
        self.include_failures = include_failures
        self.summary_interval = summary_interval
        self.initial_candidate = initial_candidate
        self.attempts = self.valid = 0

    def _summary(self, parents):
        if not self.memory_enabled:
            return ''
        mem = self.memory
        if (mem.records and self.llm.remaining >= 2 and
                (not mem.summary or mem.version - mem.summary_version >= self.summary_interval)):
            records = mem.retrieve([p.id for p in parents], 'summary', 8, self.include_failures)
            if records:
                source_version = mem.version
                reply = self.llm.draw_sample(messages=summary_prompt(
                    self.evaluation.task_description, mem.summary, records))
                try:
                    mem.update_summary(extract_tag(reply, 'summary'), source_version)
                except ValueError:
                    pass  # Keep previous summary; malformed output still consumes a call.
        return mem.summary

    def _step(self, operator):
        parents = self.population.select(self.rng, cross=operator == 'cross')
        if not parents:
            operator = 'initialization'
        # Before feature niches exist, expand via random-parent reflection.
        elif (self.population.partitions > 1 and not self.population.anchors
              and 'reflection' in self.operators):
            operator = 'reflection'
        summary = self._summary(parents) if operator == 'summary' else ''
        evidence = self.memory.retrieve([p.id for p in parents], operator, 6, self.include_failures) if self.memory_enabled else []
        ctx = context(self.evaluation.task_description, self.evaluation.template_program,
                      parents, evidence, summary)
        self.attempts += 1
        child_id = self.attempts
        hypothesis, proposal, code = None, '', ''
        metadata_errors = []
        result = {'score': None}
        try:
            if operator == 'reflection' and self.llm.remaining >= 2:
                proposal = self.llm.draw_sample(messages=reflection_prompt(ctx, self.hypothesis_enabled))
                if self.hypothesis_enabled:
                    try:
                        hypothesis = parse_hypothesis(proposal)
                    except (ValueError, TypeError) as exc:
                        metadata_errors.append('reflection_hypothesis: ' + str(exc))
                        proposal = 'No valid hypothesis was registered. Propose one before implementation.'
                    else:
                        proposal = '<hypothesis>' + json.dumps(asdict(hypothesis), ensure_ascii=False) + '</hypothesis>'
            response = self.llm.draw_sample(messages=generation_prompt(
                ctx, operator, proposal, self.hypothesis_enabled))
            if self.hypothesis_enabled and hypothesis is None:
                try:
                    hypothesis = parse_hypothesis(response)
                except (ValueError, TypeError) as exc:
                    metadata_errors.append('generation_hypothesis: ' + str(exc))
            # In a two-call reflection, the first call's stored hypothesis is
            # authoritative. Implementation replies need only concept + code;
            # any repeated/paraphrased hypothesis cannot overwrite registration.
            try:
                concept = extract_tag(response, 'concept')
            except ValueError as exc:
                metadata_errors.append('concept: ' + str(exc))
                concept = 'Generated candidate (description omitted)'
            blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, flags=re.S)
            if len(blocks) != 1:
                raise ValueError('expected one Python code block')
            code = validate_program(blocks[0], self.evaluation.template_program)
            result = normalized_result(self.evaluator.evaluate_program(code))
            if result['score'] is not None:
                self.valid += 1
        except BudgetExhausted:
            self.memory.add(child_id=child_id, parents=parents, operator=operator,
                            hypothesis=hypothesis, child_code=code,
                            result={'score': None}, error='generation_interrupted')
            raise
        except (ValueError, SyntaxError, TypeError) as exc:
            result = {'score': None, 'error': f'{type(exc).__name__}: {exc}'}
        result['metadata'] = {
            'hypothesis_status': ('registered' if hypothesis else 'missing_or_invalid')
                                  if self.hypothesis_enabled else 'disabled',
            'errors': metadata_errors,
        }
        self.memory.add(child_id=child_id, parents=parents, operator=operator,
                        hypothesis=hypothesis, child_code=code, result=result)
        retained = False
        if result['score'] is not None:
            retained = self.population.register(Candidate(child_id, code, concept, result))
        with (self.log_dir / 'candidates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'id': child_id, 'operator': operator, 'parent_ids': [p.id for p in parents],
                                'code': code, 'result': result, 'retained': retained,
                                'calls': self.llm.calls,
                                'best_score': self.population.best.score if self.population.best else None},
                               ensure_ascii=False, allow_nan=False) + '\n')

    def run(self):
        if self.initial_candidate is not None:
            self.population.register(self.initial_candidate)
        else:
            code = validate_program(self.evaluation.template_program, self.evaluation.template_program)
            result = normalized_result(self.evaluator.evaluate_program(code))
            if result['score'] is not None:
                self.population.register(Candidate(0, code, 'Task template', result))
        try:
            while self.llm.remaining > 0:
                self._step(self.operators[self.attempts % len(self.operators)])
        except BudgetExhausted:
            pass
        state = {'attempts': self.attempts, 'valid': self.valid, 'llm_calls': self.llm.calls,
                 'stop_reason': self.llm.stop_reason or 'budget_exhausted',
                 'active_partitions': self.population.active_partitions,
                 'population': [asdict(x) for x in self.population.members],
                 'best': asdict(self.population.best) if self.population.best else None,
                 'summary': self.memory.summary, 'summary_version': self.memory.summary_version}
        (self.log_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False))
        return self.population.best
