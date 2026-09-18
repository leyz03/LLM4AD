"""Offline checks for evidence, legality, budgets, and ablation behavior."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
os.environ['LLM4AD_MINIMAL_IMPORTS'] = '1'

from llm4ad.base import LLM, SecureEvaluator
from llm4ad.method.hypoevo import HypoEvo
from llm4ad.method.hypoevo.hypoevo import validate_program, parse_hypothesis
from llm4ad.method.hypoevo.memory import ExperimentMemory, Hypothesis, paired_evidence
from llm4ad.method.hypoevo.population import Candidate, NichePopulation
from llm4ad.method.hypoevo.runtime import BudgetedLLM, BudgetExhausted
from evaluation import RoutingEvaluation
from run_experiment import FakeLLM, VARIANTS


def outcome(values, protocol='fixed'):
    return {'score': float(np.mean(values)), 'instance_scores': dict(zip('abc', values)),
            'protocol_id': protocol}


def test_paired_results_not_mean_only():
    result = paired_evidence(outcome([-5, -7]), outcome([-4, -8]))
    assert result['status'] == 'unchanged'
    assert result['wins'] == result['losses'] == 1
    assert result['instance_deltas'] == [1, -1]


@pytest.mark.parametrize('child', [outcome([-4, -6], 'other'),
                                 {'score': -5}, outcome([-4])])
def test_mismatched_protocol_or_instances_not_evidence(child):
    assert paired_evidence(outcome([-5, -7]), child)['status'] == 'insufficient_evidence'


def test_invalid_is_not_evidence_against_mechanism():
    assert paired_evidence(outcome([-5]), {'score': None})['status'] == 'invalid_child'


def test_hypothesis_schema():
    with pytest.raises(ValueError):
        Hypothesis.from_dict({'prediction': 'all inputs improve'})


@pytest.mark.parametrize('wrapper', ['{}', '<hypothesis>{}</hypothesis>', '```json\n{}\n```'])
def test_real_model_json_transport_formats(wrapper):
    value = {'observation': 'conjecture', 'mechanism': 'locality', 'intervention': 'lookahead',
             'prediction': 'mean_score_increases', 'risk': 'longer routes'}
    assert parse_hypothesis(wrapper.format(json.dumps(value))).intervention == 'lookahead'
    with pytest.raises(ValueError):
        parse_hypothesis(wrapper.format('{"prediction": "unverified"}'))


def test_failure_memory_survives_retention(tmp_path):
    parent = Candidate(0, 'def f(x): return x\n', 'seed', outcome([-5, -7]))
    mem = ExperimentMemory(tmp_path / 'trials.jsonl', capacity=2)
    for i, values in enumerate([[-4, -6], [-8, -9], [-3, -5]], start=1):
        mem.add(child_id=i, parents=[parent], operator='reflection', hypothesis=None,
                child_code=f'def f(x): return x+{i}\n', result=outcome(values))
    assert len(mem.records) == 2
    assert len(mem.path.read_text().splitlines()) == 3
    context = mem.retrieve([0], 'reflection', include_failures=True)
    assert {x['evidence']['status'] for x in context} == {'improved', 'regressed'}
    assert all(x['evidence']['status'] == 'improved' for x in mem.retrieve(include_failures=False))
    assert 'code_diff' not in context[0]


def test_summary_version_does_not_erase_newer_updates(tmp_path):
    mem = ExperimentMemory(tmp_path / 'trials.jsonl')
    mem.version = 5
    mem.update_summary('snapshot from four', 4)
    assert mem.version > mem.summary_version
    mem.update_summary('', 5)
    assert mem.summary == 'snapshot from four'


class ReplyLLM(LLM):
    def __init__(self, reply='invalid'):
        super().__init__()
        self.reply = reply

    def draw_sample(self, *args, **kwargs):
        return self.reply

    def close(self):
        pass


def test_budget_counts_malformed_calls(tmp_path):
    llm = BudgetedLLM(ReplyLLM(), 2, tmp_path / 'trace.jsonl')
    llm.draw_sample(messages=[{'role': 'user', 'content': 'hi'}])
    llm.draw_sample('hi')
    with pytest.raises(BudgetExhausted):
        llm.draw_sample('excess')
    assert llm.calls == 2
    assert len(llm.trace_path.read_text().splitlines()) == 2
    assert json.loads(llm.trace_path.read_text().splitlines()[0])['messages']


def test_api_failure_stops_without_retry_or_secret_log(tmp_path):
    class Failed(ReplyLLM):
        def draw_sample(self, *args, **kwargs):
            raise RuntimeError('secret-test-value')
    llm = BudgetedLLM(Failed(), 10, tmp_path / 'trace.jsonl')
    with pytest.raises(BudgetExhausted):
        llm.draw_sample('test')
    assert llm.calls == 1 and llm.remaining == 0
    assert 'secret-test-value' not in llm.trace_path.read_text()


def test_partition_total_capacity_and_child_routing(monkeypatch):
    # Controlled feature geometry checks assignment, independent of CodeBLEU.
    def similarity(a, b):
        return float(a[0] == b[0])
    monkeypatch.setattr(NichePopulation, 'similarity', staticmethod(similarity))
    pop = NichePopulation(capacity=4, partitions=2)
    for i, code in enumerate(['a0', 'b0', 'a1', 'a2', 'b1', 'b2']):
        pop.register(Candidate(i, code, code, outcome([i + 1])))
    assert len(pop.members) == 4
    assert len({x.niche for x in pop.members}) == 2
    assert len({x.niche for x in pop.members if x.code.startswith('a')}) == 1
    assert pop.best.id == 5


def test_codebleu_real_dependencies():
    value = NichePopulation.similarity('def f(x): return x + 1', 'def f(x): return x * 2')
    assert np.isfinite(value) and 0 <= value <= 1


@pytest.mark.parametrize('task', ['tsp', 'cvrp'])
def test_fixed_data_and_positive_route_cost(task):
    evaluator = RoutingEvaluation(task, size=8, instances=3)
    copy = RoutingEvaluation(task, size=8, instances=3)
    heldout = RoutingEvaluation(task, size=8, instances=3, seed=2025, split='test')
    assert evaluator.protocol_id == copy.protocol_id != heldout.protocol_id
    a = evaluator.evaluate_program(evaluator.baseline_program())
    b = copy.evaluate_program(copy.baseline_program())
    assert a == b and a['score'] < 0
    assert len(a['instance_scores']) == 3
    assert not np.array_equal(evaluator.data[0][0], heldout.data[0][0])


def with_body(evaluator, body):
    import ast
    tree = ast.parse(evaluator.template_program)
    next(n for n in tree.body if isinstance(n, ast.FunctionDef)).body = ast.parse(body).body
    return ast.unparse(tree)


@pytest.mark.parametrize('task,body', [
    ('tsp', 'return 0'), ('tsp', 'return -1'), ('tsp', 'return 1.0'),
    ('tsp', 'return float("nan")'), ('tsp', 'return 99999'),
    ('cvrp', 'return 0'), ('cvrp', 'return -1'),
    ('cvrp', 'unvisited_nodes[0] = 0\nreturn 0'),
])
def test_invalid_actions_rejected(task, body):
    evaluator = RoutingEvaluation(task, size=8, instances=2)
    assert evaluator.evaluate_program(with_body(evaluator, body))['score'] is None


def test_cvrp_capacity_violation_rejected():
    evaluator = RoutingEvaluation('cvrp', size=3, instances=1, capacity=9)
    evaluator.data[0][1][1:] = 9
    # Ignoring the provided feasible set must not sneak through capacity checks.
    code = with_body(evaluator, 'return min(current_node + 1, 3)')
    assert evaluator.evaluate_program(code)['score'] is None


def test_interface_and_import_gate():
    evaluator = RoutingEvaluation()
    with pytest.raises(ValueError):
        validate_program('def wrong(x): return 0', evaluator.template_program)
    with pytest.raises(ValueError):
        validate_program(with_body(evaluator, 'import os\nreturn 1'), evaluator.template_program)


def test_invalid_generations_terminate_and_persist(tmp_path):
    evaluator = RoutingEvaluation('tsp', size=5, instances=1)
    # A pre-evaluated common candidate avoids extra process startup in this check.
    code = evaluator.baseline_program()
    base = Candidate(0, code, 'seed', evaluator.evaluate_program(code))
    method = HypoEvo(ReplyLLM(), evaluator, tmp_path, max_calls=3, initial_candidate=base)
    best = method.run()
    assert method.llm.calls == 3 and method.valid == 0
    assert best.id == 0
    records = [json.loads(line) for line in method.memory.path.read_text().splitlines()]
    assert len(records) == 2  # failed reflection metadata still proceeds to implementation
    assert all(r['evidence']['status'] == 'invalid_child' for r in records)


def test_spawn_timeout(tmp_path):
    evaluator = RoutingEvaluation('tsp', size=5, instances=1, timeout=1)
    result = SecureEvaluator(evaluator).evaluate_program(with_body(evaluator, 'while True:\n    pass'))
    assert result is None


def test_no_reflection_is_disabled_even_at_coldstart(tmp_path):
    evaluator = RoutingEvaluation('tsp', size=5, instances=1)
    base = Candidate(0, evaluator.baseline_program(), 'seed', outcome([-5]))
    method = HypoEvo(FakeLLM(evaluator), evaluator, tmp_path, max_calls=1,
                     initial_candidate=base, **VARIANTS['no_reflection'])
    # Avoid process startup, but retain real parsing/prompts/ledger/population.
    method.evaluator.evaluate_program = lambda code: outcome([-4])
    method.run()
    assert all(r['operator'] != 'reflection' for r in method.memory.records)


def test_no_memory_does_not_read_or_summarize_history(tmp_path):
    evaluator = RoutingEvaluation('tsp', size=5, instances=1)
    method = HypoEvo(FakeLLM(evaluator), evaluator, tmp_path, max_calls=2,
                     partitions=1, operators=('summary',), memory_enabled=False)
    method.evaluator.evaluate_program = lambda code: outcome([-4])
    method.run()
    records = [json.loads(s) for s in (tmp_path / 'llm_trace.jsonl').read_text().splitlines()]
    assert all('PAIRED EXPERIMENT RECORDS' not in json.dumps(r['messages']) for r in records)
    assert method.memory.version == 2 and method.memory.summary == ''


@pytest.mark.parametrize('repeat_hypothesis', [False, True])
def test_registered_hypothesis_survives_implementation_reply(tmp_path, repeat_hypothesis):
    import re
    evaluator = RoutingEvaluation('tsp', size=5, instances=1)

    class ImplementationLLM(FakeLLM):
        def draw_sample(self, *args, **kwargs):
            reply = super().draw_sample(*args, **kwargs)
            if 'You design executable' in kwargs['messages'][0]['content']:
                if repeat_hypothesis:
                    reply = reply.replace('Modify one distance score term.', 'A paraphrased intervention.')
                else:
                    reply = re.sub(r'<hypothesis>.*?</hypothesis>', '', reply, flags=re.S)
            return reply

    method = HypoEvo(ImplementationLLM(evaluator), evaluator, tmp_path, max_calls=2)
    method.evaluator.evaluate_program = lambda code: outcome([-4])
    method.run()
    assert method.valid == 1
    assert method.memory.records[0]['hypothesis']['intervention'] == 'Modify one distance score term.'


@pytest.mark.parametrize('operator', ['reflection', 'summary', 'cross'])
@pytest.mark.parametrize('broken_code', [False, True])
def test_metadata_failure_does_not_hide_code_outcome(tmp_path, operator, broken_code):
    import re
    evaluator = RoutingEvaluation('tsp', size=5, instances=1)

    class MissingMetadata(FakeLLM):
        def draw_sample(self, *args, **kwargs):
            reply = super().draw_sample(*args, **kwargs)
            reply = re.sub(r'<hypothesis>.*?</hypothesis>', '', reply, flags=re.S)
            reply = re.sub(r'<concept>.*?</concept>', '', reply, flags=re.S)
            if broken_code and '```python' in reply:
                reply = '```python\ndef broken(:\n```'
            return reply

    method = HypoEvo(MissingMetadata(evaluator), evaluator, tmp_path,
                     max_calls=2 if operator == 'reflection' else 1,
                     partitions=1, operators=(operator,), memory_enabled=False)
    evaluated = []
    def evaluate(code):
        evaluated.append(code)
        return outcome([-4])
    method.evaluator.evaluate_program = evaluate
    method.run()
    record = method.memory.records[0]
    assert record['hypothesis'] is None
    assert not record['hypothesis_test_eligible']
    assert record['metadata']['hypothesis_status'] == 'missing_or_invalid'
    assert record['metadata']['errors']
    assert method.valid == (0 if broken_code else 1)
    assert len(evaluated) == (1 if broken_code else 2)  # baseline plus valid child
    assert (record['evidence']['status'] == 'invalid_child') == broken_code
