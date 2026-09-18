"""Bounded retrieval memory backed by an append-only experiment ledger.

Observed paired improvements are evidence about a change, not causal proof of
the LLM's proposed mechanism. Invalid children are retained as failure events.
"""
from __future__ import annotations

import difflib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hypothesis:
    observation: str
    mechanism: str
    intervention: str
    prediction: str
    risk: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("hypothesis must be a JSON object")
        names = tuple(cls.__dataclass_fields__)
        if any(not isinstance(value.get(k), str) or not value[k].strip() for k in names):
            raise ValueError("hypothesis requires five nonempty text fields")
        if any(len(value[k]) > 2000 for k in names):
            raise ValueError('hypothesis fields are limited to 2000 characters')
        if value['prediction'] != 'mean_score_increases':
            raise ValueError("supported prediction: mean_score_increases")
        return cls(**{k: value[k].strip() for k in names})


def paired_evidence(parent, child, tolerance=1e-9):
    if child.get('score') is None:
        return {'status': 'invalid_child', 'delta': None, 'instance_deltas': []}
    if parent is None:
        return {'status': 'no_comparator', 'delta': None, 'instance_deltas': []}
    a, b = parent.get('instance_scores', {}), child.get('instance_scores', {})
    same_protocol = (parent.get('protocol_id') and
                     parent.get('protocol_id') == child.get('protocol_id'))
    if not same_protocol or not a or set(a) != set(b):
        return {'status': 'insufficient_evidence', 'delta': None, 'instance_deltas': []}
    deltas = [float(b[k]) - float(a[k]) for k in sorted(a)]
    if not all(math.isfinite(d) for d in deltas):
        return {'status': 'insufficient_evidence', 'delta': None, 'instance_deltas': []}
    delta = sum(deltas) / len(deltas)
    status = 'improved' if delta > tolerance else 'regressed' if delta < -tolerance else 'unchanged'
    return {'status': status, 'delta': delta, 'instance_deltas': deltas,
            'wins': sum(d > tolerance for d in deltas),
            'losses': sum(d < -tolerance for d in deltas),
            'scope': 'paired search instances; mechanism not causally verified'}


class ExperimentMemory:
    def __init__(self, path: Path, capacity=128):
        if capacity < 1:
            raise ValueError('memory capacity must be positive')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = deque(maxlen=capacity)
        self.version = 0
        self.summary = ''
        self.summary_version = -1

    def add(self, *, child_id, parents, operator, hypothesis, child_code, result, error=None):
        parent = parents[0] if parents else None
        evidence = paired_evidence(parent.result if parent else None, result)
        record = {
            'id': child_id, 'parent_ids': [p.id for p in parents], 'operator': operator,
            'hypothesis': asdict(hypothesis) if hypothesis else None,
            'parent_score': parent.result['score'] if parent else None,
            'child_score': result.get('score'), 'protocol_id': result.get('protocol_id'),
            'evidence': evidence,
            'error': error or result.get('error'),
            'metadata': result.get('metadata', {}),
            'hypothesis_test_eligible': bool(hypothesis) and evidence['status'] in
                {'improved', 'regressed', 'unchanged'},
            'code_diff': ''.join(difflib.unified_diff(
                parent.code.splitlines(keepends=True) if parent else [],
                child_code.splitlines(keepends=True), fromfile='parent', tofile='child')),
        }
        # Persist before retention/selection, including failed and rejected children.
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.records.append(record)
        self.version += 1
        return record

    def retrieve(self, parent_ids=(), operator='', limit=6, include_failures=True):
        pool = list(self.records)
        if not include_failures:
            pool = [r for r in pool if r['evidence']['status'] not in {'regressed', 'invalid_child'}]
        parent_ids = set(parent_ids)
        ranked = sorted(pool, key=lambda r: (
            bool(parent_ids.intersection(r['parent_ids']) or r['id'] in parent_ids),
            r['operator'] == operator, r['id']), reverse=True)
        # Prefer a relevant contrast when one exists, rather than just top scores.
        chosen = []
        for statuses in ({'improved'}, {'regressed', 'invalid_child'}):
            if len(chosen) >= limit:
                break
            match = next((r for r in ranked if r['evidence']['status'] in statuses), None)
            if match is not None:
                chosen.append(match)
        for r in ranked:
            if len(chosen) >= limit:
                break
            if r not in chosen:
                chosen.append(r)
        # Diffs and raw programs remain on disk; retrieval has bounded text size.
        return [{k: v for k, v in r.items() if k != 'code_diff'} for r in chosen]

    def update_summary(self, text, source_version):
        if text.strip():
            self.summary = text.strip()[:4000]
            self.summary_version = source_version
