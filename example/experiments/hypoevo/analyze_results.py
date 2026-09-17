"""Summarize complete and incomplete runs without claiming pilot significance."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def analyze(root):
    root = Path(root)
    config = json.loads((root / 'resolved_config.json').read_text())
    records = []
    for task in config['tasks']:
        for method in config['methods']:
            for seed in config['seeds']:
                path = root / task / method / str(seed) / 'result.json'
                r = json.loads(path.read_text()) if path.exists() else {'status': 'missing'}
                if r.get('protocol_fingerprint', config['fingerprint']) != config['fingerprint']:
                    raise ValueError('mixed experiment protocols')
                records.append({'task': task, 'method': method, 'seed': seed,
                                **{k: r.get(k) for k in ['status', 'llm_calls', 'attempts', 'valid',
                                                        'test_mean_cost', 'improvement_over_nn_percent']}})
    with (root / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    rows = []
    for task in config['tasks']:
        for method in config['methods']:
            results = [r for r in records if r['task'] == task and r['method'] == method]
            values = [r['test_mean_cost'] for r in results if r['status'] == 'completed']
            rows.append({'task': task, 'method': method, 'completed': len(values),
                         'expected': len(results), 'mean_test_cost': mean(values) if values else None,
                         'std_test_cost': stdev(values) if len(values) > 1 else None})
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# HypoEvo experiment report', '',
             '**OFFLINE FAKE LLM — plumbing verification only; scores are not research evidence.**'
             if config['backend'] == 'fake' else
             'Pilot results; no claim of significance or improvement beyond these runs.', '',
             'Uniform random TSP/CVRP construction tasks. Lower held-out tour cost is better.',
             'Search uses only the search split; one training-selected winner is tested after each run.',
             'Call caps include reflection and summary. Equal calls do not imply equal tokens or runtime.', '',
             '| Task | Method | Complete/expected | Mean test cost | Std |',
             '|---|---|---:|---:|---:|']
    for r in rows:
        m = f"{r['mean_test_cost']:.5f}" if r['mean_test_cost'] is not None else '—'
        s = f"{r['std_test_cost']:.5f}" if r['std_test_cost'] is not None else '—'
        lines.append(f"| {r['task']} | {r['method']} | {r['completed']}/{r['expected']} | {m} | {s} |")
    lines += ['', 'These are full search runs, not causal proof of individual hypothesis mechanisms.',
              'Incomplete and missing runs remain in runs.csv and are excluded from completed-run means.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output_dir', type=Path)
    analyze(p.parse_args().output_dir)
