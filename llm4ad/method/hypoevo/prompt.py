"""Stateless prompts: explicit candidates, evidence and bounded shared memory."""
import json


HYPOTHESIS_FORMAT = '''Return <hypothesis>JSON</hypothesis> with exactly these text fields:
observation: distinguish measured evidence from a conjecture based on code;
mechanism: the mechanism you suspect;
intervention: one main implementable change (cross may combine mechanisms);
prediction: exactly "mean_score_increases";
risk: a condition where the change could regress.
The prediction concerns the paired mean score, not proof of the mechanism.
Use valid JSON with double-quoted keys/values. Use plain text, not LaTeX or
backslashes, and keep each field under 100 words. Do not claim an observed
improvement unless the supplied paired evidence shows a positive score delta.
'''


def context(task, template, parents, evidence, summary):
    parts = [f'TASK\n{task}', f'INTERFACE\n```python\n{template}\n```']
    for i, p in enumerate(parents):
        parts.append(f'PARENT {i} (id={p.id}, score={p.score}, niche={p.niche})\n'
                     f'Concept: {p.concept}\n```python\n{p.code}\n```')
    if evidence:
        parts.append('PAIRED EXPERIMENT RECORDS (observations, not causal proof)\n' +
                     json.dumps(evidence, ensure_ascii=False))
    if summary:
        parts.append('CACHED GLOBAL SUMMARY (may lag recent trials)\n' + summary)
    return '\n\n'.join(parts)


def reflection_prompt(ctx, hypothesis_enabled):
    return [{'role': 'system', 'content': 'You propose a testable algorithm improvement. Do not invent observed failures.'},
            {'role': 'user', 'content': ctx + '\n\n' + (
                HYPOTHESIS_FORMAT + 'Do not output code in this reflection step.' if hypothesis_enabled else
                'Return <reflection>at most three concrete improvement suggestions</reflection>. No code.')}]


def generation_prompt(ctx, operator, proposal='', hypothesis_enabled=True):
    action = {
        'reflection': 'Improve parent 0 using the proposed reflection. Keep unrelated mechanisms stable when practical.',
        'summary': 'Improve parent 0 using relevant prior evidence. State when transferring an insight to a new context.',
        'cross': 'Use parent 0 as the foundation and transfer a specific useful mechanism from parent 1 if present.',
        'initialization': 'Generate a feasible initial algorithm.',
    }[operator]
    registered = hypothesis_enabled and '<hypothesis>' in proposal
    instructions = action + '\n' + (
        'The PROPOSAL hypothesis is already registered. Implement it; do not repeat or rewrite the hypothesis.\n'
        if registered else HYPOTHESIS_FORMAT if hypothesis_enabled else
        'Give a concrete algorithm improvement; no hypothesis record required.\n')
    instructions += (
                     'Return <concept>a concise description</concept> followed by exactly one Python code block. '
                     'Keep the supplied function name and signature; place helper definitions/imports inside it. '
                     'Use only numpy, math, random, statistics, collections, itertools, functools. '
                     'Do not read files, use the network, or change inputs. Return a valid action on every legal input.')
    return [{'role': 'system', 'content': 'You design executable heuristic algorithms. Higher evaluation scores are better.'},
            {'role': 'user', 'content': ctx + '\n\nPROPOSAL\n' + proposal + '\n\n' + instructions}]


def summary_prompt(task, previous, records):
    return [{'role': 'system', 'content': 'Summarize empirical algorithm experiments without asserting untested causes.'},
            {'role': 'user', 'content': f'TASK\n{task}\nPREVIOUS SUMMARY\n{previous}\n'
             'EXPERIMENTS\n' + json.dumps(records, ensure_ascii=False) +
             '\nReturn <summary>at most 250 words with supported observations, regressions, '
             'conditions, record IDs and unresolved conjectures</summary>. Invalid code is an implementation failure, '
             'not evidence against its proposed algorithmic mechanism.'}]
