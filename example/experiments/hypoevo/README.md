# HypoEvo: implementation and ablation experiments

HypoEvo adds a hypothesis → candidate → paired evaluation → evidence-memory
loop to niche-based LLM algorithm search. It is a separate method inspired by
PartEvo, preserving the existing PartEvo and EoH implementations for comparison.
This directory contains an **offline-tested implementation and experiment
scripts**, not real-LLM benchmark results or a claim of superiority.

## Environment and quick validation

Use Python 3.11 in a clean environment. The repository's full environment has
many optional dependencies; this experiment only needs the smaller set below.
Run from the repository root (paths are portable; no workstation path is embedded):

```bash
python3.11 -m venv .venv-hypoevo
source .venv-hypoevo/bin/activate
python -m pip install -r example/experiments/hypoevo/requirements.txt
python -m pytest -q example/experiments/hypoevo/test_hypoevo.py
python example/experiments/hypoevo/run_experiment.py --smoke \
  --output-dir /tmp/hypoevo-smoke
```

`--smoke` uses a deterministic fake LLM and small TSP/CVRP instances. It never
loads API credentials, sends API requests, or establishes research performance.
To check all ablations offline:

```bash
python example/experiments/hypoevo/run_experiment.py --smoke --suite ablations \
  --output-dir /tmp/hypoevo-ablation-smoke
```

CodeBLEU 0.7 / tree-sitter 0.22 expects a language pointer while
tree-sitter-python 0.23 returns a PyCapsule. `population.python_language()`
bridges these APIs locally for ARM macOS and Linux; it does not monkeypatch the
library. Actual CodeBLEU execution is included in the tests. Feature errors are
not silently replaced by random partitions.

The runner and tests set `LLM4AD_MINIMAL_IMPORTS=1` before importing LLM4AD.
This opt-in skips eager discovery of unrelated methods/tasks and their optional
dependencies; normal library/GUI startup is unchanged. When embedding HypoEvo
in another lightweight script, set this variable before the first `llm4ad`
import, then use `from llm4ad.method.hypoevo import HypoEvo`. On this workstation,
old bytecode in the moved/synced directory stalled reads; a fresh
`PYTHONPYCACHEPREFIX=/tmp/hypoevo-pycache` avoided that local issue. This is not
required on a normal fresh checkout.

## Search design

Three operators share the same interface, evaluator, population and ledger:

* **reflection**: propose one main modification with an explicit hypothesis,
  then implement it in a second call. When one call remains, combine proposal
  and implementation in that call. Cold start uses random-parent reflection
  when reflection is enabled; otherwise it respects the ablated operator set.
* **summary**: periodically summarize retrieved paired experiments, including
  the previous summary, then generate a candidate. A summary refresh is an LLM
  call and consumes the same budget. The summary records its source version.
* **cross**: choose a primary parent and a helper from another occupied niche
  when possible, state a transfer hypothesis, and generate the hybrid in one
  call. With only one niche, another distinct candidate can be the helper.

The hypothesis fields are `observation`, `mechanism`, `intervention`,
`prediction`, and `risk`. The currently machine-checked prediction is
`mean_score_increases` on fixed search instances. A reflection's fields must be
copied unchanged by the implementation call. This is **not** proof that the
proposed mechanism caused the result; code diffs are saved for subsequent
inspection/controlled ablations. Mechanistic or subgroup predictions are not
automatically verified. Richer counterfactual experiments are future work.

After evaluation, an append-only JSONL ledger records the parent/child IDs,
operator, hypothesis, exact code diff, paired instance-score deltas and errors
**before population selection**. Losing and invalid candidates remain in the
ledger. A bounded in-memory window retrieves related parent/operator events,
including both improvement and regression examples when available. Retrieval
does not read the test split. It is deterministic metadata-based retrieval,
not semantic-vector search. Summary text and hypothesis lengths are bounded.

Memory labels are `improved`, `regressed`, `unchanged`, `invalid_child`,
`insufficient_evidence`, or `no_comparator`. Comparisons require identical
protocol IDs and instance sets. Scalar-only evaluators can drive search, but
do not generate a false paired-evidence claim. Statistical significance is not
inferred from one paired mean improvement.

## Population and features

Default total capacity is 16, with 4 partitions. CodeBLEU syntax/dataflow match
is symmetrized and averaged, forming similarity vectors against initial
landmarks. StandardScaler → PCA (at most 10 dimensions) → K-means builds the
initial niches after enough distinct valid candidates exist. Every subsequent
child is mapped with those **same** landmarks/transforms before local selection,
so changing algorithm behavior/code does not force inheritance of its parent's
niche. If the feature matrix cannot distinguish candidates, fewer niches are
used and reported explicitly. No claim of behavior equivalence is made.

Total capacity is fixed across partition ablations (not 16 per partition).
Each niche has a fixed share; empty niches do not lend their quota. Full code
duplicates are removed; equal scores alone do not imply duplicates. The
historical global best is preserved separately. Features are fixed after
initialization; dynamic landmark refresh and behavior-space clustering are not
implemented in this version.

## Reproducible benchmark protocol

`config.json` defaults to a low-cost pilot:

* Tasks: TSP-50 and CVRP-50 construction, uniform random instances using the
  upstream task interfaces (not TSPLIB/CVRPLIB and not optimality-gap claims).
* Search data: 16 fixed instances, seed 12024. Held-out test: 64 independent
  instances, seed 22024. CVRP capacity 40, customer demands 1–9.
* Methods: `hypoevo`, `no_memory`, `eoh`.
* Search seeds: 2024, 2025, 2026. All methods see identical task datasets.
* Budget: at most 30 LLM calls per run, **540 total** for the pilot matrix.
  Reflection, summary, malformed replies and API failures count. The real
  backend uses `max_retries=1`, so automatic HTTP retries cannot hide costs.
* A nearest-neighbor seed is evaluated outside the LLM-call budget. HypoEvo
  variants start from this seed. EoH retains its original initialization and
  operators; its final winner is compared with the same common seed. EoH is
  therefore a full-method comparison, not a strictly matched initialization
  ablation. Its total population size is also 16.
* Only the search-selected winner and common seed are evaluated on held-out
  test data, after search ends. Test scores never select parents or hypotheses.
* Both methods use strict action/route validation: integer IDs, feasible
  unvisited nodes, capacity, progress, finite positive costs, and copies of
  inputs. Per-instance RNG seeds are fixed, and globals reset per instance.
* Runtime exceptions and timeouts are failures. Evaluation uses spawned child
  processes through `SecureEvaluator`. AST checks and process timeouts are
  **not an OS security sandbox**; execute untrusted generated code only in a
  suitably isolated container/account on the target machine.

Equal call counts do not mean equal token counts, runtime, or candidate counts;
those are logged. Local seeds do not guarantee deterministic remote model
responses. Three runs are a pilot, not sufficient support for broad statistical
claims. To increase repetitions, edit config or use `--seeds`.

## Ablation matrix

| Variant | Change from full HypoEvo |
|---|---|
| `no_memory` | No historical retrieval or summary calls; still writes audit records |
| `no_hypothesis` | Free reflection and code generation; no structured hypothesis required |
| `no_failures` | Exclude regressed/invalid trials from retrieved evidence and summaries |
| `no_partition` | One global population with the same total capacity |
| `no_reflection` | Remove reflection, including at cold start |
| `no_summary` | Remove the summary operator; other operators may still retrieve evidence |
| `no_cross` | Remove cross; reflection/summary remain |
| `eoh` | Existing EoH implementation using the common stricter evaluator |

`--suite ablations` runs full HypoEvo, all seven variants and EoH: 54 runs,
1,620 calls at the default three seeds. Changing operators/call overhead changes
the number of candidates within the same call budget. These ablations estimate
the effect of removing a component under an equal-call policy, not equal-token
or equal-number-of-candidates causal isolation.

## Run on the target machine

First inspect the matrix without any API calls:

```bash
python example/experiments/hypoevo/run_experiment.py --dry-run
python example/experiments/hypoevo/run_experiment.py --suite ablations --dry-run
```

Configure environment variables or the repository `.env` (never commit keys):

```text
LLM_API_KEY=...
LLM_BASE_URL=https://your-provider/compatible-mode/v1
LLM_MODEL=your-model
```

The existing `LLM4AD_API_KEY`, `LLM4AD_API_BASE_URL`/`LLM4AD_API_HOST`, and
`LLM4AD_API_MODEL` aliases also work. The config has `enable_thinking=false`
for the previously used Qwen-compatible provider; remove that entry for a
provider that does not support it. Generation options are explicitly recorded.

```bash
# Real pilot: use only when ready to consume the configured API budget.
python example/experiments/hypoevo/run_experiment.py --run \
  --output-dir example/experiments/hypoevo/outputs/pilot

# Full real ablation matrix, separately budgeted.
python example/experiments/hypoevo/run_experiment.py --run --suite ablations \
  --output-dir example/experiments/hypoevo/outputs/ablations

# Split work across machines using non-overlapping matrices and fresh outputs.
python example/experiments/hypoevo/run_experiment.py --run \
  --tasks tsp --methods hypoevo no_memory --seeds 2024 2025 2026 \
  --output-dir example/experiments/hypoevo/outputs/tsp
```

Runs execute serially, each in a separate worker with a wall-clock deadline.
Completed runs are skipped on restart. Incomplete runs are preserved; the
runner stops rather than silently retrying paid work. Choose a new output
directory for explicit retries. Mid-run population resume is not implemented.
Changing sources, dependency versions, data, model or settings changes the
protocol fingerprint and prevents mixing results in the same directory.

Outputs: `resolved_config.json`, per-run `result.json`, `best.py`,
`llm_trace.jsonl` (full prompts/replies, usage and HTTP counts),
`candidates.jsonl`; HypoEvo also writes `trials.jsonl` and `state.json`.
API keys are excluded. Traces contain generated code and task context.
`runs.csv`, `summary.csv`, and `REPORT.md` summarize results and mark fake runs
explicitly. Rebuild a partial report with:

```bash
python example/experiments/hypoevo/analyze_results.py PATH_TO_OUTPUT
```
