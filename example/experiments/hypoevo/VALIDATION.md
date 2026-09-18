# Offline validation — 2026-09-17

Implementation and experiment harness only. **No real LLM experiments were run.**

Validation was run from the actual relocated repository using a clean Python
3.11.7 environment and the versions in `requirements.txt`.

| Check | Result |
|---|---|
| Unit/regression tests | 28 passed |
| TSP/CVRP × full method, seven ablations, EoH | 18/18 offline runs completed |
| LLM budget accounting | Exactly 12 fake calls/run, 216 total |
| HTTP request audit | 0 requests |
| Completed-run restart | All 18 runs skipped; no new generation |
| Changed budget in same output directory | Rejected; trace hashes unchanged |
| Pilot dry run | 18 runs, at most 540 real calls when explicitly enabled |
| Full ablation dry run | 54 runs, at most 1,620 real calls when explicitly enabled |
| Git whitespace check | Passed |

Tests cover paired instance matching, protocol mismatches, failed-trial
persistence, bounded retrieval, summary versions, malformed generation budgets,
API error termination, feature dependency compatibility, fixed total population
capacity, child reassignment, independent search/test datasets, invalid actions,
CVRP capacity/progress, AST/interface checks, spawn timeout, and ablation flags.

Local full-matrix output is in `outputs/offline-validation/` (ignored by Git),
including the resolved protocol, prompts/responses, trial ledger, generated
programs, per-run results, CSV summaries and an explicitly marked fake-LLM
report. This artifact validates execution paths, not algorithm quality.

The original PartEvo and existing user modifications to EoH, MCTS-AHD, the HTTPS
client, and GUI were left intact. HypoEvo is added as a separate method; three
package initializers add an opt-in minimal-import mode for the new scripts.

Before real experiments on the target machine, repeat the offline checks there,
configure the model through environment variables, and inspect `--dry-run`.
See `README.md` for the pilot/full-matrix commands, limitations and cost bounds.

## Subsequent real-model validation and metadata fix

The statement above that no real calls were made describes the initial offline
milestone only. Real qwen3.5-flash experiments were subsequently completed.

- Metadata parsing/registration regressions: **39 tests passed**.
- Offline replay: 38 of 39 format-rejected candidates passed the original
  search evaluator, with no additional model calls.
- Format fix separates metadata conformance from code execution validity;
  missing hypotheses never count as eligible hypothesis tests.
- Final comparison uses 12 completed format-fix runs plus 6 historical EoH
  runs, following the user's request to reuse EoH and stop its rerun.
- Results, per-instance scores, configurations and budget
  accounting are archived in [results/20260917](results/20260917/README.md).
