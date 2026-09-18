# HypoEvo real-model pilot and metadata-format fix

The final comparison is [FINAL_REPORT.md](pilot-formatfix-20260917/FINAL_REPORT.md).
It combines **12 completed format-fix runs** (HypoEvo and no-memory) with
**6 historical EoH runs**, as requested. Newly sampled EoH results are excluded.
All use qwen3.5-flash, three seeds, 28 calls/run, and the same search/test data.

| Mean held-out route length (lower is better) | HypoEvo fixed | No-memory fixed | Historical EoH |
|---|---:|---:|---:|
| TSP-50 | 6.7841 | 6.6814 | 6.8325 |
| CVRP-50 | 14.0742 | 14.2961 | 14.0729 |

HypoEvo execution-valid candidates increased from 51/96 (53.1%) to 89/101
(88.1%). This is distinct from hypothesis conformance: candidates without a
valid hypothesis are marked ineligible for hypothesis verification. The data
are uniform random routing instances, not TSPLIB/CVRPLIB. Three seeds and a
historical baseline do not establish statistical significance or generality.

## Contents and provenance

- `pilot-real-v3-20260917`: original 18 completed runs and 39-candidate offline
  replay (38 passed evaluation; zero extra LLM calls).
- `pilot-formatfix-20260917`: final comparison, budget audit and 12 completed
  rerun artifacts. `FINAL_RUNS.csv` identifies each selected source run.
- Each archived run contains its result with per-instance scores. Candidate
  programs, trial ledgers and full traces remain in local outputs.
- Resolved configurations retain their original fingerprints/source hashes.
  Only the machine-specific `dotenv` path is removed; it was excluded from
  fingerprint computation. Full prompt/response traces remain local under
  `outputs/` and are not included in this compact archive.
- Offline replay includes aggregate and per-candidate scores.
- `SHA256.json` records hashes of archived source artifacts, not this README.
- The old replay script intentionally checks the old source hash; rerunning
  it against newer method code requires the matching historical source.

The rerun used 336 calls for the final two-method comparison. Before the user
requested reuse of historical EoH, 97 additional EoH calls were recorded and
up to 2 requests were interrupted in flight: 433 recorded / 435 upper bound.
The historical batch had its own budget, documented in its budget audit.
Do not interpret the original rerun plan of 504 as its actual final usage.

No search results were replaced with the offline replay scores. Full call
traces, diagnostic runs and excluded new EoH runs remain in local outputs.
