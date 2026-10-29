# Drug terminology benchmark: legacy matcher vs terminology resolver
Date: 2026-10-29

## Setup

- Cases: `app/tests/benchmarks/drug_terminology/cases.jsonl`, 407 labeled
  inputs (generics, US and Swiss brands, INN/USAN pairs, language variants,
  misspellings, abbreviations, salts, stereoisomers, look-alike/sound-alike
  traps, strengths, combinations, unknowns). Deterministic split: 250
  calibration, 157 held-out. Labels are author-curated from standard
  nomenclature; they are not a clinical validation.
- Legacy: the name-string pipeline at `HEAD` before this change, run on a copy
  of the populated development database (`baseline.json`).
- Terminology: `terminology.db`, the same database after migration
  `202610290001` and a full import of RxNorm Prescribable (2026-10-05),
  DrugCentral (v56, 2026-09-02), ChEMBL 37, Swissmedic (2026-09-30) and the
  curated catalog. The database did not need re-ingestion for this run: the
  code changed since its population touches only runtime normalization and
  resolution, not ingestion.
- Local run (`upgraded_local_final.json`): RxNav fallback off.
- External run (`upgraded_external_final.json`): RxNav fallback on, against a
  fresh copy of `terminology.db` (earlier external runs reused a database that
  had accumulated runtime aliases, which flattered their external figures).
- Command (from `app/server`):
  `.venv\Scripts\python.exe ..\tests\benchmarks\drug_terminology\run_benchmark.py --database <db> --output <json> [--external]`

"Wrong automatic match" means the resolver accepted an identity whose
ingredient set differs from the label. Abstaining (ambiguous or unresolved) is
never counted as wrong.

## Overall results (all 407 cases)

| Metric | Legacy | Terminology, local | Terminology, with RxNav |
| --- | --- | --- | --- |
| Top-1 accuracy | 62.7% | 97.6% | 98.2% |
| Recall@5 | 72.7% | 98.7% | 99.2% |
| Automatic matches | 297 | 375 | 377 |
| Wrong automatic matches | 58 | 2 | 2 |
| Automatic-match precision | 80.5% | 99.5% | 99.5% |
| Abstention recall (must-abstain cases) | 80.8% | 92.0% | 92.0% |
| Median latency | 1,321 ms | 2.0 ms | 2.2 ms |
| p95 latency | 6,710 ms | 5.6 ms | 747 ms |
| External calls per resolution | 0 | 0 | 0.12 |
| Index build / memory | 46 s / 69 MB | 2.1 s / 22 MB | 2.0 s / 22 MB |

Held-out split (157 cases): legacy 64.4% Top-1 with 20 wrong automatic
matches; terminology 97.3% Top-1 (local and external) with 0 wrong automatic
matches and 100% abstention recall.

The p95 with RxNav comes from network calls for inputs that no local source
knows; local resolution stays in single-digit milliseconds.

## Change since the previous run (`upgraded_local_v5.json`, 98.4%)

Top-1 fell by three cases because the spelling-correction gate now enforces
the plan's look-alike guard: a correction is refused when any other concept is
also within the edit bound. The three inputs are now **ambiguous** with the
correct drug as the first candidate (Recall@5 unchanged):

| Input | Expected | Other concept within the bound |
| --- | --- | --- |
| metfromin | metformin | merbromin |
| carbamazapine | carbamazepine | oxcarbazepine, diethylcarbamazine |
| naproxin | naproxen | norfloxacin |

No wrong automatic match was added or removed. The guard is what turns
"hydroxazine" (hydroxyzine vs hydralazine, an ISMP look-alike pair) into a
reviewer decision instead of an automatic correction.

## Breakdown (Top-1 correct / wrong automatic matches)

By alias type:

| Type | Cases | Legacy | Local | With RxNav |
| --- | --- | --- | --- | --- |
| abbreviation | 15 | 1 / 0 | 12 / 0 | 12 / 0 |
| brand | 80 | 50 / 5 | 77 / 0 | 78 / 0 |
| combination | 30 | 6 / 17 | 27 / 0 | 28 / 0 |
| generic | 55 | 51 / 3 | 55 / 0 | 55 / 0 |
| inn_usan | 30 | 7 / 7 | 30 / 0 | 30 / 0 |
| language_variant | 45 | 25 / 4 | 45 / 0 | 45 / 0 |
| lasa | 40 | 31 / 6 | 39 / 0 | 39 / 0 |
| misspelling | 45 | 39 / 3 | 41 / 0 | 41 / 0 |
| salt | 20 | 14 / 0 | 20 / 0 | 20 / 0 |
| stereoisomer | 12 | 5 / 7 | 12 / 0 | 12 / 0 |
| strength | 15 | 10 / 2 | 15 / 0 | 15 / 0 |
| unknown (must abstain) | 20 | 0 / 4 | 0 / 2 | 0 / 2 |

By input language:

| Language | Cases | Legacy | Local | With RxNav |
| --- | --- | --- | --- | --- |
| en | 324 | 202 / 51 | 298 / 2 | 300 / 2 |
| de | 41 | 17 / 4 | 37 / 0 | 37 / 0 |
| fr | 13 | 7 / 2 | 13 / 0 | 13 / 0 |
| it | 12 | 6 / 0 | 12 / 0 | 12 / 0 |
| es | 8 | 6 / 1 | 8 / 0 | 8 / 0 |
| la | 5 | 1 / 0 | 5 / 0 | 5 / 0 |

## Remaining wrong automatic matches

Both are class terms labeled "must abstain": **insulin** and **vitamin D**
resolve to the LiverTox class concepts "Insulin" and "Vitamin D". The
identity is a class rather than a single molecule, but the linked LiverTox
monograph is the appropriate one, so the clinical consequence is limited.

## Known gaps seen in the benchmark

- Brands absent from every source stay unresolved (for example Zurcal,
  Tazocin, Dafalgan Codein). The RxNav fallback recovers some US names only.
- About 23% of Swissmedic products (2,306 of 10,308) keep an unverified
  composition because a Latin ingredient name does not resolve; those brands
  return "probable, not accepted".
- DrugCentral's RxNorm cross-walk contains errors; 220 DrugCentral and 447
  ChEMBL import conflicts were recorded and overruled by exact names or
  precedence (visible in Knowledge > Terminology).

## Other validation in this change

- Unit and persistence suites, Ruff, Pyright, Alembic check, frontend build
  and Vitest: see the session summary.
- Visual check of Knowledge > Terminology, the update dialog, reviewer
  confirmation, Clinical Sessions > Detected drugs, and the two Settings pages
  on an isolated copy of this database: `screenshots/`.
