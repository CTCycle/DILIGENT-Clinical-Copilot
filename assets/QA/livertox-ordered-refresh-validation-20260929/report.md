# Live LiverTox and ordered structured-source refresh validation — 2026-09-29

## Scope

This follow-up revalidated the actionable `data.inspection.catalogs` and
`data.sources.refresh` gates against the current `develop` implementation. It
used a disposable packaged-style runtime and SQLite database under
`assets/QA/.scratch-livertox-validation`; repository resources, credentials,
and the normal development database were not modified.

Final gate outcomes for this slice:

| Gate | Final status | Boundary |
|---|---|---|
| `data.inspection.catalogs` | `VALIDATED` | Official live LiverTox archive replacement, persisted catalog reads after backend restart, DILIrank/RxNav read surfaces, and rendered LiverTox inspection/excerpt state. |
| `data.sources.refresh` | `VALIDATED` | Live RxNav → LiverTox → DILIrank success, archive reuse on the ordered run, and cancellation preservation of the last-good archive/catalog. |
| `test.automated-regression` | `PARTIAL` | The focused updater/settings suite passed; the full backend/frontend/static/build and hosted/provider lanes were not rerun by this slice. |

## Focused regression

The existing Python environment was used with isolated pytest cache/database
roots:

```text
45 passed, 3 warnings in 20.48s
```

The suite covered the LiverTox downloader and extraction paths, ordered
structured-source jobs, external-data timeouts, Settings API, and runtime
settings persistence. The warnings were the existing dependency deprecations
and a host pytest-cache permission warning; they did not fail the tests.

## Live LiverTox refresh

The disposable backend ran against the official NCBI machine-access chain:
E-utilities identity, Books-OAI metadata, LitArch `file_list.csv`, and the
LitArch archive URL resolved from the live file list.

- Job `694c14a2` reached `completed`, 100%, with no error.
- The archive was downloaded and atomically installed at
  `data/resources/sources/archives/livertox_NBK547852.tar.gz`.
- Archive size was exactly `195,497,807` bytes.
- The final source summary reported `processed_entries=1,875`; the persisted
  inspection catalog reported `total=1,594` monographs.
- The master list was downloaded from the Books-OAI candidate URL and stored
  at `data/resources/sources/LiverTox_Master_List.xlsx` with size `209,279`
  bytes.
- No `.part` file remained after completion.
- The rendered in-app Browser Data Inspection → LiverTox surface showed the
  populated catalog, enabled Update All action, and a persisted Abacavir
  excerpt modal.

## Restart and persistence

The first backend was stopped and a separate backend process was started with
the same disposable data root. Startup reported the same Alembic head and
`seeded=False`; `/api/health` returned `{"status":"ok"}`. After restart:

- LiverTox still returned `total=1,594` and Abacavir retained
  `has_excerpt=true`.
- SQLite `PRAGMA integrity_check` returned `ok`.
- SQLite `PRAGMA foreign_key_check` returned no rows.
- After the ordered refresh, direct table counts were `drugs=7,043`,
  `drug_aliases=86,197`, `livertox_monographs=1,594`, and
  `dilirank_records=1,336`.

## Ordered Update All

Job `6952f60f` completed at 100% with no error and preserved the required
source order:

1. **RxNav** completed first from
   `https://rxnav.nlm.nih.gov/REST/RxTerms/allconcepts.json`, with HTTP 200,
   `records=21,202`, `attempts=2`, and `enrichment_failures=0`.
2. **LiverTox** completed second, reused the validated archive with
   `downloaded=false`, and reported `processed_entries=1,875`.
3. **DILIrank** completed last from the official FDA download, with
   `source_records=1,336`, `persisted_records=1,336`, `linked_records=733`,
   `unmatched_records=286`, and `ambiguous_records=317`.

The unmatched and ambiguous DILIrank values are source-link outcomes observed
in the live data, not an assertion that every source row resolves to a local
drug record.

## Cancellation and last-good preservation

Before a disposable redownload, the installed archive hash was
`815D74D739C1CB3C2239516F9C6CB3FD0150B8F3A18E81E1D237B941AC578594` and the
catalog contained 1,594 rows. The runtime download timeout was temporarily
lowered to the service floor of 1 second, job `6fd6f213` was started with
`redownload=true`, and cancellation was requested during the streamed
download. The job reached terminal `cancelled` at 43.8% while extracting.

After cancellation, the archive was still exactly 195,497,807 bytes with the
same SHA-256, no `.part` file existed, the catalog remained at 1,594 rows, and
SQLite integrity remained `ok` with no foreign-key violations. The temporary
timeout was restored to 30.0 seconds. Deterministic focused tests also cover
invalid/partial payload preservation. This run did not inject an independent
live upstream outage after a connection had been established, so that
specific failure variant remains validation debt rather than a new defect
claim.

## Harness and remaining boundaries

The first non-elevated local attempt could not open a socket to NCBI because of
the host sandbox network policy and failed before any source request or data
mutation. The authorized elevated disposable run reached the official NCBI,
RxNav, and FDA endpoints successfully; the sandbox failure is recorded as a
harness limitation, not product evidence.

This slice does not change the separate `PARTIAL`/`BLOCKED` gates for the live
OpenCode Go provider, accepted current-provider revision execution, real
access-key lifecycle, automated conditional/provider lanes, desktop `v3.4.0`
release, duplicate-file policy, or spoken Narrator/Speech Recap output.
