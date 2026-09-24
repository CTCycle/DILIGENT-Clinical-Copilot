# Pre-release validation ledger
Last updated: 2026-09-24

## Scope and interpretation

This is a dated evidence register and chronological pre-release validation diary for the DILIGENT source/development tree. The high-level current operational status is canonical in [`../project_status_ledger.md`](../project_status_ledger.md); this document preserves the run-specific scope, evidence, diary, and release boundaries that explain that status. The historical feature-state register below records the validation run performed on 2026-09-18 against source revision `2e434ae1b0e276b8acbab27f65a399d306c3f691` on the `develop` branch. The current 2026-09-21 Tier 0 checkpoint is recorded above. Packaging, publication, EXE/MSI smoke tests, and clean-machine installation were intentionally outside the historical audit.

The historical 2026-09-18 run used synthetic patient content only. No patient-identifying data was entered, no source refresh or embedding update job was started, and no live cloud-provider clinical analysis was submitted. The current 2026-09-22 checkpoint below separately records the exact-provider clinical and structured-source executions.

Status meanings:

- `PASS`: the named scope was executed and behaved as expected.
- `ATTENTION`: the surface is usable, but an external dependency, partial data state, or validation limitation remains.
- `FAIL`: the named scope has a reproducible release-blocking defect.
- `NOT TESTED`: current evidence is absent; this is not a claim that the capability is broken.
- `NOT APPLICABLE`: deliberately outside this source-only audit.

## Integrated long-term evaluation strategy

The comprehensive validation roadmap is integrated into this ledger as the
stable campaign plan. It does not create a second status authority:

- [`project_status_ledger.md`](../project_status_ledger.md) remains the
  canonical current operational state.
- This document remains the dated, run-specific evidence register and diary.
- `assets/QA/` remains the supporting evidence location for logs, screenshots,
  reports, and validation notes.

### Stable campaign map

| Tier | Slices | Focus |
|---|---|---|
| Tier 0 | `V00`–`V02` | Revision/evidence baseline, source startup and migration, automated baseline gates |
| Tier 1 | `V10`–`V13` | Shell/routing, runtime Settings, model configuration/catalog cache, access-key lifecycle |
| Tier 2 | `V20`–`V24` | Clinical input/preflight, exact provider execution, multi-drug correctness, RAG grounding, job lifecycle |
| Tier 3 | `V30`–`V40` | Sessions, editing/media/deletion, Data Inspection, source mutation, RAG re-indexing |
| Tier 4 | `V41`–`V53` | Timelines, agentic revision, local Ollama, PostgreSQL, Tauri, and release lifecycle |
| Tier 5 | `V60`–`V61` | Cross-cutting resilience, accessibility, responsive behavior, and performance-sensitive UI |

The campaign order is:

`V00 → V01 → V02 → V10 → V11 → V12 → V20 → V21 → V22 → V23 → V24 → V30 → V31 → V32 → V33 → V34 → V35 → V36 → V37 → V38 → V39 → V40 → V41 → V42 → V43 → V44 → V13 → V50 → V51 → V60 → V61 → V52 → V53`

`V13` may wait for an approved disposable credential, and `V52`/`V53`
remain release-end gates. The immediate high-risk follow-up after Tier 0 is
`V21`, `V22`, `V36`–`V39`, `V42`, `V43`, and `V44`.

Every slice record must include its stable ID, capability, status, exact HEAD,
environment and preconditions, executed scenarios, `VAL-*` issues, primary
failure class, fixes and adjacent regression, evidence links, remaining gaps,
validation date, external dependencies, and cleanup result. A screen, HTTP
200, or isolated mock test is never sufficient for a `PASS` claim without the
workflow and persistence evidence required by that slice.

The shared execution loop is **Inspect → Execute → Observe → Diagnose →
Surgically Fix → Retest → Record**. Defects are classified as functional,
frontend, backend, integration, persistence/state, performance, UX/workflow,
configuration, or test/environment problems. External provider outages remain
environment evidence unless routing, classification, fallback, recovery, or
user feedback violates the product contract.

Comprehensive source validation requires one exact `develop` commit, no core
capability left `UNTESTED`, `UNKNOWN`, `FAIL`, or `BLOCKED`, current exact
provider evidence without fallback, source-refresh safety, timeline failure and
cancellation evidence, QA-clean revision finalization with reload persistence,
final automated gates at that same commit, resolvable evidence links, and no
temporary credentials or validation data left in persistent user storage.

## Tier 0 execution checkpoint — 2026-09-21

The checkpoint started from a clean `develop` worktree at
`d0e8ba1809d7a79615f300403891ae2066a574a7`. The prior status snapshot was
`ba47761aa456e78923a0df824f76ee4016ee0af6`; the exact changed-file boundary
and evidence-link audit are recorded in the [Tier 0 evidence report](../../QA/tier0-validation-20260921.md).

| Slice | Capability | Status | Executed scope and result |
|---|---|---|---|
| `V00` | Revision and evidence baseline | `PASS` | Recorded HEAD, branch, clean state, 713-entry tree, snapshot drift, and 109 status-ledger links; all referenced evidence paths resolved. |
| `V01` | Source startup, migration, health, populated-data reuse, shutdown, and port ownership | `FAIL` | Fresh and populated disposable SQLite runs passed launcher initialization, Alembic head, health, frontend readiness, read-only sessions access, integrity checks, and owned-process shutdown. The occupied-port test reproduced unsafe behavior: a controlled foreign listener on 7690 was killed, then the launcher continued with exit code 0. The finding is tracked as `ISSUE-006` in the project ledger. |
| `V02` | Automated baseline gates | `PASS` | `app\tests\run_tests.bat unit`: 742 passed; Angular/Vitest: 23 files and 96 tests passed; production build completed with `--progress=false`. |

The disposable databases and occupied-port run roots were removed after the
run. Ports `7690` and `9847` were verified free, and no matching
launcher-owned backend/frontend process remained. The fresh and populated V01
subcases are passing, but the occupied-port ownership failure is a reproducible
launcher defect and blocks treating V01 as complete.

## V01 remediation revalidation — 2026-09-21

The original V01 checkpoint above is retained as historical evidence. The
follow-up remediation was validated against the checked-out `develop` source
with the launcher changes present in the working tree. Full evidence is in
[ISSUE-006 / V01 remediation revalidation](../../QA/issue-006-v01-revalidation-20260921.md).

| Slice | Status | Evidence boundary |
|---|---|---|
| Foreign backend listener on `7690` | `PASS` | PID `42608` survived; launch returned nonzero with the port/PID remediation and did not start the application. |
| Foreign frontend listener on `9847` | `PASS` | PID `18324` survived; launch returned nonzero with the port/PID remediation and did not start the application. |
| Explicit cleanup with foreign listener | `PASS` | Confirmed `KillApplicationProcesses` returned nonzero and left PID `29992` listening. |
| Owned-process prompt and shutdown | `PASS` | Declined launch preserved the owned runtime; accepted launch stopped only the matched process tree, restarted successfully, and explicit cleanup released both ports. |
| Fresh disposable SQLite startup | `PASS` | Two launcher initialization runs reached `202609170001`; health, frontend readiness, empty sessions, and SQLite integrity passed. |
| Populated disposable SQLite startup | `PASS` | Clone initialization and launch retained 18 sessions, 50 versions, 28 revision runs, 90 artifacts, `foreign_key_check=[]`, and `integrity_check=ok`. |

Current V01 status is therefore `PASS` for the source-launcher scope. The
packaged desktop and clean-machine gates remain separate.

## Current release assessment

The original populated-database migration blocker has been remediated in source. SQLite migration transactions now suspend foreign-key enforcement only while Alembic performs the atomic parent-table rebuild, run `PRAGMA foreign_key_check` before commit, and restore the connection's prior enforcement state. In the historical 2026-09-18 audit, the fix was verified with foreign keys enabled in the migration fixture and with a task-local clone of the then-current populated database through `start_on_windows.ps1 -Action InitializeDatabase`; all existing clinical/revision row counts and SQLite integrity checks were preserved. The shared source database was intentionally not advanced by that historical run.

Release readiness remains blocked by the independent revision-acceptance, local-inference, access-key, timeline, packaging, and clean-machine gates. A current synthetic clinical workflow has passed through the exact configured OpenCode Go model with report, evidence, and restart persistence; this bounded clinical PASS does not certify every provider response or clinical adjudication.

The historical 2026-09-18 post-remediation register counts: `PASS 13`, `ATTENTION 3`, `FAIL 0`, `NOT TESTED 21`, `NOT APPLICABLE 2`.

## Clinical and structured-source validation checkpoint — 2026-09-22

This checkpoint supplements the historical 2026-09-18 feature-state register and the 2026-09-21 Tier 0 checkpoint. The baseline was clean `develop` HEAD `c61c9e02aa14a489703fce1ed09adb5c3575be8d`; the validation itself used the current working tree after surgical remediations. Synthetic data only. Structured-source mutations ran in `runtimes/cache/clinical-source-refresh-20260922-c61c9e0`; the original database was not used for mutation. It reached Alembic head `202609170001` with `integrity_check=ok` and no foreign-key violations.

The run covers the aggregate Tier 2 clinical workflow intent (`V20`–`V24`) and the structured-source portion of the Tier 3 campaign (`V30`–`V40`). It does not assign individual subtest mappings to `V21`/`V22` or `V36`–`V39` where the campaign index does not define that mapping.

| Capability | Status | Current run evidence |
|---|---|---|
| Exact-provider clinical workflow | `PASS` | Synthetic multi-drug job `3da3b0f0` completed as session 4 on exact OpenCode Go `deepseek-v4-flash`. Seven provider calls returned HTTP 200; no retry or provider/model fallback. Preflight passed, and a deliberately invalid case was blocked without a job/session. |
| Extraction, resolution, calculation, and report | `PASS` | Nitrofurantoin, amoxicillin/clavulanate, and atorvastatin remained separate with accepted RxNav identities, direct LiverTox matches, per-drug assessments and provenance. 17 observations were dated across four dates. Independently computed R-score `(420/40)/(160/120)=7.875`, hepatocellular, matched the rendered report. RAG audit was valid; 18 references retrieved, 11 bibliography entries, no citation outside the bibliography. |
| Session persistence and recovery | `PASS` | Session 4 survived browser reload and a standard-launcher application restart with report, inputs, labs, assessments, citations, metadata, and successful status intact. Case C live reload recovered the same running job `1d2a0770` without a duplicate session. |
| Post-refresh clinical regression | `PASS` | After the ordered-source failure and DILIrank retry, a RAG-on synthetic case completed as successful session 5. It resolved Nitrofurantoin with RxNav/LiverTox/DILIrank, retrieved RAG evidence, rendered citations, and independently matched R-score `(300/40)/(140/120)=6.4286`; its overall adjudication correctly remained `insufficient_data`. |
| Clinical cancellation | `PASS` | `VAL-20260922-007`: The first live Stop attempt exposed delayed cooperative cancellation. A stop-aware await boundary now cancels in-flight async extraction tasks before fallbacks can be written; its unit regression passed, and exact-provider retest job `ec61953e` reached terminal `cancelled` with `progress_status=cancelled` within two seconds. |
| Ordered structured-source Update All | `PARTIAL` | Job `b47e8264` visibly ran in order: RxNav completed after 21,202 records, LiverTox failed because NCBI Bookshelf returned a CAPTCHA challenge, and DILIrank was explicitly skipped. The combined job ended failed at 35%, one of three completed. |
| Source preservation and retry | `PASS` for the exercised boundaries | SQLite retained 1,593 LiverTox monographs and 1,336 DILIrank records and metadata through the failed combined run; integrity was `ok` and foreign-key violations were zero. Standalone DILIrank retry `eace912c` completed with 1,336 persisted rows (733 linked, 286 unmatched, 317 ambiguous), and the UI returned to Completed. Full all-source success remains externally blocked. |
| Cancellation and failure regression | `PASS` for structured-source job behavior | A live combined-source cancellation after the backend correction rendered all three sources as cancelled with zero completed. Unit coverage exercises pending cancellation, child exception during cancellation, and failure after an earlier source. The progress UI accurately reports completed source count. |

The NCBI Bookshelf CAPTCHA was not solved or bypassed. The downloader now identifies human-verification pages, reports the upstream block, and does not fall through to obsolete URLs. This is an external refresh blocker, not a clinical/provider PASS or a reason to treat the local archive as fresh data.

Current focused gates: backend `102 passed, 1 skipped` (PostgreSQL persistence needs `TEST_DATABASE_URL`), clinical extraction/cancellation slice `14 passed`, Angular tracker specs `2 files / 7 tests passed`, Ruff passed, and `git diff --check` passed with only line-ending notices. The detailed run, `VAL-*` findings, source counts, and visual observations are in [the QA report](../../QA/clinical-analysis-and-source-refresh-2026-09-22.md).

Final cleanup for this checkpoint completed after the read-only integrity checks: the disposable source clone and task-created bytecode caches were removed, the original database remained unchanged, standard launcher cleanup found no application processes, and ports `7690` and `9847` were free. Five ignored pytest cache directories under `app/tests/runtimes/cache/pytest` rejected an explicit removal attempt with an ACL denial; permissions were not widened, so that locked-path residue remains recorded for follow-up.

## Revision acceptance validation — 2026-09-22

This checkpoint continues the revision acceptance plan on local `develop` HEAD
`5960fe29d4f6b6b41dc235d5dd177cb420a9487b` in the development app.
Only synthetic session 22 (`Synthetic Case C`, source version 39) was used.
The configured Revision role remained exactly `opencode_go /
deepseek-v4-flash`; all recorded steps used that pair, with no fallback.
No application source or test code changed. Detailed run IDs, draft hashes,
QA findings, and cleanup are in the [revision acceptance report](../../QA/revision-acceptance-2026-09-22.md).

| Capability | Status | Current run evidence |
|---|---|---|
| Exact-provider revision route | `PASS` for observed routing | Runs 30–35 used exact OpenCode Go `deepseek-v4-flash` steps with no fallback. Run 29 stopped at planning with `network_unavailable` and no HTTP status. Run 34 completed the accepted path; run 35 confirmed a subsequent request selected accepted source version 61 before deliberate cancellation. Latency remained variable, so the broader provider component stays partial. |
| Deterministic edit and QA | `PASS` for the accepted path | Runs 30, 31, and 33 exercised failed and fail-closed edits. Run 32 reached the correct insertion position but was cancelled while QA remained active. Run 34 applied the unique anchored patch before `## Bibliography`; deterministic validation passed and LLM QA reported zero blocking issues. |
| QA-blocked draft persistence | `PASS` for the exercised failure path | Run 30 retained its draft and both failed QA artifacts. After navigating away and reloading, the UI still showed the QA-blocked state, 13 steps, 7 artifacts, and that the current report remained unchanged. Source report hash and synthetic session fields matched their pre-run values. |
| Accepted child and subsequent lineage | `PASS` for the synthetic development path | Run 34 created child session 24/version 61 from source version 39 with `version_status=llm_qa_passed`, `llm_qa_status=passed`, `revision_kind=llm_assisted_revision`, and the exact pipeline ID. The child report exactly matched the 9,272-character persisted draft. Navigation away, full browser reload, source-session reopen, and child-session reopen all preserved the accepted result and the source report. Run 35 then persisted `source_version_id=61` from session 24 before UI cancellation, proving current-version selection without creating a second child. |
| Focused backend/frontend regressions | `PASS` | Backend revision/provider and repository persistence suite: 88 passed, 1 skipped, 1 deprecation warning. Angular revision spec: 1 file, 4 tests passed. |
| Production frontend build | `ATTENTION` | Both `npm.cmd run build -- --progress=false` under system Node 22.23.1 and the Angular CLI build under bundled Node 22.13.0 exited `0xC0000005` before producing build output. This host did not verify a production build; no application-code diagnostic was emitted. |

The development database remained at Alembic head `202609170001`, with
`integrity_check=ok` and no foreign-key violations after the attempts. The
task-started processes were stopped and ports `7690` and `9847` verified free.
`model.provider.opencode-go` remains `PARTIAL` because its broader provider
matrix and variability requirements exceed one successful revision.
`revision.accepted-session-finalization` is `VALIDATED` for the stated
synthetic development scope. The automated regression component is `PARTIAL`
for this checkpoint because its focused test slices passed but the requested
production build did not complete.

## Production frontend build host comparison — 2026-09-22

This follow-up used `develop` HEAD
`7d797e8a7bae10e86fa28f4e53409f8f81ccbb26`, the documentation-only successor
to the revision-acceptance checkpoint above. The worktree was clean at task
start. Windows reported build host `10.0.26200.0`, PowerShell `7.6.6`, and the
launcher-provisioned Node.js `v22.13.0` / npm `10.9.2`. No application source
or test files changed.

| Gate | Result |
|---|---|
| Fresh frontend install | `npm ci --ignore-scripts --no-audit --no-fund` against a new isolated cache first failed because sandboxed registry fetches returned `EACCES`; rerunning with network access installed 471 packages successfully. |
| Frontend tests | `npm run test -- --no-watch`: 23 files, 97 tests passed, exit code 0. |
| Direct production build | `npm run build -- --progress=false` terminated with `-1073741819` (`0xC0000005`) before Angular emitted build output or a diagnostic. |
| Launcher rebuild | `start_on_windows.ps1 -Action RebuildFrontend` recognized Node `22.13.0` and reused installed dependencies, then failed on the same `0xC0000005` from `npm.cmd run build`. |
| Exact-SHA Windows CI | [Run 35764186160](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35764186160) used this exact SHA. `Setup Node`, frontend install, frontend tests, and `Build frontend` passed. The overall CI run failed later at the full browser E2E step; `security-scan` and `backend-quality` also failed, `persistence-contract` passed, and live-provider E2E was skipped. |
| Existing build reuse | Two standard launcher starts reported the build marker current and skipped frontend dependency synchronization and Angular build. `/api/health` and `/` returned HTTP 200, with page title `DILIGENT Clinical Copilot`; the served index SHA-256 was `8085087A6FFFD895D7B181377C37D369B2729AF7C88C44D29EAF063DEE6F1353`. The launcher action exited 1 only when the sandbox denied opening the default browser after both services were serving. This verifies current marker reuse, not fresh local build generation. |
| Crash diagnostics and cleanup | No matching recent Application log event 1000/1001 or user-local `node.exe` crash dump/WER report was found. The native faulting module remains unknown. Task-owned listener PIDs and executable paths were rechecked before stop; both configured ports were free after cleanup. |

The direct and launcher build failures are classified as specific to this
isolated validation host because the exact SHA passed the production build on
hosted Windows CI with Node `22.13.0`. No repository-owned cause was reproduced
and no application workaround was introduced. Keep `test.automated-regression`
`PARTIAL` until a fresh pinned local build succeeds or host-level diagnostics
resolve the native failure. The overall hosted CI run was not green, and its
other failure gates remain independent.

The focused report and raw logs are under
[`assets/QA/frontend-build-validation-2026-09-22/`](../../QA/frontend-build-validation-2026-09-22/report.md).

## Timeline cancellation validation — 2026-09-23

This checkpoint started from clean `develop` HEAD
`91404256dce1aa1d538727c80822719699576df4`. It covers the cancellation
subgate only. The overall `sessions.timeline` status remains `PARTIAL` because
dated fallback fidelity, browser retry, and controlled persistence-failure
behavior remain unverified.

The official `start_on_windows.ps1 -Action Launch` rebuilt and started the
application on ports 7690/9847 using a disposable SQLite database in the QA
folder. The Browser used synthetic session 1, `Synthetic Timeline Cancellation
QA`; the visible Timeline page showed the configured role label `qwen3.5:2b`.
A temporary environment-gated QA hook replaced the asynchronous timeline
extractor with a local awaitable that blocks until cancelled. The hook was
loaded before backend startup and removed from the source tree afterward, so
the run did not contact a provider or modify provider/model settings or
credentials. The disposable database was removed after the evidence check.

| Gate | Result |
|---|---|
| Backend cancellation and retry regression | `app/tests/unit/test_data_inspection_repository.py -k timeline_job`: 2 passed, including interruption during extraction, terminal `cancelled`, zero persistence, and a later successful generation. |
| Frontend API/component regressions | 2 focused Angular spec files, 6 tests passed; includes the typed `DELETE` request, visible Stop/stopping/terminal states, request failure, and completion-race reconciliation. |
| Ruff | Passed for the changed backend service, timeline module, and repository test. |
| Official launcher build and health | Fresh production frontend build completed; backend `/api/health` returned `{"status":"ok"}`. |
| In-app Browser cancellation flow | Stop was visible while status was `In progress`; after click the UI showed `Stopping…` while polling; the terminal view showed `Timeline generation cancelled.` and the enabled Generate Timeline control. |
| Timeline history and storage | The terminal Browser view showed `No generated timelines yet`; a read-only SQLite check while the runtime was active found 0 rows in `clinical_session_timelines`, then the disposable database was removed after shutdown. |
| Cleanup | Launcher-owned backend/frontend process trees were identity-checked and stopped; ports 7690 and 9847 were confirmed free. |

The Browser accessibility-state excerpts and reproduction details are in
[`assets/QA/timeline-cancellation-20260923/report.md`](../../QA/timeline-cancellation-20260923/report.md).
The in-app Browser screenshot was inspected inline; this Browser surface did
not expose a disk-export path, so the report preserves the observed rendered
text and state without claiming a screenshot file. This is controlled local
evidence, not live-provider validation.

## Timeline fallback date fidelity and recovery — 2026-09-23

This checkpoint started from clean `develop` HEAD
`7ceacb5919e35a7a018e985be7c60fb65b4b7afd`. It covers source-date
preservation, retry after a fallback, and persistence-failure recovery. The
overall `sessions.timeline` gate remains `PARTIAL`: the Browser run used
test-injected faults and a controlled extractor, so it does not establish live
provider behavior.

The official `start_on_windows.ps1 -Action Launch` ran against a disposable
SQLite database under `runtimes/cache/qa/timeline-recovery-20260923/` on ports
7690/9847. The seeded session contained synthetic symptom, medication, and ALT
text. The Timeline page displayed the assigned `qwen3.5:2b` local model. A
temporary, environment-gated `sitecustomize.py` hook was loaded into the
backend process only; it replaced extractor call 1 with a controlled
`network_unavailable` failure, returned a synthetic event on retry calls, and
raised a one-time persistence exception on save call 2. The hook's event log
confirmed the sequence. Provider settings, credentials, and the shared
database were not changed, and no model request was sent.

| Gate | Result |
|---|---|
| Backend date and transaction coverage | `app/tests/unit/test_data_inspection_repository.py -k "timeline or fallback_date_extraction"`: 21 passed, 7 deselected. Covers day/month/year precision, duplicate same-date tokens, missing/relative/invalid/ambiguous sources remaining undated and uncertain, no visit-timestamp inference, SQLite failure rollback, and successful retry. |
| Frontend retry coverage | Full Angular/Vitest suite: 24 files and 102 tests passed; includes Generate re-enabled after fallback and persistence failure, followed by a successful retry. |
| Ruff and whitespace | Ruff passed for all changed Python files, including the source, backend test, synthetic-session seed, and controlled fault hook; `git diff --check` passed. |
| Official launcher | Launch reused the current frontend build, started backend/frontend, and passed its `/api/health` readiness check. |
| In-app Browser fallback | After injected provider failure, the saved fallback showed 3 events: therapy `2025-01` at month precision, and symptoms and ALT at `2025-01-17` at day precision. Each event showed explicit placement and its source text. |
| In-app Browser persistence error | Generate was enabled after fallback. The second attempt showed `Controlled QA timeline persistence failure.`; Generate became enabled again and history still showed only fallback timeline #1. |
| In-app Browser successful retry | The third attempt showed `Timeline generated and saved.` and a second saved timeline. Reopening it showed the controlled synthetic event on `17 Jan 2025`. The app's `LLM generated` label reflects the exercised application success path only; the event came from the test hook, not an LLM. |
| Read-only SQLite check | Exactly two rows remained after the sequence: fallback #1 and successful retry #2. The failed save did not add history. The focused backend trigger test separately confirmed history stayed empty after a failed first save before its retry. |
| Cleanup | Launcher-owned backend/frontend trees were identity-checked and stopped; ports 7690/9847 were verified free. The disposable database, hook log, and test basetemp were removed. |

Browser accessibility-state excerpts, hook setup, seed data, and reproduction
details are in
[`assets/QA/timeline-recovery-20260923/report.md`](../../QA/timeline-recovery-20260923/report.md).
The Browser screenshot was inspected inline; the Browser surface did not
provide a disk-export path, so the report records the rendered text and state
without claiming a screenshot artifact. This is controlled local evidence,
not live-provider validation.

## Configured live Ollama timeline lane — 2026-09-23

This checkpoint started from `develop` HEAD
`7c75e51b9910a0db85ea87f225b4d542e2a76518`. The official launcher ran against
a synthetic session and a disposable SQLite database under
`runtimes/cache/qa/timeline-live-revalidation-20260923/`. Settings and the
Timeline panel both showed the configured `Local (Ollama)` / `qwen3.5:2b`
lane; the loaded Ollama catalog marked that model installed. The shared
database, `.env`, runtime model settings, and credentials were not changed.

| Gate | Result |
|---|---|
| Pre-fix live output | Two same-lane generations both persisted an unsupported `Onset of Hepatitis B infection` event dated `2022-05-18`. Its alleged acute-care/HBV evidence did not occur in the synthetic source. Both rows were incorrectly attributed to `model_provider=openai` despite `source_kind=local` and `source_model=qwen3.5:2b`. |
| Evidence guard and provenance fix | Nonempty model evidence is now required to occur within one source text field after case folding and whitespace normalization. Unsupported evidence follows `invalid_response` and the existing deterministic fallback. The prompt requests a verbatim quote; local history metadata records `ollama`. No public API or type contracts changed. |
| Focused backend regressions | Six selected tests passed, including unsupported evidence rejection, valid grounded extraction, fallback handling, and local/cloud provenance assertions. |
| Ruff and whitespace | Ruff passed for the changed Python files with pre-existing `DTZ001` naive-datetime findings ignored; the new test timestamp uses UTC. `git diff --check` passed. |
| Official launcher and preflight | `/api/health` returned `ok`. In Settings and Timeline, the effective source was Local (Ollama) and the assigned Timeline model was `qwen3.5:2b`; no fallback lane was selected. |
| In-app Browser live generation | Generate and one Regenerate ran on the same configured lane. Both saved entries showed `Fallback chronology` with `Invalid structured provider response`; no unsupported event was saved. The fallback showed therapy at `2025-01` month precision and symptoms and ALT at `2025-01-17` day precision, each with source evidence. |
| Read-only SQLite and reload | Two rows persisted, both `fallback` / `invalid_response`, model `qwen3.5:2b`, source kind `local`, provider `ollama`. After navigating away and reloading, Browser history still showed timelines #2 and #1 with three evidence-backed events each. |
| Gate boundary | No source-grounded `llm_generated` result was obtained. Timeout, authentication, and rate-limit paths were not tested or manufactured. `sessions.timeline` remains `PARTIAL`. |
| Cleanup | Both disposable databases and the focused test cache are temporary under `runtimes/cache/qa/`. Launcher cleanup misclassified its own listeners; exact DILIGENT process paths, commands, and PIDs were verified before stopping only task-started app processes. Ports 7690/9847 were verified free; the user-started Ollama process remained running. |

Browser observations and the read-only database evidence are recorded in
[`assets/QA/timeline-live-validation-20260923/report.md`](../../QA/timeline-live-validation-20260923/report.md).
The rendered fallback chronology and history list were visually inspected in
the in-app Browser. Its screenshot surface did not provide a disk-export path,
so no screenshot artifact is claimed. The post-fix error category is recorded
as exposed by the Browser/database; no timeout, authentication, or rate-limit
failure was simulated.

## Tier 4 timeline model and recovery validation — 2026-09-23

This campaign started at develop HEAD e933d3008a5773e52b21a676bd8d9b10964ca75d.
The official launcher ran on ports 7690/9847 against the isolated SQLite
backup at runtimes/cache/qa/timeline-model-matrix-20260923/timeline-model-matrix.db.
Synthetic sessions 25, 26, and 27 used identical anamnesis, medication, and
laboratory text; the seed helper and full report are in
[the timeline model matrix QA folder](../../QA/timeline-model-matrix-20260923/report.md).
The shared database and saved settings were not modified. The local catalog
refresh confirmed qwen3.5:2b and qwen3.5:9b installed. The configured
OpenCode Go route was deepseek-v4-flash, whose catalog identity is DeepSeek
V4.1 Flash. No credential value was output or changed.

| Gate | Result |
|---|---|
| Focused backend timeline regressions | 45 passed across timeline retry behavior, error diagnostics, and repository persistence tests. Synthetic timeout and rate-limit errors retried successfully; authentication did not retry; timeout, authentication, and rate-limit fallback codes persisted. No real authentication failure or rate limit was triggered. |
| Timeline component specs | 12 passed, including the timeout, authentication, and rate-limit fallback labels. |
| Ruff | All changed Python files passed with pre-existing DTZ001 findings ignored; the two focused retry/diagnostic files passed without ignores. The repository test file has 14 untouched naive-datetime findings. |
| Official launcher and browser | Launcher readiness returned backend health status ok. Settings and each Timeline panel showed the selected role model. Browser error/warning logs were empty, and the rendered timeline was visually inspected inline. |
| qwen3.5:2b | Session 25 reached local Ollama and stored source_model qwen3.5:2b, source_kind local, and model_provider ollama. It saved Fallback chronology / invalid_response with three fallback_parser events. The UI showed the failure class and retained month precision for medication and day precision for symptoms and ALT. History survived reload. |
| qwen3.5:9b | Session 26 stored the exact local Ollama provenance and an LLM-generated three-event timeline. Symptom and ALT evidence exactly matched the source at explicit day precision. The medication evidence matched the source at month precision and was marked inferred with a review note. The event Source field was Not reported and confidence was Not scored for all three events. History survived reload. |
| DeepSeek V4.1 Flash | Session 27 used cloud / opencode_go / deepseek-v4-flash and stored three LLM-generated events with exact evidence and displayed sources from anamnesis, laboratory_analysis, and drugs. The symptom and ALT dates were day-precision; medication was month-precision. History survived reload. |
| Read-only SQLite | Exactly one saved timeline was present per synthetic session. Session 25 persisted fallback / invalid_response and fallback_parser sources; session 26 persisted llm_generated with null event-level Source fields; session 27 persisted llm_generated with provider opencode_go and the three expected source fields. |
| Gate boundary | sessions.timeline remains PARTIAL because qwen3.5:2b still produced no grounded LLM timeline. The local Ollama gate remains PARTIAL. This single timeline route does not promote the broader OpenCode Go provider gate. |
| Cleanup | Task-started app processes and the isolated database/test basetemp folders were removed; ports 7690/9847 were verified free. The user-started Ollama service remained running. |

The Browser screenshot was inspected inline. The Browser surface did not
provide a disk-export path, so no screenshot file is claimed. No API, public
type, or database schema changes were made.

## Timeline source attribution validation — 2026-09-23

This slice started from clean `develop` HEAD
`309ad24558b0bafa68cd4c2827581431499b037d`. The official launcher used the
isolated SQLite database in
[`assets/QA/timeline-source-attribution-20260923/`](../../QA/timeline-source-attribution-20260923/)
and synthetic sessions 1 and 2, which had identical anamnesis, medication,
and laboratory input. The shared database, saved model settings, and
credentials were not changed. The Timeline role was switched from 2B to 9B
only inside the isolated QA database.

The prompt now asks for the canonical field containing each verbatim evidence
quote. The extractor derives the existing `source` field only after the
evidence passes validation. Direct fields and same-name copies in `sections`
count as one field; quotes missing from source text or present in multiple
distinct fields do not receive a label. Conflicting model-supplied labels are
overwritten. The fail-closed `invalid_response` guard remains intact; API,
schema, and database contracts did not change.

| Gate | Result |
|---|---|
| Focused backend timeline/retry/diagnostic/repository regressions | 56 passed, 7 deselected. Existing warnings: pytest `cache_dir` config option is unknown and Google GenAI emits a deprecation warning. |
| Timeline component specs | 12 passed. |
| Ruff and whitespace | All changed Python files passed `ruff check --no-cache`; `git diff --check` passed. |
| `qwen3.5:2b` live lane | Two generations both saved fallback / `invalid_response`; each retained three evidence-backed fallback events. Each row stored `source_kind=local`, `model_provider=ollama`, `source_model=qwen3.5:2b`. The fallback retained `2025-01` month precision for medication and `2025-01-17` day precision for symptoms and ALT. |
| `qwen3.5:9b` live lane | Two generations both saved `llm_generated` with three evidence-backed events. Each row stored `source_kind=local`, `model_provider=ollama`, `source_model=qwen3.5:9b`. The evidence mapped to `drugs`, `laboratory_analysis`, and `anamnesis`; medication remained at month precision and symptoms/ALT at day precision. All Source values rendered in the event cards and details. |
| Confidence | The 9B model omitted confidence and rationale. Both persisted as null and the UI showed “Not scored” / “Not reported”; no confidence was invented. |
| Persistence and input parity | All four timeline rows persisted in the isolated DB; each generation survived reload in history. The synthetic direct source fields for sessions 1 and 2 were equal. SQLite `integrity_check` returned `ok`; `foreign_key_check` had zero violations. |
| Browser console | The in-app Browser provided accessible UI state and screenshots but no console-log reader. The DevTools shortcut did not open a console. Errors and warnings remain unverified and are not claimed empty. |
| Gate boundary | `sessions.timeline` and `model.provider.local-ollama` remain `PARTIAL` because both exact 2B runs still returned `invalid_response`. The 9B source-label gap is resolved for the exercised synthetic case. Broader OpenCode Go and revision gates remain partial; source catalog/refresh remain partial while NCBI presents a browser verification challenge; automated regression remains partial pending host/build diagnostics and a full green run; access-key management and desktop release remain blocked on prerequisites. The RAG duplicate-file policy still needs a product decision. |
| Cleanup | Verified task-started DILIGENT processes were stopped; ports 7690/9847 were free. The user-started Ollama process remained running. The isolated DB and focused pytest basetemp were removed after evidence capture. |

The full evidence and limitations are recorded in
[`assets/QA/timeline-source-attribution-20260923/report.md`](../../QA/timeline-source-attribution-20260923/report.md).
The Browser-rendered Source values were inspected inline; no screenshot file
is claimed because the Browser surface did not export one.

## Local timeline model gate revalidation — 2026-09-23

Revalidated from clean `develop` HEAD
`07f7bba7908b202d3cc8683a7c1b6db79c45dd40`, equal to `origin/develop` at
start. The official launcher started the backend and frontend against the
isolated database in
`runtimes/cache/qa/timeline-local-gate-20260923/`; health and frontend HTTP
checks returned 200. The launcher's automatic default-browser open failed
with access denied. The embedded Browser panel was 559 pixels wide and showed
the application's 1100-pixel minimum-width message, so the rendered workflow
was inspected in the Codex-controlled full-width Chrome surface. Only
synthetic sessions were used; shared data, saved runtime configuration, and
credentials were not changed.

| Gate | Result |
|---|---|
| `qwen3.5:2b` live lane | Two generations persisted as fallback / `invalid_response`, each with three evidence-backed `fallback_parser` events. Both rows stored local / Ollama / `qwen3.5:2b` provenance; medication stayed at month precision and symptoms and ALT at day precision. The server log classified both failures as `_UnsupportedTimelineEvidenceError` after structured response parsing, at source-evidence validation. |
| `qwen3.5:9b` live lane | Two generations persisted as `llm_generated`, each with three events and local / Ollama / `qwen3.5:9b` provenance. Exact quotes mapped to `drugs`, `laboratory_analysis`, and `anamnesis`; medication stayed at month precision and symptoms and ALT at day precision. The rendered Source values matched evidence-derived fields. Confidence and rationale remained unset when omitted. |
| Browser and persistence | The rendered chronology and event inspector showed the correct model, evidence, source labels, and timing. Reload retained both 2B fallbacks and both 9B timelines. |
| Focused backend suite | 70 passed, 7 deselected across patient/lab timeline extraction, retry, diagnostics, and timeline repository tests. Pytest emitted the existing unknown `cache_dir` option warning and a Google GenAI deprecation warning. |
| Full Angular/Vitest suite | 24 test files and 105 tests passed with `npm run test -- --no-watch`. |
| Ruff and SQLite | The synthetic seeder passed `ruff check --no-cache`; SQLite `integrity_check=ok`, with zero `foreign_key_check` rows. |
| `test.automated-regression` | Remains `PARTIAL`: hosted run [35895672146](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35895672146) on validation commit `a004078879fbf3ce4da1b4350f893ad1403f2a3d` passed frontend tests/build and the SQLite/PostgreSQL persistence contract, but failed Alembic drift validation, Python dependency audit, and Windows browser E2E; live-provider E2E was skipped. The local fresh production build's `0xC0000005` also remains unresolved. |

The detailed evidence is in
[`assets/QA/timeline-local-gate-20260923/report.md`](../../QA/timeline-local-gate-20260923/report.md).
The 2B guard correctly rejected unsupported model evidence; no prompt or
extractor defect was found, so no implementation change or guard weakening was
made. `sessions.timeline` and `model.provider.local-ollama` remain `PARTIAL`
until grounded 2B output and broader model coverage are demonstrated. NCBI
catalog/refresh, OpenCode Go, API-route, RAG duplicate-file, accessibility,
access-key, and desktop release work remain separate; access-key and release
gates remain `BLOCKED` on their documented prerequisites.

## Automated regression gate revalidation — 2026-09-23

This scope started from clean `develop` HEAD `f61a9ff6`, equal to
`origin/develop`. It rechecked the current regression implementation after
hosted run [35895672146](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35895672146)
reported migration drift, dependency-audit findings, and a browser E2E
failure. All runtime and test data used a disposable SQLite database; no
shared database, saved model settings, or credentials were changed.

The audit identified two CVEs in the direct `anyio==4.9.0` pin
(`CVE-2026-63374` and `CVE-2026-64847`), with `4.14.2` listed as the fixed
version. The direct pin and `uv.lock` now use `anyio==4.14.2`. The hosted E2E
failure was an obsolete test expectation: `DiliJobTrackerService` persists an
active job in `dili-agent-active-job-v1` and reconnects after reload, while
the test expected the job to disappear. The test now checks that the saved
job resumes, completes without a duplicate submission, renders its report,
and clears the persisted marker. The reported migration drift did not
reproduce on a fresh local SQLite database.

Hosted rerun [35907570582](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35907570582)
then showed that CI had not created the parent directory for its disposable
SQLite migration database. The workflow now creates that directory before
Alembic starts; a local run from a deliberately absent parent passed all
three migration commands. Run 35908387779 passed this migration check and
surfaced two Pyright errors in cancellation helpers that scheduled a generic
`Awaitable` directly. The helpers now wrap the awaitable in a typed coroutine;
local Pyright and focused cancellation tests passed.

The final exact-SHA hosted run
[35908957190](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35908957190)
completed successfully on `3e73082f619024140cfdb92340831f450c6a3e13`.
All configured jobs passed: backend migration/Ruff/Pyright/unit, Python,
Angular, desktop, and Rust audits, SQLite/PostgreSQL persistence, and the
Windows frontend/build/browser lane. The workflow-dispatch-only live-provider
job was skipped as configured.

| Gate | Result |
|---|---|
| Locked dependency install | `uv sync --locked --project app/server --all-extras` installed `anyio 4.14.2` into the existing server venv. |
| Python dependency audit | CI-equivalent export and strict `pip-audit` completed with `No known vulnerabilities found`. |
| Fresh SQLite migration and metadata check | `upgrade head`, `current --check-heads`, and `check` passed at `202609170001`; Alembic reported no new operations. |
| Backend tests | Full unit suite: `789 passed`, with 7 existing deprecation warnings. Model-config focused suite: `40 passed`, with one Google GenAI deprecation warning. |
| Browser E2E | Before the assertion fix, the full suite reproduced the single hosted failure (`40 passed, 5 skipped`). Afterward the focused recovery test passed and the full suite passed (`41 passed, 5 skipped`). |
| In-app Browser smoke | The isolated source runtime rendered the DILI Agent shell, primary navigation, clinical input, and empty report state. No analysis was submitted. |
| Exact-SHA hosted backend and persistence | Run `35908957190` passed migration/drift (`No new upgrade operations detected`), Ruff, Pyright (`0 errors`), backend unit (`789 passed, 7 warnings`), security audits, and SQLite/PostgreSQL persistence contract. |
| Exact-SHA hosted Windows regression | Frontend: `24 files, 105 passed`; production build passed; model-config suite `40 passed, 1 warning`; browser E2E `39 passed, 7 skipped`. |
| Build boundary | The official launcher verified the stored frontend fingerprint and reused the existing output. A fresh local production build was not generated; the separate local `0xC0000005` host failure remains open, although the hosted Windows build passed. |

The seven hosted E2E skips are conditional: live-provider and pinned
multilingual embedding tests require explicit opt-in; two model API tests
need an available Ollama model; and three session/timeline UI tests need
persisted sessions absent from the isolated CI database. The separate live-
provider workflow lane also requires explicit dispatch and a repository
secret. `test.automated-regression` remains `PARTIAL` because the fresh local
production-build crash is unresolved and provider, embedding, Ollama, and
persisted-session coverage remains unrun. No provider call, access-key
mutation, package, or release gate is claimed here. The disposable database
and test caches were removed, and ports `7690` and `9847` had no listeners
after validation.

## Automated regression local-build follow-up — 2026-09-24

This follow-up started from clean develop HEAD
13f6c7a079122934e4c614452421dd1ef8792e54, equal to origin/develop. The HEAD is
a documentation-only successor to application source commit
3e73082f619024140cfdb92340831f450c6a3e13, which passed hosted CI run
35908957190. The status ledger named the fresh local production frontend build
as the next action for test.automated-regression.

On Windows NT 10.0.26200.0 with PowerShell 7.6.6, the official launcher used
Node.js 22.13.0 and npm 10.9.2. From app/client, npm run test -- --no-watch
passed with 24 test files and 105 tests. Then start_on_windows.ps1
-Action RebuildFrontend reused the installed dependencies and completed npm
run build with exit code 0. Angular generated the production bundle in 9.894
seconds. The resulting build-state marker records the current build and
dependency fingerprints and Node 22.13.0. The [focused report](../../QA/frontend-build-validation-2026-09-24/report.md)
contains the command logs and the reviewed incomplete gates.

After this checkpoint was pushed as commit
d1dc431df277a4d06e3acd745cbc25facb93ecf2, exact-SHA hosted CI run
[35967078491](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35967078491)
completed successfully on that commit. Security scan, backend quality, Windows
regression, and persistence passed. The live-provider job was skipped because
the push event does not satisfy its workflow-dispatch and run_provider_e2e
conditions. This leaves the live-provider lane unrun.

The earlier local 0xC0000005 build failure did not reproduce on this run. Its
native cause remains unknown, so this is evidence of a successful current
build, not a diagnosis of the older fault. Keep test.automated-regression
PARTIAL because the hosted live-provider lane and conditional E2E cases remain
unrun; their credential, opt-in, model-availability, and persisted-session
preconditions are unchanged. No local backend, database, provider, or
application server was started or changed; ports 7690 and 9847 were free
after the checks.

The next coherent validation slice is the isolated local timeline pair:
sessions.timeline and model.provider.local-ollama. Recheck qwen3.5:2b
grounding alongside the qwen3.5:9b comparison, with evidence, source
attribution, date precision, provenance, rendering, and reload checks. Keep
the pair PARTIAL until grounded 2B output is observed. The other active
partial and blocked gates are recorded in the focused report and remain
separate because they depend on provider credentials, upstream NCBI
availability, release prerequisites, or different workflow scope.

## Isolated local timeline pair revalidation — 2026-09-24

This slice followed the next action above and started on `develop` at clean
HEAD `85cdf3515c3d054a6aee7fc06d64816e02252e4e`, equal to `origin/develop`.
The official Windows launcher used a disposable SQLite database and isolated
data root. Two synthetic sessions contained a symptom date, medication month,
and lab date. The exact `ollama / qwen3.5:2b` and
`ollama / qwen3.5:9b` routes were exercised in the in-app Browser.

Both runs saved a fallback chronology: timeline 1 for 2B and timeline 2 for
9B, each with `generation_error_code=unknown`, exact local/Ollama/model
provenance, and three evidence-backed `fallback_parser` events. The medication
remained month precision (`2025-01`); symptom and lab remained day precision
(`2025-01-17`). The rendered history and event evidence were inspected, and
both records survived reload. No model-generated timeline succeeded in this
pair. The exact request exceptions were not captured, so the `unknown` records
cannot be attributed to model quality, provider failure, or timeout.

The isolated parser timeout setting was 3,600 seconds, but the timeline
service caps its outer wait at 300 seconds. Separate short non-clinical direct
Ollama probes took about 127 seconds for cold 9B and 96 seconds for 2B; they
are runtime-performance context only and do not establish timeline quality.
Cold-start and local throughput are plausible contributors to the long
browser runs, but remain inference without the original exception. The
in-progress browser view remained at 25% until reload; the job terminal state
was not captured, so a progress-display defect is not confirmed.

The classifier previously mapped a bare `TimeoutError()` with no message to
`unknown`. It now recognizes the exception type, and the focused diagnostics
suite passed **9 tests with 1 existing Google GenAI deprecation warning**.
This corrects a proven classification gap but is not claimed to explain either
of the two observed persisted failures. The detailed evidence, environment,
limitations, open gates, and cleanup record are in the
[local timeline pair report](../../QA/timeline-pair-validation-20260924/report.md).

Final status: `sessions.timeline` and `model.provider.local-ollama` remain
`PARTIAL`; exact-lane grounded output and request-specific failure attribution
are still required. `test.automated-regression` remains `PARTIAL` despite the
focused test passing because hosted live-provider and conditional E2E lanes
remain unrun. Catalog/source refresh, credential, OpenCode, revision, RAG,
accessibility, API catalog, and release leftovers remain separate as documented
in the focused report. The disposable runtime and test cache/data were removed;
the shared settings and database were not changed.

## Focused local timeline failure attribution — 2026-09-24

This follow-up started on `develop` at `a0566a6a77a5744c90e534c520ff94cd7d444184`,
equal to `origin/develop`, and revisited the timeline/provider leftovers from
the pair above. The official Windows launcher used a task-owned SQLite runtime
with two synthetic sessions and exact local routes `ollama / qwen3.5:2b` and
`ollama / qwen3.5:9b`.

The 2B run persisted a fallback with `generation_error_code=invalid_response`.
Its server trace records eight Pydantic validation errors for
`PatientTimelineExtraction`: the response had JSON-Schema-shaped fields rather
than a timeline instance. This is a model-response structured-output failure
for that request. It was not a timeout, API transport, job polling, or rendering
failure; the UI displayed the invalid-response classification and retained
three evidence-backed fallback events with month/day precision. A single run
does not prove that model size or performance caused the output defect.

The 9B run persisted `llm_generated`, with three source-backed events and no
generation error. The rendered detail identified evidence from the synthetic
drug, laboratory, and anamnesis inputs, preserved month/day precision, and
survived reload. Both persisted rows retained exact local/Ollama/model
provenance. Read-only SQLite checks returned `integrity_check=ok` and zero
foreign-key violations.

A first PowerShell timing wrapper exited `0xC0000005` without request output;
the following endpoint check found Ollama unavailable. That attempt is
classified as a harness/runner failure with unknown failing stage and supplies
no model-quality evidence. A Python standard-library client then completed
four direct non-clinical requests (`Reply exactly OK.`, four-token limit):
`qwen3.5:2b` cold 14.98 s / warm 0.079 s and `qwen3.5:9b` cold 39.22 s / warm
0.605 s. These timings characterize local API latency only and cannot establish
timeline schema conformance or grounding. They also do not back-attribute the
prior pair's `unknown` fallbacks.

The focused diagnostics suite passed **9 tests, 1 existing Google GenAI
deprecation warning**; Ruff passed for the timeline service and focused test.
The detailed evidence, exact failure classes, remaining independent gates, and
cleanup record are in the [local timeline diagnostics report](../../QA/timeline-local-model-diagnostics-20260924/report.md).

Final status: `sessions.timeline` and `model.provider.local-ollama` remain
`PARTIAL`: the current 9B lane generated a grounded timeline, while the current
2B lane failed the structured-output contract; the earlier pair's `unknown`
errors remain unattributed, and repeatability is not established.
`test.automated-regression` remains `PARTIAL`; hosted live-provider dispatch
and conditional E2E prerequisites are still outstanding. The separate
OpenCode/revision, NCBI source refresh, credential, API catalog, RAG policy,
accessibility, container, and release gates retain their documented statuses
and blockers.

The disposable runtime and its synthetic data were removed after listener
ownership checks. The shared settings and database were not changed.

## Feature-state register

| # | Stable capability | Area | Status | Current evidence, route, or scenario | Persistence / external dependency / gap |
|---:|---|---|---|---|---|
| 1 | Standard source launcher against the populated database | Runtime / migration | `PASS` | A task-local clone of the current populated database migrated through `202609170001` via `start_on_windows.ps1 -Action InitializeDatabase`; the same 18 sessions, 38 versions, 17 revision runs, and 54 artifacts remained available. | The shared `app/resources/database.db` was intentionally not advanced; the clone retained `PRAGMA foreign_key_check=[]` and `PRAGMA integrity_check=ok`. |
| 2 | Fresh SQLite bootstrap and Alembic head | Runtime / migration | `PASS` | Disposable source DB migrated through `202609170001`; `/api/health` returned 200. | Fresh DB evidence is complemented by the populated-clone remediation below. |
| 3 | Application shell and primary workspace navigation | UI | `PASS` | In-app Browser rendered DILI Agent, Clinical Sessions, Data Inspection, and Settings. | Browser smoke at one desktop viewport; no provider call. |
| 4 | General runtime settings persistence | Settings | `PASS` | Changed polling interval `1 -> 2`, saved, navigated, reloaded, and restored `2 -> 1`; value and persisted timestamp remained visible. | `settings/configurations.json` is back at baseline `1.0`; `.env` remained excluded. |
| 5 | Settings section surfaces | Settings | `PASS` | General, Models, Data Processing, Integrations, and Advanced routes rendered with source labels and controls. | Surface validation only for sections other than General persistence. |
| 6 | Local/cloud model configuration and access-key UX | Settings / external | `ATTENTION` | In the isolated QA database, Settings saved the Timeline role as `qwen3.5:2b` and `qwen3.5:9b` in turn; the refreshed Ollama catalog showed both installed, and the configured OpenCode Go route remained `deepseek-v4-flash`. Two 9B timeline runs now show evidence-derived Source labels; both 2B runs fell back as `invalid_response`. Cloud credential lifecycle remains unvalidated. [Timeline source attribution validation](../../QA/timeline-source-attribution-20260923/report.md); [Timeline model matrix](../../QA/timeline-model-matrix-20260923/report.md) |
| 7 | Clinical input validation and preflight | Clinical workflow | `PASS` | Empty input showed four blocking reasons; synthetic valid input showed the two unavailable structured-source blockers before execution. | No analysis job was submitted because the preflight correctly blocked it. |
| 8 | Clinical analysis submission, progress, terminal state, and recovery | Clinical workflow | `NOT TESTED` | Not reached in the current browser run. | Requires populated source catalogs and a usable model lane. |
| 9 | Session creation and persisted result | Sessions / persistence | `NOT TESTED` | Not reached because clinical execution was blocked. | Existing shared rows were inspected read-only; current UI could not start on that DB. |
| 10 | Session listing, empty state, filters, and search shell | Sessions / UI | `PASS` | Fresh runtime showed the empty state and All/Successful/Failed, search, and date-filter controls. | Populated-session interaction remains untested. |
| 11 | Session selection, CRUD, deletion, and switch isolation | Sessions / persistence | `NOT TESTED` | Not reached with a current live populated runtime. | Requires a successful end-to-end session fixture. |
| 12 | Patient anamnesis, therapy, laboratories, metadata, and propagation | Clinical input | `NOT TESTED` | Synthetic text was entered and displayed, but no session was persisted. | Image and metadata subpaths are recorded separately below. |
| 13 | Patient profile image upload and image metadata | Clinical input / media | `NOT TESTED` | Upload control was present but no file was selected during this source-only run. | Needs a controlled synthetic image fixture and persistence check. |
| 14 | Multi-drug extraction, normalization, and RxNav identity linking | Clinical pipeline | `NOT TESTED` | Not reached in a live clinical run. | Unit coverage exists; current source-catalog/provider boundary was not proven. |
| 15 | Longitudinal laboratory capture and history propagation | Clinical pipeline | `NOT TESTED` | Not reached in a live clinical run. | Unit coverage exists; rendered session evidence is absent. |
| 16 | Deterministic R-score and injury-pattern classification | Clinical calculation | `PASS` | Current source calculation for ALT 360/ULN 40 and ALP 150/ULN 120 produced `R=7.2`, `hepatocellular`; boundary and missing-ULN tests passed. | Calculation is deterministic and independently recomputed; rendered final-report result not tested. |
| 17 | Per-drug assessment and cross-drug identity isolation | Clinical pipeline | `NOT TESTED` | Not reached in a live clinical run. | Requires multi-drug clinical execution and persisted evidence review. |
| 18 | LiverTox evidence retrieval and provenance | Structured evidence | `NOT TESTED` | Fresh disposable catalog was empty; preflight reported LiverTox unavailable. | Shared DB has source rows but cannot currently start through the migration. |
| 19 | RxNav evidence retrieval and provenance | Structured evidence | `NOT TESTED` | Fresh disposable catalog was empty; preflight reported RxNav unavailable. | Shared DB has source rows but cannot currently start through the migration. |
| 20 | FDA DILIrank evidence retrieval and provenance | Structured evidence | `NOT TESTED` | Fresh disposable DILIrank view was empty. | Shared DB contains DILIrank records, but populated-runtime validation is blocked. |
| 21 | RAG document listing and vector-store readiness | RAG | `PASS` | RAG view listed supported PDFs; read-only vector-store API reported collection ready, 1,317 embeddings, 99 distinct documents, dimension 384, cosine/IVF_FLAT. | Readiness/listing only; no clinical retrieval claim. |
| 22 | RAG retrieval inside a clinical analysis | RAG / clinical workflow | `NOT TESTED` | No clinical analysis was allowed past preflight. | Requires a completed analysis with persisted citations. |
| 23 | Bibliography and source provenance in the final report | Reporting | `NOT TESTED` | No final report was generated in the current live run. | Must be rechecked with structured sources and RAG enabled. |
| 24 | Final conclusion coherence and uncertainty presentation | Reporting | `NOT TESTED` | No final report was generated in the current live run. | Must be reviewed against the rendered report and persisted result. |
| 25 | Patient timeline generation, rendering, and persistence | Sessions / timeline | `PARTIAL` | The latest in-app Browser run exercised both exact Ollama models twice against identical isolated synthetic inputs. Both 2B attempts saved fallback/invalid_response; both 9B attempts saved LLM timelines with exact evidence, evidence-derived Source labels, correct date precision, and local/Ollama/model provenance. All four history entries survived reload. | The 2B lane still lacks grounded LLM output. Confidence remained unscored when omitted. Console diagnostics were unavailable through the current in-app Browser surface. Controlled timeout, authentication, and rate-limit regressions pass. [Timeline source attribution validation](../../QA/timeline-source-attribution-20260923/report.md); [Timeline model matrix](../../QA/timeline-model-matrix-20260923/report.md) |
| 26 | Agentic revision workflow, trace, artifacts, and persisted version | Revision | `NOT TESTED` | No revision was started in the current live run. | Shared DB contains historical revision rows, but the current app cannot start on it. |
| 27 | Manual report editing and version/history behavior | Revision / UI | `NOT TESTED` | Not reached in the current live run. | Requires a completed report and persisted version transition. |
| 28 | Human-review escalation and `requires_human_review` state | Revision / clinical safety | `NOT TESTED` | Not reached in the current live run. | Unit paths exist; rendered and persisted current-run evidence is absent. |
| 29 | Cancellation, concurrency, and stale-job guards | Background jobs | `NOT TESTED` | Not reached in the current live runtime. | Automated coverage exists; current populated migration remains a prerequisite for release evidence. |
| 30 | Structured-source inspection UI and dependency messaging | Data Inspection | `ATTENTION` | Drug Catalog, LiverTox, and DILIrank pages rendered empty on the disposable DB; RAG was populated; Update All/Update Embeddings were intentionally not invoked. | External/source data is not proven current; controls were not mutated during audit. |
| 31 | Ordered structured-source update and reconciliation jobs | Data Inspection / jobs | `NOT TESTED` | No source update job was started. | Deliberately excluded to avoid mutating source caches during validation. |
| 32 | Health and inspection API boundaries on a fresh runtime | API | `PASS` | `/api/health`, sessions, RxNav, LiverTox, DILIrank, RAG documents/vector-store, model-config, and settings requests returned expected 2xx responses. | Fresh disposable DB only. |
| 33 | Browser smoke, visible error handling, and console diagnostics | UI / QA | `PASS` | DILI Agent, Settings, Clinical Sessions, and Data Inspection rendered; browser error/warn diagnostics were empty; blocking dialogs were visible and actionable. | One viewport and smoke coverage; not a full accessibility audit. |
| 34 | Backend unit and supported model-config gates | Automated QA | `PASS` | Exact-SHA hosted run `35908957190`: backend unit `789 passed`, model-config `40 passed`, migration head and metadata drift passed; Ruff and Pyright passed. | Seven existing backend warnings and one model-config warning; live-provider behavior is not certified. |
| 35 | Frontend test and production build gates | Automated QA | `ATTENTION` | Exact-SHA hosted Windows run `35908957190` passed Angular/Vitest (`24 files, 105 tests`) and production build. The official local launcher reused fingerprint-current output. | A fresh local production build still terminates with `0xC0000005`; the failure is host-specific and unresolved. |
| 36 | Exact OpenCode Go / DeepSeek end-to-end clinical provider lane | External provider | `NOT TESTED` | No live cloud clinical call was made; disposable DB had no active OpenCode key, and the shared runtime was blocked before provider resolution. | Must be validated without fallback before release. |
| 37 | Restart and reuse of existing persisted clinical data | Persistence / release | `ATTENTION` | The populated clone reopened through the launcher database-initialization path after migration, preserving 18 sessions, 45 drug mentions, 228 lab observations, 18 results, and SQLite integrity. | The shared source database was intentionally not advanced; rendered populated-session reuse remains untested. |
| 38 | EXE/MSI packaging, installer, checksum, and publication | Release packaging | `NOT APPLICABLE` | Explicitly outside this source/development audit. | Separate release gate. |
| 39 | Clean-machine install and Windows host smoke | Release packaging | `NOT APPLICABLE` | Explicitly outside this source/development audit. | Separate release gate. |
| 40 | Full browser E2E and persisted clinical-job recovery | Automated QA | `PASS` | Local Chromium suite `41 passed, 5 skipped`; exact-SHA hosted Windows suite `39 passed, 7 skipped`. Persisted-job recovery completes and clears its saved marker after reload. | Hosted skips are the opt-in provider and embedding lanes, unavailable Ollama-model checks, and session/timeline cases without persisted CI sessions; live-provider workflow job is skipped on push. |

## Release blockers and required remediation

### B1 — populated-database migration remediation (resolved)

The current populated database has 38 `clinical_session_versions`, 17 revision runs, and 54 revision artifacts. The revision-run, review, and artifact tables contain foreign keys to `clinical_session_versions`. The migration coordinator now temporarily disables SQLite enforcement inside the existing atomic migration transaction, validates `PRAGMA foreign_key_check` before commit, and restores enforcement afterward. This allows migration `202609170001` to batch-recreate the parent table without weakening post-migration integrity.

The migration fixture now enables `PRAGMA foreign_keys=ON` and exercises revision runs, review rows, and artifacts. Ten migration tests and the full 731-test backend unit suite pass. A clone of the unchanged source database was migrated through the standard launcher initialization path with all populated counts preserved. The shared database remains unchanged by this remediation run, so direct source-runtime session UI reuse is still marked `ATTENTION`.

### B2 — current clinical/provider evidence is incomplete

Because the populated runtime cannot start, the audit could not prove a current rendered clinical report, source citations, timeline, revision, manual edit/history, or exact `opencode_go / deepseek-v4-flash` provider execution. These are `NOT TESTED`, not inferred from historical reports or catalog reachability.

### B3 — test cache permissions need an explicit environment decision

The first unit invocation failed before collection because the historical split cache/database location was not writable. The same suite passed after the test runner was moved to the canonical task-owned `runtimes/cache/pytest` hierarchy. This is classified as an environment/test-runner attention item, not a product pass.

## Validation diary

### 2026-09-18, Europe/Rome, approximately 12:26–12:47

1. Recorded the clean `develop` worktree and source revision `2e434ae1b0e276b8acbab27f65a399d306c3f691`; inspected the project index, runtime settings, and existing QA documentation.
2. Started with the prescribed launcher. It found the configured environments and waited for backend health, but the populated DB stopped at migration `202609170001`. A foreground source invocation reproduced the `sqlite3.IntegrityError` and exited. The shared DB remained at `202609100001` with `PRAGMA integrity_check=ok`.
3. Re-ran the application with a disposable SQLite DB through the source backend and client preview. Startup reached head, catalog seeding completed, health/API probes returned 2xx, and the UI rendered.
4. Exercised Settings persistence, model/access-key guardrails without entering a secret, clinical validation/preflight, empty Clinical Sessions, Data Inspection views, and RAG vector-store inspection. The synthetic case was not submitted to a model. No source update or embedding update job was invoked.
5. Ran the backend unit suite with a writable QA cache (`731 passed`), the supported model-config regression slice (`40 + 10 passed`), the frontend suite (`23 files / 94 tests passed`), and the production client build (successful).
6. Rechecked the browser after reload, verified the settings baseline remained `1`, inspected browser diagnostics (`[]`), stopped the task-owned backend/frontend listeners, and retained only small source-runtime log evidence under `assets/QA/pre-release-e2e-20260918/`. Disposable DB/cache files were removed.

### 2026-09-18, populated-database migration remediation

1. Added FK-safe SQLite migration coordination: enforcement is suspended only around the atomic Alembic transaction, `PRAGMA foreign_key_check` gates commit, and the prior connection state is restored.
2. Enabled foreign keys in the migration test engine and expanded the cancellation fixture to cover revision-run, review, and artifact child relationships.
3. Migrated a clone of the source database through `start_on_windows.ps1 -Action InitializeDatabase`; the clone reached `202609170001` with 18 sessions, 38 versions, 17 revision runs, 54 artifacts, `foreign_key_check=[]`, and `integrity_check=ok`.
4. Confirmed the source database remained at `202609100001` with its original counts and `integrity_check=ok`. The provider and live clinical workflow gates remain untested.

## Evidence register

- `app/server/migrations/versions/202609170001_add_cancelled_revision_status.py:40-45` — parent-table batch alteration now covered by the FK-safe migration transaction.
- `app/tests/unit/test_database_migrations.py:14-22,186-260` — migration fixture and its foreign-key configuration gap.
- `assets/QA/pre-release-e2e-20260918/manual-backend.stderr.log` — disposable source startup and request-runtime log.
- `assets/QA/pre-release-e2e-20260918/manual-backend.stdout.log` — disposable backend process output.
- `assets/QA/pre-release-e2e-20260918/manual-frontend-2.stdout.log` — disposable frontend preview output.
- `assets/QA/release-blocker-remediation-20260917.md` — historical evidence only; it was not used as current release proof.
- `assets/QA/release-blocker-remediation-20260918.md` — current FK-safe migration remediation evidence.

The in-app Browser captures from this run were inspected inline for DILI Agent, Settings, Clinical Sessions, and Data Inspection/RAG. The browser tool did not expose a disk-export path for those captures, so this ledger records the visual assertions and the reproducible routes rather than inventing screenshot filenames.
