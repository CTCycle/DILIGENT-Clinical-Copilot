# DILIGENT browser E2E and release-readiness report

Execution window: 2026-09-14 to 2026-09-15, Europe/Rome
Repository: `DILIGENT Clinical Copilot`
Branch: `develop`
Commit: `9c5f13df7245f469dc4831c424339559d8dc3358` (`Integrate DeepSeek V4.1 provider compatibility`)
Application: frontend `127.0.0.1:9847`, backend `127.0.0.1:7690`
Browser: Chrome through the Codex in-app browser tooling, normal desktop viewport
Test data: synthetic only; no real patient identifiers

## Executive verdict

**Release readiness: NOT READY.**

The primary clinical session can be created, executed, persisted, reopened, and independently verified through the browser with the canonical OpenCode Go route for DeepSeek Flash v4.1. The completed result calculated the expected R-score and produced coherent LiverTox-backed per-drug assessments.

That is not sufficient for release. RxNav updates fail with `sqlite3.OperationalError: too many SQL variables`; the combined update stops after RxNav and marks the other sources skipped; RAG is configured to a missing directory; the clinical extractor creates false-positive drug/disease mentions and does not validate the exact amoxicillin-clavulanate RxNav alias; Review cannot reach a final draft with the tested model run; and Timeline falls back after a five-minute provider timeout, losing canonical dates. The primary clinical job took approximately 24.9 minutes.

No product source or configuration files were changed during validation. The only intended workspace addition is this QA report.

## Evidence and methodology

The main workflows were executed through the actual browser UI. The clinical session, model configuration, individual source updates, combined update, Timeline generation, Review cancellation, navigation, and refresh were all performed with visible application controls. Backend endpoints, logs, read-only database queries, and source inspection were used only to corroborate what the UI showed and to identify root causes.

CUA screenshot checkpoints were emitted and visually inspected at these stages:

1. DeepSeek model-role configuration saved and reloaded.
2. Blank and short-input clinical validation errors.
3. Populated DILI Agent form and analysis progress at 23%.
4. Completed Session 5 Preview, including R-score, detected drugs, RUCAM evidence, and lower assessment content.
5. Text Editor and session-list reopen after navigation and refresh.
6. Knowledge Base pre-update RxNav alias, LiverTox excerpt, DILIrank result, and empty RAG view.
7. RxNav, LiverTox, DILIrank, and RAG individual update progress/terminal states.
8. Update All modal configuration, in-progress state, and terminal partial-failure state.
9. Post-update RxNav, LiverTox, DILIrank, and RAG lookup screens.
10. Timeline generation progress and saved fallback detail page.
11. Revision progress, cancellation, and persisted partial agent trace.

Browser console checks on the Data Inspection and Clinical Sessions tabs were empty for `error` and `warn` entries at the final check. The backend log is [DILIGENT_20260914_223727_951483_8092.log](<G:/Projects/Repositories/Active projects/DILIGENT Clinical Copilot/app/resources/logs/DILIGENT_20260914_223727_951483_8092.log>).

## Repository and startup state

| Check | Observation | Status |
|---|---|---|
| Branch/source of truth | `develop`, HEAD and `origin/develop` at `9c5f13d` before the report artifact | PASS |
| Working tree before testing | Clean apart from existing permission warnings while Git enumerated protected historical QA cache directories | PASS |
| Relevant recent changes | DeepSeek V4.1 compatibility, PostgreSQL persistence gate, source-update contracts, LiverTox reconciliation, stale-job recovery, timeline safety, RAG settings, and structured-source orchestration are present in recent history | INFO |
| Intended launcher | `start_on_windows.ps1 -Action Launch` | PASS |
| Backend health | `GET /api/health` returned `{"status":"ok"}` | PASS |
| Frontend load | DILIGENT loaded visibly in Chrome | PASS |
| Startup console | No browser console errors/warnings observed | PASS |

The application was started with the intended launcher workflow. The backend was the repository's `app/server/.venv` Python environment. The backend log shows the expected local resources path and no startup failure.

## Mock DILI case and expected facts

Patient: `Synthetic DILI Flash 41 QA`
Visit date: 2026-09-12

The synthetic case contained:

- 38-year-old adult with fatigue, nausea, pruritus, dark urine, and right-upper-quadrant discomfort beginning around 2026-09-02.
- Stable non-cirrhotic metabolic-associated hepatic steatosis; negative viral hepatitis and autoimmune testing; no shock, biliary obstruction, alcohol misuse, or re-exposure.
- Amoxicillin-clavulanate 875/125 mg twice daily, started 2026-08-20 and stopped 2026-09-04.
- Intermittent acetaminophen 500 mg, started 2026-08-15, last dose 2026-09-03.
- Chronic atorvastatin 20 mg, started 2025-01-01 and continued.
- ALT, AST, ALP, total/direct bilirubin, GGT, INR, and albumin with reference limits.
- Four observations: 2026-08-19 baseline, 2026-09-04 first abnormal panel, 2026-09-07 repeat, and 2026-09-12 improvement.

Expected clinical invariants were defined before entry: amoxicillin-clavulanate should be the strongest candidate; acetaminophen and atorvastatin should remain alternatives/concomitant exposures; the first qualifying panel should be hepatocellular by R-score; and the report should retain uncertainty around competing causes.

## DeepSeek Flash v4.1 configuration and compatibility

Browser path: `Configurations` -> provider/model configuration -> model roles.

Actions performed:

1. Searched the catalog for the DeepSeek Flash v4.1 route.
2. Set Clinical, Text extraction, Revision, and Timeline roles through the visible controls.
3. Saved the configuration and observed the `Configuration saved.` toast.
4. Navigated away, returned, and refreshed the browser; the saved roles remained.
5. Tested the direct DeepSeek provider route through its visible provider selector. It showed no active DeepSeek access key and no available cloud models, so no key was entered and no direct-provider clinical run was attempted.
6. Switched back to the documented OpenCode Go route and ran the clinical workflow.

The successful runtime snapshot recorded by the completed browser-created job was:

| Runtime field | Value |
|---|---|
| Provider | `opencode_go` |
| Provider model | `deepseek-v4-flash` |
| Semantic model | `deepseek-v4.1-flash` |
| Clinical/text/revision/timeline model | `deepseek-v4-flash` |
| Context limit | 1,000,000 input tokens |
| Output limit | 384,000 provider capability; role-specific reserves were applied |
| JSON mode | supported |
| Native JSON schema | not supported; compatibility path used |
| Tools | native tool support recorded |
| Endpoint family | Chat Completions |
| RAG | disabled for the successful run after the UI warning was accepted |

The backend log confirmed requests to `https://opencode.ai/zen/go/v1/chat/completions` with `model=deepseek-v4-flash`. The compatibility descriptor recorded semantic ID `deepseek-v4.1-flash` and provider ID `deepseek-v4-flash`, including the provider-specific unsupported-parameter and tool-call requirements.

**Classification: PARTIAL.** The mapped canonical route works, including structured extraction and clinical synthesis, but the direct DeepSeek provider route is **BLOCKED** by missing credentials/catalog and the end-to-end latency is not release-safe. The tested secondary model duties exposed timeout and structured-parse failures.

## Browser clinical-session workflow

Browser path: `DILI Agent` -> patient/session form -> `Run` -> `Clinical Sessions` -> Session 5.

The complete mock case was entered through the same textarea and visible controls available to a normal user. RAG was left disabled after the UI displayed its readiness warning and the visible `Continue with limitations` action was selected.

Observed result:

- Session `5`, `Synthetic DILI Flash 41 QA`, Version 1 was created through the UI.
- The session list showed `Successful`.
- The primary job completed with progress 100 and session ID 5.
- The UI Preview showed the patient, visit date, anamnesis, current drugs, laboratory timeline, hepatotoxicity pattern, R-score, warnings, RUCAM evidence, per-drug assessments, limitations, references, and bibliography.
- The Text Editor showed the persisted report content.
- Navigation away, return to Clinical Sessions, selection of Session 5, and a browser refresh all restored the saved session.

The job required approximately 1,494.3 seconds (24.9 minutes). A first run using the exact semantic catalog selection stalled at Step 4/15 and was stopped/recovered at the backend. The canonical mapped route completed, but its latency is a serious usability and operational risk.

**Classification: PARTIAL.** Browser creation, execution, result rendering, persistence, and reload passed. The duration, extraction defects, and first-run stall prevent a release PASS.

### R-score independent validation

First qualifying panel entered through the UI: 2026-09-04.

| Input | Value |
|---|---:|
| ALT | 360 U/L |
| ALT ULN | 40 U/L |
| Normalized ALT | 360 / 40 = 9.00 |
| ALP | 150 U/L |
| ALP ULN | 120 U/L |
| Normalized ALP | 150 / 120 = 1.25 |
| Expected R-score | 9.00 / 1.25 = 7.20 |
| Expected classification | Hepatocellular (`R > 5`) |
| DILIGENT display | `7.20`, hepatocellular |

**Classification: PASS.** The value was calculated independently without altering the entered inputs.

### Per-drug assessment and knowledge retrieval

Observed in the persisted Preview:

- Amoxicillin-clavulanate: LiverTox score A, direct local LiverTox match `Amoxicillin-Clavulanate`, latency/dechallenge narrative, and the strongest causal assessment.
- Acetaminophen: local LiverTox match and high-dose LiverTox prior, with the report noting that the entered intermittent 500 mg exposure is dose-discordant for primary causality.
- Atorvastatin: local LiverTox match; long stable exposure and improvement while continuing it argue against primary causality.
- The extraction also produced `Synthetic` and `antibiotic` mentions from prose. `Synthetic` was unresolved and had no validated LiverTox match.
- The UI warned `RxNav alias was not validated for Amoxicillin-clavulanate`, even though the Data Inspection RxNav view contained the exact combination formulation and an alias modal showed RxNorm names including Augmentin and 875/125.
- The result contained an erroneous cirrhosis/history mention despite the entered case explicitly stating non-cirrhotic steatosis.
- Laboratory values appeared duplicated when extracted from both anamnesis and laboratory-analysis prose, and reference-limit prose generated undated `N/A` rows.

**Classification: PARTIAL.** The clinical narrative is coherent and conservatively caveated, but normalization and extraction defects must be fixed before clinical reliance.

## Validation and failure paths

| Scenario | Browser action and observation | Status |
|---|---|---|
| Missing required fields | Clicked Run with blank patient/date/input; UI displayed `Cannot start analysis` and identified missing visit date and clinical input | PASS |
| Too-short/missing sections | Entered patient/date and `ALT 20 U/L.`; UI rejected the run and identified the minimum length and missing anamnesis/drug/laboratory sections | PASS |
| Refresh during a long run | Started the full clinical run, observed Step 4/15 at 23%, clicked visible Stop, then refreshed; the in-memory form/progress was lost while the server job remained active | FAIL/PARTIAL |
| Unknown/unmatched prose | The real clinical case produced unmatched `Synthetic` and `antibiotic` mentions; the UI surfaced warnings rather than silently treating them as validated drugs | PARTIAL |
| Missing knowledge result | RAG had no documents; the UI warning and backend failure were visible | FAIL/BLOCKED |
| Failed source update | RxNav individual update visibly progressed and then showed the SQLite error; controls became usable again | FAIL |
| Partial combined update | Update All visibly showed source states and terminal failure/skips | PARTIAL |
| DeepSeek request failure/timeout | Timeline visibly saved a fallback after a five-minute provider timeout; Revision returned a structured-parse failure after retries during the canceled run | PARTIAL/FAIL |
| Reopen completed session | Navigation and refresh restored Session 5 and its report | PASS |
| Reopen partially completed Review | Revision draft artifacts and five completed agent tasks were visible after cancellation, but no final draft or approval controls were available | PARTIAL |

No production or real-patient data was used. No session was deleted.

## Review workflow

Browser path: `Clinical Sessions` -> Session 5 -> `Revision`.

The Revision page visibly showed the configured `deepseek-v4-flash` role and the instruction field. `Start revision` was clicked through the UI. The job stayed at 0% from the user-visible perspective while the backend completed five agent tasks and persisted context, plan, and tool-trace artifacts. The visible `Cancel` control was used after the extended stall.

Terminal corroboration:

- Revision pipeline run status became `cancelled` with `Revision was cancelled.`
- Five agent tasks were persisted as completed.
- No `revision_agent_draft_report` or QA artifact was persisted.
- The backend log later recorded `Structured parse failed after retries: schema=RevisionDraftResult`.
- The outer job status remained `running` for several minutes after the pipeline run was already cancelled, then became `cancelled`. This is a stale-status/recovery defect.
- No Approve/finalize action was clicked. Final clinical approval is a high-stakes action and was not authorized at the action point; the tested run did not expose a final approval control anyway.

**Classification: PARTIAL/FAIL.** Start, progress/cancellation, persistence of partial artifacts, and recovery messaging were exercised. A complete review-to-draft-to-human-approval flow was not achieved.

## Timeline workflow

Browser path: `Clinical Sessions` -> Session 5 -> `Timeline` -> `Generate Timeline` -> `Open timeline`.

The UI showed progress at 25% and then saved `Timeline #4` with the configured `deepseek-v4-flash` model, `opencode_go` provider, three events, and three evidence items. Opening it showed:

> Cloud timeline extraction using opencode-go / deepseek-v4-flash did not complete. The provider request failed unexpectedly. Check backend logs and retry.

Backend evidence shows the request was sent with a 667,533-character structured prompt and timed out after five minutes. The deterministic fallback produced three `Date not reported` events, including therapy context, symptom context, and an ALT reference-limit event, despite actual dated therapy and laboratory observations being present in the session.

**Classification: PARTIAL.** The UI safely saved and displayed a fallback with provenance, but the requested DeepSeek timeline extraction failed and date fidelity was materially degraded.

## Knowledge sources before and after updates

### RxNav / Drug Catalog

Browser evidence before and after the update showed amoxicillin formulations, including `amoxicillin clavulanate oral`, with local date `2026-09-02`. The alias inspection showed canonical/derived normalization and RxNorm names including Augmentin and 875/125.

Individual UI update:

- Settings: timeout 12 seconds, concurrency 10.
- Progress visibly reached 21,202 upserts / approximately 50.5%.
- Terminal error: `(sqlite3.OperationalError) too many SQL variables`.
- The UI re-enabled the Start control and showed the error.

Combined Update All produced the same SQLite failure at approximately 21,000 records. A read-only post-update database check found 7,028 drug rows versus 7,027 in the pre-test snapshot; the new `Enlicitide` row had no RxNav update marker. This indicates partial persistence around a failed stream and must be investigated as an atomicity defect, not reported as a clean rollback.

The official RxNorm documentation identified the current upstream release as 08-Sep-2026 during this run. The local UI marker remained 02-Sep-2026, so the local catalog was stale by approximately six days.

**Individual update: FAIL. Freshness: STALE. Post-update lookup: PASS against existing/local data only.**

### LiverTox

Browser evidence showed `Amoxicillin` and `Amoxicillin-Clavulanate` with historical monograph dates of 2020-10-20 and a pre-update excerpt describing amoxicillin-clavulanate as a common cause of clinically apparent DILI, with variable latency and cholestatic/mixed/hepatocellular presentations.

Individual UI update:

- Settings: four monograph workers, local `livertox_NBK547852.tar.gz`, reuse local archive.
- Visible download and parse milestones completed.
- Backend job completed at 100% with 1,874 processed entries.
- Archive size: 199,160,432 bytes; upstream metadata: 09-Sep-2026; local download timestamp: 14-Sep-2026.
- Database count after update: 1,593 monographs; maximum monograph date remained 2026-01-30.
- The UI terminal message remained `Update started` after backend completion, which is misleading.

The NCBI LiverTox page reported `Last updated 09 Sep 2026` during the run. The archive was refreshed, but the UI date column is a monograph publication/update field rather than the source-fetch timestamp.

**Backend operation: PASS. UI terminal-state messaging and freshness presentation: PARTIAL.**

### DILIrank

Browser evidence before and after update showed:

- `Amoxicillin` -> `LT00507` -> `vLess-DILI-concern`, severity 5, `Adverse reactions`.
- `clavulanate` returned no row, leaving a combination-coverage gap relevant to the clinical case.

Individual UI update:

- Reuse-when-unchanged FDA source handling was inspected.
- Progress reached completion and the visible modal showed 100% / `Completed`.
- Backend summary: 1,336 source records, 1,336 persisted, 737 linked, 283 unmatched, and 316 ambiguous.
- Source last-modified metadata remained `29-Aug-2025` and no fresh download was required by the selected reuse policy.

The FDA DILIrank 2.0 source page reports 1,336 drugs in the four published concern classes, matching the local record count.

**Classification: PASS for the tested individual update and lookup; PARTIAL for clinical combination coverage and freshness.**

### RAG

Browser path: `Data Inspection` -> `RAG`.

The UI displayed the configured documents path but showed `No RAG documents found`. The configured `app/resources/sources/documents` directory does not exist in the workspace and the vectors directory contains only `.gitkeep`.

Individual `Update Embeddings` was started through the visible modal using the displayed chunk/overlap/batch settings. The UI and backend job failed immediately at 4% with:

`RAG documents_path does not exist or is not a directory.`

The combined Update All modal correctly stated that RAG updates remain separate and did not include a RAG tab.

**Classification: FAIL/BLOCKED by missing configured data.** The failure is correctly surfaced by the backend, but the UI retains an `Update started` style state and does not provide a healthy knowledge base.

## Combined Update All

Browser path: `Data Inspection` -> `Update All Sources`.

The modal visibly contained LiverTox, RxNav, and DILIrank tabs and their existing configuration controls. RAG was explicitly excluded, consistent with the intended independent RAG design.

Observed sequence:

1. UI showed LiverTox Queued, RxNav Running, DILIrank Queued.
2. RxNav progressed through visible upsert milestones.
3. The aggregate text incorrectly stated `0 of 3 source updates are running or queued` while one source was running and two were queued.
4. RxNav failed with `too many SQL variables`.
5. The terminal UI showed all three sources as Failed, with LiverTox and DILIrank explicitly marked `Skipped because an earlier source update failed.`
6. The terminal aggregate stated `0 of 3 source updates completed. One or more sources failed.`

The orchestrator did not silently claim success and did preserve the failure/skip distinction at terminal state. It nevertheless did not complete the intended combined update.

**Classification: PARTIAL for failure propagation; FAIL for a successful aggregate update.**

## Post-update clinical verification

The browser was returned to Data Inspection after the update jobs. RxNav search still returned amoxicillin-clavulanate formulations; LiverTox still returned the amoxicillin and combination monographs; DILIrank still returned the amoxicillin row; and RAG still showed no documents.

These are valid UI lookups against the post-job state, but they do not prove that RxNav was refreshed. A second full clinical run was not started because the source update had failed, the first run already required approximately 25 minutes, and the tested model route had active timeout/parse defects.

**Classification: PARTIAL.** Existing/local retrieval works after the failed update; fresh post-update clinical retrieval and cache invalidation remain unproven.

## Browser UX and visual findings

- Normal desktop layout was usable. Source dialogs were vertically scrollable and controls were reachable.
- Progress bars and source cards were visible during long-running operations.
- The Update All aggregate count was wrong during execution.
- Several individual update dialogs retained the text `Update started` after backend completion or failure instead of a clear terminal label.
- The RAG page presented a configured path with no documents and allowed an update attempt that could only fail.
- Refresh during an active clinical run reset the visible form and progress instead of restoring the running job or preserving unsaved input.
- Timeline fallback displayed a safe provenance warning but exposed all three events as undated, which makes the resulting chronology clinically weak.
- Review eventually returned to an idle-looking page with partial persisted artifacts but no clear terminal cancellation/error summary in the visible panel.
- Browser console error/warning checks were empty; the main problems were application behavior and backend/provider state, not uncaught browser exceptions.

## Root causes and affected modules

| Finding | Evidence-backed likely root cause | Affected area |
|---|---|---|
| RxNav update fails on large batch | Bulk SQLite statement exceeds variable limit; streaming path persists batches through `upsert_drugs_catalog_records` and commits incrementally | `app/server/services/updater/rxnav_builder.py`, `app/server/repositories/drug_catalog_repository.py` |
| Failed RxNav run leaves partial state | Batch persistence commits before the later failing batch; `replace_rxnav_catalog_records` exists but the tested stream path uses incremental persistence | Same RxNav updater/repository path |
| Amoxicillin-clavulanate alias not validated | Drug matcher logs silent misses when alias candidates exist but no match resolves; clinical preparation emits the warning | `app/server/services/clinical/drug_matcher.py`, `app/server/services/clinical/preparation.py` |
| RAG update fails | Configured documents directory is absent and the updater rejects it | `app/server/services/updater/embeddings.py` |
| Timeline provider timeout becomes weak fallback | Cloud request waits five minutes, then deterministic fallback retains source order but does not recover canonical dates | `app/server/services/clinical/timeline.py`, `app/server/services/inspection/timeline.py` |
| Revision cannot produce final draft | Provider response/structured parse retries fail for `RevisionDraftResult`; cancellation status propagates late to the outer job wrapper | `app/server/services/inspection/revision_agent.py`, revision job/status orchestration |
| DeepSeek mapping | Semantic `deepseek-v4.1-flash` is mapped to provider `deepseek-v4-flash` in the routed gateway | `app/server/services/llm/transports/routed_gateway.py` |

## Supplemental automated checks

The following targeted unit-test slice was run from the repository server environment with cache provider disabled:

`test_deepseek_v41_compatibility.py`, `test_dili_clinical_correctness.py`, `test_dilirank_knowledge.py`, `test_structured_source_update_jobs.py`, `test_rag_bibliography_contract.py`, `test_rag_readiness.py`, `test_rag_reference_provenance.py`, `test_external_data_timeouts.py`, and `test_llm_failure_safety.py`.

Result: **50 passed, 3 warnings in 5.18 seconds.** These tests corroborate contracts but do not upgrade any browser result to PASS.

## Release-readiness summary

| Required area | Result | Evidence summary |
|---|---|---|
| 1. DeepSeek Flash v4.1 compatibility | PARTIAL | Canonical OpenCode Go mapping works; direct DeepSeek route blocked; latency/secondary failures remain |
| 2. Browser clinical session E2E | PARTIAL | Session 5 created and completed through UI, but 24.9-minute runtime and extraction defects |
| 3. R-score correctness | PASS | Independent 7.20 matched UI 7.20; hepatocellular |
| 4. Per-drug assessment | PARTIAL | Coherent LiverTox-backed assessments, but alias and false-positive extraction defects |
| 5. Review workflow | PARTIAL/FAIL | Revision started and partial artifacts persisted; no final draft or approval controls |
| 6. Timeline workflow | PARTIAL | Saved safe fallback; DeepSeek request timed out and dates were lost |
| 7. Persistence/reload | PASS for completed session; FAIL for active unsaved run | Completed Session 5 survived navigation and refresh; active form/progress did not |
| 8. LiverTox status | PARTIAL | Archive refreshed and backend completed; UI terminal message stale; monograph dates remain historical |
| 9. RxNav status | FAIL | Local catalog stale; individual and combined updates fail with SQLite variable-limit error |
| 10. DILIrank status | PASS/PARTIAL | 1,336 records and UI update succeeded; 283 unmatched/316 ambiguous and clavulanate gap |
| 11. RAG status | FAIL/BLOCKED | No documents; configured directory missing; embeddings update fails |
| 12. Knowledge-base freshness | PARTIAL/STALE | RxNav local 02-Sep vs upstream 08-Sep; LiverTox archive 09-Sep upstream; DILIrank source metadata 29-Aug-2025; RAG unavailable |
| 13. Individual update functions | PARTIAL | DILIrank PASS; LiverTox backend PASS/UI PARTIAL; RxNav FAIL; RAG FAIL/BLOCKED |
| 14. Combined Update All | PARTIAL/FAIL | Correct source scope and terminal failure propagation; no successful aggregate update and progress-count defect |
| 15. Post-update retrieval | PARTIAL | Existing/local lookups work; fresh RxNav and second full clinical rerun unproven |
| 16. Browser/UX defects | PARTIAL | No console errors; stale statuses, wrong aggregate count, refresh loss, weak fallback chronology |
| 17. Errors/regressions | FAIL | SQLite variable-limit failure, provider timeout, structured parse failure, silent alias misses, false-positive extraction |
| 18. Release blockers | FAIL | RxNav update integrity, RAG availability, long model latency, Timeline/Review failures, clinical extraction correctness |
| 19. Recommended fixes | Required before release | See ordered list below |

## Recommended fixes, ordered by severity

### P0 — block release

1. Make RxNav snapshot replacement truly atomic for the actual streaming path, or use bounded SQLite batches under the variable limit and commit only after the complete validated snapshot is ready. Add a post-failure invariant proving no partial catalog state remains.
2. Fix drug normalization/alias resolution for amoxicillin-clavulanate and add a regression that starts with the exact clinical spelling used in the case and reaches the canonical combination record.
3. Prevent false-positive clinical entities such as `Synthetic` and cirrhosis when the source text explicitly says non-cirrhotic. Deduplicate laboratory extraction across sections and keep reference limits out of the observation timeline.
4. Provide a healthy, versioned RAG documents directory or disable the RAG path until configured data exists. Do not expose a ready-looking update control for an unavailable store.
5. Establish bounded provider timeouts and cancellation that actually aborts in-flight requests. Do not allow a clinical operation to occupy the UI for approximately 25 minutes without recoverable progress.

### P1 — release-critical

6. Fix Revision structured-output handling for `RevisionDraftResult`, persist a clear terminal error, and ensure the outer job status changes immediately when the pipeline is cancelled or fails.
7. Preserve canonical dates in Timeline fallback; if the provider fails, the fallback must still use the structured session dates and must clearly label the result as deterministic.
8. Make Update All aggregate progress count running/queued/completed sources correctly and preserve a clear partial-failure state without marking skipped sources as generic failures.
9. Replace stale `Update started` text with explicit `Running`, `Completed`, `Failed`, or `Cancelled` terminal states and re-enable controls only after state reconciliation.
10. Restore or persist active clinical job state across refresh/navigation, or clearly warn that unsaved input and progress cannot be recovered.

### P2 — quality and operations

11. Add user-visible upstream version/last-modified metadata and freshness classification for RxNav, LiverTox, DILIrank, and RAG.
12. Add post-update smoke lookups and cache/index invalidation checks to each source update completion path.
13. Add browser E2E coverage for the exact DeepSeek mapping, source-update failure propagation, refresh during active analysis, Timeline timeout fallback, and Review cancellation.
14. Keep direct-provider configuration visibly blocked until a valid access key/catalog is available; do not silently substitute another provider without retaining the diagnostic.

## External source references used for freshness corroboration

- [NLM RxNorm Files](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html)
- [RxNorm API version documentation](https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getRxNormVersion.html)
- [NCBI LiverTox](https://www.ncbi.nlm.nih.gov/books/NBK547852/)
- [FDA DILIrank 2.0 dataset](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset)

## Final gate statement

The tested application demonstrates a browser-completable primary DILI session on `develop` using the canonical DeepSeek Flash v4.1 compatibility route, but the evidence does not support clinical release readiness. The release gate remains closed until the P0 items are fixed and revalidated with the same browser-first methodology.
