# Validation ledger
Last updated: 2026-10-01

## Purpose and authority

This is a sparse, dated evidence register for DILIGENT validation. The canonical
current operational status lives in [`project_status_ledger.md`](../project_status_ledger.md);
this file records the decisions, guarantees, and known limitations behind that
status and links each retained validation slice to its evidence under
`assets/QA/`. It deliberately does not reproduce run-by-run history: the QA
reports remain the detailed evidence, and the architecture, runtime, and coding
documents in this tree are the product ontology that describes how each behavior
works.

`assets/QA/` is the supporting evidence location. Only the reports referenced
here (plus the approved release screenshots) are tracked; disposable logs,
scripts, and intermediate capture batches are not committed.

## How to use this file

1. For the current status of any component, open the component ledger in
   [`project_status_ledger.md`](../project_status_ledger.md).
2. To run the mandatory regression, follow [`qa_regression.md`](qa_regression.md)
   and [`coding/testing_and_quality.md`](../coding/testing_and_quality.md).
3. For a specific validation slice, open the evidence link in the register
   below; the ontology document listed in the same row explains the behavior it
   validated.
4. Add a new slice row here (date, scope, outcome, evidence) whenever a
   meaningful validation completes. Keep the row to a few lines and move
   detailed prose into the QA report.

## Status vocabulary

The older QA reports use run-specific terms (`PASS`, `ATTENTION`, `FAIL`,
`NOT TESTED`, `NOT APPLICABLE`). The project status ledger translates those into
its current taxonomy (`VALIDATED`, `WORKING`, `PARTIAL`, `BROKEN`, `BLOCKED`,
`UNVALIDATED`, `NOT_IMPLEMENTED`, `DEPRECATED`). A `PASS` in an old report never
overrides a newer, narrower, or broader result.

## Validation campaign map

The long-term roadmap is a tiered campaign. Slices are recorded in the register
below as they complete; the campaign itself is the roadmap, not a status
authority.

| Tier | Slices | Focus |
|---|---|---|
| Tier 0 | `V00`–`V02` | Revision/evidence baseline, source startup and migration, automated baseline gates |
| Tier 1 | `V10`–`V13` | Shell/routing, runtime Settings, model configuration/catalog cache, access-key lifecycle |
| Tier 2 | `V20`–`V24` | Clinical input/preflight, exact provider execution, multi-drug correctness, RAG grounding, job lifecycle |
| Tier 3 | `V30`–`V40` | Sessions, editing/media/deletion, Data Inspection, source mutation, RAG re-indexing |
| Tier 4 | `V41`–`V53` | Timelines, agentic revision, local Ollama, PostgreSQL, Tauri, and release lifecycle |
| Tier 5 | `V60`–`V61` | Cross-cutting resilience, accessibility, responsive behavior, and performance-sensitive UI |

## Validation register

| Date | Slice / scope | Outcome and boundary | Evidence (tracked) | Product ontology |
|---|---|---|---|---|
| 2026-09-21 | Tier 0 checkpoint (`V00`–`V02`) | Baseline captured; source-launcher port-ownership defect found and remediated; automated baseline gates green. | [Tier 0 checkpoint](../../QA/tier0-validation-20260921.md); [Launcher/startup/build validation](../../QA/launcher-startup-build-validation-20260921.md) | [Startup](startup.md), [modes](modes.md) |
| 2026-09-21 | Source-launcher port ownership (`ISSUE-006`) | Foreign listeners survive; launch fails with actionable output; owned process-tree cleanup intact. | [Launcher/startup/build validation](../../QA/launcher-startup-build-validation-20260921.md) | [Startup](startup.md), [troubleshooting](troubleshooting.md) |
| 2026-09-22 | Production frontend build host comparison | Local Angular build host fault reproduced; exact-SHA hosted CI build green; host-specific issue, not a repository defect. | [Frontend build validation 2026-09-22](../../QA/frontend-build-validation-2026-09-22/report.md); [2026-09-24 follow-up](../../QA/frontend-build-validation-2026-09-24/report.md) | [Deployment](deployment.md) |
| 2026-09-23 | Automated regression gate revalidation | Migration-drift and dependency-audit findings remediated (`anyio` CVE pin); hosted run green. | [Automated regression revalidation](../../QA/automated-regression-revalidation-20260923/report.md) | [Testing and quality](../coding/testing_and_quality.md) |
| 2026-09-23/24 | Timeline local model matrix and failure attribution | Exact `qwen3.5:2b` fails the timeline structured-output contract and fail-closes to the deterministic fallback; `qwen3.5:9b` produces grounded events. Characterized as a model/task compatibility limitation. | Consolidated into [timeline Browser validation](../../QA/timeline-browser-validation-20260925/report.md) | [DILI pipeline — timeline grounding](../architecture/dili_assessment_pipeline.md) |
| 2026-09-24 | Empty timeline extraction recovery | Populated source with no evidence-backed event now raises non-retryable `invalid_response` and persists a deterministic fallback; `DILIGENT_PYTEST_CACHE_ROOT` isolates pytest state. | [Empty timeline extraction recovery](../../QA/timeline-empty-extraction-recovery-20260924/report.md) | [DILI pipeline — timeline grounding](../architecture/dili_assessment_pipeline.md) |
| 2026-09-25 | Supported timeline/Ollama boundary | Exact 2B fail-closed fallback and exact 9B grounded generation validated with provenance, evidence, date precision, rendering, and reload persistence. | [Timeline Browser validation](../../QA/timeline-browser-validation-20260925/report.md), [capture notes](../../QA/timeline-browser-validation-20260925/browser-capture-notes.md), [API/persistence evidence](../../QA/timeline-browser-validation-20260925/api-persistence-evidence.md) | [DILI pipeline — timeline grounding](../architecture/dili_assessment_pipeline.md), [background jobs](../architecture/background_jobs.md) |
| 2026-09-25 | Automated regression + live-provider dispatch | Local and hosted core gates green; live-provider E2E blocked only by missing repository secret. | [Automated regression and provider validation](../../QA/automated-regression-validation-20260925/report.md) | [Testing and quality](../coding/testing_and_quality.md) |
| 2026-09-26 | Clinical/API resilience | RAG-off, provider-failure, job/error contracts; OpenAPI catalog reconciled to 68 paths; a rendered RAG-off local run completed. | [Clinical/API resilience](../../QA/clinical-api-resilience-20260926/report.md); [Rendered local resilience](../../QA/clinical-resilience-validation-20260926/report.md) | [API surface](../architecture/api_surface.md), [DILI pipeline](../architecture/dili_assessment_pipeline.md) |
| 2026-09-26 | UI shell keyboard + responsive | Keyboard-only tab movement and focus, 1100px guard, rendered 1100/1280/1920px surfaces; focus defect fixed. | [UI shell accessibility](../../QA/ui-accessibility-validation-20260926/report.md) | [UI experience](../ui/experience.md), [UI standards](../ui/ui_standards.md) |
| 2026-09-27 | Revision lifecycle and current-tree | Persisted accepted/cancelled lineage, reload/reopen, audit rendering; `revision.accepted-session-finalization` validated for the exact synthetic path. | [Revision lifecycle](../../QA/revision-lifecycle-validation-20260927/report.md) | [Background jobs — revision agent](../architecture/background_jobs.md) |
| 2026-09-28 | RAG error boundaries + revision recovery | Expanded RAG local boundary (empty/unsupported/malformed/missing, fail-closed zero updates); missing-worker recovery and deterministic retry. | [RAG edge validation](../../QA/rag-edge-validation-20260928/report.md) | [DILI pipeline — RAG readiness](../architecture/dili_assessment_pipeline.md) |
| 2026-09-28 | Revision worker ordering + local Ollama | Job/version/recovery metadata persisted before launch (SQLite-lock race closed); unavailable-provider and timeout fail-closed with sanitized diagnostics. | [Local Ollama revision validation](../../QA/revision-local-ollama-validation-20260928/report.md) | [Background jobs — revision agent](../architecture/background_jobs.md) |
| 2026-09-28 | Runtime settings + current-tree regression | Database-backed Settings/Models save/reload/reset validated; full local regression green. | [Runtime settings and current-tree regression](../../QA/validation-followup-20260928/report.md); [Runtime configuration UI](../../QA/runtime-configuration-ui-20260928/report.md) | [Settings UI](settings_ui.md) |
| 2026-09-28 | Revision startup reconciliation + tool failure | Startup reconciles stale `running` runs; allow-listed tool failure fail-closes without leaking secret-shaped tokens. | [Revision restart/tool-failure validation](../../QA/revision-restart-recovery-validation-20260928/report.md) | [Background jobs — revision agent](../architecture/background_jobs.md) |
| 2026-09-29 | NCBI LiverTox machine access | Official E-utilities → Books-OAI → LitArch chain replaces the interactive HTML path; deterministic tests and metadata preflight passed. | [NCBI LiverTox machine access](../../QA/ncbi-livertox-machine-access-validation-20260929/report.md) | [Configuration — NCBI machine access](configuration.md) |
| 2026-09-29 | Hosted OpenCode Go live revision | Empty-secret run failed closed as infrastructure failure; approved-secret run completed a fresh non-dry revision with exact `opencode_go / deepseek-v4-flash` provenance. | [OpenCode Go live revision validation](../../QA/revision-live-provider-validation-20260929/report.md) | [Generation policy](generation_policy.md), [error handling](../coding/error_handling.md) |
| 2026-09-29 | Live LiverTox + ordered source refresh | 195 MB LitArch archive replaced the catalog, survived restart; ordered RxNav → LiverTox → DILIrank completed; cancellation preserved last-good state. | [Live LiverTox and ordered refresh](../../QA/livertox-ordered-refresh-validation-20260929/report.md) | [Desktop release — structured source updates](desktop_release.md), [background jobs](../architecture/background_jobs.md) |
| 2026-09-29 | Session/revision API local boundaries | Live session detail/version/manual-edit reads, metadata and audit mutations, timeline/revision errors; missing-session report mutation fixed to 404. | [Session and revision API validation](../../QA/api-local-boundaries-20260929/report.md) | [API surface](../architecture/api_surface.md) |
| 2026-09-30 | Final validation closure | Canonical 68-path/85-operation API matrix; RAG duplicate policy (`ISSUE-005`) resolved with raw-byte SHA-256 canonical selection; Full browser harness 53/0/0/0; backend 838 tests. | [Final validation closure](../../QA/final-validation-closure-20260930/report.md) | [QA regression](qa_regression.md), [testing and quality](../coding/testing_and_quality.md) |
| 2026-10-01 | Desktop candidate + packaged UI | Fresh v3.4.0 portable/MSI/checksum artifacts; portable smoke and deep-path replay passed; packaged interactive UI and populated workflow validated by native automation. | [Desktop release evidence](../../QA/desktop-release-validation-20261001/report.md), [host checklist](../../QA/desktop-release-validation-20261001/host-checklist.md), [D01–D14 matrix](../../QA/desktop-release-validation-20260930/D01-D14-matrix.md), [packaged-UI slice](../../QA/desktop-interactive-ui-validation-20261001/report.md), [populated-workflow slice](../../QA/desktop-populated-ui-validation-20261001/report.md) | [Desktop release](desktop_release.md) |

## Key decisions and guarantees

These are the important, validated behaviors that later work must not silently
change. Each is preserved as a product guarantee; the ontology column above and
the evidence register below show where each was established.

- **Timeline provenance and grounding.** Local timeline runs persist exact
  `source_kind=local`, `model_provider=ollama`, and `source_model` values; cloud
  runs persist `opencode_go` and the exact model. Non-empty model evidence is
  required to occur within one normalized source-text field before an event is
  accepted. Events without preserved source evidence are dropped, and fallback
  events keep `timing_type="uncertain"` without inventing dates from the visit
  timestamp. Month tokens (`YYYY-MM`) stay month-precision.
- **Timeline fail-closed behavior.** Empty or invalid structured model output
  maps to `invalid_response` and persists a deterministic, date-precise,
  evidence-backed fallback instead of silently succeeding or inventing facts.
- **Local Ollama limitation.** Exact `qwen3.5:2b` did not produce grounded
  structured timeline output under the tested conditions; this is a documented
  model/task compatibility limitation, not a pending application gate. Do not
  reopen it as a retry target unless the model, prompt, or timeline
  implementation changes.
- **Revision worker ordering.** A revision worker receives a preallocated job
  identifier and starts only after the version and pipeline run persist their
  job/version recovery fields, closing the reproducible SQLite-lock race.
- **Revision startup reconciliation.** After startup, persisted `running`
  revision runs without a live worker are marked failed with the recoverable
  message and their active steps are terminalized; the source report and draft
  shell remain intact for retry.
- **Revision fail-closed safety.** Timeout, tool-failure, and unavailable-
  provider paths fail closed with sanitized diagnostics (no secret-shaped
  content, no raw provider detail beyond provider/model/operation context).
  Deterministic patch validation and blocker-free QA are required before an
  accepted child session can be created.
- **RAG duplicate policy (`ISSUE-005`).** Supported inputs are fingerprinted by
  raw-byte SHA-256; only byte-identical files deduplicate. The canonical source
  is chosen by normalized relative-path ordering and retains the path-derived
  `document_id`; duplicates stay visible in Data Inspection and in citation
  alias metadata. Semantic deduplication is intentionally out of scope.
- **API boundary.** The Full harness derives an explicit contract matrix from
  `/openapi.json` (68 paths, 85 method/path operations); adding a public route
  requires an explicit validation disposition.
- **Settings source of truth.** Operator-editable runtime configuration is
  persisted in the `application_configuration` singleton and edited through
  Settings; `.env` and environment variables remain outside the UI/API.
- **FK-safe migration.** SQLite migration transactions suspend foreign-key
  enforcement only around the atomic parent-table rebuild, run
  `PRAGMA foreign_key_check` before commit, and restore prior enforcement.
- **Access-key lifecycle.** Keys are encrypted at rest, returned as
  metadata-only fingerprints, support one-active-key rotation, are scoped by
  provider, and are redacted from API/UI/log-visible surfaces. Provider-side
  credential validity belongs to the provider components.
- **Cooperative cancellation.** Pending jobs become terminal immediately; a
  running stop-requested worker stays `running` and occupies its concurrency
  scope until it exits. Cancellation or failure before a source's final commit
  preserves the last usable snapshot.

## Known limitations

- `qwen3.5:2b` timeline structured-output incompatibility (see above).
- MSI install/upgrade/uninstall, code signing, and clean-machine distribution
  are pending on a suitable Windows host (the current session is not an
  administrator); the packaged portable lane and packaged interactive UI are
  validated.
- The native RAG folder `IFileDialog` could not be reliably automated; the
  packaged default RAG source folder is used and the dialog remains an
  interaction boundary.
- Narrator/Speech Recap spoken output and broad component-level screen-reader
  certification are optional future enhancement work, not required closure
  scope.
- Containerized runtime is `NOT_IMPLEMENTED` and out of scope.
- Cloud provider availability, latency, and output/QA variance are validated
  only for the exact `opencode_go / deepseek-v4-flash` synthetic route and the
  supported local Ollama boundary.
- Older historical `unknown` timeline runs remain unattributed evidence and are
  not re-opened as product gates.

## Evidence register (component → evidence)

The component ledger in [`project_status_ledger.md`](../project_status_ledger.md)
is the authoritative status. This register maps each component to its primary
tracked evidence so the chain from status to report is short.

| Component | Primary evidence |
|---|---|
| `runtime.startup.source-launcher`, `runtime.database.sqlite-migrations` | [Tier 0](../../QA/tier0-validation-20260921.md); [Launcher/startup/build](../../QA/launcher-startup-build-validation-20260921.md) |
| `api.local-boundaries` | [Final closure](../../QA/final-validation-closure-20260930/report.md); [Session/revision API](../../QA/api-local-boundaries-20260929/report.md); [Clinical/API resilience](../../QA/clinical-api-resilience-20260926/report.md) |
| `ui.application-shell` | [Final closure](../../QA/final-validation-closure-20260930/report.md); [UI shell accessibility](../../QA/ui-accessibility-validation-20260926/report.md) |
| `settings.runtime-model-configuration` | [Runtime settings regression](../../QA/validation-followup-20260928/report.md); [Runtime configuration UI](../../QA/runtime-configuration-ui-20260928/report.md) |
| `auth.access-key-management` | [Final closure](../../QA/final-validation-closure-20260930/report.md); [Clinical/API resilience](../../QA/clinical-api-resilience-20260926/report.md) |
| `model.provider.opencode-go`, `revision.agentic-lifecycle`, `revision.accepted-session-finalization` | [Live provider revision](../../QA/revision-live-provider-validation-20260929/report.md); [Revision lifecycle](../../QA/revision-lifecycle-validation-20260927/report.md); [Revision restart/tool-failure](../../QA/revision-restart-recovery-validation-20260928/report.md); [Local Ollama revision](../../QA/revision-local-ollama-validation-20260928/report.md) |
| `model.provider.local-ollama`, `sessions.timeline` | [Timeline Browser validation](../../QA/timeline-browser-validation-20260925/report.md); [Empty extraction recovery](../../QA/timeline-empty-extraction-recovery-20260924/report.md) |
| `clinical.analysis.pipeline`, `clinical.input-preflight`, `clinical.drug-resolution.guardrails`, `clinical.calculation.safety-audit` | [Clinical/API resilience](../../QA/clinical-api-resilience-20260926/report.md); [Rendered local resilience](../../QA/clinical-resilience-validation-20260926/report.md); [DILI pipeline](../architecture/dili_assessment_pipeline.md) |
| `data.inspection.catalogs`, `data.sources.refresh` | [Live LiverTox + ordered refresh](../../QA/livertox-ordered-refresh-validation-20260929/report.md); [NCBI machine access](../../QA/ncbi-livertox-machine-access-validation-20260929/report.md) |
| `rag.ingestion-retrieval` | [Final closure](../../QA/final-validation-closure-20260930/report.md); [RAG edge validation](../../QA/rag-edge-validation-20260928/report.md) |
| `sessions.crud-persistence` | [Session/revision API](../../QA/api-local-boundaries-20260929/report.md); [Final closure](../../QA/final-validation-closure-20260930/report.md) |
| `test.automated-regression` | [Final closure](../../QA/final-validation-closure-20260930/report.md); [Automated regression 2026-09-23](../../QA/automated-regression-revalidation-20260923/report.md); [Automated regression 2026-09-25](../../QA/automated-regression-validation-20260925/report.md) |
| `release.desktop.v3-4-0` (release readiness) | [Desktop candidate evidence](../../QA/desktop-release-validation-20261001/report.md); [Packaged-UI slice](../../QA/desktop-interactive-ui-validation-20261001/report.md); [Populated-workflow slice](../../QA/desktop-populated-ui-validation-20261001/report.md); [D01–D14 matrix](../../QA/desktop-release-validation-20260930/D01-D14-matrix.md); [Deep-path remediation](../../QA/desktop-release-validation-20260930/deep-path-remediation.md); [Packaged suite boundary](../../QA/desktop-release-validation-20260930/packaged-suite-boundary.md); [Desktop release](desktop_release.md) |

## Maintenance

- Keep this file sparse. A new slice is one table row plus at most one new
  guarantee or limitation line.
- Always update the component ledger and, when behavior changed, the product
  ontology document named in the row, together with a `Last updated:` date.
- Only reference evidence that is actually tracked under `assets/QA/`.
- Detailed narratives belong in the QA report, not here.