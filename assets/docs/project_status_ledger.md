# Project Status Ledger
Last updated: 2026-10-18

## Purpose and authority

This ledger is the canonical current status of project components. Product behavior is defined in the concept documents linked in each row. The [validation ledger](runtime/validation_ledger.md) preserves cross-cutting evidence boundaries; it does not duplicate component statuses or maintain a dated run diary.

The latest published release is v3.4.0. Its tag, downloaded artifact hashes, hosted release gates, unsigned status, single-machine WebView2 waiver, and native acceptance are recorded in the [desktop release document](runtime/desktop_release.md) and its [release evidence report](../QA/release-gates-20261018/report.md).

Update a component status only when evidence supports the stated scope. Keep limitations with the affected concept, link to the narrowest relevant documentation, and remove stale status entries when they no longer affect current work.

## Status taxonomy

| Status | Meaning |
|---|---|
| VALIDATED | Confirmed through meaningful testing for the scope stated in the component row. |
| WORKING | Believed to work, but evidence is limited for the stated scope. |
| PARTIAL | Some expected behavior or validation breadth remains incomplete or degraded. |
| BROKEN | Known not to work correctly for the stated scope. |
| BLOCKED | An external dependency prevents completion or validation. |
| UNVALIDATED | Implementation exists, but evidence is insufficient. |
| NOT_IMPLEMENTED | The capability is absent from the current repository. |
| DEPRECATED | Intentionally retained but scheduled for removal or replacement. |
| PUBLISHED | A release is public and its tag and published bytes have been read back. This does not imply signing or clean-machine acceptance. |

A status applies only to its stated scope. In particular, PUBLISHED is not a general product-quality or distribution-safety claim.

## Component ledger

| Component | Status | Validated scope | Last validated | Concept documentation |
|---|---|---|---|---|
| runtime.startup.source-launcher | VALIDATED | Windows launcher startup, build freshness, health checks, and safe port-conflict handling. | 2026-09-22 | [Startup](runtime/startup.md); [runtime modes](runtime/modes.md) |
| runtime.database.sqlite-migrations | VALIDATED | Populated SQLite migration, Alembic head, and metadata-drift checks. | 2026-09-23 | [Persistence](architecture/persistence.md); [deployment](runtime/deployment.md) |
| api.local-boundaries | VALIDATED | Full harness classifies all 68 OpenAPI paths and 85 operations; external integrations remain separate. | 2026-09-30 | [API surface](architecture/api_surface.md); [testing and quality](coding/testing_and_quality.md) |
| ui.application-shell | VALIDATED | Visual pass: DILI, Settings, Session tabs, populated Timeline, and Inspection in both themes at 1100/1280/1440/1920px; 1099px notice, Save focus/visibility, and modal Escape/focus return checked. Broad screen-reader coverage deferred. | 2026-10-19 | [Experience](ui/experience.md); [UI standards](ui/ui_standards.md); [Component patterns](ui/components_and_patterns.md) |
| settings.runtime-model-configuration | VALIDATED | Database-backed save, reload, reset, provider/model selection, and restart persistence. | 2026-10-18 | [Settings UI](runtime/settings_ui.md); [model setup](user/model_setup.md) |
| auth.access-key-management | VALIDATED | Encrypted persistence, metadata-only responses, activation, rotation, scoping, deletion, and redaction. Provider acceptance is separate. | 2026-09-30 | [Model setup](user/model_setup.md); [testing and quality](coding/testing_and_quality.md) |
| model.provider.opencode-go | VALIDATED for exact hosted route | OpenCode Go / deepseek-v4.1-flash synthetic release flow. Other provider/model combinations are not certified. | 2026-10-18 | [Generation policy](runtime/generation_policy.md); [validation boundaries](runtime/validation_ledger.md) |
| model.provider.local-ollama | VALIDATED for supported timeline boundary | Exact qwen3.5:2b fails closed to a source-backed fallback; qwen3.5:9b has produced grounded events. Broader model coverage is not claimed. | 2026-10-18 | [Runtime modes](runtime/modes.md); [timeline workflow](user/sessions_timeline_and_data.md) |
| clinical.input-preflight | VALIDATED | Required-field and date blockers prevent job/session creation until corrected. | 2026-10-18 | [DILI workflow](user/dili_assessment_workflow.md); [DILI pipeline](architecture/dili_assessment_pipeline.md) |
| clinical.analysis.pipeline | VALIDATED for tested synthetic workflows | Exact hosted provider flow and populated native assessment; broader clinical fixtures and provider failure/retry variation remain outside this scope. | 2026-10-18 | [DILI pipeline](architecture/dili_assessment_pipeline.md); [assessment workflow](user/dili_assessment_workflow.md) |
| clinical.drug-resolution.guardrails | VALIDATED | Deterministic match acceptance and rejection safeguards for tested cases. | 2026-09-17 | [DILI pipeline](architecture/dili_assessment_pipeline.md) |
| clinical.calculation.safety-audit | VALIDATED | R-score, hepatic-pattern, Hy’s Law, RUCAM, and causality presentation safeguards for tested cases. | 2026-09-26 | [DILI pipeline](architecture/dili_assessment_pipeline.md) |
| data.inspection.catalogs | VALIDATED | Populated RxNav, LiverTox, DILIrank, and RAG inspection surfaces. | 2026-10-18 | [API surface](architecture/api_surface.md); [sessions and data](user/sessions_timeline_and_data.md) |
| data.sources.refresh | VALIDATED for ordered refresh and preservation | RxNav → LiverTox → DILIrank refresh and last-good preservation on cancellation; injected live upstream outage is unvalidated. | 2026-09-29 | [Background jobs](architecture/background_jobs.md); [persistence](architecture/persistence.md) |
| rag.ingestion-retrieval | VALIDATED for required byte-identical policy | Ingestion, retrieval, citation, and raw-byte SHA-256 deduplication with visible aliases. Semantic deduplication is not claimed. | 2026-10-18 | [DILI pipeline](architecture/dili_assessment_pipeline.md); [validation boundaries](runtime/validation_ledger.md) |
| sessions.crud-persistence | PARTIAL | Session and revision persistence works for tested flows; manual report edits currently clear saved metadata. See Open issues. | 2026-10-18 | [Persistence](architecture/persistence.md); [sessions and data](user/sessions_timeline_and_data.md) |
| sessions.timeline | VALIDATED for supported synthetic boundary | Source-backed fallback, grounded qwen3.5:9b events, event evidence, and reload persistence. Broader providers and event families are not certified. | 2026-10-18 | [Background jobs](architecture/background_jobs.md); [sessions and data](user/sessions_timeline_and_data.md) |
| revision.agentic-lifecycle | PARTIAL | Exact-provider execution and selected recovery/failure paths are covered; provider/model variance and broader accepted-versus-QA-blocked branches remain incomplete. | 2026-09-29 | [Background jobs](architecture/background_jobs.md); [generation policy](runtime/generation_policy.md) |
| revision.accepted-session-finalization | VALIDATED for tested acceptance flow | Accepted child session, persisted version, and subsequent lineage selection. | 2026-09-27 | [DILI pipeline](architecture/dili_assessment_pipeline.md); [background jobs](architecture/background_jobs.md) |
| test.automated-regression | VALIDATED | Mandatory backend, persistence, frontend, Windows Full browser, provider, and security gates passed for the published candidate. | 2026-10-18 | [Testing and quality](coding/testing_and_quality.md); [release evidence](../QA/release-gates-20261018/report.md) |
| release.desktop.v3-4-0 | PUBLISHED | Tag and downloaded artifact bytes verified; unsigned distribution disclosed; clean-host WebView2 check waived on the single available Windows machine. | 2026-10-18 | [Desktop release](runtime/desktop_release.md); [release evidence](../QA/release-gates-20261018/report.md) |
| runtime.containerized | NOT_IMPLEMENTED | No supported container deployment target. | — | [Runtime modes](runtime/modes.md) |

## Open issues

| Component | Current behavior | User impact and follow-up |
|---|---|---|
| sessions.crud-persistence | The manual report editor sends an empty metadata object. The persistence layer replaces the session metadata with that object. See [client editor](../../app/client/src/app/pages/clinical-sessions/clinical-sessions-page.component.ts) and [repository update](../../app/server/repositories/session_revision_repository.py). | A manual report edit can silently remove saved attachment and session metadata. Re-enter and save metadata after an edit if needed; fix the request/repository contract and revalidate manual-edit persistence. This issue was present in the v3.4.0 release. |

## Deferred validation

Broader provider/model behavior, additional clinical and timeline fixtures, injected live-source outages, semantic document deduplication, and broad component-level screen-reader certification remain outside the validated scopes above. They are not implied by a VALIDATED status for narrower component boundaries. Container deployment remains out of scope unless it becomes a project requirement.
