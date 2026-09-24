# Isolated local timeline pair revalidation — 2026-09-24

## Scope and starting state

This is the follow-up named by the previous validation-diary entry. It covers
the related `sessions.timeline` and `model.provider.local-ollama` gates using
the exact installed `qwen3.5:2b` and `qwen3.5:9b` Ollama models. The run started
on `develop` at `85cdf3515c3d054a6aee7fc06d64816e02252e4e`, equal to
`origin/develop`, with no pre-existing worktree changes. All clinical content
was synthetic. The official Windows launcher started an isolated runtime with
a disposable SQLite database and data root under
`runtimes/cache/qa/timeline-pair-validation-20260924`; the shared settings and
application database were not used.

The seeded sessions each contained three dated facts: symptoms beginning on
2025-01-17, acetaminophen taken in 2025-01, and ALT 75 U/L on 2025-01-17. The
reproducible fixture is
[`seed_synthetic_sessions.py`](seed_synthetic_sessions.py). The in-app Browser
was used to select the exact local model, generate each timeline, inspect the
rendered history and event detail, and reload the pages.

## Results

| Session | Configured model | Persisted result | Evidence and display |
|---|---|---|---|
| 1 | `ollama / qwen3.5:2b` | Timeline 1, `generation_status=fallback`, `generation_error_code=unknown`, `source_kind=local`, `model_provider=ollama`, `source_model=qwen3.5:2b` | Three `fallback_parser` events persisted with source evidence. The medication retained month precision (`2025-01`); symptom and lab events retained day precision (`2025-01-17`). The UI labeled the record “Fallback chronology.” |
| 2 | `ollama / qwen3.5:9b` | Timeline 2, `generation_status=fallback`, `generation_error_code=unknown`, `source_kind=local`, `model_provider=ollama`, `source_model=qwen3.5:9b` | Three `fallback_parser` events persisted with evidence and the same source date precision. The UI showed a fallback record; no successful LLM-generated timeline was observed. |

Both timeline records and their exact model/provider provenance were returned
by the session timeline API. The patient timeline rendered the event text,
dates, evidence, and `fallback_parser` source labels. Reloading each session
preserved its fallback history and event details. One in-progress view remained
at 25% with “Extracting clinical timeline events” until reload; because the
server-side job exception and terminal job state were not captured, this is
recorded as a progress-display observation for follow-up, not as a confirmed
frontend defect.

The application’s isolated `runtime.parser_llm_timeout` setting was 3,600
seconds, while the timeline service clamps its outer wait to 300 seconds. The
two browser runs were long-running relative to that five-minute service cap.
The persisted `unknown` errors do not identify the originating exception, and
no request-specific server exception log was retained. Therefore this run
cannot establish whether either fallback was caused by a timeout, another
runtime/provider failure, or a model response issue. It is not evidence that
either model failed the timeline’s grounding or schema checks.

Short, non-clinical Ollama API probes provide runtime context only. The cold
`qwen3.5:9b` probe took about 127 seconds, including an approximately 84-second
runner start and slow prompt prefill; a 16-token probe response took about 1.3
seconds. The `qwen3.5:2b` probe took about 96 seconds and its small prompt
prefill took about 18 seconds. These prompts did not exercise the application
timeline schema or evidence guard; they cannot establish timeline quality.
The direct probes and the long application runs make cold-start and local
throughput plausible contributors, but without the original exception this
remains an inference. Do not attribute the fallbacks to model capability.

## In-scope fix and regression

The error classifier previously inferred a timeout from exception text. A bare
`TimeoutError()` has an empty message, so it could be reported as `unknown`.
`_timeline_error_code()` now recognizes the `TimeoutError` type before applying
text heuristics, and the focused diagnostics test covers both empty and
message-bearing timeout instances. The focused suite passed: **9 passed, 1
warning** (the warning is the existing Google GenAI deprecated-alias warning).

This fix addresses a proven classification gap. The exact exceptions behind
the two persisted `unknown` records were not captured, so the report does not
claim that the fix would have changed those specific records.

## Gate status and remaining work

- `sessions.timeline`: **PARTIAL**. Persistence, exact local provenance,
  evidence-backed fallback events, date precision, rendered detail, and reload
  were verified for both models. No model-generated timeline succeeded in this
  pair, and the source exception for either fallback is unknown. Retest after
  capturing the server-side exception and warm/cold model timing separately;
  keep grounded-output and event-level source checks as acceptance criteria.
- `model.provider.local-ollama`: **PARTIAL**. Both exact model routes and their
  provenance were exercised, but this run adds no grounded-model success. The
  earlier 2026-09-23 grounded 9B evidence remains historical and is linked from
  the canonical ledger; it does not convert this pair to a pass.
- `test.automated-regression`: **PARTIAL**. The focused timeline diagnostics
  regression passed. The live-provider workflow still requires explicit
  dispatch and a repository secret; conditional provider/embedding/Ollama and
  persisted-session cases remain outside this run.

Other incomplete gates were reviewed and remain separate because this slice
does not satisfy their prerequisites or exercise their workflows:

- `data.inspection.catalogs` and `data.sources.refresh` remain **PARTIAL** while
  NCBI Bookshelf serves a human-verification challenge for the LiverTox master
  list.
- `model.provider.opencode-go` and `revision.agentic-lifecycle` remain
  **PARTIAL**; their broader provider, QA, and lifecycle variants were not run.
- `auth.access-key-management` remains **BLOCKED** pending an approved
  disposable credential and explicit cleanup authorization.
- `release.desktop.v3-4-0` remains **BLOCKED** pending signed artifacts,
  release authorization, hosted Windows and clean-machine evidence, and the
  final tag/main prerequisites.
- `rag.ingestion-retrieval` remains validated for its tested fixture, with
  duplicate-file policy and unsupported-file/folder behavior still open for a
  product decision and follow-up validation.
- `api.local-boundaries` remains **WORKING** because its complete route
  catalog was not exercised here; `ui.application-shell` remains validated
  without a full accessibility audit.

SQLite `PRAGMA integrity_check` returned `ok`, and `PRAGMA foreign_key_check`
returned no violations. No browser screenshot was exported to disk; the
rendered pages were inspected in the in-app Browser. The launcher’s automatic
browser-opening step was denied by Windows, so the local UI was opened
manually in the in-app Browser after the launcher reported the runtime ready.
The isolated data root and its temporary settings/database were removed after
validation; no shared runtime configuration or data was changed.
