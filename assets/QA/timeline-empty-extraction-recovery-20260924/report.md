# Empty timeline recovery validation — 2026-09-24

## Selected scope

The current ledger left `sessions.timeline`, `model.provider.local-ollama`, and `test.automated-regression` partial after two exact `qwen3.5:2b` runs persisted empty `llm_generated` timelines. The underlying acceptance gap was actionable and fit one scope: populated clinical source text could produce no evidence-backed events without being classified as an invalid extraction. This run revalidated that behavior, added a fail-closed guard, verified deterministic fallback persistence, and reran the focused regression slice. The related API boundary was exercised as part of the runtime check.

The run started from clean `develop` HEAD `3b0be42d44e047f868edb9049f3d7a9f38fdf295`, equal to `origin/develop` at start. Only synthetic patient content was used. The shared database and settings were not changed.

## Change and regression coverage

`PatientTimelineExtractor` now raises a non-retryable `invalid_response` when canonical clinical fields contain source text but normalization removes every event. This routes unsupported or empty model output through the existing deterministic fallback and prevents it from being saved as a successful empty `llm_generated` timeline. An empty result remains valid when the canonical clinical source fields are blank.

The extractor tests cover both cases. A service-level regression supplies an empty structured model result, verifies that the timeline service persists a `fallback` with `invalid_response`, and checks that the therapy, symptom, and laboratory source facts keep their month/day date precision. The focused two-file suite passed **53 tests** with one existing Google GenAI deprecation warning. Ruff passed on the production code, extractor tests, and pytest configuration. Ruff also passed on the inspection repository tests with the file's pre-existing `DTZ001` naive-datetime findings ignored; `git diff --check` passed. Logs: [combined tests](combined-tests.log), [core Ruff](ruff-core.log), [inspection-test Ruff](ruff-service-test.log), and [diff check](diff-check.log).

The root pytest fixture previously wrote its embedded database and temp fixtures under the protected shared `runtimes/cache/pytest` directory. `app/tests/conftest.py` now accepts `DILIGENT_PYTEST_CACHE_ROOT`; this run pointed it at this report's isolated `pytest-workspace` and kept the normal root fixtures active. Test cache and basetemp paths were kept under this QA directory.

## Exact local runtime check

The isolated settings reported `use_cloud_services=false` and `timeline_model=qwen3.5:2b`. Ollama's installed tag digest was verified as `324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df`; no model was downloaded or substituted. A seeded synthetic session contained three dated source facts: acetaminophen in January 2025, symptom onset on 2025-01-17, and ALT 75 U/L on 2025-01-17.

The app health endpoint returned `ok`; model configuration, session list, OpenAPI route catalog, and the timeline job endpoints were read from this isolated source runtime. Forced regeneration job `36c085d4` completed on `/api/inspection/sessions/1/timeline-jobs` and saved timeline 1. The exact 2B extraction normalized to no evidence-backed events despite populated clinical source fields. The new `_EmptyTimelineExtractionError` fired, and the inspection service persisted the deterministic fallback with `generation_error_code=invalid_response`. The saved record retained `source_kind=local`, `model_provider=ollama`, and `source_model=qwen3.5:2b`. It contains three deterministic source-evidence events: therapy at month precision (`2025-01`) and symptom/laboratory events at day precision (`2025-01-17`). A follow-up GET returned the same persisted record. Read-only SQLite checks returned `integrity_check=ok` and `foreign_key_check=[]`.

Evidence: [verified Ollama tags](ollama-tags.json), [runtime model selection](runtime-model-selection.json), [API route list](runtime-api-paths.json), [health response](health.json), [job start](job-start.json), [completed job poll](job-poll.json), [persisted timeline](timeline-1.json), [timeline list](timelines-session-1.json), [extractor guard runtime log](extractor-guard-runtime.log), [runtime API log](runtime-api-log.log), [Ollama chat request summary](ollama-chat-summary.log), [database initialization log](database-initialization.log), and [synthetic seed log](seed-sessions.log). The synthetic database was also queried read-only before cleanup.

The current live 2B request directly exercised the new guard: the backend log records `_EmptyTimelineExtractionError` for populated source fields, followed by deterministic fallback persistence. Ollama logged two successful `/api/chat` calls (1m41s and 7.05s); response bodies were not retained, so the second call is not attributed to a particular repair or retry path. No cause is inferred for the two earlier empty records or historical `unknown` failures.

## In-app Browser and environment limits

The Codex in-app Browser opened the local app at the Settings route. Both right and bottom Codex panels produced a 672-pixel-wide browser capture while the app displayed its required minimum of 1100 pixels. Browser zoom did not clear the desktop-window gate. The timeline page, rendered fallback, and UI reload therefore could not be checked in the in-app Browser during this run; this report makes no current rendered-E2E claim. The earlier Chrome rendering evidence remains in the [prior repeatability report](../timeline-local-model-repeatability-20260924/report.md) and applies only to the prior build and records.

The official launcher initialization attempt failed while accessing the protected shared UV cache at `runtimes/cache/uv/sdists-v9/.git`. The shared cache was left untouched; the isolated database and backend were initialized through the documented manual source-runtime path. Thus this run is not a launcher acceptance result. The current `gh` authentication check was invalid for the configured account, so no new hosted workflow-dispatch run is claimed.

## Gate disposition

| Gate | Final status | Evidence and remaining limitation |
|---|---|---|
| `api.local-boundaries` | `WORKING` | Health, model config, session listing, OpenAPI, timeline job start/poll, and persisted timeline read succeeded against the isolated runtime. The complete API route catalog remains uncovered. |
| `sessions.timeline` | `PARTIAL` | Empty source-backed extraction now fails closed to a persisted deterministic fallback; exact local 2B API run preserved three dated source facts. The 2B model did not produce a grounded timeline, and the in-app Browser width gate prevented current rendered/reload verification. |
| `model.provider.local-ollama` | `PARTIAL` | Exact installed 2B model and provenance were confirmed; the current extraction produced no evidence-backed events and the guard saved a deterministic fallback. Earlier independent empty 2B runs and the historical 9B control remain as recorded in the linked report. Repeatable grounded 2B output and broader provider recovery cases are still required. |
| `test.automated-regression` | `PARTIAL` | The focused current suite passed 53 tests and lint passed. Hosted CI for this working tree was not run; dispatch-only live-provider E2E still requires valid GitHub authentication, the repository secret, and its workflow inputs. Conditional E2E prerequisites remain. |
| `model.provider.opencode-go` | `PARTIAL` | Provider/retry/QA/revision matrix is independent of this local timeline run. |
| `revision.agentic-lifecycle` | `PARTIAL` | Broader live-model consistency and lifecycle branches remain outside this timeline slice. |
| `data.inspection.catalogs`, `data.sources.refresh` | `PARTIAL` | The prior NCBI LiverTox master-list human-verification challenge still blocks a full ordered refresh. |
| `auth.access-key-management` | `BLOCKED` | Requires an approved disposable credential and explicit cleanup authorization. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged release material, hosted packaging, clean-machine validation, and publication prerequisites remain absent. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | No supported container runtime is present. |

`ISSUE-005` (RAG duplicate-file policy), the full API route catalog, provider failure and RAG-off clinical cases, and the accessibility audit remain separate decisions or validation debt; this run made no claim about them. Statuses above preserve prior evidence and do not imply a fresh validation of the independent gates.

## Next-session handoff

Re-run the timeline UI check in the Codex in-app Browser with an actual viewport at least 1100 pixels wide. Page zoom did not clear the app's desktop-width gate in this run. The previous isolated database was removed, so use a fresh task-owned runtime and synthetic session with the same three dated source facts described above; do not use the shared database or settings.

1. Confirm the in-app Browser reports a supported viewport and the "Widen application window to continue" gate is gone.
2. Select the exact local `ollama / qwen3.5:2b` route, generate the synthetic session timeline, then open **Clinical Sessions → selected session → Timeline**.
3. Inspect the rendered fallback status, error/provenance labels, all three events, source evidence, and month/day precision. Reload the page and verify the same persisted timeline remains visible.
4. Retain a `qwen3.5:9b` control and record its exact route and whether its events are source-grounded. A successful fallback is safe degradation evidence, not grounded model acceptance.
5. Save browser captures and sanitized API/persistence evidence under a new dated `assets/QA/` directory; update the `sessions.timeline` and `model.provider.local-ollama` gates separately.

Rendered fallback and reload evidence will close the current UI-evidence gap only. Keep both gates `PARTIAL` until the timeline acceptance criteria are met, including repeatable grounded output on the exact 2B route for the provider gate. If the in-app Browser cannot expose a supported viewport, record the observed viewport and retain the gates as `PARTIAL`.

## Cleanup

The app and Ollama listeners were task-owned PIDs 6848, 5172, and 3788 on ports 7690, 9847, and 11434. After verifying their executable paths and listeners, all three were stopped and the ports were confirmed clear. The isolated database, runtime cache, pytest scratch directories, and Ollama profile were removed. Raw Ollama logs containing a generated public-key line and temporary browser screenshots were discarded; only the filtered runtime evidence and small API/test records listed above remain.
