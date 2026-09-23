# Local Timeline Model Gate Revalidation — 2026-09-23

## Scope and setup

Revalidated `sessions.timeline` and `model.provider.local-ollama` from clean
`develop` revision `07f7bba7908b202d3cc8683a7c1b6db79c45dd40`, which matched
`origin/develop` at the start. The official Windows launcher used the isolated
SQLite database at
`runtimes/cache/qa/timeline-local-gate-20260923/timeline-local-gate.db`.
`seed_synthetic_sessions.py` created one session for each model with identical
anamnesis, medication, and laboratory text. The shared database, saved runtime
configuration, credentials, and user-started Ollama process were not changed.
The Timeline role was selected first as `qwen3.5:2b`, then as `qwen3.5:9b`,
only in the isolated database.

`start_on_windows.ps1 -Action Launch` started the backend and preview; backend
health and the frontend route returned HTTP 200. The launcher exited with an
access-denied error when it tried to open the default browser. The Codex
embedded Browser panel was only 559 pixels wide and rendered the application's
minimum-width notice (the app requires 1100 pixels). The same user-facing
workflow was therefore inspected through the Codex-controlled Chrome surface
at 1460 pixels wide. No screenshot file is claimed; the Browser screenshot
was inspected inline.

## Live local model results

| Model | Saved timelines | Provenance and evidence | Result |
|---|---|---|---|
| `qwen3.5:2b` | Two saved fallbacks; each retained three events with evidence. | Both rows stored `source_kind=local`, `model_provider=ollama`, and `source_model=qwen3.5:2b`. The fallback parser preserved medication at month precision (`2025-01`) and symptoms and ALT at day precision (`2025-01-17`). | Both saved as `fallback` with `generation_error_code=invalid_response`. Neither counts as model success. |
| `qwen3.5:9b` | Two saved `llm_generated` timelines with three events each. | Both rows stored `source_kind=local`, `model_provider=ollama`, and `source_model=qwen3.5:9b`. The medication evidence mapped to `drugs` at month precision; ALT mapped to `laboratory_analysis` at day precision; symptoms mapped to `anamnesis` at day precision. Every source quote exactly matched its indicated synthetic source field. | Both runs passed evidence, derived Source-label, date-precision, and provenance checks. Confidence and rationale remained unset when omitted by the model. |

### 2B failure classification

The two 2B requests reached the source-evidence validation step after the
structured response had been parsed. The live backend log recorded
`error_type=_UnsupportedTimelineEvidenceError` and the message that timeline
event evidence was not supported by the session source; the persisted safe
error code was `invalid_response`. This is a post-parse evidence-guard
rejection, not a structured-schema parsing failure. The guard failed closed
and the deterministic fallback retained source-backed events. No repository
prompt or extractor defect was identified, so no source implementation change
was made and the evidence guard was not weakened.

### Browser and persistence

The rendered 9B timeline showed all three categories, the January 2025
month-level medication placement, and day-level symptom and laboratory events
on 17 January 2025. Event details displayed exact source evidence, the
evidence-derived Source value, and date precision. The 2B history visibly
identified both runs as fallback chronologies. A browser reload retained all
four history entries; reopening the 9B timeline retained `qwen3.5:9b`, the
three events, and their source labels.

Read-only SQLite inspection found two fallback rows for session 1 and two
`llm_generated` rows for session 2. Every row had three evidence-backed
events, exact model/provider provenance, and date precisions of two day-level
dates and one month-level date. `PRAGMA integrity_check` returned `ok`;
`PRAGMA foreign_key_check`
returned zero violations.

## Automated checks

| Check | Result |
|---|---|
| Focused backend timeline, extraction, retry, diagnostics, and repository tests | `70 passed, 7 deselected`. Existing warnings: pytest reports its configured `cache_dir` option as unknown, and Google GenAI reports a deprecation. |
| Full Angular/Vitest suite | `24` test files and `105` tests passed with `npm run test -- --no-watch`. |
| Ruff | The synthetic seed helper passed `ruff check --no-cache`. No production Python files changed. |
| SQLite | `integrity_check=ok`; `foreign_key_check` returned zero violations. |
| Hosted CI | Post-push status for the resulting commit is recorded in the completion update below when available. |

The local frontend suite pass does not close `test.automated-regression`.
The previously reproduced fresh-build `0xC0000005` and independent hosted-CI
browser E2E, security-scan, and migration-check failures remain separate
unresolved gates.

## Gate status and remaining limits

- `sessions.timeline` remains **PARTIAL**. The exact 2B model still falls back
  twice after its unsupported evidence is rejected; the 9B success is limited
  to this synthetic case.
- `model.provider.local-ollama` remains **PARTIAL** for the same unresolved
  2B behavior and the narrow two-model coverage.
- `test.automated-regression` remains **PARTIAL** pending a fresh local build
  or host crash diagnostics and resolution of the independent hosted-CI
  failures.
- The embedded Browser minimum-width notice prevented full-workflow validation
  in that panel; rendering and behavior were inspected in the Codex-controlled
  full-width Chrome surface.
- The broader OpenCode Go provider and revision matrices, API route coverage,
  NCBI-dependent catalog/refresh work, RAG duplicate-file policy decision,
  accessibility review, access-key lifecycle, and `v3.4.0` release remain
  separate work. NCBI Bookshelf still requires human verification for the
  LiverTox master list. Access-key management and desktop release remain
  **BLOCKED** on their stated credential, authorization, signing, CI, and host
  prerequisites.

No shared database, saved runtime settings, `.env`, or credential was mutated.
After read-only persistence and integrity checks, the isolated QA database and
focused test basetemp were removed. The seed helper and this report are the
retained reproducibility evidence. The task-started backend and frontend
preview were stopped by their verified process IDs; ports 7690 and 9847 were
confirmed free. The user-started Ollama service was left running.
