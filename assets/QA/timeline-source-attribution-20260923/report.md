# Timeline source attribution validation — 2026-09-23

## Scope

Validated `sessions.timeline` and `model.provider.local-ollama` from clean
`develop` HEAD `309ad24558b0bafa68cd4c2827581431499b037d`. The run used the
official Windows launcher, an isolated SQLite database, and identical
synthetic anamnesis, medication, and laboratory text for two generations on
each exact local model. The shared database, saved settings, credentials, and
user-started Ollama process were left untouched.

## Implementation

- The timeline prompt asks for the exact canonical source field containing
  each verbatim evidence quote.
- The extractor derives `source` from the source payload after evidence
  validation. Copies of the same field under `sections` count once; quotes
  found under multiple distinct fields and events without evidence resolve to
  no source. A conflicting model-supplied label is ignored.
- The existing fail-closed guard still rejects unsupported nonempty evidence
  as `invalid_response`; the deterministic fallback remains visible as a
  fallback rather than a successful model timeline.
- The existing normalizer continues to omit events without evidence; the
  regression captures the event before normalization and verifies that its
  model-supplied source has already been cleared.
- No public API, model schema, or database schema changed.

## Validation evidence

| Check | Result |
|---|---|
| Focused backend timeline, retry, diagnostics, and repository regressions | 56 passed, 7 deselected. The pytest run emitted the existing unknown `cache_dir` config warning and a Google GenAI deprecation warning. |
| Timeline component specs | 12 passed. |
| Ruff | All changed Python files passed with `--no-cache`. |
| Whitespace | `git diff --check` passed. |
| Official launcher | Backend health returned `ok`; local model selection and timeline generation were exercised through the in-app Browser. |
| Identical synthetic inputs | The isolated sessions for 2B and 9B had identical anamnesis, medication, and laboratory fields. |
| SQLite integrity | `PRAGMA integrity_check` returned `ok`; `PRAGMA foreign_key_check` returned zero violations. |

## Live Ollama runs

| Model | Saved results | Evidence, source, and timing | Gate outcome |
|---|---|---|---|
| `qwen3.5:2b` | Two generations both saved `fallback` with `invalid_response`. Each retained three evidence-backed fallback events. Both rows stored `source_kind=local`, `model_provider=ollama`, and `source_model=qwen3.5:2b`. | The fallback retained month precision for medication (`2025-01`) and day precision for symptoms and ALT (`2025-01-17`). Fallback event sources are `fallback_parser`. | The model still does not produce a grounded structured timeline. |
| `qwen3.5:9b` | Two generations both saved `llm_generated`, three events each. Both rows stored `source_kind=local`, `model_provider=ollama`, and `source_model=qwen3.5:9b`. | Every quote matched its source field: `drugs` — “Acetaminophen was taken in 2025-01.”; `laboratory_analysis` — “ALT was 75 U/L on 2025-01-17.”; `anamnesis` — “Symptoms started on 2025-01-17.” Medication stayed at month precision (`2025-01`); symptoms and ALT stayed at day precision (`2025-01-17`). | All three event-level source labels are present and correct on both runs. Confidence and rationale remain null because the model omitted them; the UI displays “Not scored” and “Not reported.” |

All four generation rows persisted in the isolated database. After a Browser
reload, the history showed both 2B fallbacks and both 9B LLM timelines. The
reloaded 9B timeline retained its model identity and rendered Source values;
the event details showed the exact evidence, source, date precision, and
unscored confidence. The rendered event cards were inspected in the Browser
screenshot inline. The Browser surface did not provide a disk-export path, so
no screenshot artifact is claimed.

The current in-app Browser API exposes accessibility state and screenshots,
but no console-log reader. The DevTools shortcut did not open a console in that
surface. Browser JavaScript console errors and warnings therefore remain
unverified; they are not reported as empty.

## Gate status and remaining limits

- `sessions.timeline` remains **PARTIAL** because both exact `qwen3.5:2b`
  generations still returned `invalid_response` and used the deterministic
  fallback. The 9B grounding, source attribution, date precision, provenance,
  rendering, and reload checks passed twice.
- `model.provider.local-ollama` remains **PARTIAL** for the same unresolved
  2B structured-output failure. The 9B source-label gap is closed for the
  exercised synthetic case. Confidence is intentionally left unscored when
  the model omits it.
- Broader `model.provider.opencode-go` and `revision.agentic-lifecycle` remain
  **PARTIAL**; their provider failure, QA, and revision variants are outside
  this slice.
- `data.inspection.catalogs` and `data.sources.refresh` remain **PARTIAL**.
  The NCBI Bookshelf page still presented a browser-verification challenge,
  preventing the ordered refresh from completing.
- `test.automated-regression` remains **PARTIAL** pending host/build
  diagnostics and a complete green regression run.
- `auth.access-key-management` and `release.desktop.v3-4-0` remain
  **BLOCKED** on credential, signing, authorization, hosted CI, and host
  prerequisites.
- The RAG duplicate-file policy remains unresolved pending a product decision.
- Browser console diagnostics were unavailable through the selected in-app
  Browser surface, as described above.

## Cleanup

The task-started DILIGENT backend and preview processes were stopped by their
verified PIDs, and ports 7690 and 9847 were confirmed free. The user-started
Ollama service remained running. The isolated database and temporary pytest
directory were removed after the read-only persistence and integrity checks.
