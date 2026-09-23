# Configured Ollama timeline validation — 2026-09-23

**Result: `PARTIAL`.** The configured local lane was reachable and two live
timeline requests were completed. Both post-fix requests ended as
`invalid_response` and were saved as labeled deterministic fallbacks. The
Browser and database surfaces expose the error category, but not a more
detailed cause for those post-fix responses. The exact model lane, fallback
content, date precision, provider provenance, and persisted history were
verified. A source-grounded
`llm_generated` result and timeout, authentication, and rate-limit behavior
remain unverified.

## Lane and disposable setup

The run started from clean `develop` HEAD
`7c75e51b9910a0db85ea87f225b4d542e2a76518`. The official
`start_on_windows.ps1 -Action Launch` started the backend and frontend with
process-only overrides for `EMBEDDED_DATABASE=true` and
`DATABASE_SQLITE_PATH=runtimes/cache/qa/timeline-live-revalidation-20260923/timeline.db`.
The shared database, `settings/.env`, runtime model settings, and credentials
were not changed.

The disposable session used only synthetic source fields:

- Report: `Synthetic report for timeline recovery validation.`
- Anamnesis: `Symptoms began on 2025-01-17.`
- Medication: `Acetaminophen was taken in 2025-01.`
- Laboratory: `ALT was 75 U/L on 2025-01-17.`

Before inference, Settings showed Runtime Source `Local (Ollama)` and Timeline
Model `qwen3.5:2b`. The Timeline panel also showed `qwen3.5:2b`, and the loaded
Ollama catalog marked that model `Installed`. This matched the configured lane;
no provider or model fallback was used.

## Pre-fix defect reproduction

In the first isolated run, each of two requests was labeled `llm_generated`
and returned the same unsupported event: `Onset of Hepatitis B infection`,
dated `2022-05-18`, attributed to `Acute Care Facility Records`, with evidence
claiming an admission for abdominal pain and HBV acquisition. None of those
claims appeared in the synthetic source. Both rows persisted with
`source_kind=local` and `source_model=qwen3.5:2b`, but incorrectly recorded
`model_provider=openai`.

## Guard and focused regression

`PatientTimelineExtractor` now checks every nonempty model
`source_evidence` against one source text field after case folding and
whitespace normalization. Any unsupported quote raises the existing
`invalid_response` category before the result can be saved as `llm_generated`,
so the inspection service uses its existing deterministic fallback. The
timeline prompt now asks for a verbatim source quote. Timeline provenance now
records the effective `ollama` provider on both success and fallback when the
run is local. No public API or type contract changed.

Focused backend regressions cover unsupported evidence rejection, normal
grounded extraction, deterministic fallback handling, and local/cloud model
provenance. The post-fix live requests verify the same-lane invalid-response
fallback boundary; the Browser/database do not expose a more specific cause.

## Post-fix Browser run

The in-app Browser completed Generate once and Regenerate once on the same
`Local (Ollama)` / `qwen3.5:2b` lane. Both saved timelines were visibly labeled
`Fallback chronology`; the detail page reported `Invalid structured provider
response` and the local-model fallback message. The two database payloads had
`generation_error_code=invalid_response`, `source_kind=local`,
`source_model=qwen3.5:2b`, and `model_provider=ollama`.

Each fallback contained only source-backed events with the source's date
precision:

| Event | Stored date | Precision | Source evidence |
|---|---|---|---|
| Therapy context | `2025-01` | month | `Acetaminophen was taken in 2025-01.` |
| Clinical symptom context | `2025-01-17` | day | `Symptoms began on 2025-01-17.` |
| ALT finding | `2025-01-17` | day | `ALT was 75 U/L on 2025-01-17.` |

There was no hepatitis or 2022 event in either post-fix timeline. After
navigating away, returning to Timeline, and reloading the page, the Browser
still displayed both saved entries (`Timeline #2` and `Timeline #1`), each with
three events and three evidence quotes. The detail view's rendered fallback
state and chronology were visually inspected in the Browser. The Browser
surface provided no disk-export path for its screenshot, so this report keeps
the rendered observations and does not claim a saved screenshot artifact.

## Read-only database evidence

The post-fix database was opened with SQLite `mode=ro`. It contained exactly
two timeline rows for synthetic session 1, both `fallback`; each had the
`invalid_response` error code, exact `qwen3.5:2b` model, `local` source kind,
and corrected `ollama` provider. The three events above and their evidence
strings matched the source and preserved month/day precision. The pre-fix
database independently showed two duplicate fabricated `llm_generated`
records with the incorrect `openai` provider provenance.

## Gate boundary and cleanup

`sessions.timeline` remains `PARTIAL`: this run establishes exact local model
contact, fail-closed handling of unsupported output, fallback date precision,
provenance, and history persistence. It did not produce an acceptable
`llm_generated` timeline and did not test timeout, authentication, or
rate-limit cases. No failures were injected. The separate source-refresh
CAPTCHA, automated-regression host limitation, and desktop release gates were
not changed by this run.

Both disposable SQLite directories and the focused test cache are temporary
under `runtimes/cache/qa/`. The launcher cleanup action misclassified its own
listeners as foreign; exact process paths, commands, and listener PIDs were
verified before stopping only the task-started DILIGENT backend/frontend
processes. Ports 7690 and 9847 were verified free. The user-started Ollama
service was left running.
