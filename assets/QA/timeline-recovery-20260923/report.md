# Timeline fallback date fidelity and recovery

Date: 2026-09-23
Branch: `develop`
Starting HEAD: `7ceacb5919e35a7a018e985be7c60fb65b4b7afd`
Result: controlled local recovery slice passed; `sessions.timeline` remains `PARTIAL`.

## Scope and safety

The official `start_on_windows.ps1 -Action Launch` used a disposable SQLite
database at `runtimes/cache/qa/timeline-recovery-20260923/timeline.db` and
synthetic session 1. The visible Timeline page showed the configured role
model `qwen3.5:2b` with a local label. Provider settings, credentials, and the
shared application database were left untouched.

The fault injector in `sitecustomize.py` was enabled only for the test run and
only in the backend process. The official launcher clears `PYTHONPATH` during
runtime setup, so a temporary QA-only launcher line passed the injector path
to the backend child after setup; that line was reverted after the run. The
injector intercepted the extractor before any configured provider call:

1. Extractor call 1 raised a controlled `network_unavailable` error, causing
   the normal fallback path to save timeline #1.
2. Extractor call 2 returned a deterministic synthetic event; save call 2
   raised a controlled persistence exception.
3. Extractor call 3 returned a deterministic synthetic event; the normal
   repository save succeeded as timeline #2.

The controlled event log recorded the hook install, provider-style failure,
first successful save, second extraction and failed save, then third
extraction and successful save. No live model request was sent. The app's
`LLM generated` status on timeline #2 describes the application success path;
it is not evidence of live inference.

## Browser evidence

The in-app Browser opened Clinical Sessions, selected synthetic session 1,
and used its Timeline tab.

| Step | Visible result |
|---|---|
| Controlled provider-style failure | Timeline #1 appeared as **Fallback chronology**, with the failure class **Provider network unavailable** and 3 events. |
| Saved fallback dates | Therapy text `Acetaminophen was taken in 2025-01.` appeared as an explicit month-level event in January 2025. Symptom text and ALT text each retained `2025-01-17` as an explicit day-level event. Source evidence remained visible for each. |
| Retry after fallback | Generate Timeline was enabled after the fallback had saved. |
| Controlled persistence failure | The second action showed `Controlled QA timeline persistence failure.` Generate Timeline was enabled again and the list still showed only timeline #1. |
| Successful retry | The third action showed `Timeline generated and saved.` and added timeline #2. Reopening it showed the synthetic controlled event at 17 Jan 2025. |

The Browser screenshot was inspected inline, including the rendered timeline
event. This Browser surface did not provide a disk-export path; this report
preserves the observed text/state and does not claim a screenshot artifact.

## Storage evidence

A read-only SQLite query after the sequence found exactly two rows for session
1: timeline #1 with `generation_status=fallback` and timeline #2 with
`generation_status=llm_generated`. The fallback payload retained month/day
precision and explicit certainty. The retry payload was marked with source
`qa_controlled_extractor`. No row was created for the failed persistence
attempt. The backend SQLite-trigger test separately verified no history row
after an initial failed save, then verified a successful later retry.

## Automated checks

- Backend: `app/tests/unit/test_data_inspection_repository.py -k "timeline or fallback_date_extraction"` — 21 passed, 7 deselected. Covers ISO day/month/year preservation, duplicate identical dates, missing/relative/invalid/ambiguous inputs staying undated and uncertain, no visit-timestamp inference, and persistence rollback/retry.
- Frontend: full Angular/Vitest suite — 24 files and 102 tests passed. Includes Generate being available after fallback and persistence failure and a subsequent successful retry.
- Ruff passed for all changed Python files, including `app/server/domain/timeline_dates.py`, `app/server/services/inspection/timeline.py`, the backend test, and both QA Python scripts.
- Official launcher completed its backend health readiness check and reused the current frontend build.
- `git diff --check` passed on the staged final diff.

Pytest emitted the existing unknown `cache_dir` config warning and a
`google.genai` deprecation warning; neither affected the passing tests.

## Gate boundary and cleanup

This validates deterministic fallback handling and local application
recovery under injected faults. It does not establish the behavior of the live
provider/model, actual provider timeouts, authentication failures, rate limits,
or upstream errors. The overall timeline gate remains `PARTIAL` until the
configured provider lane is exercised safely.

The launcher-owned backend/frontend process trees were identity-checked and
stopped, ports 7690 and 9847 were confirmed free, and the disposable database,
hook log, and task-owned pytest basetemp were removed.
