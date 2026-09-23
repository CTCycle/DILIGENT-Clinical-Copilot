# Session timeline cancellation validation

Date: 2026-09-23
Branch: `develop`
Starting source: `91404256dce1aa1d538727c80822719699576df4`

## Scope and setup

This report covers the timeline cancellation subgate only. The overall
`sessions.timeline` gate stays `PARTIAL`; dated fallback fidelity, browser
retry, and controlled persistence-failure behavior still need evidence.

The official `start_on_windows.ps1 -Action Launch` rebuilt the frontend and
started the app at `http://127.0.0.1:9847` with its backend at
`http://127.0.0.1:7690`. The run used a disposable SQLite database created
under this QA folder and synthetic session 1, `Synthetic Timeline Cancellation
QA`. The session content was synthetic and did not contain patient data.

Before launching the backend, an environment-gated, temporary QA hook replaced
`PatientTimelineExtractor.extract_timeline` with an async extractor that waits
until its task is cancelled. This let the Browser reach the real timeline-job
DELETE route without contacting the configured provider. The page displayed
the configured Timeline role label `qwen3.5:2b`; no provider/model settings or
credentials were changed. The hook source and synthetic session seeder are
retained here for reproducibility; the temporary startup shim and disposable
database were removed after validation.

## Browser evidence

In the in-app Browser, the Clinical Sessions page selected `Synthetic Timeline
Cancellation QA`, Session 1, then its Timeline section. The observed rendered
states were:

1. After Generate Timeline, the job showed `Generating timeline…`, progress
   `5% Preparing session timeline source`, then `25% Extracting clinical
   timeline events`, status `In progress`, and the visible `Stop timeline
   generation` button.
2. After clicking Stop, the UI showed `Stopping timeline generation…` and the
   progress status `Stopping…`; the Stop control was removed while status
   polling continued.
3. The terminal state showed `Timeline generation cancelled.`, the enabled
   Generate Timeline control, and `No generated timelines yet.` in the
   Generated timelines section.

While the disposable runtime was still active, a read-only SQLite query after
the terminal Browser state found 0 rows in `clinical_session_timelines`. The
launcher-owned process tree was then stopped, and the database and its SQLite
sidecars were deleted. The in-app Browser screenshot was visually inspected
inline, but its capture surface did not provide a disk-export path; this report
records the visible UI assertions instead of inventing a screenshot artifact.

## Automated checks

- Backend: `python -m pytest -p no:cacheprovider app/tests/unit/test_data_inspection_repository.py -k timeline_job` — 2 passed, 12 deselected. The controlled async extractor was interrupted, the job ended `cancelled`, no timeline was persisted, and a subsequent generation succeeded.
- Frontend: focused API and timeline workspace specs — 2 files, 6 tests passed. Coverage includes the typed DELETE call, visible Stop/stopping/terminal behavior, a failed cancellation request with polling retained, duplicate-click protection, and a completion race.
- Ruff passed for the changed backend files using `app/server/pyproject.toml`.
- `git diff --check` passed.
- The official launcher's fresh production frontend build completed and backend `/api/health` returned `{"status":"ok"}`.

The pytest run emitted two existing environment/dependency warnings: the test
configuration's `cache_dir` option is unknown to this installed pytest
configuration, and `google.genai` uses a deprecated typing alias. Neither
affected the passing test result.

## Boundaries and cleanup

This controlled extractor run verifies cancellation plumbing only. It is not
provider inference or fallback evidence and does not close the other timeline
gaps. The repository's current provider/model configuration and credentials
were left unchanged.

The official launcher cleanup action required an interactive console, so it
declined without changing files. I verified the backend and frontend listener
command lines and their launcher-owned process ancestry, stopped only those two
process trees, and confirmed ports 7690 and 9847 were free. The temporary
startup shim, disposable database, SQLite sidecars, and task-generated Python
bytecode were removed.
