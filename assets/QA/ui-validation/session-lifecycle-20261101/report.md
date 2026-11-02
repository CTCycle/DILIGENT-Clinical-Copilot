# Session lifecycle validation (2026-11-01)

Real-browser validation of the persisted-session workflow on an isolated stack
(`runtimes/cache/qa-lifecycle`: seeded SQLite data, backend `:7691`, built SPA
preview `:9848`, `app/tests/ci/fake_ollama.py` on `:11436` with
`--chat-delay 6` so analyses stay active for minutes). The scenarios were driven
with `validate_lifecycle.py` (Playwright/Chromium, results in `results.json`,
screenshots next to this file); layout and sticky-header behaviour were also
checked interactively in the in-app browser. Analyses run the real pipeline
against the fake model, so they never produce a model-written report; the
completed examples are the seeded session (#1) and a session that finished
after the restart.

| # | Scenario | Result | Evidence |
|---|----------|--------|----------|
| 1 | Wizard → auto-redirect to `/report/:id` | pass | `01-wizard-review`, `02-report-running-after-wizard` |
| 2 | DILI page reset and reusable | pass | `03-dili-page-reset` (second session launched straight away) |
| 3 | Several analyses run concurrently, independently | pass | three sessions `running` with independent progress, `04-sessions-three-running` |
| 4 | Limit 1 → queued with position, auto-start | pass | positions 1 and 2 (`05-report-queued-position`); the queued session started on its own after the restart (see 11) |
| 5 | Cancel one running analysis, others continue | pass | `06-cancel-confirmation`, `07-cancel-stopping`, `08-report-cancelled`; the other two stayed `running` |
| 6 | All statuses, filters and sort in Clinical Sessions | pass | `11-filter-*`, `12-sessions-all-sorted`, `21-queued-filter`, `22-all-statuses-mixed` |
| 7 | Draft: save, reopen, edit, save, launch → one session | pass | `13-draft-saved`, `14-draft-in-sessions`, `15-draft-launched-report`; one session id, edit persisted |
| 8 | Timeline from a draft; status stays `draft` | pass | `16-draft-timeline-generated` |
| 9 | Switching sessions never mixes content | pass | `17a/17b`, `18-report-input-expanded`; Back restores the previous session |
| 10 | Refresh during a running analysis | pass | `19-refresh-running` |
| 11 | Kill + restart backend | pass | running → `failed` / `interrupted` (`10-report-interrupted-after-restart`); the queued session was admitted and ran |
| 12 | Change concurrency while jobs are active | pass | limit 3 → 1 with three running: runs untouched, new launches queued |
| 13 | Older completed report still accessible | pass | `20-older-completed-report` |

Sticky header: with a tall report the shell's single scroller (`.app-main`)
scrolls while `.dili-report-header` stays pinned at the top; the report page adds
no scroll container of its own.

## Notes
- Scenario 11: the first attempt crashed after the restart in the script's
  locator, not the app; the verification was re-run against the post-restart state
  (queued → running was observed from the backend log "Admitted queued clinical
  sessions" and the session ending `completed`).
- The fake model returns empty JSON; sessions can therefore end `completed` with
  a minimal report or `failed` – both are valid terminal states for this check.

## Defects found and fixed
1. **Back reopened a launched draft's editor URL** – after launch the DILI page
   `?session=` URL stayed in history and bounced back to the report. The wizard
   now replaces the history entry (`analysis-wizard.component.ts`).
2. **Queued rows showed an indeterminate progress bar** – only `running`
   sessions show it now (`clinical-sessions-page.component.html`).
3. **Duplicated "interrupted" copy on the report** – the backend message already
   explains it, the extra paragraph was removed (`dili-report-page.component.html`).
4. **Preview tab said "No AI report preview" for drafts and active sessions** –
   it now says what to do next (`clinical-sessions-page.component.ts/html`).
5. **Status filter wrapped unevenly with seven buttons** – now a four-column grid
   (`clinical-sessions-page.component.scss`).
6. **Stale `/api/inspection/sessions` default pick in `test_app_flow.py`** – the
   timetable load test now picks a session that has a timeline.
