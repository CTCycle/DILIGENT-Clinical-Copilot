# Patient Timetable redesign validation

Date: 2026-07-22

## Implemented contract

- Canonical UTC day/month/year parsing is shared by frontend scale/layout logic and backend extraction/preview serialization.
- Inclusive date ranges retain `event_date_end`; invalid calendar tokens and reversed ranges are safely unanchored/uncertain.
- The timetable uses one shared date axis, deterministic chronology, dynamic lane packing, bounded pixel-based clusters, explicit Unanchored and Uncertain lanes, dense/compact/comfortable density, zoom, fit, and horizontal scrolling.
- The permanent rail/fixed Evidence column is removed. The inspector is a desktop overlay and a narrow-screen bottom-sheet dialog with focus restoration and Escape handling.
- Keyboard activation, roving event tab stops, non-color uncertainty/selection labels, range indicators, and cluster member selection are covered.

## Automated checks

- Backend timeline extraction and date interval tests: **10 passed**.
- Backend FastAPI/timeline focused checks previously established: **14 passed**.
- Ruff focused backend and E2E checks: **passed**.
- Angular full suite: **41 passed across 12 files**.
- Angular production build: **passed**; existing style-budget warnings remain in unrelated model-config and clinical-sessions stylesheets.
- Focused deterministic timetable E2E against a fresh current-source UI server: **1 passed**. It verifies the shared axis, no rail, Unanchored lane, four-event cluster, cluster member list, inspector, and Escape close.
- Full E2E suite was also exercised with live backend/frontend processes: **15 passed, 7 environment/service failures**. The failures were 500 responses from unrelated live API flows and session-list availability; the deterministic timetable flow passed.
- `git diff --check`: **passed**.

## Runtime note

The pre-existing app server on port 9847 was left running. Temporary isolated QA services were used for validation and are not part of the implementation.
