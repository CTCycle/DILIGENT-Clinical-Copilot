# End-to-End UI & System Validation Report

Last updated: 2026-05-25

Date: 2026-05-25  
Application: DILIGENT Clinical Copilot  
Scope: Frontend UX/UI behavior + backend/API consistency + integration state coherence

## 1. Executive Summary

Overall quality is substantially improved and the current combined frontend/backend regression gate is passing in this workspace. Remaining risk is concentrated in visual/accessibility depth and an existing style-budget warning, not in the core API/UI synchronization paths covered by this run.

Key strengths:
- Core model-config, app-flow, backend API, and unit regression paths are passing through the official launcher (`run_tests.bat`) after backend integration hardening.
- Major user flows (landing, nav, model-config save, conflict handling, form persistence) are covered by executable e2e checks.
- A high-impact backend integration defect was reproduced from UI/API flows, root-caused with logs, and fixed.

Major risks:
- Accessibility validation remains basic and targeted; no exhaustive axe-style scanner run was executed.
- `npm run build` still reports the pre-existing `clinical-sessions-page.component.scss` style budget warning.

## 2. Tested Areas

Pages and routes covered:
- `/` (DILI Assessment landing/home)
- `/model-config`
- `/data`
- `/clinical-sessions`
- `/sessions/:sessionId/timetable` (validated with `session_id=38`)

User flows covered:
- Flow A: Home page load and primary form controls present.
- Flow B: Navigation from home -> model config.
- Flow C: Navigation from home -> data inspection.
- Flow D: DILI form state persistence after page refresh.
- Flow E: Run button burst-click protection (single backend job submission).
- Flow F: Active-job conflict handling surfaced clearly in UI.
- Flow G: Model config runtime toggle + save, with backend `PUT /api/model-config` verification.
- Flow H: Timetable route-entry behavior does not auto-trigger timeline regeneration `POST`.
- Flow I: Invalid timetable route input (`session_id=0`) shows deterministic validation error state.
- Flow J: Home form state persists across browser back/forward navigation after route transition.
- Flow K: Initial home load completes without browser console errors or failed network requests.
- Flow L: Keyboard-only tab traversal reaches primary top navigation tabs.
- Flow M: Home form controls are directly focusable and operable (basic form accessibility focus check).
- Flow N: Initial `/model-config` load completes without console errors or failed network requests.
- Flow O: Initial `/data` load completes without console errors or failed network requests.
- Flow P: Initial `/clinical-sessions` load completes without console errors or failed network requests.
- Flow Q: Initial timetable route load completes without console errors or failed network requests (using available session id, fallback to invalid id when catalog is empty).
- Flow R: Clinical sessions row selection triggers matching detail API call and workspace state update to the selected session.

Backend endpoints observed/validated in this run context:
- `GET /api/health`
- `GET /api/model-config`
- `PUT /api/model-config`
- `POST /api/clinical/jobs`
- `DELETE /api/clinical/jobs/{job_id}`

Execution paths validated:
- `app\tests\run_tests.bat modelconfig`
- `app\tests\run_tests.bat modelconfigfull`
- `app\tests\run_tests.bat integration`
- `app\tests\run_tests.bat all`
- `npm run build`
- `npm test -- --watch=false`

Manual visual evidence captured:
- `artifacts/qa-shots/home-desktop.png`
- `artifacts/qa-shots/model-config-mobile.png`
- `artifacts/qa-shots/data-desktop.png`
- `artifacts/qa-shots/clinical-sessions-desktop.png`
- `artifacts/qa-shots/clinical-sessions-mobile.png`
- `artifacts/qa-shots/timetable-desktop.png`
- `artifacts/qa-shots/timetable-mobile.png`
- `artifacts/qa-shots/home-tablet.png`
- `artifacts/qa-shots/model-config-tablet.png`
- `artifacts/qa-shots/data-tablet.png`
- `artifacts/qa-shots/clinical-sessions-tablet.png`

Minimum coverage audit:
- Landing/home page: met (`test_dilu_agent_page_loads`, manual screenshot `home-desktop.png`).
- Navigation system: met (`test_model_config_navigation`, `test_data_inspection_navigation`, keyboard guard `test_keyboard_navigation_reaches_primary_tabs`).
- At least 3 full user flows: met (model-config save flow, conflict flow, timetable route-entry flow, form persistence flows).
- At least 1 form with backend validation: met (`test_model_config_runtime_toggle_enables_save_and_submits_put`; DILI conflict/invalid feedback paths).
- At least 1 error/invalid scenario: met (`test_timetable_invalid_session_id_shows_validation_error`; conflict handling and historical 429 finding).
- At least 1 refresh or navigation persistence test: met (`test_dili_form_state_restores_after_refresh`, `test_home_form_state_persists_across_back_forward_navigation`).
- Responsive behavior across 2+ viewports: met (desktop + mobile + tablet screenshots across key routes).
- Console + network inspection: met (`test_home_initial_load_has_no_console_errors_or_failed_requests`, `test_model_config_initial_load_has_no_console_errors_or_failed_requests`, `test_data_inspection_initial_load_has_no_console_errors_or_failed_requests`, `test_clinical_sessions_initial_load_has_no_console_errors_or_failed_requests` + runtime log checks).
- Console + network inspection: met (`test_home_initial_load_has_no_console_errors_or_failed_requests`, `test_model_config_initial_load_has_no_console_errors_or_failed_requests`, `test_data_inspection_initial_load_has_no_console_errors_or_failed_requests`, `test_clinical_sessions_initial_load_has_no_console_errors_or_failed_requests`, `test_timetable_initial_load_has_no_console_errors_or_failed_requests` + runtime log checks).
- Backend log verification for at least one issue: met (readonly SQLite 500 and timeline 429 correlated with logs).

## 3. Findings

### Finding 1: Backend write failures surfaced as UI/integration break in model-config operations
- Severity: critical
- Area: Integration / Backend
- Steps to reproduce:
1. Execute `cmd /c app\tests\run_tests.bat modelconfig` before isolation fix.
2. Trigger model-config writes via e2e setup and UI flow (`PUT /api/model-config`).
- Expected behavior:
  - Backend persists model config updates and returns HTTP 200.
  - UI flows remain stable.
- Actual behavior:
  - Backend returned HTTP 500.
  - Dependent UI/e2e flows failed.
- API/log evidence:
  - Backend log showed `sqlite3.OperationalError: attempt to write a readonly database` while processing `PUT /api/model-config`.
- Fix applied:
  - Added env-driven SQLite path override and per-run temporary DB isolation for regression runners.

### Finding 2: Validation pipeline can fail in restricted-network environments despite healthy app behavior
- Severity: high
- Area: Backend / Tooling integration
- Steps to reproduce:
1. Run `cmd /c app\tests\run_tests.bat modelconfigfull` in a restricted environment.
2. `uv run --with pytest ...` attempts dependency fetch.
- Expected behavior:
  - Regression runner should execute deterministically from local environment.
- Actual behavior:
  - Run fails with package-fetch error (`os error 10013`) before app-level tests complete.
- API/log evidence:
  - Runner output: failed fetch from `https://pypi.org/simple/pytest/`.
- Notes:
  - Not an application business-logic failure; it is a test infrastructure reliability risk.

### Finding 3: Pytest cache warning noise in workspace
- Severity: low
- Area: Tooling / runtime hygiene
- Steps to reproduce:
1. Execute batch model-config runners.
- Expected behavior:
  - Clean runner output.
- Actual behavior:
  - `PytestCacheWarning` about `.pytest_cache\v\cache\nodeids` with `WinError 183`.
- Notes:
  - Does not currently block pass/fail but reduces signal clarity for CI/local triage.
  - Status update (2026-05-24, post-fix): mitigated by setting per-run isolated pytest cache directories in model-config runner scripts via `PYTEST_ADDOPTS="-o cache_dir=<temp path>"`. Latest full run output no longer emitted `PytestCacheWarning`.

### Finding 4: Visual loading-state exposure on mobile model-config view
- Severity: medium
- Area: UI / Integration
- Steps to reproduce:
1. Open `/model-config` on a mobile viewport (390x844).
2. Observe initial page state.
- Expected behavior:
  - Critical configuration controls should become available promptly with clear loading-to-ready transition.
- Actual behavior:
  - Captured state shows persistent `Loading model configuration...` at screenshot time, with no visible progress indicator beyond text.
- API/log evidence:
  - Backend during same manual pass was healthy (`/api/health` responded, model-config endpoint reachable from automated suite).
- Notes:
  - This may be transient; manual screenshot confirms UX risk when data hydration is delayed.

### Finding 5: Timetable regeneration can surface backend throttling state as unstable UX
- Severity: medium
- Area: Integration / UI
- Steps to reproduce:
1. Open `/sessions/38/timetable` (or another session timetable route).
2. Observe initial timetable load/regeneration behavior.
- Expected behavior:
  - Either timetable loads deterministically, or the UI presents a clear retry model tied to backend response semantics.
- Actual behavior:
  - Timetable page remained in loading/error-like state in captured views.
  - Mobile screenshot surfaced: `[ERROR] Service is busy. Wait a moment and retry.`
- API/log evidence:
  - Backend log during the same run recorded: `POST /api/inspection/sessions/38/timeline HTTP/1.1` -> `429 Too Many Requests`.
- Notes:
  - This is not a silent failure (message is shown), but behavior suggests a fragile first-load path when regeneration is rate-limited or serialized.
  - Status update (2026-05-24, post-fix): mitigated in frontend by switching timetable route-entry to read-only fetch on initial load (no automatic regenerate POST). Route-entry validation now shows `GET /api/inspection/sessions/38/timeline` with HTTP 200 and no `POST .../timeline` during first render.

## 4. UI and UX Observations

- Confirmed:
  - Core navigation tabs and primary controls render and remain interactable.
  - Conflict UX message is explicit and user-safe when a clinical job is already running.
  - Form values on landing persist through refresh for key inputs.
  - Desktop layout hierarchy and spacing are broadly consistent across `/` and `/data` in captured states.
  - Clinical sessions desktop split-pane layout and session list/status rendering are coherent and readable.
  - Timetable desktop controls are aligned and consistently styled.

- Remaining gaps:
  - Screenshot pass now covers key pages at desktop/mobile/tablet including `/clinical-sessions` and `/sessions/:sessionId/timetable`; deeper interaction checks (filters/edit/revision/timeline item selection) remain partial.
  - Accessibility checks remain partial via targeted keyboard assertions; no full axe-style audit in this pass.

## 5. Backend and Integration Observations

- Confirmed:
  - `PUT /api/model-config` coherence is restored under isolated per-run DB.
  - End-to-end model-config + app-flow suite passes when dependencies are available.

- System behavior caveat:
  - Test execution path still depends on runtime ability to resolve/install `pytest` and `pytest-playwright` via `uv --with ...` when cache is incomplete.

## 6. Unverified Concerns

- Unverified:
- Full responsive visual quality audit across all major pages at multiple breakpoints after latest local UI edits (beyond currently captured desktop/mobile/tablet checkpoints).
- Keyboard-only traversal quality for all interactive controls beyond currently covered primary navigation and core home form inputs.
- Potential state persistence side effects in deeper clinical session/revision/timeline interactions beyond first-load and route-entry checks.

## 7. Recommended Next Actions

1. Stabilize regression runtime dependencies:
   - Pre-provision test dependencies in a reproducible local/runtime environment to avoid network-coupled `uv --with` failures.
2. Add dedicated visual-regression sweep:
   - Capture desktop + mobile screenshots for `/`, `/model-config`, `/data`, `/clinical-sessions` and track deltas against UI standards.
3. Expand integration coverage:
   - Add e2e assertions for one full clinical run completion path (start -> poll -> result render) including refresh/back-forward persistence.
4. Improve runner signal quality:
   - Resolve `.pytest_cache` warning source or redirect pytest cache into an isolated per-run path.

## Post-Fix Verification Addendum (2026-05-25)

- Remediation implemented:
  - `app/client/src/app/pages/patient-timetable/patient-timetable-page.component.ts`
  - Initial timetable load now uses `GET` fetch only.
  - Timeline generation remains an explicit user action via `Regenerate`.

- Evidence:
  - Screenshot: `artifacts/qa-shots/timetable-mobile-postfix.png`
  - Backend log excerpt from validation run:
    - `GET /api/inspection/sessions/38/timeline HTTP/1.1` -> `200 OK`
    - No automatic `POST /api/inspection/sessions/38/timeline` on route entry.
  - Automated guard:
    - `app/tests/e2e/test_app_flow.py::test_timetable_route_load_does_not_autogenerate_timeline`
    - Included in full regression run with pass signature: `21 passed` (e2e suite, latest run on 2026-05-25).
  - Invalid-route guard:
    - `app/tests/e2e/test_app_flow.py::test_timetable_invalid_session_id_shows_validation_error`
    - Confirms `/sessions/0/timetable` renders `Invalid session id.` and does not attempt timeline generation.
  - Navigation persistence guard:
    - `app/tests/e2e/test_app_flow.py::test_home_form_state_persists_across_back_forward_navigation`
    - Confirms home form data persists when navigating to `/model-config`, then browser back/forward.
  - Console/network guard:
    - `app/tests/e2e/test_app_flow.py::test_home_initial_load_has_no_console_errors_or_failed_requests`
    - Confirms initial home route load produced no console `error` events and no failed requests.
  - Additional console/network guards:
    - `app/tests/e2e/test_app_flow.py::test_model_config_initial_load_has_no_console_errors_or_failed_requests`
    - `app/tests/e2e/test_app_flow.py::test_data_inspection_initial_load_has_no_console_errors_or_failed_requests`
    - `app/tests/e2e/test_app_flow.py::test_clinical_sessions_initial_load_has_no_console_errors_or_failed_requests`
    - `app/tests/e2e/test_app_flow.py::test_timetable_initial_load_has_no_console_errors_or_failed_requests`
    - Confirms equivalent initial-load health checks on `/model-config`, `/data`, `/clinical-sessions`, and timetable route entry.
  - Keyboard navigation guard:
    - `app/tests/e2e/test_app_flow.py::test_keyboard_navigation_reaches_primary_tabs`
    - Confirms keyboard traversal reaches the primary navigation tabs without pointer interaction.
  - Form focus accessibility guard:
    - `app/tests/e2e/test_app_flow.py::test_home_form_labels_are_associated_with_inputs`
    - Confirms primary home form controls are focusable and operable via direct interaction.
  - Keyboard form traversal guard:
    - `app/tests/e2e/test_app_flow.py::test_keyboard_tab_traversal_reaches_home_form_controls`
    - Confirms deterministic keyboard traversal between primary patient fields (`#patient-name` and `#visit-date`) and keyboard focusability of `#clinical-input`.
  - Clinical-sessions selection/detail integration guard:
    - `app/tests/e2e/test_app_flow.py::test_clinical_sessions_row_selection_loads_matching_detail`
    - Confirms clicking a list row triggers `GET /api/inspection/sessions/{id}` for the selected row and updates workspace header to the same session id.
  - Runner reliability hardening:
    - `app/tests/run_model_config_regression.ps1` and `app/tests/run_model_config_full_regression.ps1` now prefer `python -m pytest` from `.venv` when available and fall back to `uv --with ...` only when necessary.

## Requirement-by-Requirement Audit (2026-05-25)

- Requirement: system understanding first (docs/API/data flow mapping).
  - Status: proven.
  - Evidence: architecture/docs review plus endpoint mapping in report `Tested Areas` and covered endpoint list.

- Requirement: initial load validation (layout/assets/console/network/basic API reflection).
  - Status: partially proven.
  - Evidence: initial-load no-console/no-failed-request tests for `/`, `/model-config`, `/data`, `/clinical-sessions`, and timetable route.
  - Gap: no automated assertion on font/icon/image decode failures beyond request-failure checks.

- Requirement: UI structure/design system audit.
  - Status: partially proven.
  - Evidence: visual captures across desktop/mobile/tablet, documented UI observations.
  - Gap: no exhaustive token-level diff audit across every component.

- Requirement: deep visual + interaction validation.
  - Status: partially proven.
  - Evidence: route-level screenshots, form persistence tests, conflict behavior tests, runtime toggle/save tests, timetable behavior tests.
  - Gap: not all interactive subflows in clinical-sessions revision/timeline editing were exhaustively exercised.

- Requirement: forms and backend validation cross-check.
  - Status: proven.
  - Evidence: model-config PUT flow, conflict handling, invalid timetable id flow, form persistence and navigation persistence checks.

- Requirement: responsiveness across at least two viewports.
  - Status: proven.
  - Evidence: desktop + mobile + tablet screenshots across key pages.

- Requirement: accessibility basics.
  - Status: partially proven.
  - Evidence: keyboard nav to primary tabs, home form focusability/traversal checks.
  - Gap: no full accessibility scanner pass (axe-like comprehensive audit not executed).

- Requirement: backend log inspection for at least one issue.
  - Status: proven.
  - Evidence: captured/triaged readonly SQLite 500 and timeline 429 with correlated logs and fixes.

- Requirement: stress/edge-case checks.
  - Status: partially proven.
  - Evidence: burst-click single-submit guard; refresh and back/forward persistence checks.
  - Gap: broader interruption chaos scenarios remain limited.

- Requirement: console/network/runtime monitoring.
  - Status: proven.
  - Evidence: dedicated console/requestfailed guards on multiple routes and repeated full-run validation.

- Requirement: minimum coverage checklist.
  - Status: proven.
  - Evidence: explicit `Minimum coverage audit` section with all required checklist items marked met.

## Continuation Addendum (2026-05-25, later run)

- Fresh reproducibility check:
  - Command rerun: `cmd /c app\tests\run_tests.bat modelconfigfull`
  - Result: `3 passed` (unit), `21 passed` (e2e)
  - Runtime output confirmed backend/frontend startup, test pass, and cleanup path completed.

- Process hygiene verification:
  - Listener check after run: `cmd /c netstat -ano | findstr LISTENING | findstr ":7690 :9847"`
  - Result: no listening entries found for ports `7690` and `9847`.

- Scope boundary acknowledged for this validation cycle:
  - Complex clinical-accuracy dynamics are treated as out-of-scope/unverifiable in local reproducible QA.
  - This report therefore focuses on observable UI behavior, integration consistency, API semantics, and backend/log-correlated execution evidence.

## Continuation Addendum (2026-05-25, newest run)

- Fresh reproducibility check:
  - Command rerun: `cmd /c app\tests\run_tests.bat modelconfigfull`
  - Result: `3 passed` (unit), `22 passed` (e2e)
  - New added coverage: `test_clinical_sessions_row_selection_loads_matching_detail`.

- New integration coverage evidence:
  - On `/clinical-sessions`, selecting a row now has explicit e2e proof that:
    - UI action selects a concrete session id from the list row.
  - Backend detail request is issued for that exact id (`GET /api/inspection/sessions/{id}`).
  - Workspace header reflects the same selected session id.

## Continuation Addendum (2026-05-25, current goal run)

- Runtime under test:
  - Backend: `http://127.0.0.1:7690`
  - Frontend preview: `http://127.0.0.1:9847`

- Timeline creation flow:
  - UI action: selected Clinical Sessions entry `39`, opened Timeline tab, and selected `Create Patient Timeline`.
  - Backend evidence:
    - `POST /api/inspection/sessions/39/timeline HTTP/1.1` -> `200 OK`
    - `GET /api/inspection/sessions/39/timeline HTTP/1.1` -> `200 OK`
  - Frontend evidence:
    - Browser navigated to `/sessions/39/timetable`.
    - Timetable rendered 11 events and evidence details from the generated timeline payload.
    - Direct API fetch to `/api/inspection/sessions/39/timeline` returned HTTP 200.

- New confirmed defect fixed:
  - Area: UI / Accessibility
  - Severity: low
  - Timetable toolbar rendered non-functional controls (`Jan 2024 - May 2025`, `Today`) as clickable buttons even though they had no handlers and did not trigger backend state changes.
  - Fix: converted them to non-interactive toolbar labels and added accessible names/selected state to timetable rail/event controls.
  - Files:
    - `app/client/src/app/pages/patient-timetable/patient-timetable-page.component.html`

- Current visual/responsive evidence:
  - Desktop timetable route has no document-level horizontal overflow at 1280px viewport.
  - Mobile timetable route at 390px viewport has no document-level horizontal overflow, hides the rail, stacks content to one column, and keeps events/evidence visible.
  - Event selection updates the evidence panel consistently with the selected event and `aria-pressed` state.

- Current command verification:
  - `npm run build`: passed.
  - `npm test -- --watch=false`: passed, 5 files / 11 tests.
  - `cmd /c app\tests\run_tests.bat unit`: did not execute tests because `app/server/.venv` has no `pytest` module and no fallback `runtimes/.venv` exists in this workspace.

- Remaining verification gap:
  - Python/backend pytest suites are currently blocked by missing local test dependencies in the mandated venv path. This is an environment readiness issue, not proof of backend test pass.

## Continuation Addendum (2026-05-25, backend validation run)

- Backend unit suite:
  - Command: `runtimes\uv\uv.exe run --frozen --offline --extra test python -m pytest ..\tests\unit -q` from `app\server`.
  - Result: `291 passed, 2 skipped`.
  - Notes: execution used the checked-in lock/cache path in offline mode; no new virtual environment was created.

- Confirmed backend defects fixed:
  - Area: Backend / Integration
  - Severity: high
  - Clinical section extraction no longer fabricates coarse fallback sections when deterministic parsing fails and LLM extraction is unavailable, incomplete, or not source-grounded. Invalid extraction now raises `ClinicalInputExtractionError` instead of allowing a false-success analysis path.
  - Files:
    - `app/server/services/session/clinical_input_extractor.py`
  - Evidence:
    - `test_clinical_input_extractor.py` passes with invalid fallback cases raising errors.

- Confirmed backend validation defects fixed:
  - Area: Backend / API validation
  - Severity: high
  - `POST /api/clinical/jobs` now rejects missing `visit_date` and an empty `selected_model_providers` list instead of falling back to implicit runtime state.
  - Files:
    - `app/server/services/session/session_request_validation.py`
    - `app/server/services/clinical/validation.py`
  - API evidence:
    - Missing visit date: HTTP `422`, body includes `{"field":"visit_date","message":"Visit date is required."}`.
    - Missing provider: HTTP `422`, body includes `{"field":"selected_model_providers","message":"At least one model provider must be selected."}`.
  - Backend log evidence:
    - `POST /api/clinical/jobs HTTP/1.1` -> `422 Unprocessable Content`.

- Confirmed UI/system-state defect fixed:
  - Area: UI / Integration
  - Severity: medium
  - Changing current DILI form metadata after a completed analysis previously left the old report rendered, including the old visit date, while the current form showed changed/missing metadata.
  - Fix: inactive form edits clear terminal job/report artifacts and show the explicit empty report state.
  - Files:
    - `app/client/src/app/pages/dili-agent/dili-agent-page.component.ts`
  - Browser evidence:
    - After clearing `#visit-date`, `Run DILI analysis` is disabled.
    - Previous `Visit date: 20 May 2026` report text is no longer present.
    - Report panel shows `No report generated yet. Run analysis to see results.`
    - Copy/Expand export actions are disabled.
    - Console error count remained `0`.

- Current command verification:
  - `npm run build`: passed with the pre-existing `clinical-sessions-page.component.scss` budget warning.
  - `npm test -- --watch=false`: passed, 5 files / 11 tests.
  - Backend unit suite: passed, 291 tests / 2 skipped.

## Final Verification Addendum (2026-05-25)

- Full combined validation:
  - Command: `cmd /c app\tests\run_tests.bat all`
  - Result: `328 passed, 5 skipped, 7 warnings`.
  - Coverage included backend API tests, browser-driven Playwright app-flow tests, and Python unit tests.

- Frontend verification:
  - Command: `npm run build`
  - Result: passed.
  - Residual warning: `clinical-sessions-page.component.scss` exceeds the configured 20 kB style budget by 4.13 kB.
  - Command: `npm test -- --watch=false`
  - Result: `5 passed` test files, `11 passed` tests.

- E2E/integration slice:
  - Command: `cmd /c app\tests\run_tests.bat integration`
  - Result: `36 passed, 3 skipped, 1 warning`.
  - Runner fix: the integration suite now starts the frontend because its target includes browser tests.

- Backend cancellation contract defect:
  - Area: Backend / Integration
  - Severity: high
  - Observed failure: `DELETE /api/clinical/jobs/{job_id}` raised a Pydantic validation error because `JobManager.cancel_job()` returns a job snapshot while API response model `JobCancelResponse.success` requires a boolean.
  - Fix: clinical job and Ollama pull cancellation now convert the returned snapshot to a boolean success value.
  - Files:
    - `app/server/services/session/session_service.py`
    - `app/server/services/llm/ollama.py`
    - `app/tests/unit/test_runtime_jobs.py`
  - Evidence:
    - `app/tests/unit/test_runtime_jobs.py::test_clinical_cancel_response_converts_job_snapshot_to_success_bool` passed.
    - Full `run_tests.bat all` completed without the previous cancellation 500 or follow-on clinical API 409 cascade.

- Cloud model testing constraint:
  - All E2E setup paths that enable cloud runtime now pin `cloud_model` to `gpt-4.1-mini`.
  - Clinical API tests reset runtime to local mode before each test to avoid dependency on persisted cloud state or active cloud keys.
  - Files:
    - `app/tests/e2e/test_app_flow.py`
    - `app/tests/e2e/test_clinical_api.py`
    - `app/tests/unit/test_model_config_persistence.py`
  - Backend log evidence from integration/full runs showed resolved runtime with `cloud_model=gpt-4.1-mini`.

- Stress-test isolation:
  - The DILI burst-click guard now mocks job start/status responses for that frontend-only stress case, preventing real long-running clinical jobs from contaminating later API tests.
  - Real backend job start/cancel behavior remains covered by clinical API tests.
