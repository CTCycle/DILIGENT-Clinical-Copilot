# QA Execution Manifest

Date: 2026-05-25
Scope: Re-runnable verification commands and evidence anchors for end-to-end UI/backend validation.

## 1. Primary Regression Command

```cmd
app\tests\run_tests.bat all
```

Expected current signature:
- `328 passed`
- `5 skipped`
- `7 warnings`

Focused browser/API slice:

```cmd
app\tests\run_tests.bat integration
```

Expected current signature:
- `36 passed`
- `3 skipped`
- `1 warning`

## 2. Core E2E Coverage Map (app/tests/e2e/test_app_flow.py)

- Home load + controls: `test_dilu_agent_page_loads`
- Home load health (console/network): `test_home_initial_load_has_no_console_errors_or_failed_requests`
- Model-config load health: `test_model_config_initial_load_has_no_console_errors_or_failed_requests`
- Data load health: `test_data_inspection_initial_load_has_no_console_errors_or_failed_requests`
- Clinical-sessions load health: `test_clinical_sessions_initial_load_has_no_console_errors_or_failed_requests`
- Timetable load health: `test_timetable_initial_load_has_no_console_errors_or_failed_requests`
- Navigation: `test_model_config_navigation`, `test_data_inspection_navigation`
- Refresh persistence: `test_dili_form_state_restores_after_refresh`
- Back/forward persistence: `test_home_form_state_persists_across_back_forward_navigation`
- Run de-duplication: `test_dili_run_burst_click_submits_single_job`
- Conflict UI handling: `test_dili_run_conflict_surfaces_clear_error_message` (deterministic mocked 409)
- Model-config save/PUT integration: `test_model_config_runtime_toggle_enables_save_and_submits_put`
- Timetable no-autopost-on-load: `test_timetable_route_load_does_not_autogenerate_timeline`
- Timetable invalid id UX: `test_timetable_invalid_session_id_shows_validation_error`
- Keyboard top-nav reachability: `test_keyboard_navigation_reaches_primary_tabs`
- Home form focusability: `test_home_form_labels_are_associated_with_inputs`
- Home form keyboard traversal: `test_keyboard_tab_traversal_reaches_home_form_controls`
- Clinical-session selection/detail sync: `test_clinical_sessions_row_selection_loads_matching_detail`

## 3. Visual Evidence Set

Location: `artifacts/qa-shots/`

- Desktop: `home-desktop.png`, `data-desktop.png`, `clinical-sessions-desktop.png`, `timetable-desktop.png`
- Mobile: `model-config-mobile.png`, `clinical-sessions-mobile.png`, `timetable-mobile.png`, `timetable-mobile-postfix.png`
- Tablet: `home-tablet.png`, `model-config-tablet.png`, `data-tablet.png`, `clinical-sessions-tablet.png`

## 4. Supporting Docs

- Main QA report: `assets/docs/QA_E2E_VALIDATION_REPORT_2026-05-25.md`
- Runner + launch troubleshooting: `assets/docs/LAUNCH_TROUBLESHOOTING.md`

## 5. Known Environment Caveat

- Current verified runner uses `app/server/.venv/Scripts/python.exe`.
- If local pytest packages are unavailable in `.venv`, older helper scripts may fall back to `uv --with ...`, which can require outbound package access.
