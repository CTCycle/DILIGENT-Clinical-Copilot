Launcher and runtime validation on 2026-06-25 / 2026-06-26

Launcher and health

- `start_on_windows.bat` completed successfully with exit code `0`.
- Initial launcher diagnosis found a stale frontend listener on `127.0.0.1:9847`; rerunning the real launcher after port cleanup restored the expected state.
- Health probe result:
  - `http://127.0.0.1:7690/api/health` -> `{"status":"ok"}`
- OpenAPI probe result:
  - `http://127.0.0.1:7690/openapi.json` -> `200`

Catalog recovery

- The first live browser run failed correctly because both local catalogs were empty.
- Real recovery path used the inspection API update jobs, not the generic database initializer:
  - `POST /api/inspection/livertox/jobs`
  - `POST /api/inspection/rxnav/jobs`
- LiverTox job completed with:
  - archive `app/resources/sources/archives/livertox_NBK547852.tar.gz`
  - `records=1869`
- Mid-validation SQLite counts after recovery:
  - `livertox_monographs 1588`
  - `drug_aliases 44178`
  - `reference_catalog_entries 817`
- Catalog spot checks confirmed the required live entries:
  - RxNav search returned `amoxicillin clavulanate oral`
  - LiverTox search returned `Amoxicillin-Clavulanate`

Real browser workflow

- In-app browser loaded `http://127.0.0.1:9847/` through the real launcher-served UI.
- Submitted a live clinical case for `Validation Case A` with visit date `2026-05-14`.
- The app completed the analysis and rendered a structured dossier, including:
  - pattern `hepatocellular`
  - detected drug `Amoxicillin-clavulanate`
  - DILIN-like causality `possible`
  - Hy's Law `possible`
  - acceptance-question block with explicit missing-data notes
- Saved browser artifacts:
  - `assets/QA/diligent-live-report.png`
  - `assets/QA/diligent-live-report.md`
  - `assets/QA/diligent-live-page-before-case.png`
  - `assets/QA/diligent-live-page-controls.png`
  - `assets/QA/diligent-homepage.png`

Focused automated validation

- Backend adjudication and workflow slice:
  - `app/server/.venv/Scripts/python.exe -m pytest -p no:cacheprovider app/tests/unit/test_dili_adjudication_engine.py app/tests/unit/test_rucam_service.py app/tests/unit/test_session_workflow_report_generation.py -q`
  - result: `16 passed`
- API contract slice:
  - `app/server/.venv/Scripts/python.exe -m pytest -p no:cacheprovider app/tests/unit/test_fastapi_contracts.py app/tests/unit/test_api_response_models.py app/tests/unit/test_error_handling_enforcement.py -q`
  - result: `15 passed`
- Additional bounded checks completed earlier in the same validation pass:
  - changed backend modules compiled successfully
  - frontend type check succeeded
  - `npm run build` succeeded outside the sandbox after the known in-sandbox `spawn EPERM` failure mode

Residual note

- `pytest-playwright` API/E2E files remained blocked in this Codex Windows environment by `PermissionError: [WinError 5] Accesso negato` during Playwright subprocess startup.
- That environment issue does not negate the successful in-app browser workflow above, which provided the required live end-to-end proof.
