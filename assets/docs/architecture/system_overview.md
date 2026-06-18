# System Overview
Last updated: 2026-06-18

## System Summary
DILIGENT is a local-first clinical application with:
- FastAPI backend in `app/server`
- Angular standalone frontend in `app/client`
- Optional Tauri desktop shell in `app/src-tauri`

Primary flow:
1. The user submits clinical data in the Angular UI.
2. The backend validates and normalizes the input, then runs clinical analysis.
3. Long-running work executes through background jobs with poll and cancel APIs.
4. Results, catalogs, and session data are persisted for later review.

## Repository Structure
Maintained source-level structure, with build and cache artifacts omitted:

```text
.
|-- start_on_windows.bat
|-- setup_and_maintenance.bat
|-- settings/
|   |-- .env
|   |-- .env.local.example
|   `-- configurations.json
|-- app/
|   |-- resources/
|   |   |-- models/
|   |   `-- sources/
|   |-- server/
|   |   |-- app.py
|   |   |-- api/
|   |   |-- configurations/
|   |   |-- domain/
|   |   |-- repositories/
|   |   |-- services/
|   |   |-- common/
|   |   |   |-- catalogs/
|   |   |   |   |-- provider.py
|   |   |   |   |-- manifest_loader.py
|   |   |   |-- utils/
|   |   |   |-- security/
|   |   |   |-- prompts/
|   |-- client/
|   |   |-- package.json
|   |   |-- src/
|   |   |   |-- main.ts
|   |   |   |-- styles.scss
|   |   |   `-- app/
|   |   |       |-- app.ts
|   |   |       |-- app.routes.ts
|   |   |       |-- core/
|   |   |       |-- components/
|   |   |       `-- pages/
|   |-- src-tauri/
|   |   |-- tauri.conf.json
|   |   `-- src/main.rs
|   `-- tests/
|       |-- run_tests.bat
|       |-- conftest.py
|       |-- unit/
|       `-- e2e/
`-- release/
    `-- tauri/
        `-- build_with_tauri.bat
```

## Application Entry Points
- Backend app: `app/server/app.py`
  - Builds the FastAPI app through `create_app()`, initializes settings, registers middleware and error handlers, mounts routers under `/api`, serves packaged SPA assets in Tauri mode, and runs startup checks through the FastAPI lifespan path.
- Frontend app: `app/client/src/main.ts`
  - Bootstraps Angular `App` with `appConfig`.
- Frontend routing: `app/client/src/app/app.routes.ts`
  - Current routes: `/`, `/clinical-sessions`, `/data`, `/model-config`, `/sessions/:sessionId/timetable`.
- Desktop runtime: `app/src-tauri/src/main.rs` plus `tauri.conf.json`.
- Windows launcher: `start_on_windows.bat`.
