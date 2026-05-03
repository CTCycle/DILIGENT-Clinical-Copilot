# DILIGENT Clinical Copilot

[![Release](https://img.shields.io/github/v/release/CTCycle/DILIGENT-Clinical-Copilot?display_name=tag)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases) [![Python](https://img.shields.io/badge/python-%3E%3D3.14-blue?logo=python&logoColor=white)](./app/server/pyproject.toml) [![Angular](https://img.shields.io/badge/angular-%5E21.2.0-DD0031?logo=angular&logoColor=white)](./app/client/package.json) [![License](https://img.shields.io/badge/license-Polyform%20Noncommercial%201.0.0-lightgrey)](./LICENSE) [![CI](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml)

## 1. Project Overview
DILIGENT Clinical Copilot supports clinicians during Drug-Induced Liver Injury (DILI) evaluations with a FastAPI backend and an Angular + TypeScript frontend. It collects anamnesis, medications, and lab values, then coordinates clinical analysis with optional RAG support and session persistence for review.

> **Work in Progress**: This project is under active development and may contain incomplete features or defects.

![DILIGENT flow schema](assets/figures/diligent-flow.png)
_Conceptual flow from case intake through guided DILI analysis, session recording, and review._

## 2. Quick Start

### 2.1 Windows (Recommended)
Run:
```cmd
start_on_windows.bat
```

The launcher prepares local runtimes/dependencies and starts backend plus frontend.

### 2.2 macOS / Linux (Manual)
Prerequisites:
- Python 3.14+
- Node.js 18+ and npm

Backend:
```bash
cd app/server
python -m pip install -e ".[test]"
uvicorn app:app --host 127.0.0.1 --port 8000
```

Frontend:
```bash
cd app/client
npm install
npm run build
npm run preview -- --host 127.0.0.1 --port 7861
```

## 3. Runtime Profiles
DILIGENT is configuration-first and uses one active runtime file: `settings/.env`.

Switch to local profile:
```cmd
copy /Y settings\.env.local.example settings\.env
```

Switch to local Tauri profile:
```cmd
copy /Y settings\.env.local.tauri.example settings\.env
```

See `assets/docs/RUNTIME_MODES.md` for full runtime and packaging details.

## 4. Using the Application
Typical workflow:
1. Enter anamnesis, exam notes, medications, and lab values.
2. Choose model/provider settings and optionally enable RAG/web search.
3. Run analysis and review the generated report.
4. Use Data Inspection for session history and data-update operations.

Detailed user journeys and feature guidance are documented in `assets/docs/USER_MANUAL.md`.

### Screenshots

#### Landing Page
![Landing page](assets/figures/home.png)
_Main DILI analysis workspace with clinical inputs and patient context panel._

#### Dashboard / Report Output
![Dashboard view](assets/figures/dashboard.png)
_Analysis dashboard focused on the report output area and execution controls._

#### Model Configuration (Settings)
![Model settings](assets/figures/settings.png)
_Runtime source selection and current model configuration summary._

#### Local Model Catalog (List)
![Model list](assets/figures/models-list.png)
_Scrollable local model catalog with install and role-assignment actions._

#### Provider Access Key Management
![Provider key detail](assets/figures/model-detail.png)
_Brave Search access-key management entry point within the configuration panel._

#### Data Inspection
![Data inspection](assets/figures/data-inspection.png)
_Session inspection table with status, timing, and record actions._

## 5. Desktop Packaging (Tauri)
Build Windows desktop artifacts:
```cmd
release\tauri\build_with_tauri.bat
```

Generated outputs:
- `release/windows/installers`
- `release/windows/portable`

## 6. Setup and Maintenance
Run:
```cmd
setup_and_maintenance.bat
```

Use this script for offline maintenance operations (for example DB initialization and cleanup tasks).

## 7. Database and Ollama Requirements
- Database schemas are not upgraded in place across this cleanup; recreate the schema (or local SQLite DB file) when upgrading.
- Runtime startup does not perform SQLite schema salvage/deletion.
- Ollama must support the chat-capable `/api/chat` API; `/api/generate` fallback behavior has been removed.

## 8. Documentation Map
- `assets/docs/USER_MANUAL.md`: end-user operation, journeys, key commands.
- `assets/docs/ARCHITECTURE.md`: system boundaries and data flow.
- `assets/docs/BACKGROUND_JOBS.md`: job lifecycle and semantics.
- `assets/docs/RUNTIME_MODES.md`: runtime profiles and packaging.
- `assets/docs/ERROR_HANDLING.md`: backend/frontend error strategy.
- `assets/docs/UI_STANDARDS.md`: frontend design standards.

## 9. License
Non-commercial use is covered by the Polyform Noncommercial License 1.0.0; commercial licensing is available separately. See `LICENSE`.

## Final validation and green-run requirement

After completing the restructure and all rewiring, run the full validation process in one uninterrupted pass until everything is green.

Do not stop to ask for confirmation between steps. If a check fails, diagnose the issue, fix it, and rerun the relevant checks until the project passes. Only stop if there is a hard blocker that cannot be resolved from the repository contents, and in that case report the exact blocker, the failing command, the error output, and the files likely involved.

Validation must include, where applicable:

* Run and verify all root-level `.bat` launcher scripts.
* Run and verify all root-level `.bat` maintenance scripts.
* Confirm every `.bat` script still works from the repository root after the restructure.
* Confirm runtime discovery and setup works with `runtimes/`.
* Run `uv sync` from `app/server/`.
* Confirm `.venv` and `uv.lock` are created or used only inside `app/server/`.
* Run backend linting, formatting checks, type checks, and tests if present.
* Run the FastAPI backend startup path and confirm imports, settings loading, resources loading, and path constants work.
* Generate or validate `app/shared/openapi.json` from the backend.
* Run frontend install, linting, type checks, tests, and production build if present.
* Confirm frontend API generation uses `app/shared/openapi.json` where applicable.
* Run all test suites under `app/tests/`.
* Validate Python imports after the move.
* Validate frontend imports after the move.
* Validate all script references under `app/scripts/`.
* Validate documentation and README references to the new layout.
* Validate Docker builds or compose files if present.
* Validate Tauri configuration if present.
* Run Tauri development checks if present.
* Run Tauri packaging or release build if present, and confirm output still lands in `release/` or the configured release output location.
* Confirm `.github/` remains unchanged.
* Confirm `assets/` and `release/` contents remain unchanged.
* Confirm `settings/` and `app/resources/` were extracted correctly and are read from their new locations.
* Confirm there are no stale references to the old root-level backend, client, tests, settings, or resources paths.

At the end, provide a concise validation report listing:

* Every command run.
* Whether it passed or failed.
* Any fixes made after failures.
* Any remaining warnings or limitations.
* Final green status once all required checks pass.

