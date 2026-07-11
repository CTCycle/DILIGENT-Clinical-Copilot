# DILIGENT Clinical Copilot

[![Release](https://img.shields.io/github/v/release/CTCycle/DILIGENT-Clinical-Copilot?display_name=tag)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases) [![Python](https://img.shields.io/badge/python-%3E%3D3.14-blue?logo=python&logoColor=white)](./app/server/pyproject.toml) [![Angular](https://img.shields.io/badge/angular-%5E21.2.0-DD0031?logo=angular&logoColor=white)](./app/client/package.json) [![License](https://img.shields.io/badge/license-Polyform%20Noncommercial%201.0.0-lightgrey)](./LICENSE) [![CI](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml?query=branch%3Adevelop)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)

## 1. Project Overview
DILIGENT Clinical Copilot supports clinicians during Drug-Induced Liver Injury (DILI) evaluations with a FastAPI backend and an Angular + TypeScript frontend. It collects anamnesis, medications, and lab values, then coordinates clinical analysis with optional RAG support and session persistence for review.

![DILIGENT flow schema](assets/figures/diligent-flow.png)
_Conceptual flow from case intake through guided DILI analysis, session recording, and review._

![DILIGENT Clinical Copilot overview](assets/figures/clinical-copilot-overview.png)
_DILIGENT Clinical Copilot overview._

## 2. Quick Start

### 2.1 Windows (Recommended)
Run:
```powershell
.\start_on_windows.ps1
```

The launcher prepares local runtimes and dependencies, then starts the backend and frontend on the configured local ports.

### 2.2 macOS / Linux (Manual)
Prerequisites:
- Python 3.14+
- Node.js 18+ and npm

Backend:
```bash
cd app/server
python -m pip install -e ".[test]"
uvicorn app:app --host 127.0.0.1 --port 7690
```

Frontend:
```bash
cd app/client
npm install
npm run build
npm run preview -- --host 127.0.0.1 --port 9847 --strictPort
```

Default local endpoints:
- UI: `http://127.0.0.1:9847`
- API health: `http://127.0.0.1:7690/api/health`

## 3. Runtime Profiles
DILIGENT is configuration-first and uses one active runtime file: `settings/.env`.

The supported runtime profile is explicit local single-user operation. Network production deployment without authentication is unsupported; runtime validation rejects non-local deployment modes until access-control work is added.

On first launch, `start_on_windows.ps1` creates `settings/.env` from `settings/.env.example` when needed. Edit the active file to change local ports, dependency options, or backend log visibility.

See `assets/docs/runtime/modes.md` for full local runtime details.

Clinical job execution is process-local. Persisted clinical sessions remain durable, but in-memory job ids are not durable across backend restarts.

## 4. Using the Application
Typical workflow:
1. Enter anamnesis, exam notes, medications, and lab values.
2. Choose model/provider settings and optionally enable RAG/web search.
3. Run analysis and review the generated report.
4. Use Data Inspection to explore current knowledge base.
5. Explore past sessions and apply manual report edits.

Detailed user journeys and feature guidance are documented in `assets/docs/user/getting_started.md`, `assets/docs/user/dili_assessment_workflow.md`, and `assets/docs/user/sessions_timeline_and_data.md`.

### Screenshots

#### Dashboard / Report Output
![Dashboard view](assets/figures/dashboard.png)
_Clinical intake workspace with structured patient input, visit metadata, and report actions._

#### Sessions overview
![Session dashboard](assets/figures/session-inspection.png)
_Clinical Sessions workspace with the persisted review layout shown and sensitive case content blurred._

#### Model Configuration (Settings)
![Model settings](assets/figures/model-detail.png)
_Runtime source, local model catalog, and active reasoning pipeline settings._

#### Data Inspection
![Data inspection](assets/figures/data-inspection.png)
_Catalog inspection view for curated drug records, update status, and maintenance actions._

## 5. Setup and Maintenance
Run:
```powershell
.\start_on_windows.ps1
```

Use menu options 2 through 7 for dependency maintenance, database initialization, tests, logs, caches, and uninstall cleanup.

### 5.1 Regression Validation Shortcuts

From repository root:

```cmd
app\tests\run_tests.bat modelconfig
app\tests\run_tests.bat modelconfigfull
```

- `modelconfig`: validated regression slice (model-config unit + focused model-config/app-flow e2e checks, including conflict-feedback handling)
- `modelconfigfull`: model-config unit + full `test_app_flow.py` + `test_model_config_api.py`
  - If `uv --with pytest-playwright` cannot access package indexes on first use, run the PowerShell runner directly after cache warmup.

These are available through `run_tests.bat`:

```cmd
app\tests\run_tests.bat modelconfig
app\tests\run_tests.bat modelconfigfull
```

## 6. Database and Ollama Requirements
- Database schemas are not upgraded in place across this cleanup; recreate the schema (or local SQLite DB file) when upgrading.
- Runtime startup does not perform SQLite schema salvage/deletion.
- Ollama must support the chat-capable `/api/chat` API; `/api/generate` fallback behavior has been removed.

## 7. Documentation Map
- `assets/docs/project_index.md`: entry point for the documentation tree.
- `assets/docs/architecture/system_overview.md`: repository layout and system boundaries.
- `assets/docs/architecture/background_jobs.md`: job lifecycle and semantics.
- `assets/docs/runtime/modes.md`: supported local runtime profile.
- `assets/docs/coding/error_handling.md`: backend and frontend error strategy.
- `assets/docs/ui/components_and_patterns.md`: frontend structure and interface patterns.

## 8. Development Status

This project is under active development and may contain incomplete features or defects. Tagged releases are stable for local evaluation.

## 9. License
Non-commercial use is covered by the Polyform Noncommercial License 1.0.0; commercial licensing is available separately. See `LICENSE`.



