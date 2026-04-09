# DILIGENT Clinical Copilot

## 1. Project Overview
DILIGENT Clinical Copilot supports clinicians during Drug-Induced Liver Injury (DILI) evaluations with a FastAPI backend and an Angular + TypeScript frontend. It collects anamnesis, medications, and lab values, then coordinates clinical analysis with optional RAG support and session persistence for review.

> **Work in Progress**: This project is under active development and may contain incomplete features or defects.

## 2. Quick Start

### 2.1 Windows (Recommended)
Run:
```cmd
DILIGENT\start_on_windows.bat
```

The launcher prepares local runtimes/dependencies and starts backend plus frontend.

### 2.2 macOS / Linux (Manual)
Prerequisites:
- Python 3.14+
- Node.js 18+ and npm

Backend:
```bash
python -m venv runtimes/.venv
source runtimes/.venv/bin/activate
pip install --upgrade pip
pip install -e .
uvicorn DILIGENT.server.app:app --host 127.0.0.1 --port 8000
```

Frontend:
```bash
cd DILIGENT/client
npm install
npm run build
npm run preview -- --host 127.0.0.1 --port 7861
```

## 3. Runtime Profiles
DILIGENT is configuration-first and uses one active runtime file: `DILIGENT/settings/.env`.

Switch to local profile:
```cmd
copy /Y DILIGENT\settings\.env.local.example DILIGENT\settings\.env
```

Switch to local Tauri profile:
```cmd
copy /Y DILIGENT\settings\.env.local.tauri.example DILIGENT\settings\.env
```

See `assets/docs/PACKAGING_AND_RUNTIME_MODES.md` for full runtime and packaging details.

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

#### Dashboard / Report Output
![Dashboard view](assets/figures/dashboard.png)

#### Model Configuration (Settings)
![Model settings](assets/figures/settings.png)

#### Local Model Catalog (List)
![Model list](assets/figures/models-list.png)

#### Provider Access Key (Detail Modal)
![Provider key detail](assets/figures/model-detail.png)

#### Data Inspection
![Data inspection](assets/figures/data-inspection.png)

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
DILIGENT\setup_and_maintenance.bat
```

Use this script for offline maintenance operations (for example DB initialization and cleanup tasks).

## 7. Documentation Map
- `assets/docs/USER_MANUAL.md`: end-user operation, journeys, key commands.
- `assets/docs/ARCHITECTURE.md`: system boundaries and data flow.
- `assets/docs/BACKGROUND_JOBS.md`: job lifecycle and semantics.
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`: runtime profiles and packaging.
- `assets/docs/ERROR_HANDLING.md`: backend/frontend error strategy.
- `assets/docs/UI_STANDARDS.md`: frontend design standards.

## 8. License
Non-commercial use is covered by the Polyform Noncommercial License 1.0.0; commercial licensing is available separately. See `LICENSE`.
