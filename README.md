# DILIGENT Clinical Copilot

## 1. Project Overview
DILIGENT Clinical Copilot guides clinicians through Drug-Induced Liver Injury (DILI) evaluations by combining a FastAPI backend with a React + TypeScript frontend (Vite). The UI collects anamnesis, medications, and ALT/ALP values; the backend parses drug mentions, classifies hepatotoxicity patterns, and runs LLM-powered consultations using either local Ollama models or approved cloud providers. Retrieval-Augmented Generation (RAG) built from LiverTox corpora grounds responses, and every clinical session is persisted to SQLite for auditability. REST endpoints mirror the UI flow, including Ollama model listing and pulling for on-prem deployments.

## 2. Installation

### 2.1 Windows (One-Click Setup - No Prerequisites Required)
Windows users get a **portable, zero-dependency install**. Launch `DILIGENT/start_on_windows.bat`; the script will:

1. Download and unpack **portable Python 3.12.10** locally (no global install).
2. Install **uv** locally for Python dependency management.
3. Download and unpack **portable Node.js v22.12.0** locally (no global install).
4. Install all Python dependencies from `pyproject.toml`.
5. Install frontend dependencies (if missing) and **build the React/Vite frontend**.
6. Launch the FastAPI backend and the Vite preview server.
7. Open your browser to the UI.

**First Run:** A few minutes while Python/Node.js/dependencies download and the UI builds. Artifacts live under `DILIGENT/resources/runtimes/` and are reused.

**Subsequent Runs:** Skip downloads/builds unless missing; startup takes seconds.

> **Note:** Everything stays inside the project folder (`DILIGENT/resources/runtimes/`), avoiding system-wide changes. You can move the folder and rerun the launcher without reinstalling.

### 2.2 macOS / Linux (Manual Setup)
**Prerequisites:**
- **Python 3.12**
- **Node.js 18+** and npm
- (Optional) Ollama running locally for on-prem models
- Cloud provider keys (OpenAI, Gemini) if you plan to use remote inference

**Setup Steps:**
1. Clone and enter the repo:
   ```bash
   git clone https://github.com/<your-org>/DILIGENT-Clinical-Copilot.git
   cd DILIGENT-Clinical-Copilot
   ```
2. Backend:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .
   ```
3. Frontend:
   ```bash
   cd DILIGENT/client
   npm install
   cd ..
   ```
4. Configuration:
   - Copy `DILIGENT/resources/templates/.env` to `DILIGENT/settings/.env`, then set API keys and any host/port overrides.
   - Optionally set `VITE_API_BASE_URL` in `DILIGENT/client/.env` if you are not relying on the Vite dev proxy.

## 3. How to use

### 3.1 Windows
Run `DILIGENT/start_on_windows.bat`.

- **First run:** Downloads runtimes, installs dependencies, builds the UI, then starts backend and preview server.
- **Later runs:** Reuse cached runtimes/builds and launch immediately.

The browser opens to `http://127.0.0.1:7861`. API root: `http://127.0.0.1:8000`. Docs: `http://127.0.0.1:8000/docs`.

### 3.2 macOS / Linux
Backend:
```bash
source .venv/bin/activate
uvicorn DILIGENT.server.app:app --host 0.0.0.0 --port 8000
```

Frontend (development with proxy to backend):
```bash
cd DILIGENT/client
FASTAPI_HOST=127.0.0.1 FASTAPI_PORT=8000 npm run dev  # UI at http://localhost:5173
```

Frontend (preview build):
```bash
cd DILIGENT/client
npm run build
npm run preview -- --host 0.0.0.0 --port 7861
```

UI: `http://localhost:5173` (dev) or `http://localhost:7861` (preview). Backend: `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

### 3.3 Using the Application
- Enter anamnesis, exam notes, current medications, and ALT/ALP values.
- Choose inference path (Ollama or cloud), select parsing/clinical models, and toggle RAG for LiverTox-backed retrieval.
- Run the clinical analysis to classify hepatotoxicity patterns, parse medications, and produce the Markdown consultation summary.
- Review and export the report; sessions and model choices persist to SQLite for later audit.

## 4. Setup and Maintenance
Run `DILIGENT/setup_and_maintenance.bat` for housekeeping:

- **Remove logs** — clear `.log` files under `DILIGENT/resources/logs`.
- **Uninstall app** — remove uv caches, embedded Python, portable Node.js, `node_modules`, `dist`, `.venv`, and `uv.lock` while keeping folder scaffolding.

## 5. Resources
`DILIGENT/resources` aggregates runtime assets, datasets, and templates:

- **database:** SQLite artifacts (`sqlite.db`) and exported evaluation data.
- **logs:** Backend and background-task logs for troubleshooting.
- **models:** Local LLM or embedding artifacts (when stored).
- **runtimes:** Portable Python/uv/Node.js downloaded by the Windows launcher.
- **templates:** Starter assets such as `.env` scaffold and `database_backup.db` snapshot for seeding.

## 6. Configuration
Backend settings live in `DILIGENT/settings/server_configurations.json` (FastAPI metadata, database mode, RAG, ingestion, LLM defaults). Runtime overrides and API keys are read from `DILIGENT/settings/.env`. Frontend builds can pin the backend via `DILIGENT/client/.env` (e.g., `VITE_API_BASE_URL`).

| Variable | Description |
|----------|-------------|
| FASTAPI_HOST | Backend host used by the Windows launcher; defined in `DILIGENT/settings/.env`; default `127.0.0.1`. |
| FASTAPI_PORT | Backend port for uvicorn; defined in `DILIGENT/settings/.env`; default `8000`. |
| UI_HOST | Host for the Vite preview server; defined in `DILIGENT/settings/.env`; default `127.0.0.1`. |
| UI_PORT | Port for the Vite preview server; defined in `DILIGENT/settings/.env`; default `7861`. |
| RELOAD | Enables uvicorn reload when `true`; defined in `DILIGENT/settings/.env`; default `false`. |
| OPENAI_API_KEY | Cloud inference key for OpenAI; defined in `DILIGENT/settings/.env`; default empty. |
| GEMINI_API_KEY | Cloud inference key for Gemini; defined in `DILIGENT/settings/.env`; default empty. |
| MPLBACKEND | Matplotlib backend for background tasks; defined in `DILIGENT/settings/.env`; default `Agg`. |
| VITE_API_BASE_URL | Backend base URL for the frontend when not using the dev proxy; defined in `DILIGENT/client/.env`; default inherits the dev proxy. |
| ollama_base_url | Ollama host for embeddings and local models; defined in `DILIGENT/settings/server_configurations.json`; default `http://localhost:11434`. |

## 7. License
Non-commercial use is covered by the Polyform Noncommercial License 1.0.0; commercial licensing is available separately. See `LICENSE` for full terms.
