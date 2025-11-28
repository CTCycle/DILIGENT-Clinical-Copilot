# DILIGENT

## 1. Introduction
DILIGENT Clinical Copilot is an LLM-powered assistant that guides clinicians through Drug-Induced Liver Injury (DILI) investigations. The stack now pairs a FastAPI backend with a React + TypeScript frontend built on Vite. The UI captures patient data and lab values, calls the backend for hepatotoxicity analysis, and renders Markdown reports. The service can reason with local Ollama models and, when configured, use approved cloud providers for advanced reasoning or document parsing. Sessions are persisted to SQLite so that outcomes, model selections, and timing data remain auditable.

Core capabilities include:

- Automated hepatotoxicity pattern classification using ALT/ALP inputs.
- Drug name extraction and LiverTox-powered risk summaries from free-text notes.
- Configurable LLM providers (local Ollama models or authorised cloud APIs such as OpenAI or Gemini).
- React UI with runtime model controls, report export, and JSON diagnostics.
- REST endpoints for submitting patient sessions programmatically and querying the available Ollama models.

## 2. Installation

### Windows quick start
The Windows onboarding flow is automated. Double-click `DILIGENT/start_on_windows.bat`; the script downloads an embeddable Python, installs dependencies with `uv`, installs/builds the React frontend (requires Node.js/npm on PATH), and launches both servers. Antivirus tools may prompt when the script creates the embedded Python interpreter—add an exception if required.

### Manual setup (macOS, Linux, or custom Windows environments)

1. **Install prerequisites**
   - Python 3.12
   - Node.js 18+ with npm (for the React/Vite frontend)
   - [Ollama](https://ollama.com/) if you plan to run local models
   - Optional: access tokens for any cloud LLM provider you intend to enable
2. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/DILIGENT-Clinical-Copilot.git
   cd DILIGENT-Clinical-Copilot
   ```

3. **Install backend dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .
   ```

4. **Install frontend dependencies**

   ```bash
   cd DILIGENT/client
   npm install
   ```

5. **Configure environment variables**
   - Copy `DILIGENT/resources/templates/.env` to `DILIGENT/setup/settings/.env`.
   - Set API keys and, if needed, override `FASTAPI_HOST/FASTAPI_PORT` or `VITE_API_BASE_URL` for the UI-to-API bridge.

6. **Verify Ollama or cloud credentials**
   - Ensure the Ollama service is running locally if you rely on on-premise models.
   - Provide cloud keys (for example, OpenAI or Gemini) only if you intend to enable remote inference.

The database schema is created automatically the first time the application starts.

## 3. How to use

- **Windows**: run `DILIGENT/start_on_windows.bat`. It launches the FastAPI backend on the configured host/port (default `127.0.0.1:8000`) and serves the production React build via `npm run preview` (default `http://127.0.0.1:7861`). API docs remain at `http://localhost:8000/docs`.
- **macOS/Linux**: activate your virtual environment, start the backend, then run the frontend:

  ```bash
  # Backend
  uvicorn DILIGENT.server.app:app --host 0.0.0.0 --port 8000

  # Frontend (development with proxy to /api -> FastAPI)
  cd DILIGENT/client
  FASTAPI_HOST=127.0.0.1 FASTAPI_PORT=8000 npm run dev  # serves at http://localhost:5173

  # Or serve the production build
  npm run build
  npm run preview -- --host 0.0.0.0 --port 7861
  ```

The UI proxies `/api` requests to FastAPI during development; in production you can set `VITE_API_BASE_URL` to the backend root (for example, `http://127.0.0.1:8000`) when building.

Once the UI is open:

1. Populate the anamnesis narrative (include exam findings as needed), current medications, and lab values.
2. Choose the preferred provider (Ollama or cloud) and model family via the side configuration panel.
3. Trigger the analysis; the agent collates lab trends, parses medications, queries the LiverTox knowledge base, and returns a Markdown consultation summary.
4. Download or copy the generated report for inclusion in the patient record. All submissions are logged to the SQLite database for later review.

The REST API mirrors the UI workflow. Submit patient payloads to `POST /clinical` to generate reports programmatically, or call `GET /models/list` and `GET /models/pull?name=<model>` to inspect or download Ollama models before selecting them in the UI.

## 3.1 Setup and Maintenance
Execute `DILIGENT/setup_and_maintenance.bat` to open the maintenance console. Available actions include:

- **Update project** - pull the latest revision from GitHub using the bundled Git client.
- **Remove logs** - clear accumulated log files stored in `DILIGENT/resources/logs`.

### 3.2 Data updaters
The reference data used for toxicity assessments changes over time. Run these scripts periodically from the project root (use the same virtual environment as the app):

```bash
python -m DILIGENT.server.scripts.update_livertox_data     # Refresh LiverTox corpus
python -m DILIGENT.server.scripts.update_drugs_catalog     # Refresh drug catalog
python -m DILIGENT.server.scripts.update_RAG               # Regenerate RAG embeddings
```

Set `REDOWNLOAD` inside `update_livertox_data.py` if you want to reuse existing archives.

### 3.3 Resources
Clinical data, configuration templates, and assets live under `DILIGENT/resources/`:

- **database/** - contains the SQLite database (`sqlite.db`) with session histories plus any exported evaluation artefacts.
- **logs/** - runtime logs produced by the FastAPI workers and background tasks.
- **templates/** - reusable templates such as the `.env` scaffold and document layouts.

Environment variables reside in `DILIGENT/setup/settings/.env` for the backend and `.env` files consumed by Vite during frontend builds. Create or edit these files after copying the template so sensitive credentials remain outside version control.

| Variable             | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| FASTAPI_HOST         | Address the FastAPI server binds to (default `127.0.0.1`).                  |
| FASTAPI_PORT         | Port for the FastAPI service (default `8000`).                              |
| UI_HOST / UI_PORT    | Host/port for the Vite preview server (defaults `127.0.0.1:7861`).          |
| VITE_API_BASE_URL    | Override for the frontend-to-backend base URL (falls back to `/api`).       |
| OLLAMA_HOST          | Base URL where the Ollama runtime is reachable.                             |
| OPENAI_API_KEY       | API key for OpenAI-backed cloud inference (if applicable).                  |
| GEMINI_API_KEY       | API key for Gemini-backed cloud inference (if applicable).                  |
| MPLBACKEND           | Matplotlib backend used by background plotting tasks.                       |

## License

This project is provided under a **dual-license model**:

- **Non-commercial use**: Licensed under the [Polyform Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).  
- **Commercial use**: Requires a separate commercial license.  

For commercial licensing inquiries, please contact the owner of this repository.

