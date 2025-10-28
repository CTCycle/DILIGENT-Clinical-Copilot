# DILIGENT

## 1. Introduction
DILIGENT Clinical Copilot is an LLM-powered assistant that guides clinicians through Drug-Induced Liver Injury (DILI) investigations. The service combines a FastAPI backend with a NiceGUI front-end to collect patient data, analyse hepatotoxicity patterns, and produce structured consultation notes. It can reason with fully local language models served by Ollama and, when configured, fall back to approved cloud providers for advanced reasoning or document parsing. All sessions are stored in a local SQLite database so that outcomes, model selections, and timing data remain auditable.

Core capabilities include:

- Automated hepatotoxicity pattern classification using ALT/ALP inputs.
- Drug name extraction and LiverTox-powered risk summaries from free-text notes.
- Configurable LLM providers (local Ollama models or authorised cloud APIs).
- A clinician-oriented user interface that captures anamnesis (including embedded exam findings) and lab values.
- REST endpoints for submitting patient sessions programmatically and querying the available Ollama models.

## 2. Installation

### Windows quick start
The Windows onboarding flow is fully automated. Double-click `DILIGENT/start_on_windows.bat`; the script installs a portable Python runtime, sets up a virtual environment, and installs all dependencies before launching the application. Antivirus tools such as Avast may prompt when the script creates the embedded Python interpreter—add an exception if required.

### Manual setup (macOS, Linux, or custom Windows environments)

1. **Install prerequisites**
   - Python 3.12
   - [Ollama](https://ollama.com/) if you plan to run local models
   - Optional: access tokens for any cloud LLM provider you intend to enable
2. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/DILIGENT-Clinical-Copilot.git
   cd DILIGENT-Clinical-Copilot
   ```

3. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .
   ```

4. **Configure environment variables**
   - Copy `DILIGENT/resources/templates/.env` to `DILIGENT/setup/.env`.
   - Adjust hosts, ports, and API keys to match your deployment.

5. **Verify Ollama or cloud credentials**
   - Ensure the Ollama service is running locally if you rely on on-premise models.
   - Provide cloud keys (for example, OpenAI) only if you intend to enable remote inference.

The database schema is created automatically the first time the application starts.

## 3. How to use

- **Windows**: run `DILIGENT/start_on_windows.bat` to launch both the FastAPI backend and the UI in a single step.
- **macOS/Linux**: activate your virtual environment, then start the web stack:

  ```bash
  uvicorn DILIGENT.app.app:app --host 0.0.0.0 --port 8000
  ```

  Start the NiceGUI client in a separate shell to access the graphical interface:

  ```bash
  python -m DILIGENT.app.client.interface
  ```

  The interactive UI will be available at `http://127.0.0.1:7861`, while the API documentation can be viewed at `http://localhost:8000/docs`.

Once the UI is open:

1. Populate the anamnesis narrative (include exam findings as needed), current medications, and lab values.
2. Choose the preferred provider (Ollama or cloud) and model family via the side configuration panel.
3. Trigger the analysis; the agent collates lab trends, parses medications, queries the LiverTox knowledge base, and returns a Markdown consultation summary.
4. Download or copy the generated report for inclusion in the patient record. All submissions are logged to the SQLite database for later review.

The REST API mirrors the UI workflow. Submit patient payloads to `POST /session` to generate reports programmatically, or call `GET /models/list` and `GET /models/pull` to inspect or download Ollama models before selecting them in the UI.

## 3.1 Setup and Maintenance
Execute `DILIGENT/setup_and_maintenance.bat` to open the maintenance console. Available actions include:

- **Update project** – pull the latest revision from GitHub using the bundled Git client.
- **Remove logs** – clear accumulated log files stored in `DILIGENT/resources/logs`.

### 3.1.1 Database updater
The LiverTox database shipped with the application changes over time as new monographs are released. Run the updater script every week or two so the toxicity assessments stay aligned with the latest guidance.

1. Activate the same virtual environment you use for the main app.
2. From the project root execute:

   ```bash
   python -m DILIGENT.app.scripts.update_database
   ```

The script downloads the newest LiverTox content (set `REDOWNLOAD = False` inside `update_database.py` if you want to reuse existing archives), refreshes the SQLite tables, and writes a brief summary to the log. It can run independently from the FastAPI server, so schedule it with cron/Task Scheduler without taking the UI offline.

### 3.1 Resources
Clinical data, configuration templates, and assets live under `DILIGENT/resources/`:

- **database/** – contains the SQLite database (`database.db`) with session histories plus any exported evaluation artefacts.
- **logs/** – runtime logs produced by the FastAPI workers and background tasks.
- **templates/** – reusable templates such as the `.env` scaffold and document layouts.

Environment variables reside in `DILIGENT/setup/.env`. Create or edit this file after copying the template so sensitive credentials remain outside version control.

| Variable             | Description                                                     |
|----------------------|-----------------------------------------------------------------|
| FASTAPI_HOST         | Address the FastAPI server binds to (default `127.0.0.1`).      |
| FASTAPI_PORT         | Port for the FastAPI service (default `8000`).                  |
| OLLAMA_HOST          | Base URL where the Ollama runtime is reachable.                 |
| OPENAI_API_KEY       | API key for the configured cloud LLM provider (if applicable).  |
| MPLBACKEND           | Matplotlib backend used by background plotting tasks.           |


## 3.2 LangSmith observability
The LangChain components bundled with DILIGENT support LangSmith tracing. When the relevant environment variables are present, every structured LLM call—including Ollama and cloud chat requests—will emit trace data to LangSmith. To enable tracing:

1. Create a free LangSmith account at [https://smith.langchain.com](https://smith.langchain.com) and generate an API key from **Settings → API Keys**.
2. Add the following keys to your `.env` file (for example `DILIGENT/setup/.env`):

   ```text
   LANGSMITH_API_KEY="sk-..."
   LANGSMITH_TRACING_V2="true"
   LANGSMITH_PROJECT="DILIGENT"
   # Optional: point to a self-hosted deployment
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   ```

3. Export the same variables in your shell before starting the server:

   ```bash
   export LANGSMITH_API_KEY="sk-..."
   export LANGSMITH_TRACING_V2="true"
   export LANGSMITH_PROJECT="DILIGENT"
   # Optional: point to a self-hosted deployment
   # export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   ```

   On Windows Command Prompt use `set`, and on PowerShell use `$Env:` instead of `export`.
4. Launch the DILIGENT application and execute any workflow that calls the LLMs.
5. Open the LangSmith web UI and select the project configured in `LANGSMITH_PROJECT` to inspect traces, prompts, model choices, and any retry/parsing attempts.

## License

This project is provided under a **dual-license model**:

- **Non-commercial use**: Licensed under the [Polyform Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).  
- **Commercial use**: Requires a separate commercial license.  

For commercial licensing inquiries, please contact the owner of this repository.

