# Getting Started
Last updated: 2026-10-21

## Purpose
DILIGENT is a local clinical copilot interface for Drug-Induced Liver Injury review workflows. It helps users enter clinical context, configure model providers, inspect local data, run DILI-oriented analysis, and review saved sessions.

DILIGENT is a decision-support application. It does not replace clinical judgment, institutional review, or clinician responsibility for final decisions.

## Intended Audience
Use this documentation if you need to:
- start DILIGENT locally
- configure model providers or API access keys
- run a DILI assessment
- review previous clinical sessions
- inspect local clinical datasets

## Before You Begin
- Choose either the Windows desktop package or source/development startup described in `README.md`.
- For desktop use, open `DILIGENT-v<version>-windows-x64-portable.exe` for no-install operation or install the matching `.msi`. Verify the matching `.sha256` file before distributing an artifact.
- The portable desktop app does not require Python, Node.js, Rust, npm, uv, or a source checkout. It stores user data under `%LOCALAPPDATA%\DILIGENT\data`.
- Source production URL (the backend serves the built SPA and API from this origin):

```text
http://127.0.0.1:7690
```

- Source-mode default health URL:

```text
http://127.0.0.1:7690/api/health
```

For frontend development, `npm run dev` serves the UI at `http://127.0.0.1:9847` and proxies `/api` to FastAPI.

- DILIGENT is designed for desktop browser windows at least `1100px` wide. If the window is narrower, enlarge it to continue.
- Windows tablets are supported when they behave like a normal desktop browser and can display the desktop interface at that width.

- If `settings/.env` uses different ports, use the local configured values instead. Packaged desktop uses a random localhost backend port recorded in `desktop-backend-ready.json`.

## Safety And Privacy Expectations
- Confirm local policy for protected health information, external model providers, and audit requirements before using real clinical material.
- Use extra caution with cloud model providers because clinical text may be sent to an external service.
- Do not enter real patient information into cloud-backed workflows unless that use is explicitly approved.
- For local-only evaluation, prefer a local provider such as Ollama and verify that the selected model is running locally.

## Application Layout

DILIGENT uses desktop-style navigation and information-dense workspaces designed for mouse and keyboard interaction. It does not provide a mobile phone or tablet layout.

Main navigation sections:
- **DILI Agent**
- **Clinical Sessions**, including the contextual **Timeline** tab
- **Data Inspection**
- **Settings**

Typical journey:
1. Start the application.
2. Confirm backend health.
3. Open **Settings → Models**, choose the provider, and assign the model roles.
4. Add and activate any required access key.
5. Open the DILI Agent.
6. Enter patient and clinical context.
7. Run the assessment.
8. Review and copy the generated report.
9. Open **Clinical Sessions** to review the saved report and its **Timeline** tab if chronology review is needed.
10. Inspect local data sources if needed.

Use **Settings → Database** to choose SQLite or PostgreSQL and enter the
connection details. A successful switch reloads settings from that database;
application settings are kept separately in each database and are not copied.

## Optional Help And Tips
The **Help** action in the existing header opens **Tips & Tricks**. It is a short, optional reference for model roles, RAG behavior, Clinical Session sections, and timeline review. Use **Show me** to restart the four-step DILI assessment walkthrough at any time. The walkthrough can be closed with its X button.

The first time the DILI Agent is opened with an empty assessment, a small **Get started with DILI Agent** callout may appear in the report area. **Show me** opens the walkthrough, **Open Settings** takes you to model setup, and the close button dismisses the callout. The callout will not return after it has been seen or dismissed in the same browser unless its content version changes.

Help state is stored in browser-local storage. It does not contain clinical input, reports, provider keys, or session data. Popovers and walkthroughs can be opened with the keyboard, closed with Escape, and follow the active light or dark theme.

## Start The Application
### Packaged Windows desktop

Open the portable EXE, or launch the application installed by the MSI. The Tauri shell verifies and extracts its embedded runtime, starts the packaged backend, waits for health readiness, and then shows the desktop window. Packaged startup does not use the development URLs below.

### Source production mode

On Windows, open PowerShell in the repository root and run:

```powershell
.\start_on_windows.ps1
```

Expected source-mode result:
- the visible launcher starts and owns one FastAPI process
- FastAPI serves the built frontend and API from one origin
- browser opens to `http://127.0.0.1:7690` after health and asset checks pass
- application loads without a blank page or connection error
- backend output is saved under `runtimes/cache/logs/`
- choose **EXIT** or press Ctrl+C in the launcher to request graceful shutdown; if needed, the launcher stops only its owned process tree
- `-Action Launch` supervises the backend until Ctrl+C or process exit

If the browser does not open automatically, open:

```text
http://127.0.0.1:7690
```

If the UI shows a backend connection error, check:

```text
http://127.0.0.1:7690/api/health
```

### Source development mode

Start FastAPI with Uvicorn reload in one terminal and run `npm run dev` from
`app/client` in another. Open the Angular development URL, normally
`http://127.0.0.1:9847`. See [Startup](../runtime/startup.md) for the commands.

## Confirm Local Configuration
Open:

```text
settings/.env
```

Confirm `FASTAPI_HOST` and `FASTAPI_PORT` for production source mode. `UI_HOST`
and `UI_PORT` configure the Angular development server only. If settings change,
restart the relevant process.

For packaged startup failures, inspect `%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log` and `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`. See [runtime troubleshooting](../runtime/troubleshooting.md) for extraction, health, and WebView2 checks.
