<p align="center"><img src="app/client/public/diligent-logo.png" alt="DILIGENT logo" width="96"></p>

<h1 align="center">DILIGENT Clinical Copilot</h1>

<p align="center">A local, single-user decision-support tool for structured review of possible drug-induced liver injury (DILI).</p>

<p align="center">
  <a href="https://github.com/CTCycle/DILIGENT-Clinical-Copilot/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/CTCycle/DILIGENT-Clinical-Copilot?style=flat-square"></a>
  <a href="https://github.com/CTCycle/DILIGENT-Clinical-Copilot/commits"><img alt="Last commit" src="https://img.shields.io/github/last-commit/CTCycle/DILIGENT-Clinical-Copilot?style=flat-square"></a>
  <a href="https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases"><img alt="Release" src="https://img.shields.io/github/v/release/CTCycle/DILIGENT-Clinical-Copilot?display_name=tag&style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/CTCycle/DILIGENT-Clinical-Copilot?style=flat-square"></a>
</p>

<p align="center">
  <a href="app/server/pyproject.toml"><img alt="Python 3.14+" src="https://img.shields.io/badge/Python-3.14%2B-3776AB?logo=python&logoColor=white&style=flat-square"></a>
  <a href="app/client/package.json"><img alt="Angular 21" src="https://img.shields.io/badge/Angular-21-DD0031?logo=angular&logoColor=white&style=flat-square"></a>
  <a href="https://www.microsoft.com/windows"><img alt="Platform: Windows" src="https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white&style=flat-square"></a>
</p>

## Overview

DILIGENT brings clinical history, medication exposure, symptoms, laboratory results, and timing information together for a reviewable DILI assessment. It supports local or cloud model providers, optional retrieval from prepared reference data, saved clinical sessions, timeline review, and report export.

> **Decision support only.** DILIGENT does not diagnose or replace clinical judgment. Verify the source information and generated statements before use. Cloud-backed assessments may send case text to the configured provider; confirm local policy before entering protected health information.

## Key Features

- Enter case context and resolve preflight feedback before running an assessment.
- Review the generated narrative alongside structured findings, medication matches, and uncertainty signals.
- Save sessions, compare revisions, review timelines, and export reports.
- Inspect local drug catalogs and reference resources, and configure model providers, roles, access keys, and retrieval options.

## Screenshots

<p align="center">
  <kbd><img src="assets/figures/01-dili-agent.png" alt="DILI Agent case entry" width="49%"></kbd>
  <kbd><img src="assets/figures/02-clinical-session-synthetic-report.png" alt="Synthetic clinical session report" width="49%"></kbd>
</p>
<p align="center"><em>DILI Agent case entry (fictional case, analysis not run) and a saved synthetic documentation example. The example does not represent a real patient, clinical assessment, or model output.</em></p>

<p align="center">
  <kbd><img src="assets/figures/03-rag-settings-autosave.png" alt="RAG settings with autosave status" width="49%"></kbd>
</p>
<p align="center"><em>Retrieval and ranking options in Settings; the autosave status is visible in the dialog.</em></p>

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.14+, FastAPI 0.141.1 |
| Web client | Angular 21 (TypeScript 5.9) |
| Database | SQLite |
| Desktop package | Tauri 2 (WebView2 required) |
| Minimum platform | Windows desktop |

## Installation and Usage

For the Windows desktop app, download the portable EXE or MSI installer from [GitHub Releases](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases). WebView2 is required. The package includes its application runtime.

To run from source on Windows, open PowerShell at the repository root and start the launcher:

```powershell
.\start_on_windows.ps1
```

Choose **Install dependencies**, then **Launch application**. The launcher starts one FastAPI process that serves both the built SPA and API. The browser opens at `http://127.0.0.1:7690` by default. Keep the launcher open; choose **EXIT** or press Ctrl+C there to stop the app. `.\start_on_windows.ps1 -Action Launch` supervises the server until Ctrl+C.

For frontend development, start FastAPI with Uvicorn reload in one terminal and use `npm run dev` in `app/client` in another. The Angular dev server runs at `http://127.0.0.1:9847` by default and proxies `/api` to FastAPI. The production launcher does not use the frontend preview server.

Manual (from source):

```powershell
cd app/server
uv sync
cd ../client
npm install
```

Then follow [Runtime startup](assets/docs/runtime/startup.md).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FASTAPI_HOST` | `127.0.0.1` | Backend host |
| `FASTAPI_PORT` | `7690` | Backend port |
| `UI_HOST` | `127.0.0.1` | Frontend host |
| `UI_PORT` | `9847` | Frontend port |
| `VITE_API_BASE_URL` | `/api` | Browser API base |
| `CORS_ALLOW_ORIGINS` | blank | Comma-separated allowed origins |
| `DILIGENT_DATA_PATH` | `data` | Source-mode data root |
| `DILIGENT_ACCESS_KEY_MATERIAL_FILE` | blank | Optional external key-material file |

Choose a local or cloud model provider and assign model roles in **Settings**. Source-mode environment settings are stored in `settings/.env`; the launcher creates it from `settings/.env.example` when needed. Keep provider credentials private.

Editing a saved report can clear its saved attachment and session metadata. Re-enter any required metadata after editing.

## Data Handling and Privacy

DILIGENT runs as a Windows desktop app or as a local web app in source mode, with application data stored in a local SQLite database. Model processing can use a local or cloud provider. When a cloud provider is configured, case text may be sent to that provider. Confirm provider, retrieval, and institutional privacy policies before entering protected health information.

## Documentation

- [Getting started](assets/docs/user/getting_started.md)
- [Model setup and access keys](assets/docs/user/model_setup.md)
- [DILI assessment workflow](assets/docs/user/dili_assessment_workflow.md)
- [Sessions, timeline, and data](assets/docs/user/sessions_timeline_and_data.md)
- [Troubleshooting](assets/docs/user/troubleshooting.md)

## Limitations

- Decision support only. DILIGENT does not diagnose or replace clinical judgment.
- Cloud-backed assessments may send case text to the configured provider; confirm local policy before entering protected health information.
- Windows desktop or local source-mode use; no mobile layout is provided.

## Contributing

Contributions are welcome. Open an issue or pull request with a clear description of the change and its motivation.

## License

DILIGENT is licensed under the GNU General Public License, version 3 or later. See [LICENSE](LICENSE) for the terms.
