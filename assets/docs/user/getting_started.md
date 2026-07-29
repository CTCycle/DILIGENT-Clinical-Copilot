# Getting Started
Last updated: 2026-07-29

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
- Default local UI URL:

```text
http://127.0.0.1:9847
```

- Default backend health URL:

```text
http://127.0.0.1:7690/api/health
```

- If `settings/.env` uses different ports, use the local configured values instead.

## Safety And Privacy Expectations
- Confirm local policy for protected health information, external model providers, and audit requirements before using real clinical material.
- Use extra caution with cloud model providers because clinical text may be sent to an external service.
- Do not enter real patient information into cloud-backed workflows unless that use is explicitly approved.
- For local-only evaluation, prefer a local provider such as Ollama and verify that the selected model is running locally.

## Application Layout
Main navigation sections:
- **DILI Agent**
- **Model Configurations**
- **Clinical Sessions**
- **Patient Timeline**
- **Data Inspection**

Typical journey:
1. Start the application.
2. Confirm backend health.
3. Configure the model provider.
4. Add and activate any required access key.
5. Open the DILI Agent.
6. Enter patient and clinical context.
7. Run the assessment.
8. Review and copy the generated report.
9. Review saved sessions if needed.
10. Inspect or update local data resources if needed.

## Start The Application
### Packaged Windows desktop

Open the portable EXE, or launch the application installed by the MSI. The Tauri shell verifies and extracts its embedded runtime, starts the packaged backend, waits for health readiness, and then shows the desktop window. Packaged startup does not use the development URLs below.

### Source/development mode

On Windows, open PowerShell in the repository root and run:

```powershell
.\start_on_windows.ps1
```

Expected source-mode result:
- backend process starts
- frontend process starts
- browser opens to the DILIGENT UI
- application loads without a blank page or connection error

If the browser does not open automatically, open:

```text
http://127.0.0.1:9847
```

If the UI shows a backend connection error, check:

```text
http://127.0.0.1:7690/api/health
```

## Confirm Local Configuration
Open:

```text
settings/.env
```

Confirm backend and frontend host or port values are correct for the local environment. If they change, restart the application so both processes use the same configuration.

For packaged startup failures, inspect `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log` and `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`. See [runtime troubleshooting](../runtime/troubleshooting.md) for extraction, health, and WebView2 checks.
