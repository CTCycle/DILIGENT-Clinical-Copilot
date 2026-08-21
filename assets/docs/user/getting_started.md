# Getting Started
Last updated: 2026-08-21

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
- Source-mode default UI URL:

```text
http://127.0.0.1:9847
```

- Source-mode default backend health URL:

```text
http://127.0.0.1:7690/api/health
```

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

## Optional Help And Tips
The **Help** action in the existing header opens **Tips & Tricks**. It is a short, optional reference for model roles, RAG behavior, Clinical Session sections, and timeline review. Use **Show me** to restart the four-step DILI assessment walkthrough at any time. The walkthrough can be closed with its X button.

The first time the DILI Agent is opened with an empty assessment, a small **Get started with DILI Agent** callout may appear in the report area. **Show me** opens the walkthrough, **Open Configurations** takes you to model setup, and the close button dismisses the callout. The callout will not return after it has been seen or dismissed in the same browser unless its content version changes.

Help state is stored in browser-local storage. It does not contain clinical input, reports, provider keys, or session data. Popovers and walkthroughs can be opened with the keyboard, closed with Escape, and follow the active light or dark theme.

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
