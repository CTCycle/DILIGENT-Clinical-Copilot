# DILIGENT Clinical Copilot User Manual

Last updated: 2026-05-18

This manual describes practical use of DILIGENT Clinical Copilot for clinical DILI assessment workflows.

## 1. Intended users

- Clinicians or clinical researchers performing DILI assessments.
- Technical operators configuring local or cloud model providers.
- Reviewers auditing prior sessions and data-update history.

## 2. Accessing the application

Typical local startup (Windows):
```cmd
DILIGENT\start_on_windows.bat
```

Typical local URLs:
- Frontend UI: `http://127.0.0.1:9847`
- Backend API: `http://127.0.0.1:7690`

Desktop packaged mode is available through Tauri builds. See `assets/docs/RUNTIME_MODES.md`.

## 3. Primary user journeys

### Journey A: Run a clinical DILI assessment

1. Open the main DILI workflow page.
2. Enter patient/session context (for example patient name and visit date).
3. Enter anamnesis and exam narrative.
4. Add drug exposure information with timing details.
5. Enter laboratory findings (ALT/ALP and related context).
6. Choose model/provider settings and optionally enable RAG/web search.
7. Start analysis and wait for job completion.
8. Review the generated report, hepatotoxicity pattern, and per-drug RUCAM outputs.
9. Export or copy outputs if needed for downstream reporting.

### Journey B: Configure models and provider keys

1. Open Model Configuration.
2. Choose inference path (local/Ollama vs cloud provider).
3. Enter and activate provider API keys where needed.
4. Select parser and clinical model choices.
5. Save/update configuration and run a test analysis.

### Journey C: Maintain reference data and RAG resources

1. Open Data Inspection.
2. Run RxNav update wizard when medication datasets need refresh.
3. Run LiverTox update wizard when source corpus requires refresh.
4. Run RAG update wizard to rebuild embeddings/vector data.
5. Review job status and resulting document/vector-store summaries.

### Journey D: Audit historical sessions

1. Open Clinical Sessions.
2. Filter/search historical sessions.
3. Open a session to inspect the AI-generated preview, detected drugs, and any recorded revision audit.
4. Manually edit the persisted clinical text when the source material needs correction.
5. Add session metadata, including document and image references.
6. Generate or open the patient timeline from the session workspace.
7. Use findings for quality review, reproducibility checks, and documentation.

### Journey E: Revise a clinical session

1. Open Clinical Sessions and select the session to revise.
2. Use the editor to save any manual text corrections.
3. Optionally paste or select the specific text section that should receive model focus.
4. Add a revision instruction when the model should scrutinize a specific concern in the selected section while still reprocessing the full session text.
5. Open Revision Mode and override model settings when the second pass should use a different model.
6. Start the revision job and monitor progress.
7. Review the newly versioned session result after the revision pipeline completes.
8. Check the revision audit for parser cross-validation, focused selection, revision instruction, newly identified drugs, missing previous drugs, and the conclusion action.

## 4. Primary commands (operator workflow)

Local launch:
```cmd
DILIGENT\start_on_windows.bat
```

Offline setup and maintenance menu:
```cmd
DILIGENT\setup_and_maintenance.bat
```

Switch runtime profile to local mode:
```cmd
copy /Y DILIGENT\settings\.env.local.example DILIGENT\settings\.env
```

Switch runtime profile to local Tauri mode:
```cmd
copy /Y DILIGENT\settings\.env.local.tauri.example DILIGENT\settings\.env
```

Build desktop artifacts (Windows):
```cmd
release\tauri\build_with_tauri.bat
```

## 5. Usage patterns and best practices

- Clinical input textbox accepts plain text only.
- Preferred template:
  - `## Anamnesis`
  - `## Therapy`
  - `## Laboratory history`
- Accepted deterministic heading variants:
  - Anamnesis: `Clinical history`, `Medical history`, `Patient history`, `Storia clinica`
  - Therapy: `Current medications`, `Drug therapy`, `Terapia farmacologica`, `Farmaci`
  - Laboratory history: `Laboratory tests`, `Blood tests`, `Esami di laboratorio`, `Esami ematochimici`
- Sections must be clearly separated and titled.
- The backend does not infer missing sections from paragraph prose.
- Exact preferred headings are recommended but not mandatory.
- Numbered headings and Markdown headings are supported.
- Enter complete medication timing details before running analysis.
- Provide structured, clinically specific lab context to improve pattern derivation.
- Treat missing-core-field warnings as blockers and resolve before re-running.
- Use one provider/model configuration per evaluation session for reproducibility.
- Re-run RAG updates after major LiverTox source refreshes.
- Validate model/provider configuration with a small test case before production use.

## 6. Key features

- Guided DILI analysis workflow from input collection to report generation.
- Deterministic per-drug RUCAM scoring support.
- Optional RAG enrichment from local LiverTox resources.
- Optional web-search augmentation when enabled by configuration.
- Background job lifecycle with status polling and cancellation.
- Encrypted-at-rest provider key storage and activation controls.
- Clinical Sessions workspace for persisted sessions, metadata, revision, and patient timelines.
- Inspection views for datasets and vector-store assets.
- Local-first execution with optional desktop packaging.

## 7. Troubleshooting quick checks

- If UI cannot reach backend, verify `FASTAPI_HOST`/`FASTAPI_PORT` and `UI_HOST`/`UI_PORT` in `DILIGENT/settings/.env`.
- If analysis quality is poor, verify model selection and provider-key activation.
- If RAG appears stale, run LiverTox update followed by RAG update.
- If startup fails on Windows, re-run launcher and confirm bundled runtimes exist under `runtimes/`.

## 8. Related documentation

- `README.md` for installation and setup overview.
- `assets/docs/ARCHITECTURE.md` for module boundaries and data flow.
- `assets/docs/BACKGROUND_JOBS.md` for job behavior.
- `assets/docs/RUNTIME_MODES.md` for local/desktop runtime modes.

- Report output is formatted in-app; downloads remain raw Markdown (.md).
- Copy action includes formatted HTML with plain-text fallback.
- Expanded report opens a full-page reading view.
- RUCAM may be unavailable when criteria-level evidence is insufficient; explicit trusted-source RUCAM scores are used directly when present.
