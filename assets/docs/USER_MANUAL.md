# DILIGENT User Manual

Last updated: 2026-05-29

## Purpose

DILIGENT is a local clinical copilot interface for Drug-Induced Liver Injury, DILI, review workflows. It helps users enter clinical context, configure model providers, inspect local clinical data, run DILI-oriented analysis, and review saved sessions.

DILIGENT is a decision-support application. It does not replace clinical judgment, institutional review, or a licensed clinician's responsibility for final decisions.

## Who should use this manual

Use this manual if you need to:

- Start DILIGENT locally.
- Configure model providers or API access keys.
- Run a DILI assessment.
- Review previous clinical sessions.
- Inspect local clinical datasets.
- Understand what to expect during the main application journey.

## Before you begin

Make sure the application has been installed and started according to `README.md`.

For the current `develop` branch configuration, the local browser UI normally opens at:

```text
http://127.0.0.1:9847
```

The backend health endpoint normally runs at:

```text
http://127.0.0.1:7690/api/health
```

If your local `settings/.env` uses different ports, use those local values instead.

## Safety and privacy expectations

Before using the application with real clinical material, confirm your local policies for protected health information, external model providers, and audit requirements.

Use particular caution with cloud model providers. When a cloud model is selected, clinical text may be sent to that external provider depending on the configured provider and backend behavior. Do not enter real patient information into cloud-backed workflows unless your organization has explicitly approved that use.

For local-only evaluation, use a local provider such as Ollama where applicable and verify that the selected model is running locally.

## Application layout

The application uses a left navigation sidebar. The current main sections are:

- **DILI Agent**, the primary assessment workflow.
- **Model Configurations**, model provider and model selection.
- **Clinical Sessions**, saved session review.
- **Patient Timeline**, timeline-oriented patient review.
- **Data Inspection**, local catalog and database inspection.

A typical user journey is:

1. Start the application.
2. Confirm the backend is healthy.
3. Configure the model provider.
4. Add or activate any required access key.
5. Open the DILI Agent.
6. Enter patient and clinical context.
7. Run the assessment.
8. Review and copy the generated report.
9. Review saved sessions if needed.
10. Inspect or update local data resources if needed.

## Step 1, start the application

On Windows, double-click:

```text
start_on_windows.bat
```

The launcher checks required runtimes, starts the backend, starts the frontend, waits for health checks, and opens the browser UI.

Expected result:

- A backend console window or process starts.
- A frontend console window or process starts.
- The browser opens to the DILIGENT UI.
- The application loads without a blank page or connection error.

If the browser does not open automatically, open:

```text
http://127.0.0.1:9847
```

If the UI shows a backend connection error, check:

```text
http://127.0.0.1:7690/api/health
```

A healthy backend should return a health response rather than a browser connection failure.

## Step 2, confirm configuration

Open the repository file:

```text
settings/.env
```

Confirm these values are appropriate for your local environment:

```text
BACKEND_HOST=127.0.0.1
BACKEND_PORT=7690
FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=9847
```

If you change these values, restart the application so the backend and frontend use the same configuration.

## Step 3, configure models

Open **Model Configurations** from the sidebar.

This page is where you select how the application should call a model during analysis.

You should expect to see controls for:

- Local provider configuration.
- Cloud provider configuration.
- Model selection.
- Saving or applying model settings.
- Provider access key management for providers that require credentials.

Recommended workflow:

1. Decide whether you want a local or cloud provider.
2. For local testing, select an Ollama-compatible model if available.
3. For cloud use, select the intended cloud provider and model.
4. Save or apply the configuration.
5. Add and activate an access key if the selected provider requires one.

## Step 4, manage access keys

Some providers require access keys. The access key workflow is available from **Model Configurations**.

The backend supports access key operations for:

- OpenAI.
- Gemini.
- Brave.

The application stores provider keys through the backend access key service. The UI should display fingerprints and metadata rather than exposing the full secret after saving.

Recommended workflow:

1. Open **Model Configurations**.
2. Choose the provider that needs a key.
3. Open the access key dialog or key management control.
4. Paste the provider key.
5. Save the key.
6. Activate the key that should be used.
7. Confirm the active key indicator is shown for that provider.

Expected result:

- The saved key appears as a key record or fingerprint.
- Only one key should be active for a provider at a time.
- The configured provider can be used by analysis workflows after activation.

Do not paste keys into screenshots, chat messages, issue reports, or shared logs.

## Step 5, open the DILI Agent

Open **DILI Agent** from the sidebar.

This is the main assessment page. It collects clinical context and sends a structured request to the backend analysis endpoint.

You should expect fields or controls for clinical case material such as:

- Patient identifier or case identifier.
- Patient age or demographic context.
- Suspected medication or exposure.
- Clinical history.
- Laboratory values.
- Symptoms.
- Timing information.
- Notes or free-text clinical context.
- Optional file upload, where available.
- Run or submit action.
- Generated assessment output.

The exact enabled fields may vary as the UI evolves, but the workflow is to provide enough structured and narrative context for a DILI-focused assessment.

## Step 6, enter clinical context

Before running an assessment, enter the most relevant clinical context.

Use clear and specific text. Prefer this style:

```text
Patient: 54-year-old adult
Suspected medication: ExampleDrug
Exposure timing: Started 21 days before liver enzyme rise
Labs: ALT 820, AST 610, ALP 160, total bilirubin 3.2
Symptoms: fatigue, jaundice
Relevant negatives: no known viral hepatitis in available records
Clinical question: assess whether this pattern is compatible with DILI
```

Avoid vague entries such as:

```text
Patient has liver issue. Check DILI.
```

The better the input context, the easier it is to interpret the generated report.

## Step 7, run the DILI assessment

After completing the input fields:

1. Review the entered information.
2. Confirm the selected model configuration is correct.
3. Select the run or submit action on the DILI Agent page.
4. Wait for the progress indicator or loading state to finish.
5. Do not refresh the browser while the assessment is running unless the application is unresponsive.

Expected result:

- The application sends the structured clinical input to the backend.
- The backend uses the configured provider and model.
- The UI displays a generated DILI assessment report or a clear error message.

If the run fails:

- Confirm the backend health endpoint is reachable.
- Confirm model provider configuration is saved.
- Confirm the selected provider has an active access key, if required.
- Confirm local Ollama is running if a local Ollama model is selected.
- Review the backend console for a structured error message.

## Step 8, review the generated report

Read the generated report carefully.

Treat the report as a clinical decision-support draft, not a final diagnosis.

Review for:

- Whether the timeline is consistent with the entered medication exposure.
- Whether the liver chemistry pattern is interpreted correctly.
- Whether confounders and alternative causes are mentioned.
- Whether the generated conclusion matches the supplied evidence.
- Whether the output includes unsupported assumptions.
- Whether any model-generated statements need human correction.

If the report is incomplete or clearly wrong:

1. Add missing clinical details.
2. Correct the input fields.
3. Re-run the assessment.
4. Compare the new output with the previous one.

## Step 9, copy or export output

Use available copy or export controls on the DILI Agent page to move the generated report into your local workflow.

Before copying output into clinical documentation, remove unsupported statements and verify all facts against the medical record.

Recommended review before reuse:

- Confirm dates.
- Confirm lab values and units.
- Confirm drug names and dosing details.
- Confirm patient-specific context.
- Remove placeholders.
- Remove irrelevant model text.
- Add human reviewer attribution according to local policy.

## Step 10, review saved clinical sessions

Open **Clinical Sessions** from the sidebar.

This page is intended for reviewing prior sessions and their generated output.

You should expect controls for:

- Viewing a list of sessions.
- Selecting a session.
- Reviewing session metadata.
- Reviewing generated content.
- Filtering or refreshing records where supported by the UI.

Recommended workflow:

1. Open **Clinical Sessions**.
2. Locate the target session by patient identifier, date, or available metadata.
3. Select the session.
4. Review the saved input and output.
5. Use available copy or inspection controls if you need to reuse content.

If a session is missing, confirm that the assessment completed successfully and that local persistence is initialized.

## Step 11, use Patient Timeline

Open **Patient Timeline** from the sidebar.

This page is intended for timeline-oriented review. Use it to understand event order, clinical sequence, or patient-specific chronology where data is available.

Recommended workflow:

1. Open **Patient Timeline**.
2. Select or locate the relevant patient or case.
3. Review timeline entries in chronological order.
4. Compare medication exposure dates against lab abnormalities and symptoms.
5. Use timeline context to refine the DILI Agent input if needed.

If the page does not show expected timeline data, confirm that the relevant data has been loaded into the local database or resource catalog.

## Step 12, inspect local data

Open **Data Inspection** from the sidebar.

This page is intended for viewing local data resources and database-backed records used by the application.

You should expect data-oriented controls such as:

- Resource or table selection.
- Refresh controls.
- Record counts or metadata.
- Table-style inspection.
- Search, filter, or pagination where supported.
- Embedding or resource update status where supported.

Recommended workflow:

1. Open **Data Inspection**.
2. Select the resource or dataset you want to inspect.
3. Refresh the view.
4. Confirm that expected records are present.
5. Use filters or page controls to inspect specific records.
6. If records are missing, run the setup or maintenance workflow that initializes or refreshes resources.

Do not edit local database files manually while the application is running.

## Step 13, update local resources

Some data resources or embeddings may need initialization or refresh through maintenance scripts.

Use:

```text
setup_and_maintenance.bat
```

Use the menu options for database initialization, dependency maintenance, or embedding updates.

Expected result:

- The script reports progress.
- Long-running jobs should show status.
- The application can read refreshed resources after completion.
- Restarting the application after maintenance is recommended.

## Step 14, troubleshoot common issues

### Browser page does not load

Check whether the frontend is running at:

```text
http://127.0.0.1:9847
```

If not, restart with:

```text
start_on_windows.bat
```

### Backend health check fails

Open:

```text
http://127.0.0.1:7690/api/health
```

If the endpoint is unreachable, restart the application and check the backend console output.

### Model call fails

Check:

- Provider is selected.
- Model is selected.
- Required access key is saved and active.
- Local Ollama service is running for local models.
- Network access is available for cloud providers.
- Provider quota or billing is available, if applicable.

### No saved sessions appear

Check:

- Database initialization has been run.
- The previous assessment completed successfully.
- The application was not interrupted during save.
- You are looking at the correct local repository and database.

### Data inspection is empty

Check:

- Local resources exist under the expected resource directories.
- Database initialization was completed.
- Embedding or catalog update jobs completed successfully.
- The backend was restarted after maintenance.

### Ports are already in use

The current default ports are:

```text
Backend: 7690
Frontend: 9847
```

Close conflicting processes or update `settings/.env`, then restart the application.

## Step 15, recommended end-to-end user journey

Use this workflow for a complete assessment:

1. Start DILIGENT with `start_on_windows.bat`.
2. Confirm the UI opens.
3. Confirm backend health at `/api/health`.
4. Open **Model Configurations**.
5. Select provider and model.
6. Add and activate any required access key.
7. Open **DILI Agent**.
8. Enter patient context, medication exposure, labs, symptoms, and timing.
9. Run the assessment.
10. Review the generated report.
11. Correct or enrich the input if needed.
12. Re-run if needed.
13. Copy the final reviewed output.
14. Open **Clinical Sessions** to confirm the session was saved.
15. Use **Patient Timeline** or **Data Inspection** for supporting review when needed.

## Good input checklist

Before running an assessment, try to include:

- Patient age or relevant demographic context.
- Suspected drug or exposure.
- Exposure start date or approximate timing.
- Symptom onset date or approximate timing.
- ALT, AST, ALP, bilirubin, and INR if available.
- Relevant baseline labs if available.
- Relevant comorbidities.
- Alcohol, viral, biliary, autoimmune, ischemic, or other alternative causes if known.
- Dechallenge or rechallenge information if available.
- Current clinical question.

## Output review checklist

Before relying on generated output:

- Verify all dates.
- Verify all lab values.
- Verify units and reference ranges.
- Verify medication names.
- Verify that alternative causes were considered.
- Verify that the model did not invent facts.
- Verify that the conclusion follows from the entered evidence.
- Add human clinical interpretation before use in any formal setting.

## What not to do

Do not:

- Treat generated output as a final diagnosis.
- Enter real patient data into cloud workflows without approval.
- Share provider access keys.
- Manually edit the local database while the app is running.
- Ignore backend or provider errors.
- Assume saved sessions exist unless the assessment completed successfully.
- Copy model output into clinical documentation without human review.

## Where to go next

For installation and developer setup, read:

```text
README.md
```

For deeper project documentation, read the documents under:

```text
assets/docs/
```