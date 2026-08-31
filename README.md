# DILIGENT Clinical Copilot
Last updated: 2026-08-31

[![Release](https://img.shields.io/github/v/release/CTCycle/DILIGENT-Clinical-Copilot?display_name=tag)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases) [![Python](https://img.shields.io/badge/python-%3E%3D3.14-blue?logo=python&logoColor=white)](./app/server/pyproject.toml) [![Angular](https://img.shields.io/badge/angular-%5E21.2.0-DD0031?logo=angular&logoColor=white)](./app/client/package.json) [![License](https://img.shields.io/badge/license-GNU%20GPL%20v3-lightgrey)](./LICENSE) [![CI](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml?query=branch%3Adevelop)
[![CTCycle Portfolio](https://img.shields.io/badge/CTCycle-Portfolio-58a6ff?style=flat-square)](https://ctcycle.github.io/CTCycle/)

DILIGENT Clinical Copilot is a local, single-user decision-support application for structured Drug-Induced Liver Injury (DILI) review. It helps you bring clinical history, medication exposure, symptoms, laboratory results, and timing information together, then use a configured local or cloud model to create a reviewable assessment draft.

The application is designed to make a complex review more organized and easier to inspect. It is not a diagnostic device and does not replace clinical judgment, institutional review, or clinician responsibility. A qualified clinician must check every input, generated statement, conclusion, and use of exported text.

## How DILIGENT works

DILIGENT follows a clear review flow:

1. You enter the clinical context, exposure history, symptoms, laboratory values, and timing information.
2. The application performs rule-based pre-flight checks so missing or inconsistent information can be addressed before analysis begins.
3. The configured local or cloud model organizes the supplied information and produces a DILI-oriented draft. Optional retrieval support can add prepared reference evidence when it is available.
4. The application keeps the narrative report alongside structured details such as timelines, laboratory-pattern information, competing-cause states, and review warnings.
5. You verify the result, refine the case when needed, and save, compare, copy, or export the reviewed work.

The application deliberately separates model-generated suggestions from saved evidence and human decisions. A polished paragraph is still a draft until a clinician has checked it against the source record.

![DILIGENT assessment flow (v3.3.0)](assets/figures/diligent-flow-v3.2.0.png)
_The assessment flow moves from structured input through pre-flight checks, optional evidence support, a generated draft, and human review._

![DILIGENT Clinical Copilot system overview (v3.3.0)](assets/figures/clinical-copilot-overview-v3.2.0.png)
_The application brings the user interface, local data, configured model services, and saved review sessions into one local workspace._

### Principles behind the assessment

DILI review depends on several kinds of evidence rather than one isolated value:

- **Timing and exposure:** The order of medication starts, dose changes, symptoms, laboratory abnormalities, stopping a medicine, and any later restart can support or weaken a possible relationship.
- **Liver-chemistry pattern:** ALT and ALP are considered relative to their laboratory-specific upper limits of normal. In other words, each result is first expressed as a multiple of its own upper limit, and those normalized elevations can be compared using the commonly used R ratio. This helps describe a predominantly hepatocellular, cholestatic, or mixed pattern; it does not identify the cause by itself.
- **Causality frameworks:** Structured evidence inspired by RUCAM, Hy's Law, and DILIN-style reasoning helps organize timing, severity, alternative causes, and supporting or missing evidence. RUCAM is a structured causality aid, while Hy's Law is a warning pattern involving liver-cell injury and jaundice when another likely cause has not been established. These frameworks support review; they are not automatic proof that a drug caused liver injury.
- **Uncertainty:** Approximate dates, incomplete source evidence, ambiguous drug matches, and unavailable retrieval are shown as limitations or review signals instead of being silently converted into facts.

### Technology at a glance

At a high level, DILIGENT combines an Angular user interface, a Python/FastAPI local application service, local persistence, optional evidence resources, and either a local Ollama model or a supported cloud provider. The Windows desktop build is packaged with Tauri. These components are coordinated for you when you use the packaged desktop application or the official Windows launcher.

## What DILIGENT helps with

- Capture a DILI-focused clinical narrative, medication exposure, laboratory data, symptoms, and timing information in one structured request.
- Check the request before analysis so incomplete sections and blocking input issues can be corrected early.
- Produce a DILI-oriented decision-support draft through the selected model configuration, with optional retrieval-supported evidence when available.
- Review the reasoning alongside exposure and laboratory timelines, liver-pattern information, competing-cause states, and drug-match review flags.
- Save completed work as clinical sessions, edit report text directly, compare official revisions, and record human clinical-review status.
- Inspect locally available datasets and resource status through **Data Inspection**.
- Review patient chronology in **Patient Timeline** and use it to refine later assessments.

## v3.3.0 release highlights

The v3.3.0 release provides Windows x64 desktop packages in two forms:

- a portable executable for no-install use
- an MSI installer for an installed application and shortcut

The packaged application contains the runtime it needs, starts its local services automatically, and keeps user data separate from the downloaded package. A matching SHA-256 file is published so a downloaded package can be checked before use or distribution.

## Before you use it

Use DILIGENT only in an environment approved for the information you plan to enter. In particular:

- Confirm your organisation's policy before entering protected health information.
- A cloud-backed run may send clinical text to an external provider. Do not use real patient information with a cloud provider unless that transfer is explicitly authorised.
- Check the selected provider, model roles, active access key, and retrieval choice before each assessment.
- Verify drug names, dates, doses, laboratory values, units, alternative causes, and conclusions before relying on a report.
- Use a desktop-width window. The interface is information-dense and is designed for a window about 1100 pixels wide or wider; it does not provide a mobile-phone layout.

DILIGENT is intended for local, single-user operation. Unauthenticated network deployment is not supported.

## Get the source release

For source-based use, download `DILIGENT-Clinical-Copilot-3.3.0.zip` from the [v3.3.0 GitHub release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.3.0), extract it, and follow the source-startup instructions below.

The source archive is intended for local launcher-based operation. It is separate from the Windows desktop packages described next.

## Windows desktop release

Download the package you need from the [GitHub Releases page](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases):

- **Portable EXE:** run it directly without an installation step.
- **MSI installer:** install the application and use the resulting shortcut.

The portable executable and MSI do not require a separate Python, Node.js, Rust, npm, uv, or source checkout on the target machine. On first launch, the application may take a little longer while it prepares its local runtime. Windows WebView2 is also required; an MSI may need network access to obtain it on a system where it is not already installed.

Windows desktop startup does not use the source-mode development addresses shown below. User settings, sessions, logs, models, source documents, evidence resources, exports, and access-key material are kept in DILIGENT's local application-data area.

## Start the application

### Windows

#### Packaged desktop mode

Open the downloaded portable EXE, or launch the application installed by the MSI. The desktop window appears after the package has prepared its local services and completed its health check. No PowerShell launcher or development server is needed.

#### Source/launcher mode

Open PowerShell in the extracted repository folder and run:

```powershell
.\start_on_windows.ps1
```

On a fresh checkout:

1. Choose **Install dependencies** from the launcher menu (option 4). This prepares the local runtimes, dependencies, database, and frontend output.
2. Choose **Launch application** (option 1).

For later launches, you can run the launcher directly with its launch action:

```powershell
.\start_on_windows.ps1 -Action Launch
```

The launcher can recover missing or unusable setup during launch. When source mode starts successfully, the interface is normally available at [http://127.0.0.1:9847](http://127.0.0.1:9847). If the page says that the local service is unavailable, check [http://127.0.0.1:7690/api/health](http://127.0.0.1:7690/api/health), then restart the launcher if necessary.

### macOS and Linux

There is no packaged desktop build or one-step launcher for macOS and Linux in the current release. Source use requires Python 3.14, Node.js, npm, and uv. Keep the two terminals open while using the application.

In the first terminal, from the repository root:

```bash
cd app/server
uv sync --locked --all-extras
uvicorn app:app --host 127.0.0.1 --port 7690
```

In a second terminal:

```bash
cd app/client
npm install
npm run build
npm run preview -- --host 127.0.0.1 --port 9847 --strictPort
```

Open the interface at [http://127.0.0.1:9847](http://127.0.0.1:9847). The local service health check is available at [http://127.0.0.1:7690/api/health](http://127.0.0.1:7690/api/health). If you use different local ports, use the same values consistently for both parts of the application and restart them after changing the setup.

## Configure models and access keys

Open **Configurations** from the sidebar before starting an assessment. This is where you choose how DILIGENT should use models.

1. Choose a **local** or **cloud** provider.
2. Choose compatible models for the four roles used by the application: clinical analysis, text extraction, report revision, and timeline generation. One model may be used for more than one role.
3. Save the configuration.
4. If the provider needs credentials, add the appropriate access key and explicitly activate the key you want to use.
5. Confirm that the selected setup is shown as valid before returning to **DILI Agent**.

For local use, Ollama must be running with a compatible chat-capable model installed. For cloud use, the active provider key is used when the model list is refreshed. After a key is saved, the interface shows a fingerprint and metadata rather than the full secret. Do not paste access keys into screenshots, chat messages, issue reports, or shared logs.

Opening **Configurations** uses the last saved model list. Use **Refresh** when you explicitly want a new provider listing. If a refresh fails, the last valid list may remain visible so that you can understand the previous setup. Changing between local and cloud modes requires compatible role selections; incompatible choices are cleared rather than silently reused.

![Configurations](assets/figures/models-configuration.png)
_Configurations brings provider selection, model roles, retrieval settings, and access-key management into one workspace._

## Run a DILI assessment

Open **DILI Agent** and enter a concise but complete case description. Useful information normally includes:

- a case or patient identifier appropriate to your local policy
- age or relevant demographic context
- suspected drug or exposure, dose, and timing
- symptom onset and relevant history
- liver laboratory values with units and reference limits when available
- concomitant medication and relevant competing causes
- dechallenge or rechallenge information when available
- the clinical question you want the assessment to address

Use clear, specific wording. For example, a useful fictional case summary could include:

> **Suspected medication:** ExampleDrug<br>
> **Exposure timing:** Started 21 days before the enzyme rise<br>
> **Symptoms:** Fatigue and jaundice<br>
> **Laboratory values:** ALT 820 U/L (upper limit 50), AST 610 U/L, ALP 160 U/L (upper limit 120), bilirubin 3.2 mg/dL<br>
> **Relevant context:** No known viral hepatitis in the available record<br>
> **Clinical question:** Is this pattern compatible with DILI?

Then:

1. Select the configured provider or providers and choose whether to use retrieval-supported evidence for this assessment.
2. Start the run.
3. Read the pre-flight feedback. Correct blocking items before proceeding. Continue past warnings only when you understand the limitation they describe.
4. Wait for the progress indicator. Some assessments take time. You may navigate away, refresh, or return to the DILI Agent while a run is still active.
5. Review the completed report and its structured evidence before copying or exporting anything.

**Run without RAG** (retrieval-augmented evidence) affects only the current assessment. It does not change the saved retrieval preference for later work. If evidence preparation is unavailable or takes too long, DILIGENT may continue without that prepared evidence and will report the limitation for review.

## Review the result

The generated report is a decision-support draft, not a final diagnosis. Compare it with the source record, especially:

- exposure chronology, dose changes, stopping or restarting medication, and timing of symptoms
- laboratory values, units, reference limits, and the described liver-chemistry pattern
- alternative and competing causes
- causal statements and the evidence supporting them
- matched-drug identity and any ambiguous, missing, or unvalidated match warnings
- missing data, assumptions, placeholders, and unsupported claims

The report is accompanied by structured information that can help you check the narrative, including longitudinal exposure and laboratory events, Hy's Law status and rationale, competing-cause states, supportive RUCAM evidence, and DILIN-style causality reasoning. A drug reference or monograph excerpt indicates that reference text was available; it is not automatic proof that the drug identity or clinical conclusion is correct.

If the report is incomplete or incorrect:

1. Add the missing clinical details or correct the input.
2. Run the assessment again.
3. Compare the new result with the previous saved work.
4. Record the human review and attribution required by local policy before formal use.

![Dashboard view](assets/figures/dashboard.png)
![Dashboard view](assets/figures/dashboard-dark-theme.png)
_The DILI Agent workspace combines structured case input with assessment actions and report output._

## Work with saved sessions

Open **Clinical Sessions** to find persisted work by identifier, date, or available metadata. Select a session to review its content, evidence, and revision history.

- **Text Editor** lets you make direct manual changes to report text. **Rendered** shows a read-only preview of the same draft.
- **LLM Revision** creates a new draft revision and leaves the previous official version unchanged.
- **Official Version History** and **Manual Edit History** show different kinds of change and should be read separately.
- **Version Comparison** helps compare persisted official versions and the changes they contain.
- **Human Clinical Review** records whether a revision is under review, approved by a human, or rejected. This status is separate from automated or model-quality checks.

Manual report edits do not create a new official version. Before approving or reusing a model-assisted revision, review its evidence, revision history, and provenance.

![Session dashboard](assets/figures/session-inspection.png)
_Clinical Sessions provides a saved review workspace for reports, evidence, revisions, and human review._

## Use the timeline and data inspection views

### Patient Timeline

Open **Patient Timeline** to review event order and clinical chronology. Generate a timeline when needed, or reopen a previously saved timeline instead of regenerating it. Compare medication exposure dates with symptoms and laboratory changes before refining the DILI Agent input.

Timeline generation uses the model assigned to the Timeline role in **Configurations**. If model extraction is unavailable, DILIGENT may build a deterministic fallback from saved session fields. Approximate dates, missing source evidence, and fallback chronology are labeled warnings and navigation aids, not clinically established facts.

### Data Inspection

Open **Data Inspection** to view available local resources, records, metadata, and update status. Use the available search, filtering, refresh, or pagination controls to confirm that expected data is present. Do not edit local database files while the application is running.

![Data inspection](assets/figures/data-inspection.png)
_Data Inspection presents curated resource records, status, and maintenance information for local review._

## Important limitations and expected behaviour

- DILIGENT is designed for local, single-user use on a desktop-width screen. It is not an authenticated shared web service or a mobile application.
- Cloud providers may receive the clinical text needed for a run. Confirm that this is allowed before using protected information.
- Local providers and optional evidence resources can be unavailable. The application reports the limitation or shows a fallback rather than treating missing evidence as certainty.
- A completed session remains available for later review, but an assessment that is still running may need to be started again if the local service is restarted.
- Generated text, timelines, drug matches, and causality signals always require human verification before clinical or formal use.

## Troubleshooting

| Symptom | Likely cause | Try this |
| --- | --- | --- |
| The Windows desktop app does not open | The download may be incomplete, WebView2 may be unavailable, or packaged startup may have failed. | Verify the matching SHA-256 file, download the package again if needed, and make sure WebView2 is available. See the [desktop troubleshooting guide](assets/docs/runtime/troubleshooting.md) for the packaged log location. |
| The source-mode page is blank or does not load | The launcher has not finished preparing the local services or frontend. | Run `.\start_on_windows.ps1` again. On a fresh checkout, complete **Install dependencies** before **Launch application**, then open [http://127.0.0.1:9847](http://127.0.0.1:9847). |
| The UI says that the local service is unavailable | The service is still starting, stopped, or another local application is using its port. | Open [http://127.0.0.1:7690/api/health](http://127.0.0.1:7690/api/health), then restart the launcher. |
| A model cannot be selected or saved | The provider mode and model are incompatible, or the saved catalog is out of date. | In **Configurations**, choose the correct local or cloud mode, use **Refresh**, assign compatible roles, and save again. |
| A cloud model catalog is unavailable | The active key, network connection, provider quota, or provider service may be unavailable. | Confirm that the correct key is active, check network access and provider status, then use **Refresh** again. A previously valid catalog may remain visible. |
| A local run fails | Ollama is not running or the selected model is not installed or chat-capable. | Start Ollama, confirm that the selected model is installed, and retry the run. |
| The assessment is blocked before it starts | Pre-flight checks found missing or inconsistent input. | Follow the feedback, add or correct the requested history, timing, medication, or laboratory details, and retry. |
| The report is incomplete, uncertain, or appears wrong | The input may be missing important evidence, or the model may have returned a fallback or ambiguous match. | Check the warnings and source record, enrich the case, run it again, and obtain human review before reuse. |
| No saved sessions appear | The assessment may not have completed, or first-time local setup may be incomplete. | Complete the launcher's dependency and database setup options, restart the application, and confirm that a later assessment finishes successfully. |
| A timeline shows fallback chronology | Model extraction did not complete or the available dates are uncertain. | Use it only as a navigation aid, review the warning, and retry after correcting the provider or model condition. |
| Data Inspection is empty | Local resources have not been initialized or refreshed. | Use the launcher's setup or maintenance options, wait for them to finish, and restart DILIGENT. |
| A source-mode port is already in use | Another local application is using one of the default development ports. | Close the conflicting application or use the local port arrangement already approved for your environment, then restart both source-mode services. |

For source mode, the default interface address is `http://127.0.0.1:9847` and the default health-check address is `http://127.0.0.1:7690/api/health`. Packaged Windows desktop startup uses its own local address, so these source-mode addresses are not a packaged-app health check.

## Further documentation

For deeper technical and maintenance reference, start with the [documentation index](assets/docs/project_index.md). It links to the [system overview](assets/docs/architecture/system_overview.md), [backend layers](assets/docs/architecture/backend_layers.md), [persistence model](assets/docs/architecture/persistence.md), [background jobs](assets/docs/architecture/background_jobs.md), [DILI pipeline](assets/docs/architecture/dili_assessment_pipeline.md), and the detailed [desktop release documentation](assets/docs/runtime/desktop_release.md).

## Project status and license

DILIGENT is under active development and may contain incomplete features or defects. Tagged releases are intended for local evaluation.

This project is licensed under the GNU General Public License, version 3 or any later version. See [LICENSE](LICENSE) for the terms.
