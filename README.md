# DILIGENT Clinical Copilot
Last updated: 2026-10-18

[![Release](https://img.shields.io/github/v/release/CTCycle/DILIGENT-Clinical-Copilot?display_name=tag)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases)
[![License](https://img.shields.io/github/license/CTCycle/DILIGENT-Clinical-Copilot)](./LICENSE)
[![CI](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/workflows/ci.yml?query=branch%3Adevelop)

DILIGENT is a local, single-user decision-support application for structured Drug-Induced Liver Injury (DILI) review. It brings clinical history, medication exposure, symptoms, laboratory results, and timing information together for a reviewable assessment draft.

DILIGENT does not diagnose or replace clinical judgment. A qualified clinician must verify the source information, generated statements, conclusions, and any exported text before clinical use.

## Review workflow

1. Enter the case and review pre-flight feedback.
2. Run an assessment with the configured local or cloud model. Retrieval-supported reference evidence is optional.
3. Review the narrative alongside structured findings, timelines, drug-match flags, and uncertainty warnings.
4. Save and compare session revisions, make reviewed edits, or export the result.

The main workspaces are **DILI Agent**, **Clinical Sessions**, **Data Inspection**, and **Settings**. Patient Timeline is opened from the Timeline tab of a selected session.

## Windows release

The latest published release is [v3.4.0](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/latest), published on 2026-10-18. It includes a portable EXE and an MSI installer. Use the published SHA-256 manifest to verify downloads.

The v3.4.0 binaries are unsigned. The release notes disclose that they have no publisher signature and may produce an unknown-publisher or SmartScreen warning. Windows WebView2 is required. Release verification and its recorded limits are in the [desktop release record](assets/docs/runtime/desktop_release.md).

The packaged app starts its local services and stores settings, sessions, resources, and keys in the user's local application-data area. It is intended for a desktop-width window; authenticated shared-network deployment and mobile use are not supported.

## Start DILIGENT

### Packaged desktop app

Download the portable EXE to run without installation, or install the MSI from the [GitHub Releases page](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases). The package includes the application runtime; a separate Python, Node.js, or source checkout is not needed.

### Source mode

Clone the repository on Windows and run:

~~~powershell
.\start_on_windows.ps1
~~~

On first use, choose **Install dependencies**, then **Launch application** in the launcher. See [Getting started](assets/docs/user/getting_started.md) for prerequisites and startup details.

## Safe use and privacy

- Use DILIGENT only where the data handling is approved. A cloud-backed run may send clinical text to an external provider.
- Check the selected provider, model roles, active access key, and retrieval setting before each assessment.
- Verify drug identity, dates, doses, laboratory values and units, alternative causes, and conclusions against the source record.
- Treat model output, timelines, drug matches, and causality signals as decision-support material that requires human review.

DILIGENT is designed for local, single-user operation. It is not an authenticated shared web service.

## User guides

- [Getting started](assets/docs/user/getting_started.md)
- [Model setup and access keys](assets/docs/user/model_setup.md)
- [DILI assessment workflow](assets/docs/user/dili_assessment_workflow.md)
- [Sessions, timeline, and data](assets/docs/user/sessions_timeline_and_data.md)
- [Troubleshooting](assets/docs/user/troubleshooting.md)

## Project documentation

Start at the [documentation index](assets/docs/project_index.md). The [project status ledger](assets/docs/project_status_ledger.md) records current component status; the [validation ledger](assets/docs/runtime/validation_ledger.md) records retained evidence boundaries and links them to the product concepts they support. For test and release procedures, see [testing and quality](assets/docs/coding/testing_and_quality.md) and [desktop release](assets/docs/runtime/desktop_release.md).
