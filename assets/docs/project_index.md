# Project Overview
Last updated: 2026-06-03

## Purpose
This file is the root index for `assets/docs`. Read it first, then open the smallest topic file that matches the task.

## How To Navigate
1. Start with this file only.
2. Choose the topic branch that matches the task.
3. Open the narrowest leaf document that answers the question.
4. Expand to sibling files only when the task clearly crosses topic boundaries.
5. Keep documentation updates aligned with implementation changes.

## Naming Rules
- All files and folders under `assets/docs` use lower-case names.
- Topic folders group related leaf documents by subject.
- Root-level files are reserved for entry-point documents only.

## Documentation Ontology
### Root
- `project_index.md`
  - Entry point and master index for the full documentation tree.

### Architecture
- `architecture/system_overview.md`
  - Repository layout, maintained code structure, and entry points.
- `architecture/api_surface.md`
  - Backend route catalog and stable HTTP boundaries.
- `architecture/backend_layers.md`
  - Layer responsibilities, request flow, and async or sync execution rules.
- `architecture/persistence.md`
  - Database, vectors, resources, and persisted artifact locations.
- `architecture/background_jobs.md`
  - Centralized job lifecycle, polling, cancellation, and active job types.

### Coding
- `coding/shared_rules.md`
  - Cross-language rules for scope, boundaries, imports, and cleanup.
- `coding/python.md`
  - Python runtime, typing, validation, async, and structure rules.
- `coding/typescript.md`
  - Angular TypeScript architecture, UI-state, and UX behavior rules.
- `coding/testing_and_quality.md`
  - Test expectations and cross-language quality gates.
- `coding/error_handling.md`
  - Failure-class, timeout, logging, cleanup, and safe-error rules.

### Runtime
- `runtime/modes.md`
  - Supported runtime targets and differences between local and packaged execution.
- `runtime/startup.md`
  - Launcher-first startup procedures for local development, Codex sessions, browser-driven UI work, and manual fallback commands only when the launcher path is unsuitable or already diagnosed as failing.
- `runtime/configuration.md`
  - Environment variables, ports, runtime settings, and catalog inputs.
- `runtime/deployment.md`
  - Packaging constraints, release outputs, and dependency notes.
- `runtime/troubleshooting.md`
  - Startup failures, port conflicts, backend launch recovery steps, and Angular sandbox-build fallback guidance for `spawn EPERM`.
- `runtime/qa_regression.md`
  - Repeatable regression slice for model configuration and app-flow validation.

### UI
- `ui/design_tokens.md`
  - Typography, spacing, sizing, and color tokens.
- `ui/components_and_patterns.md`
  - Controls, navigation, modal, and page composition rules.
- `ui/experience.md`
  - Core journeys, responsiveness, accessibility, and design principles.

### User
- `user/getting_started.md`
  - Purpose, audience, prerequisites, safety, layout, and startup basics.
- `user/model_setup.md`
  - Model provider selection and access-key workflows.
- `user/dili_assessment_workflow.md`
  - DILI Agent data entry, run flow, report review, and export guidance.
- `user/sessions_timeline_and_data.md`
  - Clinical Sessions, Patient Timeline, Data Inspection, and maintenance usage.
- `user/troubleshooting.md`
  - User-facing troubleshooting for startup, health, providers, sessions, and data.
- `user/checklists.md`
  - End-to-end journey, input checklist, output checklist, and usage cautions.

## Reading Order
1. Read this root index.
2. Open the smallest leaf file that covers the current question.
3. Expand to adjacent files only when the task crosses topic boundaries.
4. Return here when switching branches.

## Context Rules
- Read documentation files only when required by the active task.
- Defer reading until the task proves the file is needed.
- Keep all affected documents updated whenever behavior, architecture, runtime, or UX changes.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Pre-select files to read by folder structure and task intent before opening them.

## Environment Rules
- Windows is the default operating environment for this repository.
- Support both PowerShell and CMD guidance where commands differ.
- Keep runtime guidance aligned with `start_on_windows.bat`, `setup_and_maintenance.bat`, `release/tauri/*.bat`, and `app/tests/*.ps1` or `.bat`.
