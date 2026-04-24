# DILIGENT Clinical Copilot Documentation Overview

Last updated: 2026-04-24

## FILES INDEX

- ARCHITECTURE.md  
  System architecture: source structure, API surface, entry points, layering, persistence, and sync/async boundaries.

- BACKGROUND_JOBS.md  
  Thread-based background job lifecycle, state contract, and job endpoint mapping.

- CODING_RULES.md  
  Consolidated coding standards for Python and TypeScript, plus shared tooling/testing rules.

- ERROR_HANDLING.md  
  Error-handling requirements across backend/frontend boundaries and user-safe failure rules.

- PROJECT_OVERVIEW.md  
  Master documentation index and context/environment rules for maintaining docs.

- RUNTIME_MODES.md  
  Supported runtime targets, startup commands, configuration differences, interoperability, and deployment constraints.

- UI_STANDARDS.md  
  Enforceable frontend design system based on current Angular implementation.

- USER_MANUAL.md  
  End-user workflows and operational usage guidance for the application.

## CONTEXT RULES

- Only read documents when necessary for the active task.
- Defer document reading until a task needs that information.
- Keep all impacted documents updated whenever behavior, architecture, runtime, or UX changes.
- Always include a `Last updated: YYYY-MM-DD` line when modifying documents.
- Avoid reading all `SKILL.md` files indiscriminately.
- Pre-select relevant files using folder structure and user intent before reading.

## ENVIRONMENT RULES

- Windows is the default operating environment for this repository.
- Support both `cmd` and PowerShell command usage in docs when startup/maintenance flows require it.
- Prefer PowerShell for scripting/search unless a `.bat`/CMD workflow is explicitly required.
- Update this section whenever new environment-specific constraints or proven solutions are identified.
