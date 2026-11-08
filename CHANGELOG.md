# Changelog

Release notes for DILIGENT Clinical Copilot. Earlier releases (v3.5.1 and before) are described on the [GitHub Releases](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases) page.

## 3.6.1 (unreleased)

- Recover desktop startup after an interrupted runtime extraction and verify the cached runtime before executing it.
- Block Windows MSI downgrades and include the project license notice and frontend dependency notices in desktop packages.
- Use stable inputs for freezing the backend and preserve the development environment when dependency downloads fail.

## 3.6.0 (2026-11-03)

Compared with v3.5.1 (2026-10-22). Minor release: the stack, packaging, and data locations are unchanged, and existing data is upgraded automatically.

### Highlights

- **Session lifecycle.** The DILI Agent now creates a session for every run. Sessions move through draft, queued, running, completed, failed, and cancelled; queued or running analyses can be cancelled, and runs interrupted by a restart are marked as such. The report page follows progress, keeps working through brief connection loss, and reconnects automatically.
- **Database-backed drug terminology.** Drug names resolve against a locally imported terminology (RxNorm Prescribable, DrugCentral, ChEMBL, Swissmedic, curated aliases) with explicit LiverTox links, a strict spelling-correction gate, and a bounded RxNav fallback. Ambiguous matches can be confirmed by a reviewer in Knowledge → Terminology.
- **FDA DILIrank 2.0** is available as a structured drug-level hepatotoxicity source; unlinked or ambiguous rows stay inspectable but never become patient-level causality evidence.
- **Per-session evidence sources and model overrides** can be chosen in the pre-run setup dialog.
- **Redesigned report, sessions, and timeline views.** A full-width report page with sticky actions; Clinical Sessions with multi-select status and date filters, server-side search, and a 90-day recent view with "Search older"; a Patient Timeline rail with collapsible panels and a horizontal timeline view, with dates and evidence grounded in the clinician-entered source.
- **Persisted attachments** on sessions (images, PDF, JSON; active content such as HTML is rejected).
- **Status bar and jobs monitor** showing backend, provider, model, local Ollama, and resource state; local Ollama can be started and stopped from the status bar.
- **Database and network settings.** Settings gains a Database page (SQLite or PostgreSQL, with test and controlled live switching) and runtime network configuration.

### Improvements

- Clinical logic: competing-cause, dechallenge, and date handling in DILI adjudication; hepatic injury pattern calculation shared with the evidence bundle; dated same-episode Hy's Law pairing with baseline awareness; laboratory normalization and RUCAM extraction; narrative claim checks limited to patient-level statements; merge of duplicate laboratory observations.
- Resumable clinical input drafts and a compact DILI workspace with a setup dialog.
- Revision workflow with lineage, evidence-safety audit, and QA artifacts.
- Launcher: timing reports, supervised backend console, hardened shutdown, and maintenance cleanup.
- Design-token cleanup, improved contrast, compact controls, and settings autosave feedback.
- NCBI contact email for LiverTox updates is now optional.

### Fixes and security

- Pre-release audit remediation: atomic session deletion that refuses parents with revisions, `expected_version` on manual report edits (stale saves return 409 and keep the draft), cancelled-before-start sessions finalized and the next queue entry dispatched, attachment downloads with `no-store` and restrictive CSP headers.
- Manual edits keep saved session metadata.
- Patched dependency advisories (pypdf, certifi).
- Unified single-page-app startup and shutdown.

### Upgrade notes

- Database migrations run automatically on first start and are forward-only. Back up `%LOCALAPPDATA%\DILIGENT\data` (or `data/` in source mode) before upgrading; a database upgraded to 3.6.0 cannot be used with an earlier release.
- Database access is configured only through Settings → Database (`<data-root>/database_config.json`); `.env` and process environment variables no longer select the database.
- The legacy draft routes and the draft `use_rag` flag were removed; drafts are now sessions. This affects only API clients other than the bundled interface.
- Both MSI and portable builds remain unsigned. The MSI upgrades an existing installation in place; the portable build is independent of any installation.
