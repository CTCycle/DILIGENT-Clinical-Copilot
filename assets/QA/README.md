# QA Evidence Ontology
Last updated: 2026-11-02

`assets/QA/` holds validation evidence. It is git-ignored except this file and compact reports that substantiate a claim made in `assets/docs/` (force-tracked). Durable conclusions belong in the documentation ontology ([`project_status_ledger.md`](../docs/project_status_ledger.md)); this tree is the evidence behind them. Do not add new top-level folders.

| Folder | Contents | Naming | Tooling that writes here |
| --- | --- | --- | --- |
| `audits/` | Clinical, release, and remediation audits with reports, harnesses, and outputs | `<topic>-YYYYMMDD/` | manual |
| `release-gates/` | Hosted and native release-gate reports and screenshots, CI caches | `YYYYMMDD/` per gate; CI uses `browser-cache`, `provider-cache`, `baseline`, `msi`, `published-*` | `.github/workflows/release.yml`, `release_candidate.ps1` |
| `desktop/` | `staging/` (Tauri/PyInstaller build workspace), `smoke/` (packaged-app smoke reports), `launcher-logs/` | `smoke/<version>-<run>` | `start_on_windows.ps1`, `smoke_release.ps1` |
| `ui-validation/` | Browser-driven UI validation captures per feature | `<feature>-YYYYMMDD/` | manual |
| `e2e/` | Browser E2E suite run reports | `browser-YYYYMMDD/` | `run_browser_e2e.ps1` |
| `terminology/` | Drug-terminology benchmark reports and captures | `drug-terminology-YYYYMMDD.md` | manual |
| `pytest/` | Pytest cache, databases, and fixtures (scratch) | fixed | `app/tests/conftest.py` |
| `scratch/` | Obsolete local leftovers awaiting deletion; never referenced | any | none |
