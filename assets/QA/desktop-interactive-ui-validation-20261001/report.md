# Packaged interactive-UI validation — 2026-10-01

Date: 2026-10-01 (Europe/Rome)

## Scope and baseline

This slice revisits the previously BLOCKED packaged desktop interactive-UI
gate (`D06`–`D11` in the desktop release matrix) from the exact prepared
v3.4.0 candidate. The prior blockers were: the token is not
administrator-capable, and no native UI automation surface was available to
drive the packaged Tauri/WebView2 window. This run used the available native
Windows UI-automation driver to actually operate the packaged window.

Baseline:

- Repository HEAD: `738a14bdfab9627cc9e38461f374d10cacd32edc` (equal to
  `origin/develop`), docs-only successor to candidate
  `2f804d251769f4e01375d02459e41dd9970ec8eb`.
- Artifacts verified against the release manifest:
  - `DILIGENT-v3.4.0-windows-x64-portable.exe` SHA-256
    `0209cf25a8acf3a39762ffeec48488c28103e1b96c7c5a57ed8beb813574e167`
  - `DILIGENT-v3.4.0-windows-x64.msi` SHA-256
    `1df49329361b600edb2e0c07949fe00bc113b25122d55af9ba93cb52f67fd508`
- Current identity is `BLUEGREEN\TV`, `IsInRole(Administrator) = False`.
- Ports `7690`, `9847` free; the user-started Ollama service on `11434`
  remained running and was used read-only for catalog discovery.

## Environment and data root

The portable EXE was launched with an **isolated data root** at
`C:\Users\THOMAS~1\AppData\Local\Temp\opencode\dcui-iso` (short path; a
first attempt under the deep repository path reproduced the documented
Windows DLL-load filename-length failure for the isolated root, so the short
root matching `smoke_release.ps1` semantics was used). Only synthetic data
was used. The shared database, settings, credentials, and provider caches
were not touched.

First launch extracted the runtime to
`DILIGENT\rt\3.4.0\105727fc84c900b82f9d98f3f3d6b2200e984e88e98a60ebcbf5afd6af333536`
(1109 files + `extraction.complete`), wrote
`DILIGENT\data\state\desktop-backend-ready.json` with
`{"pid": 19280, "port": 65516, "release_version": "3.4.0"}`, and served the
packaged backend.

## API boundary (D12 recheck on packaged surface)

| Check | Result |
|---|---|
| `/api/health` | HTTP 200, `{"status":"ok"}` |
| `/` (root/SPA) | HTTP 200 |
| `/api/settings` (no session cookie) | HTTP 401 (desktop boundary) |

## Rendered workspaces and navigation (D06)

The packaged window (native title `DILIGENT Clinical Copilot`) was driven via
the native UI-automation driver. The following surfaces rendered and were
read from the accessibility tree:

- **DILI Agent** — clinical input, patient name, visit date, RAG toggle,
  Run DILI analysis, Clear all, report panel, tips.
- **Clinical Sessions** — search, All/Successful/Failed filters, date filter,
  empty state (`No clinical sessions found.`) on the fresh packaged data.
- **Data Inspection** — Drug Catalog / LiverTox / DILIrank / RAG tabs,
  Update all, Update Catalog, empty RxNav state.
- **Settings → General** — Source `Database / Settings UI`, Job polling
  interval, Last persisted timestamp, Reset/Discard/Save.
- **Settings → Models** — runtime source radio, RAG settings modal entry,
  reasoning slider, local model catalog with install/assigned status
  (`qwen3.5:2b`, `qwen3.5:9b`, and others marked Installed; not-installed
  entries present), cloud provider keys (OpenAI, Google Gemini, DeepSeek,
  Anthropic Claude, OpenCode Zen/Go), current configuration
  (`Local (Ollama)`, qwen3.5:2b for all roles, RAG model `Granite Embedding
  97M Multilingual R2`), generation behavior, Save Configuration.

## Settings persistence, reload, and reset (D06)

From General Settings in the packaged UI:

1. Changed Job polling interval `1 → 2`, saved. The UIA tree showed
   `value="2"`, the `Unsaved changes` indicator cleared, and the timestamp
   advanced (`Last persisted: 1 Oct 2026, 10:45`). Read-only SQLite in the
   isolated data root confirmed `jobs.polling_interval = 2.0` and
   `updated_at` advanced from `2026-10-01 10:36:12` to
   `2026-10-01 10:45:13`.
2. Navigated to Data Inspection and back to Settings: the spinner still
   showed `2` (reload persistence).
3. Pressed Reset to defaults: the spinner returned to `1`. Read-only SQLite
   confirmed `polling_interval = 1.0` and `updated_at` advanced to
   `2026-10-01 12:02:40`.
4. SQLite `PRAGMA integrity_check` returned `ok`; `foreign_key_check`
   returned no rows throughout.

## Clean close and restart persistence

- The packaged app was closed with `Alt+F4`. No DILIGENT/backend process
  remained and the assigned port had no LISTEN listener (only transient
  TIME_WAIT connections).
- Relaunch used the same isolated data root. Ready payload reported
  `{"pid": 11452, "port": 54285, "release_version": "3.4.0"}` and
  `/api/health` returned 200.
- The backend log shows `Alembic schema check: current=('202609170001',)
  target=('202609170001',) tables=23`, model-configuration initialization
  completed with `seeded=False`, and `SQLite database is synchronized`.
- Read-only SQLite after restart: `polling_interval = 1.0` with the same
  `updated_at` as the reset — durable persistence across a real process
  restart.
- The packaged UI reopened on DILI Agent; navigating to Settings showed the
  spinner value `1`, matching the DB.
- The second process (11452) was closed cleanly with `Alt+F4` and the same
  no-process/no-listener cleanup was observed.

## Preflight boundary on the packaged surface (D07 preflight)

With synthetic clinical text entered, Run DILI analysis opened the preflight
dialog in the packaged UI:

- Title `Cannot start analysis`; summary `5 blocking`, `1 warning`.
- Blocking items: Visit date missing (twice listed), LiverTox catalog empty,
  RxNav catalog empty, clinical input too brief (under 60 words) — each with
  section, attention, and consequence text.
- Warning: medication chronology incomplete (no drug with explicit
  source-reported timing).
- The dialog returned to input on request. No job was created: read-only
  SQLite showed `clinical_sessions = 0` and `clinical_session_versions = 0`
  after the attempt.

This confirms the packaged UI fails closed on an empty fresh install exactly
as the source mode does. A full synthetic multi-drug analysis, session CRUD,
timeline, revision, and RAG native-folder workflows in the packaged window
remain unexercised because a fresh packaged data root has no populated
structured sources, and cloning the credential-bearing shared database into
the disposable root was intentionally avoided.

## Deliverables in this QA folder

- `01-packaged-window-dili-agent.png` — packaged DILI Agent window.
- `02-settings-general.png` — packaged Settings → General.
- `03-settings-models.png` — packaged Settings → Models (catalog + config).
- `04-data-inspection.png` — packaged Data Inspection.
- `05-preflight-blocked.png` — packaged preflight dialog.
- `06-restart-dili-agent.png` — packaged window after clean restart.
- This report.

## Gate disposition

| Gate | Status after this run | Remaining boundary |
|---|---|---|
| `D06` workspaces + Settings | `PASS` for the exercised packaged scope | Workspaces render; General save/reload/reset/restart persistence proven; Models surface (catalog, roles, keys, RAG) renders. |
| `D07` preflight | `PASS` for the packaged preflight boundary | Full synthetic multi-drug analysis not run (no populated sources in fresh packaged root; credential-bearing DB clone intentionally avoided). |
| `D12` API boundary | `PASS` (recheck on packaged surface) | health 200, root 200, unauthenticated settings 401. |
| `D03` startup/restart/cleanup | `PASS` for the packaged portable lane | Fresh portable launch, clean close, restart persistence, no leftover processes/listeners. |
| `D08` sessions/timeline/revision | `UNTESTED` | Requires a populated packaged data root and a usable model lane. |
| `D09` RAG native folder picker | `UNTESTED` | Native dialog not exercised; requires a populated RAG collection. |
| `D10` file dialogs / WebView2 edge cases | `UNTESTED` | Not part of this slice. |
| `D11` failure/cancellation/retry in packaged UI | `UNTESTED` | Not part of this slice. |
| `D13`/`D14` MSI lifecycle + upgrade | `BLOCKED` | Non-administrator token; no elevation attempted. |
| Signing / clean-machine distribution | `PENDING` | Separate distribution procedures. |

## Cleanup

- The second packaged process was closed cleanly; no task-owned DILIGENT
  processes remained.
- The disposable data root (`dcui-iso`) and temporary screenshots are under
  the temp area; the isolated database and settings were removed after the
  read-only checks. Ports `7690`, `9847` and the packaged random ports were
  free; the user-started Ollama service remained running.
- The shared database, settings, `.env`, and protected caches were untouched.