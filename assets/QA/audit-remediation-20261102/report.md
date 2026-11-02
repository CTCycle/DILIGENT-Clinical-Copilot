# Audit remediation validation, 2026-11-02 (uncommitted changes on `develop`)

## Automated matrix
| Check | Result |
|---|---|
| `compileall` backend + tests | pass |
| Alembic `upgrade head`, `current --check-heads`, `check` | pass (head 202611010003, no drift) |
| `ruff check . ../tests` | pass |
| `pyright .` | 0 errors |
| Backend unit + persistence | 1034 passed, 26 skipped (PostgreSQL) |
| Vitest | 24 files / 194 tests passed (193 before the fix below) |
| Angular production build | pass |
| Full browser E2E harness | first run 62 passed / **1 failed**; after test fix 63 passed |
| npm audit (client, desktop; `--audit-level=high`) | exit 0 (client lists 18 moderate) |
| cargo-audit (CI flags) | exit 0 (9 allowed warnings) |
| pip-audit | not run (not installed; CI uses `uvx`) |
| `app/shared/openapi.json` | nothing in CI/tests checks it; it is already stale against the generated schema (`/api/inspection/sessions` params), only the version string is bumped in the diff |

## Defects found and fixed
1. **E2E test not updated for M-04** (`app/tests/e2e/test_api_local_boundaries.py`): manual-edit PUTs omitted the now-required `expected_version` (422). The test now sends the current session `version`; the missing-session 404 case sends `expected_version: 1`. No assertion was weakened.
2. **Misleading delete-conflict message (H-04 UI)**: the API returns 409 "This session has revised child sessions. Delete them first.", but the client maps every 409 to "Another operation is already running. Please wait and retry." (generic mapping in `http-api.ts`, pinned by an existing spec, so left alone). Fix: `deleteSession` in `clinical-sessions-page.component.ts` shows a specific 409 message, plus one spec.

## Rendered browser checks (isolated data dir, launcher `-Action Launch`, built-in browser, 1440x900)
| Check | Result | Evidence |
|---|---|---|
| a. PNG, PDF, JSON upload; downloads | pass. Responses: png `inline`, pdf/json `attachment` with the right filenames, `no-store`, `nosniff`, `default-src 'none'; sandbox`. Request URLs are `/api/inspection/sessions/1/attachments/{id}` with no token or query. Source mode has no access token, so header-based auth was not exercised. | a1 |
| a. `.html` upload | pass: "[ERROR] Attachments of type .html are not allowed because they can contain active content." | a2 |
| b. Report banner | pass: backend stopped, "Connection lost, retrying…" with last snapshot visible; later "Updates paused" + Reconnect; after restart + Reconnect the state updated to Interrupted | b1-b3 |
| c. Stale manual edit | pass: tab 2 save gives 409 message and "Reload latest"; reload moves to Version 3 and keeps the draft | c1, c2 |
| d. Delete parent with child | pass at API (409 + detail) and UI (after fix 2). Seeded the child through repositories (version row linked to the parent). Before the fix the UI showed the generic message. | d |
| e. Search | pass: `zebracanary` matches the recent session; "Search older" adds the 200-day-old session (`search`+`created_before` query) | e1, e2 |

Notes: file inputs were driven through JS `DataTransfer` (no native picker). Selecting a PDF and a JSON in one `change` while a PNG upload was in flight uploaded only the PNG, not reproduced with sequential selections; multi-file selection in a single input was not re-tested. A completed session does not poll, so check b used a seeded `running` session. The delete error banner shifts the list down (pre-existing layout).

## Not run
LiveProvider and LocalOllama lanes, embedding integration, portable/MSI builds, hosted PostgreSQL persistence, pip-audit, anything needing credentials. The Full harness was not re-run after fix 2 (client-only, covered by Vitest and a rendered re-check).

## Cleanup
Backend stopped; port 7690 free. Port 9847 is held by a pre-existing `node scripts/preview-server.mjs` (started 13:34, not by this session), left running. Temp data dir and seed script removed.
