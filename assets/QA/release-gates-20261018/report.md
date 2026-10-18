# v3.4.0 release gate validation
Last updated: 2026-10-18

## Disposition

**RELEASED — 2026-10-18.** Publish workflow
[39125173765](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39125173765)
succeeded after the required `desktop-release` approval. Public release
[v3.4.0](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0)
is also the GitHub latest release. Tag `v3.4.0` resolves to the accepted
candidate SHA `d4efa5e0bccfa552c80a5edeff2168bfaa718639`; publication reused the
selected immutable candidate without rebuilding. Published and downloaded
portable EXE/MSI bytes match the candidate hashes below. The EXE launched from
the downloaded release bytes and returned `/api/health` status `ok`. The user
authorized an unsigned release on 2026-10-18; the EXE/MSI are NotSigned and the
public release notes disclose that they have no publisher signature.

## Changes

- Preflight maps the clinical `missing_visit_date` code to `visit_date_missing`
  and emits that requirement once per field. Clinical submission validation and
  all other blockers remain enforced.
- Following the user's updated policy, candidate receipts explicitly declare
  `signing=unsigned`; preparation and promotion verify `NotSigned` status on the
  EXE/MSI and installed shell. The signing-only helper and launcher hooks were
  removed. No certificate or signing credentials are required or fabricated.
- A live-provider run exposed a deterministic revision-normalization defect:
  `Read the session context once, then append exactly...` was not recognized
  because the append parser only accepted instructions starting with `Append`.
  The explicit context-read prefix is now supported, with regression coverage
  for forced append, negation, and truncated canonical context. No live assertion
  was relaxed. The 39-test revision-agent suite passed locally.
- MSI smoke now covers install, launch, uninstall, reinstall, relaunch, and
  uninstall. Genuine v3.3.0 upgrade uses published checksum-verified MSI bytes,
  the baseline's runtime-directory convention, and synthetic data persisted
  through the v3.3.0 repositories. Checks cover sessions, report versions,
  nondefault polling settings, SQLite integrity/foreign keys, and Alembic head.
  This fixture is software-persistence evidence, not a clinical/provider run.
- Release preparation and publication are separate manual workflow modes.
  Preparation retains immutable candidate assets, exact source/run/attempt,
  checksums, explicit unsigned status, successful gate results, and lifecycle receipts.
  Publication verifies the selected successful run and promotes those bytes
  without a rebuild. A draft is read back and launch-tested before publication.
  Tag/release absence checks fail closed on API errors.
- GitHub environment `desktop-release` requires CTCycle review, disallows
  administrator bypass, and permits only `main`. The agent does not approve
  the environment on the operator's behalf.

## Local execution evidence

| Check | Result | Boundary |
|---|---|---|
| Backend unit + SQLite persistence | 855 passed; 14 skipped | PostgreSQL cases require hosted service; new release-guard tests were added afterward. |
| Focused final preflight, launcher, release guards | 20 passed | Includes seven behavioral release-guard cases: accepted candidate, stale source, existing tag, wrong run, skipped provider, absent reviewer protection, and modified candidate bytes. No network or publication occurs in these tests. |
| Angular/Vitest | 105 passed | Existing frontend suite. |
| Angular production build | Passed | Source bundle; not a packaged desktop candidate. |
| Canonical Windows Full browser harness | 53 passed; 0 skips | Includes real API-backed missing-date preflight and zero job/session creation after rejection. |
| Ruff / Pyright | Passed / 0 errors, 0 warnings | Project backend configuration. |
| Workflow and PowerShell | actionlint v1.7.12 passed; PowerShell parser passed | Workflow execution and administrator MSI evidence remain separate. |
| Baseline fixture replay | Passed | v3.3.0 repository seed -> current migration -> unchanged sessions/report/settings, integrity/foreign keys, current Alembic head. No MSI was installed locally. |

The in-app browser displayed **3 blocking** items with exactly one missing-date
entry. Supplying the date reduced the count to **2 blocking**, preserving the
empty LiverTox and RxNav blockers. No analysis was started. See
[rendered single-date blocker](preflight-single-date.png) and
[date supplied, other blockers preserved](preflight-date-supplied.png).

Raw local logs and disposable fixtures are in this directory. Candidate
identity and current disposition are recorded below.

## Hosted implementation evidence

At implementation commit `104c72b5200983d25239f1fc4f129aae98e3505e`,
[main CI run 39108642370](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39108642370)
passed backend quality, Windows frontend/browser regression, SQLite/PostgreSQL
persistence, and security scanning. Its push-triggered provider job was skipped;
that skip is not provider evidence. The separate
[release preparation run 39108643638](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39108643638)
executed and passed the live-provider E2E using the configured hosted credential
and exact existing OpenCode Go/model lane. No provider secret was downloaded.
The unsigned-policy run [39109135829](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39109135829)
passed source, browser/frontend/backend, persistence, and security gates, but
provider E2E failed because its requested append was absent. That failure led to
the normalization fix above. No candidate from that failed run was accepted.
The repaired exact source was subsequently prepared successfully as run
39109627873 at commit d4efa5e0bccfa552c80a5edeff2168bfaa718639.

The first hosted attempts exposed a PowerShell error-message wrapping assumption
in a new guard test and a missing fresh-host SQLite parent directory. Both were
fixed and revalidated; cancelled/superseded runs are not release acceptance.

## Exact-source candidate and hosted gates

Accepted source: d4efa5e0bccfa552c80a5edeff2168bfaa718639.
Successful prepare run [39109627873](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39109627873), dispatched from main.
Local main and develop both resolved to this SHA; no commit was made before publication.

| Artifact | SHA-256 |
|---|---|
| Portable EXE | 702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26 |
| MSI | 3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d |

Hosted prepare passed 864 backend tests, 53 browser tests, live-provider E2E, PostgreSQL persistence, MSI lifecycle, and portable/deep-path smoke. Hosted results are distinct from the native evidence below. The workflow verifies the unsigned receipt and includes no-publisher-signature disclosure in generated public release notes.

## Publication and downloaded-byte verification

| Check | Result | Evidence |
|---|---|---|
| Publish workflow | Run 39125173765 succeeded after environment approval. | [GitHub Actions run](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39125173765) |
| Tag/source | `v3.4.0` resolves directly to `d4efa5e0bccfa552c80a5edeff2168bfaa718639`. | GitHub tag ref readback |
| Release/latest | Release is public (`draft=false`); GitHub `/releases/latest` returns `v3.4.0`, published 2026-10-18 11:59:42 UTC. | [Public release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0); GitHub latest-release API |
| Portable EXE | Published digest and downloaded SHA-256 both `702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26`; 171,714,048 bytes. | `native-downloads/DILIGENT-v3.4.0-windows-x64-portable.exe` (temporary local copy; removed after verification) |
| MSI | Published digest and downloaded SHA-256 both `3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d`; 170,172,416 bytes. | `native-downloads/DILIGENT-v3.4.0-windows-x64.msi` (temporary local copy; removed after verification) |
| Downloaded EXE launch | DILI Agent rendered from the downloaded portable EXE; `/api/health` returned `{"status":"ok"}`; app closed gracefully and listeners were gone. | [Downloaded-release startup](native-17-release-downloaded-start.png) |

The GitHub publish job also completed its downloaded-distribution smoke and
readback. The MSI was hash-verified locally; local MSI installation was not
repeated because the hosted prepare run already passed the full MSI lifecycle.

## Native environment and populated validation

| Field | Value |
|---|---|
| Date | 2026-10-18 |
| Windows build | 26200.9457 |
| WebView2/Edge | 155.0.4283.60 |
| Local analysis/timeline | Ollama qwen3.5:9b |
| RAG embedding | ONNX Runtime granite-embedding-97m-multilingual-r2 |
| Hosted provider | OpenCode Go deepseek-v4.1-flash |
| Data | Synthetic; LOCALAPPDATA, APPDATA, TEMP, TMP isolated under assets/QA/native340 |

| Check | Result | Evidence |
|---|---|---|
| Populated analysis/export | Autosaved report; export 8,743 bytes, SHA-256 D064C1822835AD83FB32C021059E8465D52AC56C9B8DA77EBB2112921B9A181C. | native-01, native-02, native-03 screenshots; native-export-clinical_report.md |
| Missing date | Exactly one blocker; no job/session created. DB remained at 2 sessions, 3 versions, 2 results, 2 timelines. Setting date to 2026-10-18 removed blocker. | native-06-missing-date-blocker.png; native-07-date-restored.png |
| Image picker | Opened and cancelled; app remained usable. | native-08-image-cancel.png |
| Job cancellation | Stop reached cancelled terminal state with no partial session; second job also cancelled; DB counts unchanged. | native-09-analysis-running.png; native-10-analysis-cancelled.png |
| Settings/restart | Value 3 saved and survived relaunch; reset restored default 1. | native-11-settings-saved.png; native-12-settings-reset.jpg |
| Session/timeline restart | Session 1 v2, session 2 v1, and session 1 timelines reloaded. | native-13, native-14, native-15 screenshots |
| Session deletion | Deleted only session 2 through UI. DB then held 1 session, 2 versions, 1 result, 2 timelines; session 1 remains. | native-16-session2-deleted.jpg |
| Published-byte launch | Downloaded release EXE rendered DILI Agent; `/api/health` returned `ok`; graceful shutdown left no app listeners. | native-17-release-downloaded-start.png |
| Shutdown | App/backend PIDs absent; ports 60056, 9223, 62172 free. | Process/listener check recorded for this report |

Manual report edit created version 2 with manual_edit lineage and a matching previous-text audit hash. Known issue: editing a report after saving metadata submits metadata: {} from clinical-sessions-page.component.ts and overwrites saved metadata in session_revision_repository.py. Metadata was saved again afterward and persisted. The issue is documented for follow-up and was not fixed in v3.4.0.

The local timeline model timed out twice; the application saved its deterministic fallback: 3 grounded events, 3 with evidence, including Amoxicillin 2026-09-01 through 2026-09-08 and ALT 210 U/L on 2026-09-08. Two fallback timelines survived restart. This is local-model degradation evidence; hosted provider E2E passed independently.

Native RAG vectorized 2 canonical documents and skipped 1 byte-identical duplicate; LanceDB held 2 rows. Session 2 cited 01-synthetic-exposure.txt, p. 1 and 02-synthetic-laboratory.txt, p. 1. Evidence: native-04-timeline.png, native-05-rag-documents.png, native-13-restart-session-list.jpg.

**Dated waiver — 2026-10-18:** The clean-host WebView2 installation check is waived because validation was limited to one Windows machine. The observed WebView2 version is from that machine, not clean-host compatibility evidence.

## Operator evidence submitted to publish run

Exact free-text field submitted with the publish dispatch:

Prepare run 39109627873 succeeded on main at d4efa5e0bccfa552c80a5edeff2168bfaa718639. Portable EXE SHA-256 702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26; MSI SHA-256 3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d. Hosted gates passed: 864 backend, 53 browser, live-provider E2E, PostgreSQL persistence, MSI lifecycle, and portable/deep-path smoke. Native synthetic validation: exactly one missing-date blocker and no database write; image picker cancelled; cancelled jobs saved no partial session; Settings value 3 survived restart and reset to default; sessions, version 2, and timelines survived restart; disposable session 2 was deleted, leaving session 1, 2 versions, 1 result, and 2 timelines. RAG vectorized 2 unique documents, skipped 1 byte-identical duplicate, and report cited both sources. Local qwen3.5:9b timeline timeouts produced the saved 3-event, 3-evidence fallback. WAIVER dated 2026-10-18: clean-host WebView2 check waived because only one Windows machine was available. Known issue: a manual report edit sends metadata={} and clears previously saved metadata; re-saving metadata persisted, and the issue is documented for follow-up. The EXE/MSI are NotSigned under the policy authorized 2026-10-18; generated public release notes disclose no publisher signature. Evidence: assets/QA/release-gates-20261018/report.md and operator-checklist.md.

The required environment approval was recorded and publish run 39125173765
completed successfully. The public release notes disclose unsigned status, and
the tag, latest-release metadata, asset digests, downloaded bytes, and portable
launch were verified above.
