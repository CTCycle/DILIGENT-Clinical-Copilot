# v3.4.0 release gate implementation and validation
Last updated: 2026-10-18

## Disposition

**NOT RELEASE READY.** A new exact-source candidate and operator acceptance
remain required. The user explicitly authorized an **unsigned** release on
2026-10-18 because no signing identity is available. Signing is waived for this
release; hashes and build provenance remain required, and public release notes
must disclose the lack of a publisher signature. Windows may show an
unknown-publisher or SmartScreen warning. Earlier 3.4.0 binaries remain historical.

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

Raw local logs and disposable fixtures are in this directory and remain
untracked. Hosted results and current branch/candidate disposition are recorded
below when available.

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
the normalization fix above. No candidate from that failed run was accepted;
the repaired exact source requires fresh preparation before promotion.

The first hosted attempts exposed a PowerShell error-message wrapping assumption
in a new guard test and a missing fresh-host SQLite parent directory. Both were
fixed and revalidated; cancelled/superseded runs are not release acceptance.

## Remaining prerequisites

- A successful exact-source `prepare` run must supply portable/MSI artifacts and
  genuine hosted MSI lifecycle receipts. `NotSigned` is the approved disposition.
- The operator must complete the [native and clean-machine checklist](operator-checklist.md)
  against those exact hashes before approving `publish`.
- Until those requirements pass, no v3.4.0 tag or GitHub Release may be created.
