# v3.4.0 release gate implementation and validation
Last updated: 2026-10-18

## Disposition

**NOT RELEASE READY.** The release-gate implementation is complete; a new
signed candidate and operator acceptance remain required. This report does
not certify MSI execution, signing, populated native workflows, or clean-machine
distribution. Earlier v3.4.0 binaries represent earlier source and are historical.

## Changes

- Preflight maps the clinical `missing_visit_date` code to `visit_date_missing`
  and emits that requirement once per field. Clinical submission validation and
  all other blockers remain enforced.
- The launcher can sign the shell before MSI bundling, sign the MSI, and produce
  final checksums. Signature verification requires the expected trusted signer
  and a timestamp, including on the installed executable.
- MSI smoke now covers install, launch, uninstall, reinstall, relaunch, and
  uninstall. Genuine v3.3.0 upgrade uses published checksum-verified MSI bytes,
  the baseline's runtime-directory convention, and synthetic data persisted
  through the v3.3.0 repositories. Checks cover sessions, report versions,
  nondefault polling settings, SQLite integrity/foreign keys, and Alembic head.
  This fixture is software-persistence evidence, not a clinical/provider run.
- Release preparation and publication are separate manual workflow modes.
  Preparation retains immutable signed assets, exact source/run/attempt,
  checksums, signing identity, successful gate results, and lifecycle receipts.
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
| Angular production build | Passed | Source bundle; not a signed desktop candidate. |
| Canonical Windows Full browser harness | 53 passed; 0 skips | Includes real API-backed missing-date preflight and zero job/session creation after rejection. |
| Ruff / Pyright | Passed / 0 errors, 0 warnings | Project backend configuration. |
| Workflow and PowerShell | actionlint v1.7.12 passed; PowerShell parser passed | Workflow execution and administrator MSI evidence remain separate. |
| Baseline fixture replay | Passed | v3.3.0 repository seed -> current migration -> unchanged sessions/report/settings, integrity/foreign keys, current Alembic head. No MSI was installed locally. |

The in-app browser displayed **3 blocking** items with exactly one missing-date
entry. Supplying the date reduced the count to **2 blocking**, preserving the
empty LiverTox and RxNav blockers. No analysis was started. See
[rendered single-date blocker](preflight-single-date.png).

Raw local logs and disposable fixtures are in this directory and remain
untracked. Hosted results and current branch/candidate disposition are recorded
below when available.

## Remaining prerequisites

- GitHub secret names currently include `OPENCODE_GO_API_KEY`; the two signing
  secrets and two signing variables were absent when inspected. A trusted
  certificate must be configured securely; no certificate was fabricated or
  copied from another project.
- A successful exact-source `prepare` run must supply signed portable/MSI
  artifacts and genuine hosted MSI lifecycle receipts.
- The operator must complete the [native and clean-machine checklist](operator-checklist.md)
  against those exact hashes before approving `publish`.
- Until those requirements pass, no v3.4.0 tag or GitHub Release may be created.
