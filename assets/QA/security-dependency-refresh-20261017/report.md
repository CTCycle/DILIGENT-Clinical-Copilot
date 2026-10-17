# Security dependency refresh — 2026-10-17

## Scope

Remediate dependency audit findings against the current `develop` source and
validate the automated regression gate. This report covers source dependencies
at commit `15632376d804d68e3bc1be9a8fc884b9fa97ebb1` and hosted CI run
[39033116041](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39033116041).

## Findings and changes

- The Python audit identified vulnerabilities in `pypdf==6.18.1`. Updated the
  exact pin and lock to `pypdf==6.21.0`.
- The Angular audit identified high/critical findings in the locked CLI/build
  dependency tree. Updated `@angular/cli` and `@angular/build` from `21.2.24` to
  `21.2.27`, which brings in fixed `@modelcontextprotocol/sdk` and `piscina`
  releases; refreshed compatible transitive lock entries.
- The Rust audit identified three advisories for `zip==2.4.2`. Updated the
  direct dependency and lock to `zip==9.0.4`. The runtime extractor uses
  `ZipArchive` entry reads with its own path and symlink checks; the new API
  reports filename decoding errors, which now fail the extraction closed.

The remediation was split across source commits
[`fabe9c35`](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/commit/fabe9c35f3b16042a552e80eb568ccec061c4f30),
[`4c49af4e`](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/commit/4c49af4e7234e4fbd2b0610f5aaa50e67c7a5cc9),
and [`15632376`](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/commit/15632376d804d68e3bc1be9a8fc884b9fa97ebb1), all pushed to
`develop`.

## Validation

- `uv lock --check` passed.
- Full backend unit suite: **838 passed**, with 7 existing deprecation warnings.
- Strict `pip-audit` over the exported locked Python requirements reported no
  known vulnerabilities.
- A one-page PDF smoke through `DocumentSerializer` extracted the expected
  text and page metadata with `pypdf 6.21.0`.
- `npm audit --audit-level=high --package-lock-only` reported **0
  vulnerabilities**.
- `cargo test --locked --bin diligent-desktop`: **2 passed**. `rustfmt
  --edition 2021 --check src/runtime.rs` passed.
- Hosted CI run [39033116041](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39033116041)
  completed successfully for backend quality, Windows regression, persistence
  contract, and security scan (Python, Angular, desktop, and Rust audits).
  `live-provider-e2e` was skipped because this was a push event.

## Boundaries

The dependency changes were not rebuilt into fresh portable, MSI, or checksum
artifacts. The previously prepared v3.4.0 candidate predates this source update;
rebuild and candidate-specific validation are required before release use. This
run also provides no live-provider E2E evidence.
