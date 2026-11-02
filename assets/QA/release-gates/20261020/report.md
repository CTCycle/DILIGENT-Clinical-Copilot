# DILIGENT Clinical Copilot v3.5.0 Release Evidence
Last updated: 2026-10-20

## Release identity and publication

- Source and tag: `7a1b23e6f27b3d1987f1106eb2fb5f9aa40b2820`.
- Prepare run: [39382912130](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39382912130), succeeded on `main`.
- Protected publish run: [39419567722](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39419567722), succeeded after the `desktop-release` environment approval.
- Public release: [v3.5.0](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.5.0), published 2026-10-20.
- Tag `v3.5.0` resolves directly to the source commit above. The publish workflow downloaded and promoted the immutable prepare candidate without rebuilding.
- The release is unsigned. Its public notes disclose the missing publisher signature and possible Windows unknown-publisher or SmartScreen warning.

| Published artifact | Candidate and published SHA-256 |
|---|---|
| `DILIGENT-v3.5.0-windows-x64-portable.exe` | `981fe5e3bf1db072f0e210de85c63b6baaa17be5f6a1027915124ac846435204` |
| `DILIGENT-v3.5.0-windows-x64.msi` | `4c6ff76d69693564b7889c26bec4784602324b404d539886ded5ab7716df8916` |
| `DILIGENT-v3.5.0-windows-x64.sha256` | `8ee62e76c9f1e175a975f01dc46e4616899e4259d8cc314dde4554031286487e` |

The successful publish run retained distribution verification. Public release asset digests match the candidate receipt for the EXE, MSI, checksum manifest, and `candidate.json`.

## Validation evidence

- The hosted prepare run passed its required release security, release preflight, LiveProvider E2E, persistence preflight, and MSI lifecycle checks. The immutable candidate receipt is attached to the prepare run.
- On the local Windows 11 Pro build 26200 host, the exact portable EXE hash was rechecked. The app rendered DILI Agent and other inspected pages, `/api/health` returned `{"status":"ok"}`, and the app closed cleanly with no remaining candidate process or listener.
- The local smoke captures below were taken from the exact portable candidate in a fresh isolated profile and document launch/navigation only. The DILI Agent page shows the untouched default form with no generated assessment; Clinical Sessions reports no sessions; Data Inspection reports no RxNav rows. No analysis or catalog update was started. These screenshots do not evidence a populated clinical workflow, imported data, or catalog behavior.

  - [DILI Agent: default launch view, no assessment](native-home-page.jpg)
  - [Clinical Sessions: empty isolated profile](native-sessions-page.jpg)
  - [Data Inspection: empty RxNav list in isolated profile](native-data-inspection.jpg)
- The canonical [project status ledger at the candidate source commit](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/blob/7a1b23e6f27b3d1987f1106eb2fb5f9aa40b2820/assets/docs/project_status_ledger.md) records previously validated synthetic analysis, UI, catalog, and supporting feature coverage.

## Explicit waivers and limits

At the release owner's direction, the protected approval relied on the existing validation ledger and waived a new populated native assessment and a separate clean-host Windows 11/WebView2 run for v3.5.0. The approval comment records those waivers. The local smoke verifies package startup and health; it does not establish a populated assessment or clean-host compatibility for this candidate.

The synthetic clinical case entered during the local smoke was not submitted. Its fresh profile had empty source catalogs; the combined source refresh was cancelled at 11,000 RxNav records before any clinical analysis job was created. The host had WebView2 installed, but the operating system itself was not a clean Windows image.
