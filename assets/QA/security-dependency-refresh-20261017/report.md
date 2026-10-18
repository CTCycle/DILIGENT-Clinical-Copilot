# Security dependency refresh — 2026-10-17

Last updated: 2026-10-18

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

## Fresh packaged candidate validation

The v3.4.0 candidate was rebuilt from clean `develop` commit
`37624def8a0f3eba0dfb9990f536aea4b3c0e8a9` with the updated dependency locks.
The pinned desktop build used Rust 1.95.0, Python 3.14.7, uv 0.11.30, Node
22.13.0, and PyInstaller 6.21.0. The command
`start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.4.0 -DesktopTarget All`
completed, including the frontend build, frozen-backend first-run health smoke,
runtime archive validation, Rust/Tauri release build with `zip 9.0.4`, MSI
bundle, PE/MSI metadata checks, and checksum manifest generation.

- Runtime archive: 1,108 files; SHA-256
  `9375d0600bbe6f4b714633ad85894be31a56f4e4ca0272d4d6dcb5a01103a929`.
- Portable EXE: 171,780,096 bytes; SHA-256
  `a74bbfd53a6f3c30672a44f9ec24ca871aac4412b0d8dc0ffbf56c3d040bbcb1`.
- MSI: 170,233,856 bytes; SHA-256
  `808d766ed11f61196fa308aa012d653e190b0d6d13e7571b348ba6cc6643c6c2`.
- The SHA-256 manifest matches both artifact hashes. The portable smoke also
  independently matched the EXE hash against the manifest.

The fresh portable EXE passed two launches with the packaged smoke harness.
Each launch reported version 3.4.0, health 200, HTTP 401 for all tested
unauthenticated protected endpoints, the branded window title, and clean
backend and listener shutdown. Runtime extraction completed, and the second
launch replaced the stale ready-file payload.

The exact deep-path boundary was replayed twice with the fresh EXE copied to an
executable path of 230 characters and `%LOCALAPPDATA%` set to a 107-character
path. Both launches returned health 200 and settings 401, showed the branded
title, completed runtime extraction, and closed their backend process and
port. No ONNX Runtime import or Windows filename-length failure occurred.
Temporary executable and data roots were removed.

Read-only MSI metadata is valid: ProductName `DILIGENT Clinical Copilot`,
ProductVersion `3.4.0`, Manufacturer `CTCycle`, ProductCode
`{7705CE15-1681-4B44-A6FF-D40066B3194B}`, and the established UpgradeCode
`{2CF8EF35-4160-59EB-89D8-01EC7D19A887}`. Both artifacts report `NotSigned`.

The portable smoke printed Chromium `Failed to unregister class
Chrome_WidgetWin_0` Win32 error 1412 during normal window close. The harness
still observed normal shell exit, backend exit, and port closure on every
launch; no packaged launch or health assertion failed.

## Portable browser-surface workflow parity — 2026-10-17

### Method

Compared the current source backend at `37624def8a0f3eba0dfb9990f536aea4b3c0e8a9`
with the frozen backend extracted from the fresh portable candidate. Both were
opened in the in-app browser and served the exact candidate Angular bundle:
**16 files, zero mismatches**. The generated `app/client/dist/browser` in the
checkout differed from the candidate in five of its 16 files, so it was not
used as the comparison frontend. The two apps used separate disposable SQLite
roots with identical synthetic catalog rows and runtime settings. Cloud
provider keys were empty/unset, and RAG retrieval was off.

### Results

- Shared navigation rendered DILI Agent, Clinical Sessions, Data Inspection,
  and Settings in both. Initial session/catalog empty states matched.
- Settings → General changed job polling from 1 to 2 seconds, saved the same
  success message, retained 2 after reload, then reset to 1 with the same
  default-reset message. Settings → Models listed installed local
  `qwen3.5:2b` and `qwen3.5:9b`; the clinical model was set to `qwen3.5:9b` in
  both isolated databases.
- With empty catalogs, the same intentionally brief synthetic input produced
  the same four preflight blockers: empty LiverTox catalog, empty RxNav catalog,
  input under 60 words, and missing `anamnesis`, `drugs`, and
  `laboratory_analysis` sections. No session was created. Matching synthetic
  amoxicillin catalog rows then appeared in Data Inspection on both sides.
- The same 143-word explicitly synthetic amoxicillin input completed through
  the local Ollama workflow on both backends. Both showed identical report
  content, enabled Copy/Expand/Download report, and navigated to the same
  `Successful` Session 1 / Version 1 preview. Browser console error logs were
  empty for both tabs.
- The UI classified the synthetic case as indeterminate with R-score N/A;
  numerical RUCAM was not calculated, Hy's Law was not assessable, and the
  report required clinician review. Read-only SQLite inspection confirmed one
  session/version per side, `session_status=successful`,
  `version_status=current`, `llm_qa_status=not_run`, and
  `clinical_review_status=not_reviewed`. Each stored report was 8,294
  characters with identical SHA-256
  `4915cdcc980beba707aa52227f9a40f3f018e6254850238a70a3149b7096f30a`.
  No provider keys were stored. This is workflow/software parity evidence, not
  clinical-quality evidence.

The browser comparison does not cover Tauri-only chrome, native window
behavior, OS integration, or native file dialogs. UI automation on this host
exposed browser tabs but no native-window inventory; overall package UI parity
therefore remains **PARTIAL**. Detailed machine-readable observations are in
the local-only
[`portable-ui-parity-evidence.json`](portable-ui-parity-evidence.json) QA file.

## Non-elevated MSI static inspection — 2026-10-17

Read-only Windows Installer inspection confirmed ProductName `DILIGENT Clinical
Copilot`, ProductVersion `3.4.0`, Manufacturer `CTCycle`, ProductCode
`{7705CE15-1681-4B44-A6FF-D40066B3194B}`, UpgradeCode
`{2CF8EF35-4160-59EB-89D8-01EC7D19A887}`, and embedded `app.cab`. The MSI File
table declares `diligent-desktop.exe`, version `3.4.0.0`, and 171,780,096
bytes. Its cabinet was extracted to a disposable user-space path without
installing the MSI or changing machine state. The extracted shell PE is valid,
version 3.4.0, and the expected size; its SHA-256 is
`0eb5ba98b8e7597508ccad93bf70559176a90a2725885c66eef684256cbace96`, which
differs from the portable EXE SHA-256. The embedded runtime archive validates
to the same digest as the portable runtime:
`9375d0600bbe6f4b714633ad85894be31a56f4e4ca0272d4d6dcb5a01103a929` (1,108
files, 400,259,530 bytes). The extracted MSI payload also passed a user-space
backend launch and `/api/health` 200 check. Both EXE and MSI remain `NotSigned`.
These checks do not establish MSI install, upgrade, uninstall, reinstall, code
signing, or distribution.

## Native portable Tauri smoke — 2026-10-18

Launched the v3.4.0 portable candidate from `develop` commit
`37624def8a0f3eba0dfb9990f536aea4b3c0e8a9` with an isolated temporary
`LOCALAPPDATA` root. The native window opened, and the DILI Agent, Clinical
Sessions, Data Inspection, and Settings areas rendered. In Settings, changing
the job polling interval from 1 to 2 seconds survived navigation and a clean
restart; resetting defaults restored 1 second across restart.

The native RAG folder chooser opened and selected an empty disposable QA
directory. The native patient-image chooser opened and was dismissed without
selecting a file. Closing the app left no package process and no listening
socket on either observed runtime port. The temporary data root was isolated
from user app data.

An intentionally short synthetic draft was submitted through the native DILI
Agent form. The native preflight refused to start analysis with 5 blocking
items and 1 warning: empty LiverTox and RxNav catalogs, input under 60 words,
and two visit-date blockers. The date blockers share the same visible title
(`Visit date is missing`), so the current UI displays that requirement twice.
After returning to input, Clinical Sessions showed `No clinical sessions
found`; no session was created and no model/provider call was made. Screenshots:
[preflight blockers](native-empty-preflight-20261018.jpg) and
[empty session list](native-empty-session-list-20261018.jpg).

This closes the native-window and basic native-dialog availability gap for
this candidate. It is a shell smoke: the current packaged native window did
not repeat populated analysis, session/revision, or RAG ingestion workflows.
The duplicate date blocker is a small preflight UX cleanup item. Overall
packaged UI parity therefore remains **PARTIAL** pending broader native
workflow evidence.

## Boundaries

Fresh local v3.4.0 portable, MSI, and checksum artifacts now represent the
current source and the portable runtime/deep-path gates pass. Shared
browser-surface workflow parity is demonstrated, and a current-candidate
native shell/dialog smoke passed; overall packaged UI parity remains `PARTIAL`
pending broader native workflow evidence. The current host token is not
administrator-capable, so MSI install, launch, upgrade, uninstall, and
reinstall remain pending on a suitable administrator host. No DILIGENT entry
was found in the current-user or machine Add/Remove Programs registry paths,
so there is no registered prior MSI install on this host to use as an upgrade
baseline. Both artifacts
remain `NotSigned`, and no local signing certificate or `signtool.exe` is
available. Signing, clean-machine distribution, offline WebView2 certification,
and tag/release publication remain separate procedures.
Provider-dependent checks remain pending; the push-triggered `live-provider-e2e` job in run
[39033116041](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39033116041)
was skipped; no external provider credentials were used and no live-provider
evidence is claimed. Documentation-push CI run
[39033542775](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39033542775)
also passed. A read-only check of the [latest GitHub release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/latest)
on 2026-10-18 returned `v3.3.0`; lookup of the
[`v3.4.0` Git tag ref](https://api.github.com/repos/CTCycle/DILIGENT-Clinical-Copilot/git/ref/tags/v3.4.0)
returned 404. The remote `develop` and `main` branch endpoints returned
`37624def8a0f3eba0dfb9990f536aea4b3c0e8a9` and
`3e341c5a84def4805c6d4ddbf36b67a9071dad3a`, respectively. No tag, v3.4.0
GitHub release, upload, or publication was performed.

## Superseded candidate — release-gate follow-up

The 2026-10-18 release-gate source changes supersede these unsigned 3.4.0
binaries. Retain this report and its hashes as historical evidence. The duplicate
visit-date finding is fixed and revalidated in source; signing, MSI execution,
and native/clean-machine acceptance require the new exact candidate. See the
[release-gate follow-up](../release-gates-20261018/report.md).
