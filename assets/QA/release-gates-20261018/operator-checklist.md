# v3.4.0 operator acceptance
Last updated: 2026-10-18

Status: **UNRUN — awaiting the exact candidate.** Do not approve publication based
on the earlier portable candidate or source/browser checks alone.

Signing was explicitly waived by the user on 2026-10-18. This checklist retains
the native, MSI, and clean-machine gates; it does not require a publisher certificate.

## Candidate identity

Download `desktop-candidate-<prepare-run-id>-<attempt>` from a successful
Windows Desktop Release `prepare` run. Extract into a disposable QA directory.
Keep `candidate.json`, `qa/smoke.json`, and the checksum manifest together.
Record source SHA, run ID/attempt, EXE/MSI SHA-256, explicit `NotSigned` status,
Windows build, WebView2 version, model IDs, test date, and operator identity.
Verify artifact hashes and confirm the approved unsigned disposition before executing.
Store the completed report and screenshots under `assets/QA/`; include the
candidate hashes in the evidence reference supplied to `publish`.

## Populated native portable workflow

Use an isolated `LOCALAPPDATA` root and explicitly synthetic clinical inputs.
Use exact configured provider/model lanes and record their provenance; avoid
provider fallbacks. Existing local `qwen3.5:9b` is the analysis/timeline control.
Use the configured `opencode_go / deepseek-v4-flash` lane for revision when
available. Manage credentials through approved runtime/CI mechanisms; never
paste secrets into screenshots, reports, or fixtures.

- [ ] Open all four native workspaces; save/restart/reset Settings.
- [ ] Confirm one missing-date blocker, no job/session after rejection, and
  disappearance of that blocker when the date is supplied.
- [ ] Complete a populated synthetic analysis; inspect report/evidence and export.
- [ ] Inspect the saved session/version, update metadata, edit the report, and
  verify manual-edit lineage and audit. Exercise deletion on a disposable copy.
- [ ] Generate and render a grounded timeline; verify evidence, dates, and reload.
- [ ] Complete revision or reach an explicit persisted QA rejection. A timeout
  alone leaves revision acceptance pending. Verify source preservation and lineage.
- [ ] Select a real QA document folder with the native picker; ingest two
  distinct supported documents and a byte-identical duplicate. Verify two unique
  documents, deduplication, retrieval, and citations in a RAG-enabled analysis.
- [ ] Exercise native image selection/cancellation and return to the application.
- [ ] Cancel a running job, verify terminal state and absence of partial saves,
  then successfully start another job.
- [ ] Close/relaunch, verify persisted state, close again, and confirm shell,
  backend, and observed listening ports are gone.

## Clean Windows 11 x64 / MSI distribution

Use an administrator-capable disposable Windows 11 host with no DILIGENT
installation. Preserve a VM snapshot if available. The supported release mode
is the existing **online** WebView2 bootstrapper; offline certification is excluded.

- [ ] On a clean host without WebView2, install the exact unsigned MSI with network
  access; record bootstrapper/installer behavior and exit code.
- [ ] Launch as a standard user, inspect the window/health contract, and close
  cleanly without administrator credentials or development tools.
- [ ] Confirm the installed executable is `NotSigned`, matching the approved policy.
- [ ] Confirm install/upgrade/uninstall/reinstall receipts from the prepare run.
  Re-run on the disposable client host when needed using the checksum-verified
  published v3.3.0 MSI and the maintained smoke harness. Preserve installer logs.
- [ ] Confirm sessions, report versions, and settings survive a genuine upgrade
  and uninstall; confirm program files/registration are removed after uninstall.

## Publication approval

Review the candidate hashes, hosted gate receipts, and completed native/client
evidence. Dispatch `publish` on synchronized `main` with the same version,
successful prepare run ID, and the completed evidence reference. Approve the
`desktop-release` environment only after every required item passes. Record
the approval's attestation in the GitHub review comment. Leave missing or failed
items pending; do not approve using a source smoke or a different candidate.
