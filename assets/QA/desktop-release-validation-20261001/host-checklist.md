# DILIGENT 3.4.0 host-dependent release-readiness checklist

Candidate source SHA: 2f804d251769f4e01375d02459e41dd9970ec8eb

Candidate artifacts:

- release/DILIGENT-v3.4.0-windows-x64-portable.exe
- release/DILIGENT-v3.4.0-windows-x64.msi
- release/DILIGENT-v3.4.0-windows-x64.sha256

This checklist is intentionally unclaimed on the current workstation. It
requires an administrator-capable, interactive Windows host and must be
completed before a later release decision.

## Interactive packaged UI

Progress on the packaged portable surface is recorded in
[the 2026-10-01 packaged-UI validation report](../desktop-interactive-ui-validation-20261001/report.md),
which drove the packaged Tauri/WebView2 window with the available native UI
automation and confirmed workspaces, Settings save/reload/reset/restart
persistence, the preflight boundary, the API boundary, and clean
close/restart behavior.

- [x] Launch the exact portable candidate and capture the branded window.
- [x] Exercise DILI Agent startup, required-input preflight (blocked on fresh
      data), and the preflight dialog.
- [ ] Exercise Sessions, Timeline, Data Inspection (rendered; populated
      session/timeline flows still require a populated packaged data root).
- [x] Verify provider/model settings, reset/reload, and restart persistence
      (General polling interval; Models catalog/config surface).
- [ ] Use the native RAG folder picker and verify ingestion/retrieval state.
- [x] Exercise normal close, restart, and no-leftover-process/port behavior.
- [ ] Exercise a full synthetic analysis, cancellation/retry, and failure
      recovery in the packaged window.
- [x] Record screenshots and this interaction log in this QA folder.
- [x] Confirm no packaged backend process or listener remains after shutdown.

## MSI lifecycle

- [ ] Verify the v3.3.0 input MSI and checksum from the existing upgrade-input
      evidence.
- [ ] Install v3.3.0 on the disposable administrator-capable host.
- [ ] Launch it, create a representative database/settings state, and record
      the persistence marker and before-upgrade database evidence.
- [ ] Upgrade to the exact v3.4.0 MSI above with verbose Windows Installer
      logging.
- [ ] Verify ProductVersion 3.4.0, launch, settings, database state, and
      persistence-marker survival.
- [ ] Uninstall v3.4.0 and verify the documented user-data retention policy.
- [ ] Reinstall v3.4.0 once and repeat the launch/health check.
- [ ] Store sanitized MSI logs and before/after comparison results in this QA
      folder.

## Distribution procedures

- [ ] Sign the EXE and MSI if signed distribution is required.
- [ ] Verify Authenticode signatures and compare signed-file hashes with the
      final checksum manifest.
- [ ] Build and test Offline WebView2 only if that distribution mode is
      selected.
- [ ] Run one clean-machine install/launch/shutdown smoke with no repository
      development environment.
- [ ] Record the final distribution decision separately from source validation.

## Current host boundary

The current token is not administrator-capable and the available Computer Use
surface exposes no native desktop window. The local candidate therefore has
portable process/API, exact deep-path, and now a native-automation-driven
packaged-UI slice (workspaces, Settings persistence, preflight, clean
close/restart), but no claimed MSI lifecycle pass. MSI install, launch,
upgrade, uninstall, and reinstall still require an administrator-capable
host; full synthetic analysis and RAG folder flows require a populated
packaged data root.
