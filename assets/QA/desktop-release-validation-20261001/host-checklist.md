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

- [ ] Launch the exact portable candidate and capture the branded window.
- [ ] Exercise DILI Agent startup, required-input preflight, and a synthetic
      analysis.
- [ ] Exercise Sessions, Timeline, Data Inspection, and Settings.
- [ ] Verify provider/model settings, reset/reload, and restart persistence.
- [ ] Use the native RAG folder picker and verify ingestion/retrieval state.
- [ ] Exercise normal close, restart, cancellation/retry, and failure recovery.
- [ ] Record screenshots or an interaction log in this QA folder.
- [ ] Confirm no packaged backend process or listener remains after shutdown.

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
portable process/API and exact deep-path evidence, but no claimed screen-level
UI or MSI lifecycle pass.
