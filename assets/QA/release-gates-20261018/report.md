# v3.4.0 Release Evidence
Last updated: 2026-10-18

## Release identity and publication

- Source and tag: d4efa5e0bccfa552c80a5edeff2168bfaa718639.
- Prepare run: [39109627873](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39109627873), succeeded.
- Approved publish run: [39125173765](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39125173765), succeeded.
- Public release: [v3.4.0](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0), latest when read back on 2026-10-18.

| Artifact | Published and downloaded SHA-256 |
|---|---|
| Portable EXE | 702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26 |
| MSI | 3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d |

The published and downloaded bytes matched. The downloaded portable EXE rendered DILI Agent, returned an OK health response, and shut down cleanly. The hosted prepare run passed 864 backend tests, 53 browser tests, Angular/Vitest and production build, PostgreSQL persistence, LiveProvider E2E, dependency/security gates, portable/deep-path smoke, and the MSI install/launch/upgrade/uninstall/reinstall lifecycle.

The release is unsigned under the policy authorized on 2026-10-18. Public release notes disclose that the binaries have no publisher signature and may trigger an unknown-publisher or SmartScreen warning.

## Native populated acceptance

The published candidate was exercised on Windows 11 build 26200.9457 with WebView2 155.0.4283.60. Inputs and app data were synthetic and isolated. Local analysis/timeline used Ollama qwen3.5:9b; RAG used ONNX Runtime granite-embedding-97m-multilingual-r2. The hosted provider lane used OpenCode Go deepseek-v4.1-flash.

- A populated DILI assessment completed, persisted, and exported. The report remained a decision-support draft requiring human review.
- Missing visit date produced exactly one blocker and no job/session; supplying the date cleared the blocker.
- The image picker opened and was cancelled. Running jobs reached cancelled terminal state without creating partial sessions.
- Settings survived restart and reset to the default. Sessions, versions, and timelines survived restart; the disposable second session was deleted through the UI.
- RAG indexed two unique documents, skipped one byte-identical duplicate, and the report cited both sources.
- The local timeline model timed out twice. The application saved the deterministic three-event fallback with source evidence; this did not replace the independent passing hosted provider flow.
- App and backend closed cleanly with no remaining process or observed listener.

Representative captures:

- [Completed report](native-03-report-complete.png)
- [Saved timeline](native-04-timeline.png)
- [RAG documents](native-05-rag-documents.png)
- [Single missing-date blocker](native-06-missing-date-blocker.png)
- [Settings saved](native-11-settings-saved.png)
- [Downloaded release launch](native-17-release-downloaded-start.png)

## Known issue and waiver

A manual report edit sends metadata as an empty object from the client. The repository replaces saved session metadata with that object. Re-entering metadata restores it. This issue was present in the released version and remains in the current source; see the [project status ledger](../../docs/project_status_ledger.md) and [session guide](../../docs/user/sessions_timeline_and_data.md).

**Waived on 2026-10-18:** clean-host WebView2 installation was not tested because only one Windows machine was available. The recorded WebView2 version is evidence about that machine only, not clean-host compatibility.
