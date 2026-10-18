# v3.4.0 operator acceptance
Last updated: 2026-10-18

Status: **RELEASED — 2026-10-18.** The approved publish workflow completed successfully, and the public release, tag, asset digests, downloaded bytes, and portable launch were verified.

The unsigned release policy was authorized on 2026-10-18. The candidate EXE/MSI are NotSigned. The workflow-generated public release notes state that the binaries have no publisher signature and may trigger an unknown-publisher or SmartScreen warning.

## Candidate identity

| Field | Value |
|---|---|
| Source | d4efa5e0bccfa552c80a5edeff2168bfaa718639 |
| Prepare run | 39109627873 — successful, dispatched from main |
| Portable EXE SHA-256 | 702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26 |
| MSI SHA-256 | 3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d |
| Windows build / WebView2 | 26200.9457 / 155.0.4283.60 |
| Local analysis and timeline model | Ollama qwen3.5:9b |
| RAG embedding model | ONNX Runtime granite-embedding-97m-multilingual-r2 |
| Hosted live-provider lane | OpenCode Go deepseek-v4.1-flash |
| Test date | 2026-10-18 |

Artifacts, report, export, screenshots, and isolated synthetic data are under assets/QA/release-gates-20261018/ and assets/QA/native340/. The exact export hash and other detailed results are in report.md.

## Publication readback

- [x] Required `desktop-release` approval recorded; publish run [39125173765](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39125173765) succeeded.
- [x] Public release [v3.4.0](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0) is GitHub's latest release; published 2026-10-18 11:59:42 UTC.
- [x] Tag `v3.4.0` resolves to the accepted source SHA `d4efa5e0bccfa552c80a5edeff2168bfaa718639`; publish promoted the selected candidate without a rebuild.
- [x] Published EXE/MSI digests and downloaded byte hashes match the candidate hashes above.
- [x] Downloaded portable EXE launched, rendered DILI Agent, returned `/api/health` status `ok`, and closed gracefully. See native-17-release-downloaded-start.png.
- [x] Generated public release notes disclose that the binaries have no publisher signature.

## Populated native portable workflow

All flows used the exact portable candidate and synthetic data. The app profile and temp paths were isolated.

- [x] DILI Agent, Clinical Sessions, Settings, and RAG/Data Inspection flows exercised; populated report, export, session/version, and timeline evidence recorded. See native-01-populated-form.png, native-02-analysis-running.png, native-03-report-complete.png, native-04-timeline.png, native-05-rag-documents.png, and native-export-clinical_report.md.
- [x] Missing-date submission showed exactly one blocker, created no job/session, and cleared when date 2026-10-18 was restored. See native-06-missing-date-blocker.png and native-07-date-restored.png.
- [x] Patient image picker opened and was cancelled; app remained usable. See native-08.
- [x] Running analysis cancelled through Stop with terminal cancelled state and no partial session. A second started job was also cancelled; the database remained unchanged. See native-09 and native-10.
- [x] Settings interval 3 saved, survived close/relaunch, and was reset to default 1. See native-11 and native-12.
- [x] Close/relaunch retained session 1 version 2, session 2 version 1, and session 1 timelines. See native-13 through native-15.
- [x] Disposable RAG session 2 deleted in the UI. Read-only database check showed 1 session, 2 versions, 1 result, and 2 timelines; session 1 remains. See native-16.
- [x] Final app/backend shutdown verified; ports 60056, 9223, and 62172 were free.
- [x] Candidate hosted LiveProvider E2E passed in prepare run 39109627873. Native local timeline generation timed out twice and saved the expected deterministic fallback: 3 grounded events, 3 with evidence. This local timeout does not replace or invalidate the hosted provider result.
- [x] RAG ingested 2 canonical documents, skipped 1 byte-identical duplicate, and cited both source documents in session 2.
- [x] Manual report edit produced version 2 with manual_edit lineage and matching previous-text audit hash.

Known issue in v3.4.0: after metadata is saved, a manual report edit submits metadata: {} and overwrites it. Metadata was saved again and persisted. The issue is documented for follow-up and was not changed in this release; details are in report.md.

## Hosted Windows and MSI gates

The successful prepare run 39109627873 passed 864 backend tests, 53 browser tests, PostgreSQL persistence, MSI lifecycle, portable and deep-path smoke, and live-provider E2E. The unsigned candidate receipt and artifact hashes are retained with that run.

- [x] Portable and deep-path smoke — prepare run 39109627873.
- [x] MSI install/launch/upgrade/uninstall/reinstall lifecycle and populated persistence — prepare run 39109627873.
- [x] NotSigned status — accepted under the 2026-10-18 unsigned release policy.
- [x] Public release-note disclosure — generated release notes explicitly say the binaries have no publisher signature.
- [WAIVED — 2026-10-18] Clean-host WebView2 installation check. Validation was limited to one Windows machine. The observed WebView2 version is recorded above; this is not clean-host compatibility evidence.

## Approval and publication

The user completed the single approval click for the `desktop-release`
environment. Publication finished successfully before post-publish documentation
was committed. The dated single-machine WebView2 waiver remains in force as
recorded above and in report.md.
