# Automated regression local-build follow-up
Last updated: 2026-09-24

## Scope and selection

This follow-up started from clean develop HEAD
13f6c7a079122934e4c614452421dd1ef8792e54, equal to origin/develop. Its
parent, 3e73082f619024140cfdb92340831f450c6a3e13, is the source commit covered
by hosted CI run 35908957190; the current HEAD adds validation documentation
only.

The current status ledger named the fresh local production frontend build as
the next action for test.automated-regression. This run repeated that exact
gate through the official RebuildFrontend launcher action on the pinned Node
runtime. The local frontend unit suite was rerun alongside it. No application
source, settings, database, credentials, or provider state was changed.

## Environment and results

Host: Windows NT 10.0.26200.0, PowerShell 7.6.6. The launcher used Node.js
22.13.0 and npm 10.9.2.

| Gate | Result |
|---|---|
| Frontend unit suite | From app/client, npm run test -- --no-watch passed with exit code 0: 24 test files and 105 tests. |
| Official fresh production build | start_on_windows.ps1 -Action RebuildFrontend passed with exit code 0. It reused installed dependencies and ran npm run build. Angular completed production bundle generation in 9.894 seconds and wrote app/client/dist. |
| Build-state marker | Present after the successful forced rebuild. It records build fingerprint 4edd07b8da14d6a099c9e5c9484a2091d5368c666ceef3befa0c9b9b2f446f5e, dependency fingerprint 559eff725b1f551222e2ffe3624a7fa70769fdc6182192fb81df30ff1d95cc22, and Node 22.13.0. |
| Exact-SHA hosted CI | [Run 35966513019](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35966513019) completed successfully on commit 1d1cfa8f068fe4e81717e59e11af27d55aaf5721. Security scan, backend quality, Windows regression, and persistence passed. The live-provider job was skipped because the run event was push; the workflow requires workflow_dispatch with run_provider_e2e enabled. |
| Launcher log | [launcher-rebuild.log](launcher-rebuild.log) records the action and successful Angular output. |
| Frontend test log | [frontend-test.log](frontend-test.log) records the 24-file, 105-test pass. |
| Cleanup | No application server was started. Ports 7690 and 9847 were free after the checks; no task-owned build or test process remained. |

The previous 0xC0000005 local production-build failure did not reproduce on
this checkout and host. This single successful rebuild does not identify the
earlier native fault or prove it cannot recur.

## Final gate status

test.automated-regression remains PARTIAL. The previously missing fresh local
build now passes, as do the current local frontend tests. Hosted run 35908957190
passed its configured jobs on the same application source commit. The
subsequent exact-SHA run 35966513019 passed all configured jobs on the commit
that carries this report. Its live-provider job was skipped because it ran on
push rather than workflow_dispatch with run_provider_e2e enabled. Seven
browser E2E cases were conditionally skipped in the prior detailed report for
live-provider and embedding opt-in, unavailable Ollama models, or absent
persisted CI sessions. Those provider, embedding, model-availability, and
persisted-session boundaries still need their own evidence.

## Other incomplete ledger items reviewed

These gates were considered while selecting the build slice. Their linked
evidence and status remain the latest recorded evidence; this build run did
not exercise them.

| Component or gate | Current status | Remaining work and next action |
|---|---|---|
| api.local-boundaries | WORKING | The complete route catalog lacks one current browser/API pass. Run a focused route-contract slice when it is next needed. |
| auth.access-key-management | BLOCKED | Requires an approved disposable credential and explicit cleanup authorization. No credential was read or changed. |
| model.provider.opencode-go | PARTIAL | Broader provider failure, retry, QA, and revision variants remain uncertified. Run the workflow-dispatch provider lane and retain exact route/no-fallback evidence when its credential and dispatch prerequisites are available. |
| model.provider.local-ollama and sessions.timeline | PARTIAL | The latest two qwen3.5:2b runs failed source-evidence validation and saved invalid_response fallbacks; two qwen3.5:9b runs passed grounded evidence and attribution checks. The next coherent model slice is a fresh isolated 2B/9B timeline run with evidence, date, provenance, rendering, and reload checks; keep both rows partial until grounded 2B output is observed. |
| data.inspection.catalogs and data.sources.refresh | PARTIAL | The last source report records an NCBI Bookshelf CAPTCHA stopping the ordered refresh at LiverTox. Retry the full ordered update and failure-preservation checks when the master-list endpoint is available; do not bypass human verification. |
| revision.agentic-lifecycle | PARTIAL | One accepted synthetic child and subsequent source-version selection are covered. Broader live-model consistency and lifecycle branches remain separate from the build gate. |
| release.desktop.v3-4-0 | BLOCKED | Release authorization, signed material, synchronized main/tag, hosted Windows packaging, and clean-machine install/upgrade/uninstall evidence remain prerequisites. |
| clinical.analysis.pipeline | VALIDATED, with validation debt | The named exact-provider synthetic workflow passed. Additional provider failure/retry variants and a RAG-off lane remain. |
| rag.ingestion-retrieval | VALIDATED, with validation debt | Unsupported-file, arbitrary-folder, and duplicate-file policy coverage remains. ISSUE-005 still needs a deduplication or canonical-citation policy decision before related remediation. |
| ui.application-shell | VALIDATED, with validation debt | Full keyboard, screen-reader, and responsive-viewport review remains outside the exercised desktop surfaces. |

## Evidence boundary

This follow-up validates the local frontend test and fresh production-build
gates. It does not claim new backend, provider, clinical, source-refresh,
access-key, release, accessibility, or full browser E2E evidence. The next
actionable coherent validation slice is the isolated local Ollama timeline
pair covering sessions.timeline and model.provider.local-ollama.
