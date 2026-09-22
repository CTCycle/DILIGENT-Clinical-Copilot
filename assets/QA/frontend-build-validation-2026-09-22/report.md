# Production frontend build host validation
Last updated: 2026-09-22

## Scope

This run investigated `test.automated-regression` at `develop` HEAD
`7d797e8a7bae10e86fa28f4e53409f8f81ccbb26`. The prior ledger snapshot was at
`5960fe29d4f6b6b41dc235d5dd177cb420a9487b`; the intervening commit changed
only the project and validation ledgers. The worktree was clean at task start.
No application source or test files changed during this run.

Host: Windows 10.0.26200.0, PowerShell 7.6.6. The launcher-provisioned runtime
reported Node.js `v22.13.0` and npm `10.9.2`.

## Local frontend gates

| Gate | Result |
|---|---|
| Fresh dependency installation | The first `npm ci --ignore-scripts --no-audit --no-fund` used a new isolated cache and was denied registry fetches with `EACCES`; npm then reported `Exit handler never called`. Retrying the same command with network access installed 471 packages successfully. |
| Frontend tests | `npm run test -- --no-watch` passed: 23 test files, 97 tests, exit code 0. |
| Direct production build | `npm run build -- --progress=false` terminated with exit code `-1073741819` (`0xC0000005`) under the pinned Node runtime. No Angular diagnostic or build output appeared. |
| Launcher rebuild | `start_on_windows.ps1 -Action RebuildFrontend` confirmed Node `v22.13.0`, reused the installed dependencies, then terminated on the same `0xC0000005` while running `npm.cmd run build`. |

Command transcripts and the crash-diagnostic output were captured in the workspace during this run; the gate outcomes and relevant runtime details are summarized above.

The queried Windows Application log had no recent matching event 1000/1001,
and no recent `node.exe` crash dump or WER report was found in the user-local
locations checked. The native faulting module remains unknown.

## Exact-SHA hosted Windows comparison

GitHub Actions [CI run 35764186160](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35764186160)
ran on `develop` at the exact HEAD above. Its Windows job passed `Setup Node`,
frontend dependency installation, frontend tests, and `Build frontend` using
the workflow-pinned Node.js `22.13.0`. The overall run was not green: the
Windows job failed later at `Run full browser E2E suite`; the separate
`security-scan` and `backend-quality` jobs also failed, while
`persistence-contract` passed and `live-provider-e2e` was skipped. This
comparison validates the hosted production-build step only, not the complete
CI or release matrix.

## Launcher output reuse

Before the launch checks, both `app/client/dist/browser/index.html` and
`app/client/dist/.diligent-build-state.json` existed. The marker recorded:

- Build fingerprint: `00c6a91871a036c7184efd802d3f00a51da8744102e5ac445fd986b0764e2907`
- Dependency fingerprint: `559eff725b1f551222e2ffe3624a7fa70769fdc6182192fb81df30ff1d95cc22`
- Node version: `22.13.0`
- Build command: `npm run build`

Two executions of `start_on_windows.ps1 -Action Launch` reported that the
frontend build was current and skipped dependency synchronization and Angular
build. Both backend health and frontend requests returned HTTP 200; the served
page title was `DILIGENT Clinical Copilot`. The served index SHA-256 was
`8085087A6FFFD895D7B181377C37D369B2729AF7C88C44D29EAF063DEE6F1353`.

The launcher action then exited 1 because the sandbox denied its attempt to
open `http://127.0.0.1:9847` in the default browser. The backend and frontend
were already serving successfully. This validates reuse of the existing
fingerprinted output; it does not establish that this host generated fresh
production output during this run.

Task-owned listeners were stopped only after their PIDs and executable paths
were rechecked; the backend console wrappers were also stopped. Ports `7690`
and `9847` were free after cleanup.

## Classification

The local access violation is classified as specific to this isolated
validation host: direct and launcher builds fail locally, while the exact same
SHA passes the production-build step on hosted Windows CI with the supported
Node version. No repository-owned build cause was reproduced, so no
application workaround was added. Because this host still cannot produce a
fresh build and has no crash dump identifying the native fault, keep
`test.automated-regression` `PARTIAL` until a fresh pinned build succeeds on
this host or equivalent host-level diagnostics resolve the execution failure.
