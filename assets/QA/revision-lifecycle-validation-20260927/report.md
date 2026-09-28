# Revision lifecycle validation — 2026-09-27

## Scope and evidence boundary

- Repository: `develop`, source revision `f68ab5cfc86c27d9c40329a1b452d5339568aaa8`, equal to `origin/develop` at validation start.
- Scope: current-tree revision-agent contracts, revision API/OpenAPI boundaries, persisted accepted and cancelled revision lineage, reload/reopen behavior, rendered Revision audit UI, and adjacent focused regression coverage.
- Data: a copy of the existing synthetic revision database was used at task-local runtime scope. The shared `resources/database.db`, settings, credentials, and provider caches were not mutated.
- Provider boundary: no new cloud or local LLM request was made. The persisted accepted run exposes the exact historical route `opencode_go / deepseek-v4-flash`. The local validation process did not receive `OPENCODE_GO_API_KEY`; repository/GitHub secret availability was not independently inspected, and a repository secret is not automatically injected into a local PowerShell or manually launched runtime. The installed Ollama client was not substituted for that route and no Ollama service was available for a fresh revision run.
- The official launcher was attempted first. It stopped safely before process creation because a pre-existing listener occupied source UI port `9847` and the invocation was noninteractive, so it could not request termination. The isolated manual source fallback used backend port `7690` and preview port `9848`; no launcher defect is inferred.

## Current implementation checks

| Check | Result |
|---|---|
| Revision-agent and persistence suite | `31 passed`, 2 existing warnings |
| Adjacent context/generation/transport/job/repository/session-contract suite | `71 passed`, `1 skipped`, 2 existing warnings |
| Clinical safety resilience and OpenAPI/revision-route suite | `18 passed`, 3 existing warnings |
| Canonical Angular/Vitest runner | `24 files`, `105 tests passed` |
| Backend health | `GET /api/health` returned `{"status":"ok"}` |
| OpenAPI catalog | `68` paths exposed; revision paths and response models present |
| SQLite integrity | `PRAGMA integrity_check=ok`; `PRAGMA foreign_key_check` returned no rows |

The first direct `vitest` invocation omitted Angular's test-environment setup and failed before meaningful component execution. It was not used as product evidence; the repository-standard `npm run test -- --no-watch` runner passed the complete current frontend suite.

## Persisted revision/API evidence

The isolated clone contained the following current readable records:

- Accepted version `61`: `session_id=24`, `root_session_id=19`, `source_version_id=39`, `version_status=llm_qa_passed`, `llm_qa_status=passed`, `clinical_review_status=not_reviewed`.
- Accepted run `3c2f2186c2d443759ff1ce871dc7f91e`: `status=completed`, source session `22`, exact provider/model metadata `opencode_go / deepseek-v4-flash`, seven completed pipeline steps, and five artifacts: context, plan, tool trace, generated draft, and passed QA.
- Follow-on version `62`: `source_version_id=61`, no child session, `version_status=cancelled`, `llm_qa_status=not_run`; run `886026e90e37416b9a9188947b6be9e4` remains `cancelled`.
- Read-only API calls returned the same version, run, step, artifact, and review records. The accepted version has no clinical review action, and no approve/reject action was performed.

## Rendered Browser evidence

At `1280 × 720`, the in-app Browser rendered the synthetic source session's Revision tab. The UI visibly showed:

- the configured Revision model and central model-management boundary;
- the notice that the generated draft is retained and the current report remains unchanged until a human decision;
- the accepted revision's ordered trace: planning, four revision tasks, report editor, and quality review;
- five persisted artifacts with the Quality review marked `Passed`;
- after browser reload and manual reopening of the source session, the same accepted draft, trace, artifacts, and review state;
- the separate child session's follow-on cancellation as terminal, with its cancellation message, three retained steps, and four retained artifacts.

These observations are rendered UI evidence, not a claim of spoken screen-reader output. The Browser tool exposed inline captures but no durable screenshot export path; no screenshot filename is invented.

## Final gate disposition

| Gate | Final status | Boundary after this slice |
|---|---|---|
| `revision.agentic-lifecycle` | `PARTIAL` | Current contracts, accepted/cancelled persistence, lineage, reload/reopen, manual-review boundary, and rendered audit state are revalidated. Fresh current provider execution, broader model variability, timeout/tool-failure/backend-restart injection, and every lifecycle branch remain incomplete. |
| `revision.accepted-session-finalization` | `VALIDATED` for the exact synthetic accepted path | Version `61` remains QA-clean, linked to child session `24`, and reloadable without replacing the source. Broader provider coverage remains under `model.provider.opencode-go`. |
| `api.local-boundaries` | `WORKING` for the exercised revision route scope | Revision version/run/step/artifact/review reads and the 68-path OpenAPI catalog passed; all response/error variants and mutating route groups remain outside this slice. |
| `test.automated-regression` | `PARTIAL` | The focused current-tree backend/frontend/API suites passed. Live-provider and conditional hosted/browser lanes remain separate. |
| `model.provider.opencode-go` | `PARTIAL` | Exact historical provider identity is preserved and visible; this local run made no current provider request because the key was not injected into the local process. A hosted or explicitly injected live-provider run is still required. |
| `auth.access-key-management` | `BLOCKED` | No approved disposable credential was available; no credential material was entered or mutated. |
| `data.sources.refresh` | `PARTIAL` | NCBI LiverTox human-verification remains an independent upstream blocker; no refresh was attempted. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged artifact, hosted packaging, clean-machine, and publication gates were not part of this source-mode slice. |
| `ui.application-shell` | `VALIDATED` for previously stated scope | The revision surface rendered without a regression at the tested desktop viewport; spoken screen-reader output remains unvalidated. |

## Cleanup and next action

The task-owned backend, preview, cloned database, logs, and pytest workspace were stopped or removed after evidence capture. The next actionable revision slice is a fresh non-dry run using the approved repository secret through its hosted workflow or an explicitly injected local process, followed by retry/recovery coverage; a working local Ollama service configured explicitly for the Revision role is the alternative. Do not approve the persisted revision without an authorized clinical reviewer.
