# Automated regression and live-provider validation — 2026-09-25

Last updated: 2026-09-25

## Selected scope

This run revisited the next actionable incomplete slice from the validation
ledger: `test.automated-regression` and the explicit live-provider boundary of
`model.provider.opencode-go`. It started from clean `develop` HEAD
`2081a076a0b02d13bc6305939c13658b1fe80494`, equal to `origin/develop`.

The current implementation was checked locally and through the repository CI
workflow. The regular push run [36151424087](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36151424087)
passed its configured core jobs. The explicit provider dispatch was then run
against the same SHA as [36153314442](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36153314442)
with `run_provider_e2e=true`.

No application code change was required by this slice. The only failure was an
external provider prerequisite, confirmed before any provider request was made.

## Current implementation and regression evidence

| Check | Result |
|---|---|
| Local backend unit suite | `793 passed`, 7 existing dependency/framework deprecation warnings |
| Hosted backend quality | `793 passed`, Alembic upgrade/head/drift checks, Ruff, Pyright all passed |
| Hosted persistence contract | `29 passed`, SQLite and PostgreSQL |
| Local exact CI Ruff target | Passed: `app/server` plus `app/tests` |
| Local exact Pyright target | Passed: `0 errors, 0 warnings, 0 informations` |
| Local Angular/Vitest suite | `24 files, 105 tests passed` |
| Local frontend production build | Passed: Angular bundle generation completed |
| Hosted Windows frontend/build gate | Frontend `24 files/105 tests`; production build passed |
| Hosted Windows browser E2E | Model-config unit `40 passed`; full browser E2E `39 passed, 7 skipped` |
| Hosted persistence/security gates | Security Python, Angular, desktop, and Rust audits passed; Python audit reported no known vulnerabilities |

The seven hosted browser skips remain conditional coverage: live provider,
multilingual embedding, unavailable Ollama-model, and persisted-session
scenarios whose documented prerequisites are absent from the isolated CI
database. They are not counted as passes.

## Live-provider result

The dispatched `live-provider-e2e` job completed setup, dependency installation,
frontend build, and Playwright installation successfully. Its single browser
test stopped at the explicit prerequisite check because the masked
`OPENCODE_GO_API_KEY` workflow secret was empty:

`DILIGENT_LIVE_PROVIDER_E2E=1 requires the OPENCODE_GO_API_KEY secret.`

The test therefore made no OpenCode Go request and produced no live provider,
latency, clinical-result, or model-output evidence. This is an external
credential blocker, not an application regression. The historical exact
`opencode_go / deepseek-v4-flash` clinical, revision, and timeline evidence
remains valid only for its documented runs and does not promote the broader
provider matrix.

## Final gate disposition

| Gate | Final status | Evidence and remaining limitation |
|---|---|---|
| `test.automated-regression` | `PARTIAL` | Current local unit/static/frontend/build checks and hosted core CI passed. The explicit provider dispatch failed at its missing-secret prerequisite, and conditional browser lanes remain unrun or skipped. |
| `model.provider.opencode-go` | `PARTIAL` | No current live request was possible because `OPENCODE_GO_API_KEY` is absent. No provider/model fallback was silently accepted. Re-run only after an approved repository secret and the documented live-provider inputs are available. |
| `runtime.database.sqlite-migrations` | `VALIDATED` for the exercised CI path | Hosted Alembic head/drift checks passed; the persistence contract also passed on SQLite/PostgreSQL. Deployment-specific and clean-machine upgrade evidence remain separate. |
| `ui.application-shell` | `VALIDATED` for the exercised CI/browser scope | Hosted frontend tests/build and full browser E2E passed; accessibility and screen-reader coverage remain separate. |

Independent incomplete gates remain unchanged: `auth.access-key-management`
is `BLOCKED` pending an approved disposable credential and cleanup
authorization; `data.inspection.catalogs` and `data.sources.refresh` are
`PARTIAL` while NCBI Bookshelf presents the LiverTox human-verification
challenge; `revision.agentic-lifecycle` is `PARTIAL` for the broader provider
and lifecycle matrix; `release.desktop.v3-4-0` is `BLOCKED` pending signed,
tagged, hosted-packaging, clean-machine, and publication prerequisites; and
`runtime.containerized` is `NOT_IMPLEMENTED`. The RAG duplicate-file policy,
complete API catalog, clinical provider-failure/RAG-off cases, and full
accessibility audit remain separate validation debt.

## Cleanup

The task-owned pytest workspace was removed after validation. The pre-existing
protected cache residue elsewhere in the repository was not modified.
