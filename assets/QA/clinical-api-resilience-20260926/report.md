# Clinical/API resilience and blocker revalidation

Last updated: 2026-09-26

## Scope and baseline

This validation revisited the locally actionable clinical/API resilience slice
and the incomplete provider, credential, and LiverTox boundaries identified in
the current ledgers. It started from clean `develop` HEAD
`6083c27d454e6a0ff974453b1c454364c2ef3c12`, equal to `origin/develop`.

The runtime used a task-owned SQLite database, access-key material file, and
pytest cache. The backend ran at `http://127.0.0.1:7690`; no shared database,
settings file, credential, source cache, or vector store was mutated. This was
an API/backend validation slice; it makes no rendered UI or screen-reader
claim.

## Current implementation evidence

| Check | Result |
|---|---|
| Focused backend contract, provider-safety, preflight, job, access-key, RAG, and workflow tests | `102 passed`, 6 existing dependency/framework deprecation warnings |
| Live HTTP model, model-config, clinical, and access-key API tests | `15 passed` |
| Backend health | `GET /api/health` returned `200` |
| OpenAPI route surface | `68` API paths returned by `/openapi.json` |
| Read-only API probes | `/api/settings`, `/api/inspection/rag/browse`, `/api/models/list`, `/api/inspection/jobs`, and `/api/inspection/sessions` returned `200`; `/api/clinical/jobs/latest` returned expected `404` with no active job |
| Route-catalog consistency | After the documentation fix, all `68` OpenAPI API paths matched `assets/docs/architecture/api_surface.md` |
| SQLite integrity | `PRAGMA integrity_check=ok`; `PRAGMA foreign_key_check=[]` |
| Cleanup state | Isolated access-key count `0`; isolated clinical-session count `0` after test cleanup |

The focused tests covered provider HTTP error classification for 401, 429,
503, 530, timeout, and network failures; bounded retry behavior; sanitized
provider details; direct OpenCode Go route selection; RAG-off and RAG-readiness
preflight behavior; clinical job terminal/cancellation semantics; workflow
report and bibliography safeguards; metadata-only access-key CRUD; and
structured-source ordering, cancellation, and failure preservation.

## Finding and remediation

`VAL-20260926-001` — The API catalog was stale: five current paths were absent
(`settings`, settings reset, desktop bootstrap/shutdown, and RAG browse), and
the access-key query parameter was incorrectly embedded in the documented path.
`assets/docs/architecture/api_surface.md` now documents all `68` OpenAPI paths
and describes the optional provider query parameter separately. No application
API or database schema change was required.

## External and incomplete gate dispositions

| Gate | Final status | Evidence and limitation |
|---|---|---|
| `api.local-boundaries` | `WORKING` | The selected clinical/model/access-key/inspection route group passed live HTTP checks and the documented catalog now matches OpenAPI. The complete response/error matrix remains open. |
| `clinical.analysis.pipeline` | `VALIDATED` for existing scope | Local failure/retry and RAG-off contracts passed. No new live clinical result was claimed without the approved provider secret. |
| `model.provider.opencode-go` | `PARTIAL` | `OPENCODE_GO_API_KEY` was absent; no live provider request was made. Exact provider/output/latency acceptance remains historical and bounded. |
| `test.automated-regression` | `PARTIAL` | The selected local slice passed; hosted CI, live-provider E2E, and conditional browser lanes remain independent prerequisites. |
| `auth.access-key-management` | `BLOCKED` | Synthetic isolated CRUD passed without exposing plaintext, but no approved disposable credential or cleanup authorization was available. |
| `data.inspection.catalogs` | `PARTIAL` | The read-only NCBI master-list preflight returned HTTP 200 HTML containing a human-verification/CAPTCHA marker; no source mutation was attempted. |
| `data.sources.refresh` | `PARTIAL` | The NCBI blocker persists, so no all-source refresh can be promoted. Existing ordered failure-preservation evidence remains valid. |
| `revision.agentic-lifecycle` | `PARTIAL` | Existing accepted-child evidence remains bounded; the broader provider/lifecycle matrix was not reopened without live provider prerequisites. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged packaging, hosted release CI, clean-machine smoke, and publication remain outside this slice. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | No supported container workflow exists. |

The unchanged Ollama 2B compatibility limitation was not retried. No NCBI
CAPTCHA bypass, live credential mutation, desktop publication, or container
work was performed.

## Cleanup

The task-owned backend process was stopped by exact executable-path check, ports
7690 and 9847 were free, and the temporary database, access-key material, and
pytest workspace were removed. Protected pre-existing pytest cache residue was
preserved. No secret values or patient-identifying data were written to this
report.
