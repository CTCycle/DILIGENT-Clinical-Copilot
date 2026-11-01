# API Surface
Last updated: 2026-11-01

`/api/model-config` manages provider, model, reasoning, and RAG selection; it
does not expose sampling temperature. `GET` returns the rich catalog and
runtime-status view. `PUT` is a persistence-only response containing the
saved configuration and timestamp; catalog and embedding refreshes remain
explicit GET/status operations.

The persisted role assignments are independent: `clinical_model`,
`text_extraction_model`, `revision_model`, and `timeline_model`. Revision and
Timeline job requests do not accept active model overrides; each job resolves
its configured role when it starts. Cloud model descriptors expose capability
data through the canonical `model_capabilities` object only.

## Stable Boundary
All business APIs are mounted under `/api`. The frontend uses `/api` as the stable frontend-backend boundary.

## Root and OpenAPI Routes
- `GET /`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

## Session And Clinical Routes
- `GET /api/health`
- `GET /api/system/status` (non-cacheable, always HTTP 200: provider availability, Ollama reachability and `ollama_control` state/ownership, and system-wide CPU, RAM and GPU VRAM; unavailable readings are `null`)
- `POST /api/system/ollama/start` (launches `ollama serve` on a loopback endpoint; HTTP 409 with a message when it cannot start)
- `POST /api/system/ollama/stop` (stops only the process DILIGENT launched; HTTP 409 otherwise)
- `GET /api/jobs` (non-cacheable summary of active and recent jobs for the global Jobs monitor)
- `DELETE /api/jobs/{job_id}` (cooperative cancel; refused for `session_revision` and `session_timeline`, which are cancelled from their own pages). Clinical rows carry `session_id`.
- `GET /api/clinical/section-template`
- `POST /api/clinical/validate-input`
- `POST /api/clinical/sessions` (create a `draft` session; 201)
- `GET /api/clinical/sessions/{session_id}/input`
- `PUT /api/clinical/sessions/{session_id}/input` (draft only; `expected_updated_at` token)
- `POST /api/clinical/sessions/{session_id}/launch` (draft → queued; 202)
- `GET /api/clinical/sessions/{session_id}/status`
- `POST /api/clinical/sessions/{session_id}/cancel`

`POST /api/clinical/validate-input` and `POST .../launch` accept the
same session configuration:

- `evidence_sources`: any of `livertox`, `dilirank`, `documents` (the indexed
  document library). The default is `["livertox"]`; an explicit `[]` means LLM
  knowledge only. Unknown values are rejected.
- `model_overrides`: optional `text_extraction_model` and `clinical_model`
  names that apply to that session only. They are validated against the
  active provider's cached catalog and are never written to the persisted
  model configuration.

Preflight validates the exact configuration that will run. The response adds
an `evidence_sources` list (`source`, `label`, `selected`, `available`,
`message`) and keeps `rag_readiness` for the document library. Selecting
`documents` while the embedding cache is unavailable is blocking (code from
`rag_readiness`, field `evidence_sources`); an unimported DILIrank is the
non-blocking `dilirank_unavailable`; an invalid override is the blocking
`session_model_unavailable` (field `model_overrides`).

Every lifecycle route returns `ClinicalSessionState`: `session_id`, `status`
(`draft|queued|running|completed|cancelled|failed`), patient name and visit
date, `created_at`/`updated_at`/`queued_at`/`started_at`/`finished_at`,
`queue_position` (queued only), `progress` (null = indeterminate),
`progress_stage` (canonical step key such as `drugs.resolving` or
`report.generating`), `progress_message`, `stop_requested`, `error`
(`{code, message}`; `interrupted` marks a run stopped by an application
shutdown), `has_report`, a monotonic `version`, and `poll_interval`. Clients
poll only `/status`; there are no clinical job ids on the frontend.

- Draft input is created and edited without preflight or model calls. Edits
  require the `updated_at` the client last read and return `409` when the
  session is no longer a draft or the token is stale.
- `launch` runs the same launch checks as before (overrides, provider key,
  catalogs, sections, RAG readiness). Failures return `4xx` and leave the
  draft unchanged; a session that is no longer a draft returns `409`, so a
  repeated request can never start a second analysis.
- `cancel` makes a queued session `cancelled` at once. A running session keeps
  `running` with `stop_requested: true` until its worker exits, then becomes
  `cancelled`. Drafts and terminal sessions return `409`.
- Responses use `Cache-Control: no-store`; validation errors on these routes
  do not echo submitted clinical text.

## Model Catalog And Pull Routes
- `GET /api/models/list`
- `POST /api/models/pull/jobs`
- `GET /api/models/jobs/{job_id}`
- `DELETE /api/models/jobs/{job_id}`

## Model Configuration Routes
- `GET /api/model-config`
- `PUT /api/model-config`
- `POST /api/model-config/catalogs/{provider}/load`
- `POST /api/model-config/catalogs/{provider}/refresh`
- `GET /api/model-config/embedding-status`
- `POST /api/model-config/connectivity-check`

`GET /api/model-config` returns each cloud provider's model catalog together with
freshness status and a user-safe message. Catalogs are refreshed through the
provider's official model-list API; a successful in-process result is returned
as `cached` if a later refresh fails. Successful and failed catalog attempts are
persisted in the SQLAlchemy-backed provider catalog cache, so cached catalog
state survives a backend restart.
If a persisted model is no longer present in its provider's refreshed catalog,
the GET response still returns that saved selection together with the current
catalog so clients can present a valid replacement. Catalog membership remains
strictly validated when a new provider or model selection is saved. Responses
are non-cacheable so model selectors and their Retry actions always observe the
latest saved configuration and provider-catalog state.

## Access Key Routes
- `GET /api/access-keys`
- `POST /api/access-keys`
- `PUT /api/access-keys/{key_id}/activate`
- `DELETE /api/access-keys/{key_id}`

`GET /api/access-keys` accepts the optional `provider` query parameter with
one of `openai`, `gemini`, `deepseek`, `anthropic`, `opencode`, or `brave`.

## Runtime Settings Routes
- `GET /api/settings`
- `PATCH /api/settings`
- `POST /api/settings/reset/{category}`

## Desktop Runtime Routes
- `POST /api/desktop/bootstrap`
- `POST /api/desktop/shutdown`

## Inspection Routes
- `GET /api/inspection/jobs`
- `GET /api/inspection/sessions` (`status` is any lifecycle status; `sort` is `updated_desc` (default), `created_desc` or `created_asc`; the date filter applies to the creation date)
- `GET /api/inspection/sessions/{session_id}`
- `GET /api/inspection/sessions/{session_id}/versions`
- `GET /api/inspection/sessions/{session_id}/versions/{version_id}`
- `PUT /api/inspection/sessions/{session_id}`
- `PUT /api/inspection/sessions/{session_id}/report`
- `GET /api/inspection/sessions/{session_id}/manual-edits`
- `GET|POST /api/inspection/sessions/{session_id}/attachments` and `GET|DELETE .../attachments/{attachment_id}` (base64 JSON upload, 10 MB limit, idempotent by content hash)
- `POST /api/inspection/sessions/{session_id}/revision/jobs`
- `GET /api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}`
- `POST /api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/retry`
- `GET /api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/steps`
- `PUT /api/inspection/sessions/{session_id}/versions/{version_id}/clinical-review`
- `GET /api/inspection/sessions/revision/jobs/{job_id}`
- `DELETE /api/inspection/sessions/revision/jobs/{job_id}`
- `GET /api/inspection/sessions/{session_id}/versions/{version_id}/entities`
- `GET /api/inspection/sessions/{session_id}/versions/{version_id}/reviews`
- `GET /api/inspection/sessions/{session_id}/versions/{version_id}/artifacts`
- `GET /api/inspection/sessions/{session_id}/timelines`
- `GET /api/inspection/sessions/{session_id}/timelines/{timeline_id}`
- `DELETE /api/inspection/sessions/{session_id}/timelines/{timeline_id}`
- `POST /api/inspection/sessions/{session_id}/timeline-jobs`
- `GET /api/inspection/sessions/{session_id}/timeline-jobs/{job_id}`
- `DELETE /api/inspection/sessions/{session_id}/timeline-jobs/{job_id}`
- `DELETE /api/inspection/sessions/{session_id}`
- `GET /api/inspection/rxnav`
- `GET /api/inspection/rxnav/{drug_id}/aliases`
- `DELETE /api/inspection/rxnav/{drug_id}`
- `GET /api/inspection/rxnav/update-config`
- `POST /api/inspection/rxnav/jobs`
- `GET /api/inspection/rxnav/jobs/{job_id}`
- `DELETE /api/inspection/rxnav/jobs/{job_id}`
- `GET /api/inspection/terminology/summary`
- `GET /api/inspection/terminology/runs`
- `GET /api/inspection/terminology/conflicts`
- `POST /api/inspection/terminology/resolve`
- `POST /api/inspection/terminology/aliases/confirm`
- `POST /api/inspection/terminology/aliases/reject`
- `GET /api/inspection/terminology/update-config`
- `POST /api/inspection/terminology/jobs`
- `GET /api/inspection/terminology/jobs/{job_id}`
- `DELETE /api/inspection/terminology/jobs/{job_id}`
- `GET /api/inspection/livertox`
- `GET /api/inspection/livertox/{drug_id}/excerpt`
- `DELETE /api/inspection/livertox/{drug_id}`
- `GET /api/inspection/livertox/update-config`
- `POST /api/inspection/livertox/jobs`
- `GET /api/inspection/livertox/jobs/{job_id}`
- `DELETE /api/inspection/livertox/jobs/{job_id}`
- `GET /api/inspection/dilirank`
- `GET /api/inspection/dilirank/{drug_id}`
- `GET /api/inspection/dilirank/update-config`
- `POST /api/inspection/dilirank/jobs`
- `GET /api/inspection/dilirank/jobs/{job_id}`
- `DELETE /api/inspection/dilirank/jobs/{job_id}`
- `POST /api/inspection/structured-sources/jobs`
- `GET /api/inspection/structured-sources/jobs/{job_id}`
- `DELETE /api/inspection/structured-sources/jobs/{job_id}`
- `GET /api/inspection/reference-catalogs/runtime-observations`
- `GET /api/inspection/reference-catalogs/runtime-observations/{category}`
- `PUT /api/inspection/reference-catalogs/runtime-observations/{category}`
- `DELETE /api/inspection/reference-catalogs/runtime-observations/{category}/{term}`
- `GET /api/inspection/rag/update-config`
- `GET /api/inspection/rag/documents`
- `GET /api/inspection/rag/vector-store`
- `GET /api/inspection/rag/browse`
- `POST /api/inspection/rag/jobs`
- `GET /api/inspection/rag/jobs/{job_id}`
- `DELETE /api/inspection/rag/jobs/{job_id}`

## Notes
- DILIrank inspection is read-only at record level. `GET /api/inspection/dilirank` exposes the complete persisted FDA snapshot, including rows that are intentionally unlinked to the local canonical drug catalog. The update routes use the same centralized inspection job manager as RxNav, LiverTox, terminology, the combined structured-source workflow, and RAG. `redownload=true` forces a fresh FDA workbook; otherwise conditional HTTP metadata is used when a validated cache exists.
- `POST /api/inspection/structured-sources/jobs` starts the dependency-aware `RxNav -> LiverTox -> Terminology -> DILIrank` sequence and returns child source state in the job result. The optional `include_terminology` flag (default true) omits the terminology step; the skipped source is reported with status `skipped`. RAG is intentionally excluded and remains independently controllable.
- Terminology routes: `summary` reports per-source versions, last runs, counts, license and attribution, open conflicts and fuzzy-index statistics; `resolve` is the try-it resolver (`allow_external` controls the RxNav fallback); `aliases/confirm` and `aliases/reject` store reviewer decisions as `reviewer` aliases (`404` for an unknown concept). `POST /api/inspection/rxnav/jobs` accepts only `force_full`; `POST /api/inspection/terminology/jobs` accepts `force_full`, `drugcentral_enabled`, `chembl_enabled` and `swissmedic_enabled`. See [drug_terminology.md](drug_terminology.md).
- `POST /api/inspection/sessions/{session_id}/timeline-jobs` accepts
  `force_regenerate` and returns a job for polling. The job resolves the
  persisted `timeline_model` role at start; provider/model controls are managed
  from Settings → Models. Timeline previews include source-evidence,
  missing-evidence, uncertain, and undated counts, plus `is_stale` when the
  session input changed after generation. Drafts can generate timelines from
  the deterministic section split of their input; this never changes the
  session status. A non-forced request reuses the latest timeline only while
  it is current. Timeline job status and cancel resolve only for the job's own
  session (admission scope `session_timeline:{session_id}`).
- `DELETE /api/inspection/sessions/{session_id}` returns 409 while the session
  is queued or running. Revision jobs and manual report edits return 409 until
  the session has a report.
- Timeline deletion is scoped by both session and timeline identifiers and returns 404
  when that exact persisted timeline does not exist.
- Clinical and inspection workflows rely on job polling for long-running work.
  Angular feature trackers use the shared `JobPollingService`.
- Revision jobs expose the implemented bounded revision workflow: start and
  status, persisted run/context/plan/step/tool-trace/artifact reads, draft
  report validation, QA, optional `agentic_revision` session creation, and
  lineage/review reads. Dry runs and QA-blocked runs retain an auditable draft;
  an accepted non-dry run finalizes the revision lineage.
- Research has no active route inventory in the current architecture source and should not be documented as an active API surface until implemented.

### Session revision agent
`POST /api/inspection/sessions/{session_id}/revision/jobs` starts an agentic
revision job. Existing status, cancellation, pipeline-run, step, artifact,
entity, and clinical-review routes remain stable. The job returns
revision/version identifiers and task, tool, QA, and human-review summaries.

```mermaid
sequenceDiagram
    participant UI as Angular UI
    participant API as Revision API
    participant S as Inspection service
    participant A as RevisionAgentRunner
    participant RR as Revision persistence
    participant CR as Clinical persistence

    UI->>API: Start revision
    API->>S: Request
    S->>RR: Create version and run shell
    S->>A: Run bounded agent
    A->>RR: Context, plan, steps, traces
    A->>A: Draft and deterministic patch validation
    alt no-op or QA blocker
        A->>A: One bounded evidence-focused repair
        A->>RR: Repair attempt and latency metadata
    end
    A->>RR: QA and final artifacts
    alt accepted non-dry run
        A->>CR: Save agentic_revision session
        A->>RR: Finalize lineage
    else dry run or QA blocked
        A->>RR: Keep auditable draft
    end
    API-->>UI: Job status and revision identifiers
```
