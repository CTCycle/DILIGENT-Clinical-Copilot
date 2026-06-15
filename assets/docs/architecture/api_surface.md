# API Surface
Last updated: 2026-06-15

## Stable Boundary
All business APIs are mounted under `/api`. The frontend uses `/api` as the stable frontend-backend boundary.

## Root and OpenAPI Routes
- `GET /`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`
- In packaged Tauri mode, `/` and `/{full_path:path}` serve SPA assets.

## Session And Clinical Routes
- `GET /api/health`
- `GET /api/clinical/section-template`
- `POST /api/clinical/validate-input`
- `POST /api/clinical/jobs`
- `GET /api/clinical/jobs/{job_id}`
- `DELETE /api/clinical/jobs/{job_id}`

## Model Catalog And Pull Routes
- `GET /api/models/list`
- `POST /api/models/pull/jobs`
- `GET /api/models/jobs/{job_id}`
- `DELETE /api/models/jobs/{job_id}`

## Model Configuration Routes
- `GET /api/model-config`
- `PUT /api/model-config`
- `POST /api/model-config/openai-connectivity-check`

## Access Key Routes
- `GET /api/access-keys`
- `POST /api/access-keys`
- `PUT /api/access-keys/{key_id}/activate`
- `DELETE /api/access-keys/{key_id}`

## Inspection Routes
- `GET /api/inspection/sessions`
- `GET /api/inspection/sessions/{session_id}`
- `GET /api/inspection/sessions/{session_id}/versions`
- `GET /api/inspection/sessions/{session_id}/versions/{version_id}`
- `GET /api/inspection/sessions/{session_id}/versions/{left_version_id}/compare/{right_version_id}`
- `PUT /api/inspection/sessions/{session_id}`
- `PUT /api/inspection/sessions/{session_id}/report`
- `GET /api/inspection/sessions/{session_id}/manual-edits`
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
- `POST /api/inspection/sessions/{session_id}/timelines`
- `GET /api/inspection/sessions/{session_id}/timelines/{timeline_id}`
- `GET /api/inspection/sessions/{session_id}/timeline`
- `POST /api/inspection/sessions/{session_id}/timeline`
- `DELETE /api/inspection/sessions/{session_id}`
- `GET /api/inspection/rxnav`
- `GET /api/inspection/rxnav/{drug_id}/aliases`
- `DELETE /api/inspection/rxnav/{drug_id}`
- `GET /api/inspection/rxnav/update-config`
- `POST /api/inspection/rxnav/jobs`
- `GET /api/inspection/rxnav/jobs/{job_id}`
- `DELETE /api/inspection/rxnav/jobs/{job_id}`
- `GET /api/inspection/livertox`
- `GET /api/inspection/livertox/{drug_id}/excerpt`
- `DELETE /api/inspection/livertox/{drug_id}`
- `GET /api/inspection/livertox/update-config`
- `POST /api/inspection/livertox/jobs`
- `GET /api/inspection/livertox/jobs/{job_id}`
- `DELETE /api/inspection/livertox/jobs/{job_id}`
- `GET /api/inspection/reference-catalogs/runtime-observations`
- `GET /api/inspection/reference-catalogs/runtime-observations/{category}`
- `PUT /api/inspection/reference-catalogs/runtime-observations/{category}`
- `DELETE /api/inspection/reference-catalogs/runtime-observations/{category}/{term}`
- `GET /api/inspection/rag/update-config`
- `GET /api/inspection/rag/documents`
- `GET /api/inspection/rag/vector-store`
- `POST /api/inspection/rag/jobs`
- `GET /api/inspection/rag/jobs/{job_id}`
- `DELETE /api/inspection/rag/jobs/{job_id}`

## Notes
- Clinical and inspection workflows rely on job polling for long-running work.
- Research has no active route inventory in the current architecture source and should not be documented as an active API surface until implemented.
