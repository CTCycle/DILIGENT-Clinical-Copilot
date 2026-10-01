# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Live HTTP checks for the current session and revision API boundary."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import APIRequestContext


SEED_PATIENT_NAME = "CI Browser Regression Subject"

# Keep this matrix explicit so a newly published operation fails the Full
# harness until it is assigned a deterministic success/error contract or an
# already-validated external integration boundary.
API_OPERATION_MATRIX: dict[tuple[str, str], str] = {
    ("GET", "/api/access-keys"): "access-keys.synthetic-lifecycle",
    ("POST", "/api/access-keys"): "access-keys.synthetic-lifecycle",
    ("DELETE", "/api/access-keys/{key_id}"): "access-keys.synthetic-lifecycle",
    ("PUT", "/api/access-keys/{key_id}/activate"): "access-keys.synthetic-lifecycle",
    ("POST", "/api/clinical/jobs"): "clinical.validation-and-lifecycle",
    ("DELETE", "/api/clinical/jobs/{job_id}"): "clinical.validation-and-lifecycle",
    ("GET", "/api/clinical/jobs/{job_id}"): "clinical.validation-and-lifecycle",
    ("GET", "/api/clinical/section-template"): "clinical.success",
    ("POST", "/api/clinical/validate-input"): "clinical.validation-and-lifecycle",
    ("POST", "/api/desktop/bootstrap"): "desktop.runtime-contract",
    ("POST", "/api/desktop/shutdown"): "desktop.runtime-contract",
    ("GET", "/api/health"): "root-runtime.success",
    ("GET", "/api/inspection/dilirank"): "inspection.catalog-reads",
    ("GET", "/api/inspection/dilirank/{drug_id}"): "inspection.not-found-contracts",
    ("POST", "/api/inspection/dilirank/jobs"): "inspection.job-boundary",
    ("DELETE", "/api/inspection/dilirank/jobs/{job_id}"): "inspection.job-boundary",
    ("GET", "/api/inspection/dilirank/jobs/{job_id}"): "inspection.job-boundary",
    ("GET", "/api/inspection/dilirank/update-config"): "inspection.catalog-reads",
    ("GET", "/api/inspection/jobs"): "inspection.job-boundary",
    ("GET", "/api/inspection/livertox"): "inspection.catalog-reads",
    ("DELETE", "/api/inspection/livertox/{drug_id}"): "inspection.not-found-contracts",
    ("GET", "/api/inspection/livertox/{drug_id}/excerpt"): "inspection.not-found-contracts",
    ("POST", "/api/inspection/livertox/jobs"): "inspection.job-boundary",
    ("DELETE", "/api/inspection/livertox/jobs/{job_id}"): "inspection.job-boundary",
    ("GET", "/api/inspection/livertox/jobs/{job_id}"): "inspection.job-boundary",
    ("GET", "/api/inspection/livertox/update-config"): "inspection.catalog-reads",
    ("GET", "/api/inspection/rag/browse"): "rag.filesystem-boundaries",
    ("GET", "/api/inspection/rag/documents"): "rag.inspection-and-dedup",
    ("POST", "/api/inspection/rag/jobs"): "rag.job-boundary",
    ("DELETE", "/api/inspection/rag/jobs/{job_id}"): "rag.job-boundary",
    ("GET", "/api/inspection/rag/jobs/{job_id}"): "rag.job-boundary",
    ("GET", "/api/inspection/rag/update-config"): "rag.inspection-and-dedup",
    ("GET", "/api/inspection/rag/vector-store"): "rag.inspection-and-dedup",
    ("GET", "/api/inspection/reference-catalogs/runtime-observations"): "inspection.runtime-observations",
    ("GET", "/api/inspection/reference-catalogs/runtime-observations/{category}"): "inspection.runtime-observations",
    ("PUT", "/api/inspection/reference-catalogs/runtime-observations/{category}"): "inspection.runtime-observations",
    ("DELETE", "/api/inspection/reference-catalogs/runtime-observations/{category}/{term}"): "inspection.runtime-observations",
    ("GET", "/api/inspection/rxnav"): "inspection.catalog-reads",
    ("DELETE", "/api/inspection/rxnav/{drug_id}"): "inspection.not-found-contracts",
    ("PUT", "/api/inspection/rxnav/{drug_id}"): "inspection.catalog-mutation",
    ("GET", "/api/inspection/rxnav/{drug_id}/aliases"): "inspection.not-found-contracts",
    ("POST", "/api/inspection/rxnav/jobs"): "inspection.job-boundary",
    ("DELETE", "/api/inspection/rxnav/jobs/{job_id}"): "inspection.job-boundary",
    ("GET", "/api/inspection/rxnav/jobs/{job_id}"): "inspection.job-boundary",
    ("GET", "/api/inspection/rxnav/update-config"): "inspection.catalog-reads",
    ("GET", "/api/inspection/sessions"): "session.timeline-revision-slice",
    ("DELETE", "/api/inspection/sessions/{session_id}"): "session.not-found-contracts",
    ("GET", "/api/inspection/sessions/{session_id}"): "session.timeline-revision-slice",
    ("PUT", "/api/inspection/sessions/{session_id}"): "session.mutation-persistence",
    ("GET", "/api/inspection/sessions/{session_id}/manual-edits"): "session.mutation-persistence",
    ("PUT", "/api/inspection/sessions/{session_id}/report"): "session.mutation-persistence",
    ("POST", "/api/inspection/sessions/{session_id}/revision/jobs"): "revision.error-and-persistence-boundaries",
    ("POST", "/api/inspection/sessions/{session_id}/timeline-jobs"): "timeline.job-lifecycle",
    ("DELETE", "/api/inspection/sessions/{session_id}/timeline-jobs/{job_id}"): "timeline.job-lifecycle",
    ("GET", "/api/inspection/sessions/{session_id}/timeline-jobs/{job_id}"): "timeline.job-lifecycle",
    ("GET", "/api/inspection/sessions/{session_id}/timelines"): "timeline.job-lifecycle",
    ("DELETE", "/api/inspection/sessions/{session_id}/timelines/{timeline_id}"): "timeline.not-found-contracts",
    ("GET", "/api/inspection/sessions/{session_id}/timelines/{timeline_id}"): "timeline.job-lifecycle",
    ("GET", "/api/inspection/sessions/{session_id}/versions"): "revision.success-reads",
    ("GET", "/api/inspection/sessions/{session_id}/versions/{version_id}"): "revision.success-reads",
    ("GET", "/api/inspection/sessions/{session_id}/versions/{version_id}/artifacts"): "revision.success-reads",
    ("PUT", "/api/inspection/sessions/{session_id}/versions/{version_id}/clinical-review"): "revision.error-and-persistence-boundaries",
    ("GET", "/api/inspection/sessions/{session_id}/versions/{version_id}/entities"): "revision.success-reads",
    ("GET", "/api/inspection/sessions/{session_id}/versions/{version_id}/reviews"): "revision.success-reads",
    ("DELETE", "/api/inspection/sessions/revision/jobs/{job_id}"): "revision.error-and-persistence-boundaries",
    ("GET", "/api/inspection/sessions/revision/jobs/{job_id}"): "revision.error-and-persistence-boundaries",
    ("GET", "/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}"): "revision.error-and-persistence-boundaries",
    ("POST", "/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/retry"): "revision.error-and-persistence-boundaries",
    ("GET", "/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/steps"): "revision.error-and-persistence-boundaries",
    ("POST", "/api/inspection/structured-sources/jobs"): "external-boundary.start-status-cancel",
    ("DELETE", "/api/inspection/structured-sources/jobs/{job_id}"): "external-boundary.start-status-cancel",
    ("GET", "/api/inspection/structured-sources/jobs/{job_id}"): "external-boundary.start-status-cancel",
    ("GET", "/api/model-config"): "model-config.persistence-and-errors",
    ("PUT", "/api/model-config"): "model-config.persistence-and-errors",
    ("POST", "/api/model-config/catalogs/{provider}/load"): "external-boundary.start-status-cancel",
    ("POST", "/api/model-config/catalogs/{provider}/refresh"): "external-boundary.start-status-cancel",
    ("POST", "/api/model-config/connectivity-check"): "model-config.validation-errors",
    ("GET", "/api/model-config/embedding-status"): "model-config.success",
    ("DELETE", "/api/models/jobs/{job_id}"): "models.pull-lifecycle",
    ("GET", "/api/models/jobs/{job_id}"): "models.pull-lifecycle",
    ("GET", "/api/models/list"): "models.pull-lifecycle",
    ("POST", "/api/models/pull/jobs"): "models.pull-lifecycle",
    ("GET", "/api/settings"): "settings.persistence-and-reset",
    ("PATCH", "/api/settings"): "settings.persistence-and-reset",
    ("POST", "/api/settings/reset/{category}"): "settings.persistence-and-reset",
}


def _seed_session(api_context: APIRequestContext) -> tuple[int, dict[str, Any]]:
    response = api_context.get(
        "/api/inspection/sessions",
        params={"search": SEED_PATIENT_NAME, "offset": 0, "limit": 10},
    )
    assert response.status == 200, response.text()
    items = response.json().get("items", [])
    assert len(items) == 1, f"Expected one deterministic seed session, got {items!r}"
    session_id = items[0].get("session_id")
    assert isinstance(session_id, int)

    detail_response = api_context.get(f"/api/inspection/sessions/{session_id}")
    assert detail_response.status == 200, detail_response.text()
    detail = detail_response.json()
    assert detail["session_id"] == session_id
    return session_id, detail


def _assert_not_found(response: Any) -> None:
    assert response.status == 404, response.text()
    assert isinstance(response.json().get("detail"), str)


def test_openapi_operations_are_explicitly_classified(
    api_context: APIRequestContext,
) -> None:
    response = api_context.get("/openapi.json")
    assert response.status == 200, response.text()
    schema = response.json()
    operations = {
        (method.upper(), path)
        for path, path_item in schema.get("paths", {}).items()
        for method in path_item
        if method.lower()
        in {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
    }

    assert operations == set(API_OPERATION_MATRIX), (
        "OpenAPI operations must have an explicit local-boundary classification. "
        f"Missing={sorted(operations - set(API_OPERATION_MATRIX))}; "
        f"stale={sorted(set(API_OPERATION_MATRIX) - operations)}"
    )
    assert all(API_OPERATION_MATRIX[operation] for operation in operations)


def test_root_runtime_contracts_are_reachable(
    api_context: APIRequestContext,
) -> None:
    root = api_context.get("/")
    assert root.status in {200, 307}
    assert api_context.get("/docs").status == 200
    assert api_context.get("/redoc").status == 200

    openapi = api_context.get("/openapi.json")
    assert openapi.status == 200
    assert openapi.json().get("openapi")

    health = api_context.get("/api/health")
    assert health.status == 200
    assert health.json().get("status") == "ok"

    bootstrap = api_context.post("/api/desktop/bootstrap", data={"token": "invalid"})
    assert bootstrap.status in {401, 422}
    shutdown = api_context.post("/api/desktop/shutdown")
    assert shutdown.status in {404, 503}


def test_catalog_rag_and_runtime_observation_boundaries(
    api_context: APIRequestContext,
) -> None:
    for path in (
        "/api/inspection/jobs",
        "/api/inspection/rxnav",
        "/api/inspection/rxnav/update-config",
        "/api/inspection/livertox",
        "/api/inspection/livertox/update-config",
        "/api/inspection/dilirank",
        "/api/inspection/dilirank/update-config",
        "/api/inspection/rag/update-config",
        "/api/inspection/rag/documents",
        "/api/inspection/rag/vector-store",
        "/api/inspection/reference-catalogs/runtime-observations",
        "/api/inspection/reference-catalogs/runtime-observations/api-boundary",
    ):
        response = api_context.get(path)
        assert response.status == 200, f"{path}: {response.text()}"

    for response in (
        api_context.get("/api/inspection/rxnav/999999/aliases"),
        api_context.put(
            "/api/inspection/rxnav/999999",
            data={"drug_name": "missing"},
        ),
        api_context.delete("/api/inspection/rxnav/999999"),
        api_context.get("/api/inspection/livertox/999999/excerpt"),
        api_context.delete("/api/inspection/livertox/999999"),
        api_context.get("/api/inspection/dilirank/999999"),
        api_context.get("/api/inspection/rag/browse", params={"path": "relative"}),
    ):
        assert response.status in {404, 422}, response.text()

    term = "api-boundary-dedup-marker"
    category = "api-boundary"
    upsert = api_context.put(
        f"/api/inspection/reference-catalogs/runtime-observations/{category}",
        data={
            "term": term,
            "replacement": "canonical",
            "source": "api-local-boundaries",
            "is_active": True,
        },
    )
    assert upsert.status == 200, upsert.text()
    assert upsert.json()["term"] == term
    deleted = api_context.delete(
        f"/api/inspection/reference-catalogs/runtime-observations/{category}/{term}"
    )
    assert deleted.status == 200, deleted.text()
    assert deleted.json()["deleted"] is True


def test_route_family_validation_and_missing_job_contracts(
    api_context: APIRequestContext,
) -> None:
    rag_start = api_context.post(
        "/api/inspection/rag/jobs",
        data={"documents_path": "relative"},
    )
    assert rag_start.status == 202, rag_start.text()
    rag_job_id = rag_start.json()["job_id"]
    rag_status = api_context.get(f"/api/inspection/rag/jobs/{rag_job_id}")
    assert rag_status.status == 200, rag_status.text()
    rag_cancel = api_context.delete(f"/api/inspection/rag/jobs/{rag_job_id}")
    assert rag_cancel.status in {200, 404}, rag_cancel.text()

    invalid_requests = (
        api_context.post("/api/clinical/jobs", data={"name": "invalid"}),
        api_context.post("/api/inspection/rxnav/jobs", data={"rxnav_request_timeout": 0}),
        api_context.post(
            "/api/inspection/livertox/jobs",
            data={"livertox_monograph_max_workers": 0},
        ),
        api_context.post("/api/inspection/dilirank/jobs", data={"redownload": "invalid"}),
        api_context.post(
            "/api/inspection/structured-sources/jobs",
            data={"rxnav": {"rxnav_request_timeout": 0}},
        ),
        api_context.post("/api/models/pull/jobs"),
        api_context.post(
            "/api/model-config/catalogs/not-a-provider/load"
        ),
        api_context.post(
            "/api/model-config/catalogs/not-a-provider/refresh"
        ),
        api_context.post(
            "/api/model-config/connectivity-check",
            data={"provider": "not-a-provider", "model": "missing"},
        ),
        api_context.patch("/api/settings", data={}),
    )
    for response in invalid_requests:
        assert response.status in {400, 409, 422}, response.text()

    missing_job_requests = (
        ("/api/clinical/jobs/missing", "get"),
        ("/api/clinical/jobs/missing", "delete"),
        ("/api/models/jobs/missing", "get"),
        ("/api/models/jobs/missing", "delete"),
        ("/api/inspection/rxnav/jobs/missing", "get"),
        ("/api/inspection/rxnav/jobs/missing", "delete"),
        ("/api/inspection/livertox/jobs/missing", "get"),
        ("/api/inspection/livertox/jobs/missing", "delete"),
        ("/api/inspection/dilirank/jobs/missing", "get"),
        ("/api/inspection/dilirank/jobs/missing", "delete"),
        ("/api/inspection/rag/jobs/missing", "get"),
        ("/api/inspection/rag/jobs/missing", "delete"),
        ("/api/inspection/structured-sources/jobs/missing", "get"),
        ("/api/inspection/structured-sources/jobs/missing", "delete"),
    )
    for path, method in missing_job_requests:
        response = getattr(api_context, method)(path)
        _assert_not_found(response)


def test_session_version_detail_and_timeline_job_lifecycle(
    api_context: APIRequestContext,
) -> None:
    session_id, _detail = _seed_session(api_context)
    versions = api_context.get(f"/api/inspection/sessions/{session_id}/versions")
    assert versions.status == 200, versions.text()
    version_id = versions.json()["items"][0]["version_id"]
    version_detail = api_context.get(
        f"/api/inspection/sessions/{session_id}/versions/{version_id}"
    )
    assert version_detail.status == 200, version_detail.text()
    assert version_detail.json()["version"]["version_id"] == version_id

    timeline_start = api_context.post(
        f"/api/inspection/sessions/{session_id}/timeline-jobs",
        data={"force_regenerate": True},
    )
    assert timeline_start.status == 202, timeline_start.text()
    job_id = timeline_start.json()["job_id"]
    timeline_status = api_context.get(
        f"/api/inspection/sessions/{session_id}/timeline-jobs/{job_id}"
    )
    assert timeline_status.status == 200, timeline_status.text()
    timeline_cancel = api_context.delete(
        f"/api/inspection/sessions/{session_id}/timeline-jobs/{job_id}"
    )
    assert timeline_cancel.status == 200, timeline_cancel.text()


def test_session_inspection_mutations_and_timeline_errors(
    api_context: APIRequestContext,
) -> None:
    session_id, original_detail = _seed_session(api_context)
    original_report = (
        original_detail.get("report")
        or original_detail.get("official_report_text")
        or (original_detail.get("result_payload") or {}).get("report")
    )
    assert isinstance(original_report, str) and original_report.strip()

    versions = api_context.get(f"/api/inspection/sessions/{session_id}/versions")
    assert versions.status == 200, versions.text()
    assert isinstance(versions.json().get("items"), list)

    manual_edits = api_context.get(
        f"/api/inspection/sessions/{session_id}/manual-edits"
    )
    assert manual_edits.status == 200, manual_edits.text()
    assert isinstance(manual_edits.json(), list)

    timelines = api_context.get(
        f"/api/inspection/sessions/{session_id}/timelines"
    )
    assert timelines.status == 200, timelines.text()
    timeline_items = timelines.json().get("items", [])
    assert len(timeline_items) == 1
    timeline_id = timeline_items[0].get("timeline_id")
    assert isinstance(timeline_id, int)

    timeline = api_context.get(
        f"/api/inspection/sessions/{session_id}/timelines/{timeline_id}"
    )
    assert timeline.status == 200, timeline.text()
    assert timeline.json()["timeline_id"] == timeline_id

    metadata_update = api_context.put(
        f"/api/inspection/sessions/{session_id}",
        data={"metadata": {"validation_scope": "api-local-boundaries"}},
    )
    assert metadata_update.status == 200, metadata_update.text()
    assert metadata_update.json()["session_id"] == session_id

    edited_report = f"{original_report}\n\nAPI boundary validation marker."
    try:
        report_update = api_context.put(
            f"/api/inspection/sessions/{session_id}/report",
            data={
                "report_text": edited_report,
                "edited_fields": ["report_text"],
                "reviewer_note": "Disposable API contract check.",
                "edited_by": "validation",
                "metadata": {"scope": "api-local-boundaries"},
            },
        )
        assert report_update.status == 200, report_update.text()
        report_payload = report_update.json()
        assert report_payload["session"]["report"] == edited_report
        assert (
            report_payload["audit"]["new_text_hash"]
            != report_payload["audit"]["previous_text_hash"]
        )

        after_edit = api_context.get(
            f"/api/inspection/sessions/{session_id}/manual-edits"
        )
        assert after_edit.status == 200, after_edit.text()
        assert any(
            row.get("reviewer_note") == "Disposable API contract check."
            for row in after_edit.json()
        )
    finally:
        restore = api_context.put(
            f"/api/inspection/sessions/{session_id}/report",
            data={
                "report_text": original_report,
                "edited_fields": ["report_text"],
                "edited_by": "validation-cleanup",
            },
        )
        assert restore.status == 200, restore.text()

    for response in (
        api_context.get("/api/inspection/sessions/999999"),
        api_context.put(
            "/api/inspection/sessions/999999",
            data={"metadata": {"should_not": "persist"}},
        ),
        api_context.put(
            "/api/inspection/sessions/999999/report",
            data={"report_text": "missing session"},
        ),
        api_context.delete("/api/inspection/sessions/999999"),
        api_context.get(
            f"/api/inspection/sessions/{session_id}/timelines/999999"
        ),
        api_context.delete(
            f"/api/inspection/sessions/{session_id}/timelines/999999"
        ),
        api_context.get(
            f"/api/inspection/sessions/{session_id}/timeline-jobs/missing"
        ),
        api_context.delete(
            f"/api/inspection/sessions/{session_id}/timeline-jobs/missing"
        ),
    ):
        _assert_not_found(response)


def test_revision_route_error_contracts(
    api_context: APIRequestContext,
) -> None:
    session_id, _detail = _seed_session(api_context)

    invalid_start = api_context.post(
        f"/api/inspection/sessions/{session_id}/revision/jobs",
        data={"max_tasks": 0},
    )
    assert invalid_start.status == 422, invalid_start.text()

    unknown_session_start = api_context.post(
        "/api/inspection/sessions/999999/revision/jobs"
    )
    _assert_not_found(unknown_session_start)

    for response in (
        api_context.get("/api/inspection/sessions/revision/jobs/missing"),
        api_context.delete("/api/inspection/sessions/revision/jobs/missing"),
        api_context.get(
            "/api/inspection/sessions/revision/pipeline-runs/missing"
        ),
        api_context.post(
            "/api/inspection/sessions/revision/pipeline-runs/missing/retry"
        ),
        api_context.get(
            "/api/inspection/sessions/revision/pipeline-runs/missing/steps"
        ),
        api_context.get(
            f"/api/inspection/sessions/{session_id}/versions/999999/artifacts"
        ),
        api_context.get(
            f"/api/inspection/sessions/{session_id}/versions/999999/entities"
        ),
        api_context.get(
            f"/api/inspection/sessions/{session_id}/versions/999999/reviews"
        ),
        api_context.put(
            f"/api/inspection/sessions/{session_id}/versions/999999/clinical-review",
            data={"clinical_review_status": "approved_by_human"},
        ),
    ):
        _assert_not_found(response)
