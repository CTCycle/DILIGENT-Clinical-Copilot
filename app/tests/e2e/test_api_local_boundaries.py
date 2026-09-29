"""Live HTTP checks for the current session and revision API boundary."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import APIRequestContext


SEED_PATIENT_NAME = "CI Browser Regression Subject"


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
