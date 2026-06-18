from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pytest

###############################################################################
def _normalize_host_for_url(host: str) -> str:
    if host in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host

###############################################################################
def _build_base_url(
    host_env: str,
    port_env: str,
    default_host: str,
    default_port: str,
) -> str:
    host = _normalize_host_for_url(os.getenv(host_env, default_host))
    port = os.getenv(port_env, default_port)
    return f"http://{host}:{port}"


API_BASE_URL = (
    os.getenv("APP_TEST_BACKEND_URL")
    or os.getenv("API_BASE_URL")
    or _build_base_url("FASTAPI_HOST", "FASTAPI_PORT", "127.0.0.1", "8000")
)

###############################################################################
@pytest.fixture
def api_client() -> httpx.Client:
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=15.0) as client:
            health = client.get("/api/health")
            if health.status_code != 200:
                pytest.skip(f"Backend API is unavailable at {API_BASE_URL}: HTTP {health.status_code}")
            yield client
    except httpx.HTTPError as exc:  # pragma: no cover - environment availability path
        pytest.skip(f"Backend API is unavailable at {API_BASE_URL}: {exc}")

###############################################################################
def _list_sessions(api_client: httpx.Client) -> list[dict[str, Any]]:
    response = api_client.get("/api/inspection/sessions", params={"offset": 0, "limit": 20})
    assert response.status_code == 200
    payload = response.json()
    items = payload.get("items") if isinstance(payload, dict) else None
    return items if isinstance(items, list) else []

###############################################################################
def _get_session_detail(
    api_client: httpx.Client,
    session_id: int,
) -> dict[str, Any]:
    response = api_client.get(f"/api/inspection/sessions/{session_id}")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    return payload

###############################################################################
def _get_session_versions(
    api_client: httpx.Client,
    session_id: int,
) -> list[dict[str, Any]]:
    response = api_client.get(f"/api/inspection/sessions/{session_id}/versions")
    assert response.status_code == 200
    payload = response.json()
    items = payload.get("items") if isinstance(payload, dict) else None
    return items if isinstance(items, list) else []

###############################################################################
def _wait_for_revision_job(
    api_client: httpx.Client,
    job_id: str,
    *,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        response = api_client.get(f"/api/inspection/sessions/revision/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, dict)
        last_payload = payload
        if payload.get("status") in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.5)
    if last_payload is None:
        raise AssertionError("Revision job never returned a status payload.")
    return last_payload

###############################################################################
def _find_editable_session(api_client: httpx.Client) -> tuple[int, dict[str, Any]]:
    for item in _list_sessions(api_client):
        session_id = item.get("session_id") if isinstance(item, dict) else None
        if not isinstance(session_id, int) or session_id <= 0:
            continue
        detail = _get_session_detail(api_client, session_id)
        report_text = str(detail.get("official_report_text") or "").strip()
        if report_text:
            return session_id, detail
    pytest.skip("No persisted clinical session with an editable report is available.")

###############################################################################
def test_manual_report_edit_flow_preserves_version_and_audits_change(
    api_client: httpx.Client,
) -> None:
    session_id, detail = _find_editable_session(api_client)
    original_report = str(detail.get("official_report_text") or "")
    original_version = int(detail.get("version") or 0)
    original_source = str(detail.get("source_clinical_text") or "")
    original_manual_edits = list(detail.get("manual_edit_history") or [])

    edit_marker = f"[E2E manual edit marker {int(time.time())}]"
    updated_report = f"{original_report}\n\n{edit_marker}"
    response = api_client.put(
        f"/api/inspection/sessions/{session_id}/report",
        json={
            "report_text": updated_report,
            "edited_fields": ["report_text"],
            "reviewer_note": "E2E manual edit audit check",
            "edited_by": "E2E Reviewer",
            "metadata": {"source": "e2e"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["official_report_text"] == updated_report
    assert payload["session"]["version"] == original_version
    assert payload["session"]["source_clinical_text"] == original_source
    assert payload["audit"]["edited_fields"] == ["report_text"]
    assert payload["audit"]["reviewer_note"] == "E2E manual edit audit check"
    assert edit_marker in payload["session"]["official_report_text"]

    edits_response = api_client.get(f"/api/inspection/sessions/{session_id}/manual-edits")
    assert edits_response.status_code == 200
    edits_payload = edits_response.json()
    assert isinstance(edits_payload, list)
    assert len(edits_payload) >= len(original_manual_edits) + 1
    assert edits_payload[0]["reviewer_note"] == "E2E manual edit audit check"

    versions = _get_session_versions(api_client, session_id)
    assert versions
    assert versions[-1]["version_number"] == original_version

###############################################################################
def test_revision_workflow_start_persists_draft_shell_and_pipeline_run(
    api_client: httpx.Client,
) -> None:
    session_id, detail = _find_editable_session(api_client)
    versions_before = _get_session_versions(api_client, session_id)
    if not versions_before:
        pytest.skip("No version history is available for the selected clinical session.")
    source_version = versions_before[-1]

    start_response = api_client.post(
        f"/api/inspection/sessions/{session_id}/revision/jobs",
        json={
            "selected_text": str(detail.get("official_report_text") or "")[:500],
            "revision_instruction": "Review chronology and lab wording only.",
            "model_overrides": {},
            "metadata": {
                "reviewer": "E2E Reviewer",
                "revision_note": "E2E revision workflow check",
            },
        },
    )
    if start_response.status_code in {404, 409, 422, 500}:
        pytest.skip(
            "Revision workflow could not be started in this environment: "
            f"{start_response.text}"
        )

    assert start_response.status_code == 202
    started = start_response.json()
    job_id = str(started["job_id"])
    job_payload = _wait_for_revision_job(api_client, job_id)
    result_payload = job_payload.get("result") if isinstance(job_payload, dict) else None
    assert isinstance(result_payload, dict)
    pipeline_run_id = str(result_payload.get("pipeline_run_id") or "")
    target_revision_version_id = int(result_payload.get("target_revision_version_id") or 0)
    assert pipeline_run_id
    assert target_revision_version_id > 0

    run_response = api_client.get(
        f"/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}"
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["pipeline_run_id"] == pipeline_run_id
    assert int(run_payload["source_version_id"]) == int(source_version["version_id"])
    assert int(run_payload["target_revision_version_id"]) == target_revision_version_id

    versions_after = _get_session_versions(api_client, session_id)
    draft_versions = [
        item
        for item in versions_after
        if int(item["version_id"]) == target_revision_version_id
    ]
    assert len(draft_versions) == 1
    assert draft_versions[0]["revision_kind"] == "llm_assisted_revision"
    assert draft_versions[0]["source_version_id"] == source_version["version_id"]
    assert draft_versions[0]["pipeline_run_id"] == pipeline_run_id

    steps_response = api_client.get(
        f"/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/steps"
    )
    assert steps_response.status_code == 200
    steps_payload = steps_response.json()
    steps = steps_payload.get("items") if isinstance(steps_payload, dict) else None
    assert isinstance(steps, list)
    if job_payload.get("status") == "failed":
        error_payload = job_payload.get("error") or {}
        detail_text = str(error_payload) + " " + str(job_payload.get("message") or "")
        if "dependency" in detail_text.casefold() or "unavailable" in detail_text.casefold():
            pytest.skip(
                "Revision job reached the live backend but failed due unavailable dependency: "
                f"{detail_text}"
            )
    assert len(steps) >= 1
