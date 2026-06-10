from __future__ import annotations

from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute

from app import app


###############################################################################
def test_openapi_schema_generation_succeeds() -> None:
    schema = app.openapi()
    assert "paths" in schema
    assert "/api/health" in schema["paths"]


###############################################################################
def test_health_route_uses_response_model() -> None:
    schema = app.openapi()
    response = schema["paths"]["/api/health"]["get"]["responses"]["200"]
    content = response["content"]["application/json"]
    assert content["schema"]["$ref"].endswith("/HealthResponse")


###############################################################################
def test_clinical_job_route_advertises_response_model() -> None:
    schema = app.openapi()
    response = schema["paths"]["/api/clinical/jobs"]["post"]["responses"]["202"]
    assert "application/json" in response["content"]


###############################################################################
def test_stable_json_routes_declare_response_models() -> None:
    violations: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.include_in_schema:
            continue
        if route.status_code == 204:
            continue
        if route.response_class is PlainTextResponse:
            continue
        if route.response_model is None:
            methods = ",".join(sorted(route.methods or []))
            violations.append(f"{methods} {route.path}")

    assert not violations, "Routes missing response_model:\n" + "\n".join(violations)


###############################################################################
def test_inspection_revision_routes_are_present_in_openapi() -> None:
    schema = app.openapi()
    expected_paths = [
        "/api/inspection/sessions/{session_id}/versions",
        "/api/inspection/sessions/{session_id}/versions/{version_id}",
        "/api/inspection/sessions/{session_id}/versions/{left_version_id}/compare/{right_version_id}",
        "/api/inspection/sessions/{session_id}/report",
        "/api/inspection/sessions/{session_id}/manual-edits",
        "/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}",
        "/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/retry",
        "/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/steps",
    ]

    missing_paths = [path for path in expected_paths if path not in schema["paths"]]
    assert not missing_paths, "OpenAPI missing inspection paths:\n" + "\n".join(
        missing_paths
    )
