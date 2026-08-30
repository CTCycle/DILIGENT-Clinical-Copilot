from __future__ import annotations

import json
from typing import get_args

from domain.keys import ProviderName
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.routing import APIRoute

from app import app


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
    schema = app.openapi()
    assert schema["openapi"]
    assert schema["paths"]
    violations: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.include_in_schema:
            continue
        if route.status_code == 204:
            continue
        response_class = route.response_class
        if response_class in (FileResponse, PlainTextResponse, StreamingResponse) or (
            isinstance(response_class, type)
            and issubclass(
                response_class, (FileResponse, PlainTextResponse, StreamingResponse)
            )
        ):
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


###############################################################################
def test_clean_break_routes_are_canonical() -> None:
    paths = app.openapi()["paths"]
    assert "/api/model-config/connectivity-check" in paths
    assert "/api/model-config/openai-connectivity-check" not in paths
    assert "/api/inspection/sessions/{session_id}/timelines" in paths
    assert "/api/inspection/sessions/{session_id}/timelines/{timeline_id}" in paths
    assert "/api/inspection/sessions/{session_id}/timeline-jobs" in paths
    assert "/api/inspection/sessions/{session_id}/timeline-jobs/{job_id}" in paths
    assert "post" not in paths["/api/inspection/sessions/{session_id}/timelines"]
    assert "/api/inspection/sessions/{session_id}/timeline" not in paths


###############################################################################
def test_model_catalog_routes_are_explicit_and_provider_scoped() -> None:
    paths = app.openapi()["paths"]
    assert "/api/model-config/catalogs/{provider}/load" in paths
    assert "/api/model-config/catalogs/{provider}/refresh" in paths
    get_parameters = paths["/api/model-config"]["get"].get("parameters", [])
    assert not any(
        item.get("name") == "include_local_availability" for item in get_parameters
    )


###############################################################################
def test_provider_descriptions_match_supported_providers() -> None:
    assert set(get_args(ProviderName)) == {
        "openai",
        "gemini",
        "deepseek",
        "anthropic",
        "opencode",
        "brave",
    }


###############################################################################
def test_access_key_openapi_schema_excludes_openrouter() -> None:
    schema = app.openapi()
    assert "openrouter" not in json.dumps(schema).lower()
