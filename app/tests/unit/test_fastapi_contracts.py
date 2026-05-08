from __future__ import annotations

from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute

from app import app


def test_openapi_schema_generation_succeeds() -> None:
    schema = app.openapi()
    assert "paths" in schema
    assert "/api/health" in schema["paths"]


def test_health_route_uses_response_model() -> None:
    schema = app.openapi()
    response = schema["paths"]["/api/health"]["get"]["responses"]["200"]
    content = response["content"]["application/json"]
    assert content["schema"]["$ref"].endswith("/HealthResponse")


def test_clinical_plain_text_route_does_not_advertise_json_response_model() -> None:
    schema = app.openapi()
    response = schema["paths"]["/api/clinical"]["post"]["responses"]["202"]
    assert "text/plain" in response["content"]


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

