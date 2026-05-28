from __future__ import annotations

from fastapi.routing import APIRoute

from app import app


def test_schema_exposed_api_routes_define_response_model() -> None:
    missing_response_models: list[str] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.include_in_schema:
            continue
        if route.path == "/openapi.json":
            continue
        if route.response_model is None:
            missing_response_models.append(f"{sorted(route.methods)} {route.path}")

    assert missing_response_models == []
