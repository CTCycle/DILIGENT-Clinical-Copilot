from __future__ import annotations

from typing import Any

import pytest
from api import data_inspection
from domain.inspection import (
    MAX_SEARCH_LENGTH,
    CatalogListFilters,
    InspectionLiverToxOverrideRequest,
    InspectionRagOverrideRequest,
    InspectionRxNavOverrideRequest,
    SessionListFilters,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from repositories.schemas.models import Base
from repositories.serialization.data import (
    _RepositorySerializationService,
)
from services.inspection import DataInspectionService
from sqlalchemy import create_engine


def get_route_owner(router: Any, route_path: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", "").endswith(route_path):
            owner = getattr(route.endpoint, "__self__", None)
            if owner is not None:
                return owner
    raise AssertionError(f"Route not found: {route_path}")


# -----------------------------------------------------------------------------
def test_session_search_filter_strips_control_characters() -> None:
    filters = SessionListFilters(search=" \x00  metformin\t\n ")

    assert filters.search == "metformin"


# -----------------------------------------------------------------------------
def test_catalog_search_filter_rejects_oversized_values() -> None:
    oversized = "a" * (MAX_SEARCH_LENGTH + 1)

    with pytest.raises(ValidationError):
        CatalogListFilters(search=oversized)


# -----------------------------------------------------------------------------
def test_search_pattern_escapes_like_wildcards() -> None:
    service = object.__new__(_RepositorySerializationService)

    pattern = service.build_search_pattern(r"  100%_match\check  ")

    assert pattern == r"%100\%\_match\\check%"


# -----------------------------------------------------------------------------
def test_schema_guard_rejects_missing_required_columns() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE drugs")
        connection.exec_driver_sql(
            "CREATE TABLE drugs ("
            "id INTEGER NOT NULL PRIMARY KEY, "
            "canonical_name TEXT NOT NULL, "
            "canonical_name_norm VARCHAR NOT NULL, "
            "rxnorm_rxcui VARCHAR, "
            "livertox_nbk_id VARCHAR"
            ")"
        )

    service = object.__new__(_RepositorySerializationService)
    service.engine = engine

    with pytest.raises(RuntimeError, match="missing required column"):
        service.ensure_session_result_table()


# -----------------------------------------------------------------------------
def test_new_inspection_models_validate_shapes() -> None:
    rxnav_request = InspectionRxNavOverrideRequest(
        rxnav_request_timeout=10.0,
        rxnav_max_concurrency=4,
    )
    assert rxnav_request.rxnav_max_concurrency == 4

    livertox_request = InspectionLiverToxOverrideRequest(
        livertox_archive="livertox-current.zip",
        redownload=True,
    )
    assert livertox_request.redownload is True

    rag_request = InspectionRagOverrideRequest(
        chunk_size=512,
        chunk_overlap=64,
        use_cloud_embeddings=False,
    )
    assert rag_request.chunk_size == 512


# -----------------------------------------------------------------------------
def test_livertox_update_config_route_is_not_shadowed() -> None:
    class ServiceStub:
        @staticmethod
        def build_update_config_response(target: str) -> dict[str, object]:
            assert target == "livertox"
            return {
                "target": "livertox",
                "defaults": {"redownload": False},
                "allowed_fields": ["redownload"],
            }

    app = FastAPI()
    endpoint = get_route_owner(data_inspection.router, "/livertox/update-config")
    original_service = endpoint.service
    endpoint.service = ServiceStub()  # type: ignore[assignment]
    try:
        app.include_router(data_inspection.router)
        client = TestClient(app)
        response = client.get("/inspection/livertox/update-config")
    finally:
        endpoint.service = original_service

    assert response.status_code == 200
    assert response.json()["target"] == "livertox"


# -----------------------------------------------------------------------------
def test_livertox_update_config_exposes_only_supported_overrides() -> None:
    service = object.__new__(DataInspectionService)

    payload = service.build_update_config_response("livertox")

    assert payload["target"] == "livertox"
    assert "redownload" in payload["allowed_fields"]
    assert "redownload" in payload["defaults"]


# -----------------------------------------------------------------------------
def test_reference_catalog_runtime_observation_routes_are_registered() -> None:
    app = FastAPI()
    app.include_router(data_inspection.router)
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/inspection/"):
            continue
        for method in getattr(route, "methods", set()) or set():
            routes.add((method, path))
    assert ("GET", "/inspection/reference-catalogs/runtime-observations") in routes
    assert (
        "GET",
        "/inspection/reference-catalogs/runtime-observations/{category}",
    ) in routes
    assert (
        "PUT",
        "/inspection/reference-catalogs/runtime-observations/{category}",
    ) in routes
    assert (
        "DELETE",
        "/inspection/reference-catalogs/runtime-observations/{category}/{term}",
    ) in routes


# -----------------------------------------------------------------------------
def test_legacy_text_normalization_routes_are_removed() -> None:
    app = FastAPI()
    app.include_router(data_inspection.router)
    legacy_paths = {
        "/inspection/text-normalization",
        "/inspection/text-normalization/{category}",
        "/inspection/text-normalization/{category}/{term}",
    }
    current_paths = {getattr(route, "path", "") for route in app.routes}
    assert legacy_paths.isdisjoint(current_paths)
