from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from api import data_inspection
from domain.inspection import (
    MAX_SEARCH_LENGTH,
    CatalogListFilters,
    InspectionLiverToxOverrideRequest,
    InspectionRagUpdateRequest,
    InspectionRxNavOverrideRequest,
    SessionListFilters,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from repositories.drug_catalog_repository import _build_search_pattern
import services.inspection.service as inspection_service_module
from services.inspection import DataInspectionService

###############################################################################
def get_route_owner(router: Any, route_path: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", "").endswith(route_path):
            owner = getattr(route.endpoint, "__self__", None)
            if owner is not None:
                return owner
    raise AssertionError(f"Route not found: {route_path}")

###############################################################################
def test_session_search_filter_strips_control_characters() -> None:
    filters = SessionListFilters(search=" \x00  metformin\t\n ")

    assert filters.search == "metformin"

###############################################################################
def test_catalog_search_filter_rejects_oversized_values() -> None:
    oversized = "a" * (MAX_SEARCH_LENGTH + 1)

    with pytest.raises(ValidationError):
        CatalogListFilters(search=oversized)

###############################################################################
def test_search_pattern_escapes_like_wildcards() -> None:
    pattern = _build_search_pattern(r"  100%_match\check  ")

    assert pattern == r"%100\%\_match\\check%"

###############################################################################
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

    rag_request = InspectionRagUpdateRequest(documents_path="C:/data/rag")
    assert rag_request.documents_path == "C:/data/rag"

###############################################################################
def test_livertox_update_config_route_is_not_shadowed() -> None:

    ###############################################################################
    class ServiceStub:

        # -------------------------------------------------------------------------
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

###############################################################################
def test_livertox_update_config_exposes_only_supported_overrides() -> None:
    service = object.__new__(DataInspectionService)

    payload = service.build_update_config_response("livertox")

    assert payload["target"] == "livertox"
    assert "redownload" in payload["allowed_fields"]
    assert "redownload" in payload["defaults"]

###############################################################################
def test_rag_update_config_exposes_read_only_vectorization_summary() -> None:
    service = object.__new__(DataInspectionService)

    payload = service.build_update_config_response("rag")

    assert payload["target"] == "rag"
    assert payload["read_only"] is True
    assert payload["defaults"] == {}
    assert payload["allowed_fields"] == []
    assert "chunk_size" in payload["summary"]
    assert "documents_path" not in payload["summary"]
    assert "retrieval_candidate_count" not in payload["summary"]

###############################################################################
def test_rag_update_job_route_rejects_removed_vectorization_overrides() -> None:

    ###############################################################################
    class ServiceStub:
        RAG_JOB_TYPE = "rag_update"

        # -------------------------------------------------------------------------
        @staticmethod
        def start_update_job(
            job_type: str, overrides: dict[str, object] | None = None
        ) -> dict[str, object]:
            assert job_type == "rag_update"
            assert overrides == {"documents_path": "C:/data/rag"}
            return {
                "job_id": "job-1",
                "job_type": "rag_update",
                "status": "pending",
                "poll_interval": 1.0,
            }

    app = FastAPI()
    endpoint = get_route_owner(data_inspection.router, "/rag/jobs")
    original_service = endpoint.service
    endpoint.service = ServiceStub()  # type: ignore[assignment]
    try:
        app.include_router(data_inspection.router)
        client = TestClient(app)
        rejected = client.post(
            "/inspection/rag/jobs",
            json={"chunk_size": 256},
        )
        accepted = client.post(
            "/inspection/rag/jobs",
            json={"documents_path": "C:/data/rag"},
        )
    finally:
        endpoint.service = original_service

    assert rejected.status_code == 422
    assert accepted.status_code == 202
    assert accepted.json()["job_type"] == "rag_update"

###############################################################################
def test_rag_directory_browse_lists_canonical_child_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "RAG Démo"
    root.mkdir()
    (root / "zeta").mkdir()
    (root / "Alpha folder").mkdir()
    (root / "document.txt").write_text("not a directory", encoding="utf-8")
    service = object.__new__(DataInspectionService)
    monkeypatch.setattr(
        inspection_service_module,
        "get_server_settings",
        lambda: SimpleNamespace(
            rag=SimpleNamespace(allow_local_filesystem_access=True)
        ),
    )

    payload = service.browse_rag_directories(str(root))

    assert payload["current_path"] == str(root.resolve())
    assert payload["parent_path"] == str(root.resolve().parent)
    assert [item["name"] for item in payload["items"]] == [
        "Alpha folder",
        "zeta",
    ]
    assert all(item["is_dir"] for item in payload["items"])
    assert all(Path(item["path"]).is_absolute() for item in payload["items"])

###############################################################################
def test_rag_directory_browse_handles_roots_and_rejects_invalid_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(DataInspectionService)
    monkeypatch.setattr(
        inspection_service_module,
        "get_server_settings",
        lambda: SimpleNamespace(
            rag=SimpleNamespace(allow_local_filesystem_access=True)
        ),
    )

    roots = service.browse_rag_directories()

    assert roots["current_path"] == ""
    assert roots["parent_path"] is None
    assert roots["drives"]
    assert roots["items"]

    with pytest.raises(ValueError, match="must be absolute"):
        service.browse_rag_directories("relative-folder")
    with pytest.raises(FileNotFoundError):
        service.browse_rag_directories(str(tmp_path / "missing"))
    file_path = tmp_path / "document.txt"
    file_path.write_text("document", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        service.browse_rag_directories(str(file_path))

###############################################################################
def test_rag_directory_browse_respects_filesystem_access_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(DataInspectionService)
    monkeypatch.setattr(
        inspection_service_module,
        "get_server_settings",
        lambda: SimpleNamespace(
            rag=SimpleNamespace(allow_local_filesystem_access=False)
        ),
    )

    with pytest.raises(PermissionError):
        service.browse_rag_directories()

###############################################################################
def test_rag_directory_browse_route_returns_typed_payload() -> None:

    ###############################################################################
    class ServiceStub:

        # -------------------------------------------------------------------------
        @staticmethod
        def browse_rag_directories(path: str) -> dict[str, object]:
            assert path == ""
            return {
                "current_path": "",
                "parent_path": None,
                "items": [{"name": "C:\\", "path": "C:\\", "is_dir": True}],
                "drives": ["C:\\"],
            }

    app = FastAPI()
    endpoint = get_route_owner(data_inspection.router, "/rag/browse")
    original_service = endpoint.service
    endpoint.service = ServiceStub()  # type: ignore[assignment]
    try:
        app.include_router(data_inspection.router)
        response = TestClient(app).get("/inspection/rag/browse")
    finally:
        endpoint.service = original_service

    assert response.status_code == 200
    assert response.json()["items"][0]["path"] == "C:\\"
    response_schema = app.openapi()["paths"]["/inspection/rag/browse"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/RagDirectoryBrowseResponse")

###############################################################################
@pytest.mark.parametrize(
    ("service_error", "expected_status", "expected_detail"),
    [
        (
            PermissionError("permission denied: C:\\private"),
            403,
            "Local filesystem browsing is not available.",
        ),
        (
            FileNotFoundError("missing: C:\\private"),
            404,
            "The selected folder was not found.",
        ),
        (
            ValueError("invalid path: C:\\private"),
            422,
            "The selected folder path is invalid or cannot be read.",
        ),
    ],
)
def test_rag_directory_browse_route_sanitizes_filesystem_errors(
    service_error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:

    ###############################################################################
    class ServiceStub:

        # -------------------------------------------------------------------------
        @staticmethod
        def browse_rag_directories(path: str) -> dict[str, object]:
            _ = path
            raise service_error

    app = FastAPI()
    endpoint = get_route_owner(data_inspection.router, "/rag/browse")
    original_service = endpoint.service
    endpoint.service = ServiceStub()  # type: ignore[assignment]
    try:
        app.include_router(data_inspection.router)
        response = TestClient(app).get(
            "/inspection/rag/browse",
            params={"path": "C:\\private"},
        )
    finally:
        endpoint.service = original_service

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert "C:\\private" not in response.text

###############################################################################
def test_rag_cancel_route_uses_delete_only() -> None:
    app = FastAPI()
    app.include_router(data_inspection.router)
    routes = {
        (method, getattr(route, "path", ""))
        for route in app.routes
        for method in getattr(route, "methods", set()) or set()
    }

    assert ("DELETE", "/inspection/rag/jobs/{job_id}") in routes
    assert ("POST", "/inspection/rag/jobs/{job_id}/cancel") not in routes

###############################################################################
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

###############################################################################
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
