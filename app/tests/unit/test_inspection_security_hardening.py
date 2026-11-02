# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import services.inspection.service as inspection_service_module
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
from services.inspection import DataInspectionService
from services.inspection.attachment_policy import validate_upload


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
    rxnav_request = InspectionRxNavOverrideRequest(force_full=True)
    assert rxnav_request.force_full is True
    with pytest.raises(ValidationError):
        InspectionRxNavOverrideRequest.model_validate({"rxnav_max_concurrency": 4})

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
@pytest.mark.parametrize(
    ("path", "request_body"),
    [
        ("/livertox/jobs", {}),
        ("/structured-sources/jobs", {}),
    ],
)
@pytest.mark.parametrize(
    "contact_email", ["developer@example.org", "", "legacy@pharmagent.local"]
)
def test_ncbi_update_routes_start_with_or_without_contact(
    path: str,
    request_body: dict[str, object],
    contact_email: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:


    ###############################################################################
    class JobsProbe:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.created: list[dict[str, object]] = []

        # -------------------------------------------------------------------------
        def is_job_running(self, *args: object, **kwargs: object) -> bool:
            return False

        # -------------------------------------------------------------------------
        def start_job(self, **kwargs: object) -> str:
            self.created.append(kwargs)
            return "contact-configured-job"

        # -------------------------------------------------------------------------
        def get_job_status(self, job_id: str) -> dict[str, object]:
            job = self.created[0]
            return {
                "job_id": job_id,
                "job_type": job["job_type"],
                "status": "pending",
            }

    service = object.__new__(DataInspectionService)
    jobs = JobsProbe()
    service.jobs = jobs
    monkeypatch.setattr(
        inspection_service_module,
        "get_server_settings",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(ncbi_contact_email=contact_email),
            jobs=SimpleNamespace(polling_interval=1.0),
        ),
    )
    owner = get_route_owner(data_inspection.router, path)
    original_service = owner.service
    owner.service = service
    try:
        test_app = FastAPI()
        test_app.include_router(data_inspection.router)
        response = TestClient(test_app).post(
            f"/inspection{path}", json=request_body
        )
    finally:
        owner.service = original_service

    assert response.status_code == 202
    assert len(jobs.created) == 1


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
def test_rag_document_listing_exposes_unsupported_files_as_not_ingestible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supported_path = tmp_path / "nested" / "study.txt"
    supported_path.parent.mkdir()
    supported_path.write_text("TITLE\n\nBody text.", encoding="utf-8")
    duplicate_path = tmp_path / "nested" / "z-copy.txt"
    duplicate_path.write_bytes(supported_path.read_bytes())
    unsupported_path = tmp_path / "nested" / "legacy.doc"
    unsupported_path.write_bytes(b"legacy binary Word document")


    ###############################################################################
    class FakeVectorDatabase:

        # -------------------------------------------------------------------------
        def __init__(self, **_: object) -> None:
            pass

        # -------------------------------------------------------------------------
        def has_collection(self) -> bool:
            return False

    service = object.__new__(DataInspectionService)
    monkeypatch.setattr(
        service,
        "get_effective_rag_documents_path",
        lambda: str(tmp_path),
        raising=False,
    )
    monkeypatch.setattr(
        inspection_service_module,
        "build_effective_rag_settings",
        lambda: SimpleNamespace(
            vector_collection_name="documents",
            vector_index_metric="cosine",
            vector_index_type="IVF_FLAT",
            vector_stream_batch_size=16,
        ),
    )
    monkeypatch.setattr(
        inspection_service_module,
        "read_active_collection_name",
        lambda value: value,
    )
    monkeypatch.setattr(
        inspection_service_module,
        "LanceVectorDatabase",
        FakeVectorDatabase,
    )

    payload = service.list_rag_documents(search=None, offset=0, limit=10)

    rows = {item["file_name"]: item for item in payload["items"]}
    assert payload["total"] == 3
    assert rows["study.txt"]["supported_for_ingestion"] is True
    assert rows["study.txt"]["is_canonical_source"] is True
    assert rows["z-copy.txt"]["is_duplicate"] is True
    assert rows["z-copy.txt"]["duplicate_of"] == "nested/study.txt"
    assert rows["study.txt"]["duplicate_source_paths"] == [
        "nested/study.txt",
        "nested/z-copy.txt",
    ]
    assert rows["legacy.doc"]["supported_for_ingestion"] is False


###############################################################################
def test_rag_cancel_route_uses_delete_only() -> None:
    app = FastAPI()
    app.include_router(data_inspection.router)
    routes = {
        (method, getattr(route, "path", ""))
        for route in data_inspection.router.routes
        for method in getattr(route, "methods", set()) or set()
    }

    assert ("DELETE", "/inspection/rag/jobs/{job_id}") in routes
    assert ("POST", "/inspection/rag/jobs/{job_id}/cancel") not in routes


###############################################################################
def test_reference_catalog_runtime_observation_routes_are_registered() -> None:
    app = FastAPI()
    app.include_router(data_inspection.router)
    routes: set[tuple[str, str]] = set()
    for route in data_inspection.router.routes:
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
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


###############################################################################
class AttachmentServiceStub:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.rows: dict[int, tuple[dict[str, Any], bytes]] = {}

    # -------------------------------------------------------------------------
    def add_session_attachment(
        self, session_id: int, *, file_name: str, content: bytes
    ) -> dict[str, Any]:
        policy = validate_upload(file_name, content)
        attachment_id = len(self.rows) + 1
        summary = {
            "attachment_id": attachment_id,
            "session_id": session_id,
            "version_id": 1,
            "kind": policy.kind,
            "file_name": policy.file_name,
            "media_type": policy.media_type,
            "size_bytes": len(content),
            "sha256": "0" * 64,
            "created_at": None,
        }
        self.rows[attachment_id] = (summary, content)
        return summary

    # -------------------------------------------------------------------------
    def get_session_attachment_content(
        self, session_id: int, attachment_id: int
    ) -> tuple[dict[str, Any], bytes] | None:
        return self.rows.get(attachment_id)


###############################################################################
@pytest.fixture
def attachment_client() -> Any:
    stub = AttachmentServiceStub()
    endpoint = get_route_owner(data_inspection.router, "/sessions/{session_id}/attachments")
    original_service = endpoint.service
    endpoint.service = stub
    app = FastAPI()
    app.include_router(data_inspection.router)
    try:
        yield TestClient(app), stub
    finally:
        endpoint.service = original_service


# -----------------------------------------------------------------------------
def upload_attachment(client: TestClient, file_name: str, content: bytes) -> Any:
    return client.post(
        "/inspection/sessions/1/attachments",
        json={
            "file_name": file_name,
            "media_type": "text/html",
            "content_base64": base64.b64encode(content).decode(),
        },
    )


###############################################################################
@pytest.mark.parametrize(
    "file_name", ["page.html", "logo.SVG", "feed.xml", "run.js", "x.hta", "a.xhtml"]
)
def test_attachment_upload_rejects_active_formats(
    attachment_client: Any, file_name: str
) -> None:
    client, stub = attachment_client

    response = upload_attachment(client, file_name, b"<script>alert(1)</script>")

    assert response.status_code == 422
    assert "not allowed" in response.json()["detail"]
    assert not stub.rows


# -----------------------------------------------------------------------------
def test_attachment_upload_rejects_spoofed_image(attachment_client: Any) -> None:
    client, stub = attachment_client

    response = upload_attachment(client, "photo.png", b"<html><script>1</script>")

    assert response.status_code == 422
    assert "image format" in response.json()["detail"]
    assert not stub.rows


# -----------------------------------------------------------------------------
def test_attachment_png_is_served_inline_with_verified_type(
    attachment_client: Any,
) -> None:
    client, _ = attachment_client
    stored = upload_attachment(client, "scan.png", PNG_BYTES).json()

    response = client.get(f"/inspection/sessions/1/attachments/{stored['attachment_id']}")

    assert stored["kind"] == "image"
    assert stored["media_type"] == "image/png"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"


# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("file_name", "kind", "media_type"),
    [
        ("report.pdf", "document", "application/pdf"),
        ("data.json", "json", "application/json"),
        ("blob.bin", "document", "application/octet-stream"),
    ],
)
def test_attachment_documents_are_downloaded_not_inlined(
    attachment_client: Any, file_name: str, kind: str, media_type: str
) -> None:
    client, _ = attachment_client
    stored = upload_attachment(client, file_name, b"%PDF-1.4").json()

    response = client.get(f"/inspection/sessions/1/attachments/{stored['attachment_id']}")

    assert stored["kind"] == kind
    assert response.headers["content-type"] == media_type
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"


# -----------------------------------------------------------------------------
def test_legacy_html_attachment_is_served_as_inert_download(
    attachment_client: Any,
) -> None:
    client, stub = attachment_client
    summary = {
        "attachment_id": 7,
        "file_name": "legacy.html",
        "media_type": "text/html",
        "kind": "document",
    }
    stub.rows[7] = (summary, b"<script>alert(1)</script>")

    response = client.get("/inspection/sessions/1/attachments/7")

    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["x-content-type-options"] == "nosniff"


# -----------------------------------------------------------------------------
def test_attachment_file_name_cannot_inject_headers(attachment_client: Any) -> None:
    client, _ = attachment_client
    hostile_name = '..\\evil/rep\r\nX-Injected: 1"ört.pdf'
    stored = upload_attachment(client, hostile_name, b"%PDF-1.4").json()

    response = client.get(f"/inspection/sessions/1/attachments/{stored['attachment_id']}")

    disposition = response.headers["content-disposition"]
    assert stored["file_name"] == 'repX-Injected: 1"ört.pdf'
    assert "x-injected" not in response.headers
    assert '/' not in stored["file_name"] and "\r" not in stored["file_name"]
    ascii_part, _, encoded_part = disposition.partition("; filename*=UTF-8''")
    assert ascii_part == 'attachment; filename="repX-Injected: 1__rt.pdf"'
    assert encoded_part == "repX-Injected%3A%201%22%C3%B6rt.pdf"
