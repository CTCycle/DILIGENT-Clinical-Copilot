# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from api.inspection.terminology import InspectionTerminologyEndpoint
from repository_fixtures import build_repository_graph
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from services.terminology.index import get_terminology_index_cache
from terminology_fixtures import build_sqlite_engine, seed_reference_terminology


###############################################################################
@pytest.fixture
def client(tmp_path) -> TestClient:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=build_sqlite_engine(tmp_path / "t.db"))
    seed_reference_terminology(graph.context)
    # The process-wide index is keyed by revision, which every test database reuses.
    get_terminology_index_cache().invalidate()
    service = DataInspectionService(
        clinical_session_repository=graph.clinical_session_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
        knowledge_repository=graph.knowledge_repository,
        session_timeline_repository=graph.session_timeline_repository,
        session_revision_repository=graph.session_revision_repository,
        jobs=JobManager(),
    )
    router = APIRouter(prefix="/inspection")
    InspectionTerminologyEndpoint(router=router, service=service).add_routes()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


###############################################################################
def test_summary_reports_sources_with_license_and_coverage(client: TestClient) -> None:
    response = client.get("/inspection/terminology/summary")

    assert response.status_code == 200
    payload = response.json()
    sources = {item["source"]: item for item in payload["sources"]}
    assert sources["rxnorm"]["last_status"] == "completed"
    assert sources["rxnorm"]["aliases"] > 0
    assert "CC BY-SA" in sources["drugcentral"]["license"]
    assert (payload["products"], payload["verified_products"]) == (3, 2)
    assert payload["linked_monographs"] == payload["total_monographs"]

    runs = client.get("/inspection/terminology/runs", params={"limit": 5}).json()["items"]
    assert {run["source"] for run in runs} == {"rxnorm", "curated", "swissmedic"}
    assert client.get("/inspection/terminology/conflicts").json()["total"] == 0


###############################################################################
def test_reviewer_confirmation_and_rejection_apply_to_later_resolutions(client: TestClient) -> None:
    ambiguous = client.post(
        "/inspection/terminology/resolve", json={"name": "hydroxazine", "allow_external": False}
    ).json()
    assert ambiguous["status"] == "ambiguous"
    hydroxyzine = next(item["drug_id"] for item in ambiguous["candidates"] if item["name"] == "hydroxyzine")

    confirmed = client.post(
        "/inspection/terminology/aliases/confirm", json={"alias": "Hydroxazine", "drug_id": hydroxyzine}
    )
    assert confirmed.json() == {"alias": "Hydroxazine", "drug_id": hydroxyzine, "validation_status": "confirmed"}
    resolved = client.post(
        "/inspection/terminology/resolve", json={"name": "hydroxazine 25 mg", "allow_external": False}
    ).json()
    assert (resolved["status"], resolved["method"], resolved["preferred_name"]) == (
        "verified",
        "confirmed_local",
        "hydroxyzine",
    )

    lipitor = client.post("/inspection/terminology/resolve", json={"name": "Lipitor", "allow_external": False}).json()
    client.post("/inspection/terminology/aliases/reject", json={"alias": "Lipitor", "drug_id": lipitor["drug_id"]})
    rejected = client.post("/inspection/terminology/resolve", json={"name": "Lipitor", "allow_external": False}).json()
    assert rejected["accepted"] is False

    missing = client.post("/inspection/terminology/aliases/confirm", json={"alias": "x", "drug_id": 999999})
    assert missing.status_code == 404


###############################################################################
def test_update_config_exposes_source_flags(client: TestClient) -> None:
    payload = client.get("/inspection/terminology/update-config").json()

    assert payload["target"] == "terminology"
    assert set(payload["allowed_fields"]) == {
        "force_full",
        "drugcentral_enabled",
        "chembl_enabled",
        "swissmedic_enabled",
    }
    assert client.post("/inspection/terminology/jobs", json={"unknown": True}).status_code == 422
