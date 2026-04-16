from __future__ import annotations

import pytest
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from DILIGENT.server.api import data_inspection
from DILIGENT.server.domain.inspection import (
    CatalogListFilters,
    DiliPriorCatalogResponse,
    DiliPriorDetailResponse,
    DrugLabelCatalogResponse,
    DrugLabelSectionsResponse,
    InspectionDiliPriorsOverrideRequest,
    InspectionDrugLabelsOverrideRequest,
    MAX_SEARCH_LENGTH,
    SessionListFilters,
)
from DILIGENT.server.services.inspection import DataInspectionService
from DILIGENT.server.repositories.serialization.data import _RepositorySerializationService
from DILIGENT.server.repositories.schemas.models import Base


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
    priors_request = InspectionDiliPriorsOverrideRequest(redownload=True)
    assert priors_request.redownload is True

    labels_request = InspectionDrugLabelsOverrideRequest(
        dailymed_request_timeout=10.0,
        dailymed_max_concurrency=4,
        redownload=False,
    )
    assert labels_request.dailymed_max_concurrency == 4

    priors_catalog = DiliPriorCatalogResponse(
        items=[
            {
                "drug_id": 1,
                "drug_name": "Acetaminophen",
                "dilirank_class": "Most-DILI-Concern",
                "dilist_class": "Known",
                "linked_source_count": 2,
            }
        ],
        total=1,
        offset=0,
        limit=10,
    )
    assert priors_catalog.total == 1

    prior_detail = DiliPriorDetailResponse(
        drug_id=1,
        drug_name="Acetaminophen",
        annotations=[{"source_dataset": "dilirank"}],
    )
    assert len(prior_detail.annotations) == 1

    label_catalog = DrugLabelCatalogResponse(
        items=[
            {
                "drug_id": 1,
                "drug_name": "Acetaminophen",
                "source": "dailymed",
                "effective_date": "2025-01-01",
                "retained_section_count": 1,
            }
        ],
        total=1,
        offset=0,
        limit=10,
    )
    assert label_catalog.total == 1

    label_detail = DrugLabelSectionsResponse(
        drug_id=1,
        drug_name="Acetaminophen",
        source="dailymed",
        set_id="set-1",
        spl_version=1,
        effective_date="2025-01-01",
        sections=[
            {
                "section_key": "boxed_warning",
                "section_title": "Boxed Warning",
                "text": "Hepatic warning.",
                "contains_hepatic_keywords": True,
                "display_order": 0,
            }
        ],
    )
    assert label_detail.sections[0]["section_key"] == "boxed_warning"


# -----------------------------------------------------------------------------
def test_dili_priors_update_config_route_is_not_shadowed() -> None:
    class ServiceStub:
        @staticmethod
        def build_update_config_response(target: str) -> dict[str, object]:
            assert target == "dili_priors"
            return {
                "target": "dili_priors",
                "defaults": {"dili_priors_request_timeout": 10.0, "redownload": False},
                "allowed_fields": ["dili_priors_request_timeout", "redownload"],
            }

    app = FastAPI()
    original_service = data_inspection.endpoint.service
    data_inspection.endpoint.service = ServiceStub()  # type: ignore[assignment]
    try:
        app.include_router(data_inspection.router)
        client = TestClient(app)
        response = client.get("/inspection/dili-priors/update-config")
    finally:
        data_inspection.endpoint.service = original_service

    assert response.status_code == 200
    assert response.json()["target"] == "dili_priors"


# -----------------------------------------------------------------------------
def test_dili_priors_update_config_exposes_only_supported_overrides() -> None:
    service = object.__new__(DataInspectionService)

    payload = service.build_update_config_response("dili_priors")

    assert payload["target"] == "dili_priors"
    assert payload["allowed_fields"] == ["redownload"]
    assert set(payload["defaults"]) == {"redownload"}
