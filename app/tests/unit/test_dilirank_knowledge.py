from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.inspection.dilirank import InspectionDiliRankEndpoint
from common.prompts.clinical_context import build_dilirank_knowledge_fragment
from domain.clinical.entities import DrugClinicalAssessment
from repositories.context import RepositoryContext
from repositories.dilirank_repository import (
    DILIRANK_IDENTIFIER_SYSTEM,
    DiliRankRepository,
)
from repositories.knowledge_repository import KnowledgeRepository
from repositories.schemas import Base
from repositories.schemas.knowledge import Drug, DrugAlias, DrugIdentifier
from services.clinical.knowledge import ClinicalKnowledgeComposer
from services.clinical.report_finalizer import ReportFinalizer
from services.updater.dilirank import DiliRankUpdater

###############################################################################
def build_repository() -> tuple[DiliRankRepository, RepositoryContext]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    context = RepositoryContext.create(
        engine=engine,
        session_factory=session_factory,
    )
    return DiliRankRepository(context), context

###############################################################################
def seed_drugs(context: RepositoryContext) -> None:
    with context.session_factory() as db_session:
        acetaminophen = Drug(
            canonical_name="Acetaminophen",
            canonical_name_norm="acetaminophen",
        )
        ibuprofen = Drug(
            canonical_name="Ibuprofen",
            canonical_name_norm="ibuprofen",
        )
        collision_a = Drug(canonical_name="Alpha", canonical_name_norm="alpha")
        collision_b = Drug(canonical_name="Beta", canonical_name_norm="beta")
        session_only = Drug(canonical_name="Gamma", canonical_name_norm="gamma")
        db_session.add_all(
            [acetaminophen, ibuprofen, collision_a, collision_b, session_only]
        )
        db_session.flush()
        db_session.add_all(
            [
                DrugAlias(
                    drug_id=acetaminophen.id,
                    alias="Paracetamol",
                    alias_norm="paracetamol",
                    alias_kind="synonym",
                    source="livertox",
                    term_type=None,
                ),
                DrugAlias(
                    drug_id=collision_a.id,
                    alias="Collision",
                    alias_norm="collision",
                    alias_kind="synonym",
                    source="livertox",
                    term_type=None,
                ),
                DrugAlias(
                    drug_id=collision_b.id,
                    alias="Collision",
                    alias_norm="collision",
                    alias_kind="synonym",
                    source="rxnorm",
                    term_type="SCD",
                ),
                DrugAlias(
                    drug_id=session_only.id,
                    alias="Session-only mention",
                    alias_norm="session only mention",
                    alias_kind="observed_query",
                    source="session",
                    term_type=None,
                ),
            ]
        )
        db_session.commit()

###############################################################################
def test_replace_records_links_canonical_and_unique_alias_without_guessing() -> None:
    repository, context = build_repository()
    seed_drugs(context)

    summary = repository.replace_records(
        [
            {
                "ltkb_id": "LT001",
                "compound_name": "Acetaminophen",
                "severity_class": 8,
                "label_section": "Warnings and Precautions",
                "dili_concern": "vMost-DILI-concern",
                "comment": None,
                "source_url": "https://www.fda.gov/example",
                "source_last_modified": "Wed, 01 Jan 2025 00:00:00 GMT",
            },
            {
                "ltkb_id": "LT002",
                "compound_name": "Paracetamol",
                "severity_class": "7",
                "label_section": "Warnings and Precautions",
                "dili_concern": "vMost-DILI-concern",
                "comment": None,
                "source_url": "https://www.fda.gov/example",
                "source_last_modified": None,
            },
            {
                "ltkb_id": "LT003",
                "compound_name": "Collision",
                "severity_class": None,
                "label_section": None,
                "dili_concern": "Ambiguous-DILI-concern",
                "comment": None,
                "source_url": "https://www.fda.gov/example",
                "source_last_modified": None,
            },
            {
                "ltkb_id": "LT004",
                "compound_name": "Unmapped compound",
                "severity_class": None,
                "label_section": None,
                "dili_concern": "vLess-DILI-concern",
                "comment": None,
                "source_url": "https://www.fda.gov/example",
                "source_last_modified": None,
            },
        ]
    )

    assert summary["linked_records"] == 2
    assert summary["ambiguous_records"] == 1
    assert summary["unmatched_records"] == 1
    acetaminophen_records = repository.list_catalog(
        search="acetaminophen", offset=0, limit=10
    )[0]
    assert {item["ltkb_id"] for item in acetaminophen_records} == {"LT001", "LT002"}
    with context.session_factory() as db_session:
        identifiers = db_session.execute(
            select(DrugIdentifier).where(
                DrugIdentifier.identifier_system == DILIRANK_IDENTIFIER_SYSTEM
            )
        ).scalars().all()
    assert {item.identifier_value for item in identifiers} == {"LT001", "LT002"}

###############################################################################
def test_replace_records_ignores_session_observed_aliases_for_linkage() -> None:
    repository, context = build_repository()
    seed_drugs(context)

    summary = repository.replace_records(
        [
            {
                "ltkb_id": "LT005",
                "compound_name": "Session-only mention",
                "severity_class": 4,
                "dili_concern": "vLess-DILI-concern",
            }
        ]
    )

    assert summary["linked_records"] == 0
    assert summary["unmatched_records"] == 1
    rows, total = repository.list_catalog(search="Session-only", offset=0, limit=10)
    assert total == 1
    assert rows[0]["drug_id"] is None

###############################################################################
def test_replace_records_preserves_unlinked_source_rows() -> None:
    repository, _ = build_repository()
    summary = repository.replace_records(
        [
            {
                "ltkb_id": "LT001",
                "compound_name": "Acetaminophen",
                "severity_class": 5,
                "dili_concern": "vMost-DILI-concern",
            }
        ]
    )

    assert summary["persisted_records"] == 1
    assert summary["linked_records"] == 0
    rows, total = repository.list_catalog(search=None, offset=0, limit=10)
    assert total == 1
    assert rows[0]["drug_id"] is None
    assert rows[0]["compound_name"] == "Acetaminophen"

###############################################################################
def test_dilirank_parser_preserves_official_schema_and_rejects_unknown_category(
    tmp_path: Path,
) -> None:
    repository, _ = build_repository()
    updater = DiliRankUpdater(repository=repository, archives_path=tmp_path)
    valid_frame = pd.DataFrame(
        [
            {
                "LTKBID": "LT001",
                "CompoundName": "Acetaminophen",
                "SeverityClass": 8.0,
                "LabelSection": "Warnings and Precautions",
                "vDILI-Concern": "vMost-DILI-concern",
                "Comment": None,
            }
        ]
    )
    records = updater._parse_records(
        valid_frame,
        {"source_url": "https://www.fda.gov/example", "last_modified": "today"},
    )
    assert records[0]["severity_class"] == 8
    assert records[0]["dili_concern"] == "vMost-DILI-concern"

    invalid_frame = valid_frame.copy()
    invalid_frame.loc[0, "vDILI-Concern"] = "New-unvalidated-category"
    with pytest.raises(RuntimeError, match="unsupported concern category"):
        updater._parse_records(invalid_frame, {})

###############################################################################
def test_dilirank_parser_normalizes_official_category_capitalization(
    tmp_path: Path,
) -> None:
    repository, _ = build_repository()
    updater = DiliRankUpdater(repository=repository, archives_path=tmp_path)
    frame = pd.DataFrame(
        [
            {
                "LTKBID": "LT040",
                "CompoundName": "Abacavir sulfate",
                "SeverityClass": 8,
                "LabelSection": "Warnings & precautions",
                "vDILI-Concern": "vMOST-DILI-concern",
                "Comment": "Unchanged",
            }
        ]
    )

    records = updater._parse_records(frame, {})

    assert records[0]["dili_concern"] == "vMost-DILI-concern"

###############################################################################
def test_dilirank_updater_reads_official_version_two_sheet_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _ = build_repository()
    updater = DiliRankUpdater(repository=repository, archives_path=tmp_path)
    frame = pd.DataFrame(
        [
            {
                "LTKBID": "LT001",
                "CompoundName": "Acetaminophen",
                "SeverityClass": 8,
                "LabelSection": "Warnings & precautions",
                "vDILI-Concern": "vMost-DILI-concern",
                "Comment": "Unchanged",
            }
        ]
    )
    read_excel_kwargs: dict[str, object] = {}

    def fake_download() -> dict[str, object]:
        updater.candidate_path.write_bytes(b"candidate")
        return {
            "downloaded": True,
            "source_url": "https://www.fda.gov/example",
            "last_modified": None,
            "etag": None,
            "size": 9,
        }

    def fake_read_excel(path: Path, **kwargs: object) -> pd.DataFrame:
        assert path == updater.candidate_path
        read_excel_kwargs.update(kwargs)
        return frame

    monkeypatch.setattr(updater, "_download_workbook", fake_download)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = updater.update_from_fda()

    assert result["persisted_records"] == 1
    assert read_excel_kwargs == {
        "sheet_name": "version 2",
        "header": 1,
        "engine": "openpyxl",
    }

###############################################################################
def test_failed_download_candidate_preserves_last_known_good_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _ = build_repository()
    updater = DiliRankUpdater(repository=repository, archives_path=tmp_path)
    updater.workbook_path.write_bytes(b"stable-workbook")
    old_metadata = {"etag": "old-etag", "source_url": "stable-source"}
    updater.metadata_path.write_text(json.dumps(old_metadata), encoding="utf-8")

    def fake_download() -> dict[str, object]:
        updater.candidate_path.write_bytes(b"PK-invalid-xlsx")
        return {
            "downloaded": True,
            "etag": "new-etag",
            "source_url": "new-source",
            "last_modified": "now",
            "size": 15,
        }

    monkeypatch.setattr(updater, "_download_workbook", fake_download)

    with pytest.raises(Exception):
        updater.update_from_fda()

    assert updater.workbook_path.read_bytes() == b"stable-workbook"
    assert json.loads(updater.metadata_path.read_text(encoding="utf-8")) == old_metadata
    assert not updater.candidate_path.exists()

###############################################################################
def test_cancelled_download_candidate_preserves_last_known_good_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _ = build_repository()
    updater = DiliRankUpdater(repository=repository, archives_path=tmp_path)
    updater.workbook_path.write_bytes(b"stable-workbook")
    old_metadata = {"etag": "old-etag", "source_url": "stable-source"}
    updater.metadata_path.write_text(json.dumps(old_metadata), encoding="utf-8")

    def fake_download() -> dict[str, object]:
        updater.candidate_path.write_bytes(b"PK-candidate")
        return {
            "downloaded": True,
            "etag": "new-etag",
            "source_url": "new-source",
            "last_modified": "now",
            "size": 12,
        }

    stop_checks = iter([False, True])
    monkeypatch.setattr(updater, "_download_workbook", fake_download)

    with pytest.raises(RuntimeError, match="cancelled"):
        updater.update_from_fda(should_stop=lambda: next(stop_checks))

    assert updater.workbook_path.read_bytes() == b"stable-workbook"
    assert json.loads(updater.metadata_path.read_text(encoding="utf-8")) == old_metadata
    assert not updater.candidate_path.exists()

###############################################################################
def test_dilirank_prompt_keeps_drug_level_prior_separate_from_patient_causality() -> None:
    fragment = build_dilirank_knowledge_fragment(
        records=[
            {
                "ltkb_id": "LT001",
                "compound_name": "Acetaminophen",
                "severity_class": 8,
                "label_section": "Warnings and Precautions",
                "dili_concern": "vMost-DILI-concern",
                "comment": "Curated classification",
            }
        ]
    )
    assert "vMost-DILI-concern" in fragment
    assert "patient-specific causality score" in fragment
    assert "do not override chronology" in fragment

###############################################################################
def test_dilirank_inspection_routes_use_shared_update_contract() -> None:

    ###############################################################################
    class ServiceStub:
        DILIRANK_JOB_TYPE = "dilirank_update"

        # -------------------------------------------------------------------------
        @staticmethod
        def build_update_config_response(target: str) -> dict[str, object]:
            assert target == "dilirank"
            return {
                "target": "dilirank",
                "defaults": {"redownload": False},
                "allowed_fields": ["redownload"],
                "summary": {},
                "read_only": False,
            }

        # -------------------------------------------------------------------------
        @staticmethod
        def list_dilirank_catalog(
            *, search: str | None, offset: int, limit: int
        ) -> dict[str, object]:
            assert search == "acetaminophen"
            return {
                "items": [
                    {
                        "drug_id": 1,
                        "drug_name": "Acetaminophen",
                        "ltkb_id": "LT00004",
                        "compound_name": "Acetaminophen",
                        "severity_class": 5,
                        "label_section": "Warnings & precautions",
                        "dili_concern": "vMost-DILI-concern",
                        "comment": "Unchanged",
                        "source_url": "https://www.fda.gov/example",
                        "source_last_modified": None,
                    }
                ],
                "total": 1,
                "offset": offset,
                "limit": limit,
            }

        # -------------------------------------------------------------------------
        @staticmethod
        def get_dilirank_records(drug_id: int) -> dict[str, object] | None:
            assert drug_id == 1
            return {
                "drug_id": 1,
                "drug_name": "Acetaminophen",
                "records": [
                    {
                        "drug_id": 1,
                        "drug_name": "Acetaminophen",
                        "ltkb_id": "LT00004",
                        "compound_name": "Acetaminophen",
                        "severity_class": 5,
                        "label_section": "Warnings & precautions",
                        "dili_concern": "vMost-DILI-concern",
                        "comment": "Unchanged",
                        "source_url": "https://www.fda.gov/example",
                        "source_last_modified": None,
                    }
                ],
            }

        # -------------------------------------------------------------------------
        @staticmethod
        def start_update_job(
            job_type: str,
            overrides: dict[str, object] | None = None,
        ) -> dict[str, object]:
            assert job_type == "dilirank_update"
            assert overrides == {"redownload": True}
            return {
                "job_id": "job-1",
                "job_type": "dilirank_update",
                "status": "pending",
                "poll_interval": 1.0,
            }

        # -------------------------------------------------------------------------
        @staticmethod
        def get_job_status(
            job_id: str,
            *,
            expected_type: str,
        ) -> dict[str, object] | None:
            assert job_id == "job-1"
            assert expected_type == "dilirank_update"
            return {
                "job_id": "job-1",
                "job_type": "dilirank_update",
                "status": "completed",
                "progress": 100.0,
                "result": {"summary": {"persisted_records": 1336}},
                "error": None,
                "created_at": 1.0,
                "completed_at": 2.0,
                "version": 1,
                "stop_requested": False,
            }

        # -------------------------------------------------------------------------
        @staticmethod
        def cancel_job(job_id: str, *, expected_type: str) -> bool:
            assert job_id == "job-1"
            assert expected_type == "dilirank_update"
            return True

    router = APIRouter(prefix="/inspection")
    InspectionDiliRankEndpoint(router=router, service=ServiceStub()).add_routes()  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    config = client.get("/inspection/dilirank/update-config")
    catalog = client.get("/inspection/dilirank", params={"search": "acetaminophen"})
    detail = client.get("/inspection/dilirank/1")
    started = client.post("/inspection/dilirank/jobs", json={"redownload": True})
    status_response = client.get("/inspection/dilirank/jobs/job-1")
    cancelled = client.delete("/inspection/dilirank/jobs/job-1")

    assert config.status_code == 200
    assert config.json()["target"] == "dilirank"
    assert catalog.status_code == 200
    assert catalog.json()["items"][0]["ltkb_id"] == "LT00004"
    assert detail.status_code == 200
    assert started.status_code == 202
    assert started.json()["job_type"] == "dilirank_update"
    assert status_response.status_code == 200
    assert cancelled.status_code == 200

###############################################################################
def test_clinical_composer_adds_linked_dilirank_evidence_and_provenance() -> None:
    dilirank_repository, context = build_repository()
    seed_drugs(context)
    dilirank_repository.replace_records(
        [
            {
                "ltkb_id": "LT00004",
                "compound_name": "Acetaminophen",
                "severity_class": 5,
                "label_section": "Warnings & precautions",
                "dili_concern": "vMost-DILI-concern",
                "comment": "Unchanged",
                "source_url": "https://www.fda.gov/example",
            }
        ]
    )
    with context.session_factory() as db_session:
        drug_id = int(
            db_session.execute(
                select(Drug.id).where(Drug.canonical_name_norm == "acetaminophen")
            ).scalar_one()
        )
    composer = ClinicalKnowledgeComposer(
        knowledge_repository=KnowledgeRepository(context)
    )
    resolved = {
        "acetaminophen": {
            "matched_livertox_row": {"drug_id": drug_id},
            "extracted_excerpts": ["LiverTox evidence."],
        }
    }

    composer.enrich_resolved_drugs(resolved)

    payload = resolved["acetaminophen"]
    assert payload["dilirank_records"][0]["ltkb_id"] == "LT00004"
    assert "LiverTox evidence." in payload["knowledge_prompt"]
    assert "vMost-DILI-concern" in payload["knowledge_prompt"]
    assert "patient-specific causality score" in payload["knowledge_prompt"]
    provenance = [
        item
        for item in payload["extraction_metadata"]
        if item.get("knowledge_source") == "fda_dilirank_2"
    ]
    assert provenance == [
        {
            "knowledge_source": "fda_dilirank_2",
            "record_count": 1,
            "ltkb_ids": ["LT00004"],
        }
    ]

###############################################################################
def test_report_source_attribution_is_conditional_on_dilirank_provenance() -> None:
    without_dilirank = DrugClinicalAssessment(drug_name="Acetaminophen")
    with_dilirank = DrugClinicalAssessment(
        drug_name="Acetaminophen",
        extraction_metadata=[
            {
                "knowledge_source": "fda_dilirank_2",
                "record_count": 1,
                "ltkb_ids": ["LT00004"],
            }
        ],
    )

    assert ReportFinalizer.bibliography_source_label(without_dilirank) == "LiverTox"
    assert ReportFinalizer.bibliography_source_label(with_dilirank) == (
        "LiverTox; FDA DILIrank 2.0"
    )
