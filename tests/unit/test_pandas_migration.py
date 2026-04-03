from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.services.clinical.livertox import LiverToxData
from DILIGENT.server.repositories.serialization.data import DataSerializer
from DILIGENT.server.repositories.schemas.models import (
    Base,
    ClinicalSession,
    ClinicalSessionLab,
    ClinicalSessionResult,
)
from DILIGENT.server.services.updater.livertox import LiverToxUpdater


###############################################################################
class QueryStub:
    def __init__(self, engine: Any) -> None:
        self.database = SimpleNamespace(backend=SimpleNamespace(engine=engine))


###############################################################################
class LookupStub:
    # -------------------------------------------------------------------------
    def normalize_name(self, value: str) -> str:
        return value.strip().lower()

    # -------------------------------------------------------------------------
    def parse_synonyms(self, value: Any) -> dict[str, str]:
        return {}

    # -------------------------------------------------------------------------
    def collect_tokens(self, name: str, synonyms: list[str]) -> set[str]:
        return set()

    # -------------------------------------------------------------------------
    def iter_alias_variants(self, value: str):
        yield value


# -----------------------------------------------------------------------------
def build_serializer() -> tuple[Any, Any]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(queries=QueryStub(engine))
    return serializer, engine


# -----------------------------------------------------------------------------
def test_normalize_date_uses_explicit_units_for_numeric_timestamps() -> None:
    serializer, _ = build_serializer()
    assert serializer.normalize_date("1735689600") == "2025-01-01"
    assert serializer.normalize_date("1735689600000") == "2025-01-01"
    assert serializer.normalize_date("20250101") == "2025-01-01"


# -----------------------------------------------------------------------------
def test_save_clinical_session_preserves_row_append_order() -> None:
    serializer, engine = build_serializer()
    serializer.save_clinical_session(
        {
            "patient_name": "existing",
            "session_timestamp": "2025-01-01T00:00:00",
        }
    )
    serializer.save_clinical_session(
        {
            "patient_name": "incoming",
            "session_timestamp": "2025-01-02T00:00:00",
        }
    )

    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        rows = (
            db_session.execute(select(ClinicalSession).order_by(ClinicalSession.id))
            .scalars()
            .all()
        )

    assert len(rows) == 2
    assert rows[0].patient_name == "existing"
    assert rows[1].patient_name == "incoming"


# -----------------------------------------------------------------------------
def test_save_clinical_session_persists_raw_result_payload() -> None:
    serializer, engine = build_serializer()
    raw_payload = {
        "report": "# Clinical Visit Summary",
        "issues": [{"severity": "warning", "message": "Example issue"}],
        "matched_drugs": [{"raw_drug_name": "acetaminophen"}],
    }

    serializer.save_clinical_session(
        {
            "patient_name": "payload-patient",
            "session_timestamp": "2025-01-03T00:00:00",
            "session_result_payload": raw_payload,
        }
    )

    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        session_row = (
            db_session.execute(select(ClinicalSession).order_by(ClinicalSession.id.desc()))
            .scalars()
            .first()
        )
        result_row = (
            db_session.execute(
                select(ClinicalSessionResult).order_by(ClinicalSessionResult.id.desc())
            )
            .scalars()
            .first()
        )

    assert session_row is not None
    assert result_row is not None
    assert result_row.session_id == session_row.id
    assert json.loads(result_row.payload_json) == raw_payload


# -----------------------------------------------------------------------------
def test_save_clinical_session_deduplicates_labs_per_marker() -> None:
    serializer, engine = build_serializer()
    serializer.save_clinical_session(
        {
            "patient_name": "lab-patient",
            "session_timestamp": "2025-01-04T00:00:00",
            "session_result_payload": {
                "lab_timeline": [
                    {
                        "marker_name": "ALT",
                        "sample_date": "2025-01-01",
                        "value": "120",
                        "upper_limit_normal": "40",
                    },
                    {
                        "marker_name": "ALT",
                        "sample_date": "2025-01-03",
                        "value": "150",
                        "upper_limit_normal": "40",
                    },
                    {
                        "marker_name": "AST",
                        "sample_date": "2025-01-02",
                        "value": "90",
                        "upper_limit_normal": "35",
                    },
                ],
            },
        }
    )

    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        labs = (
            db_session.execute(
                select(ClinicalSessionLab).order_by(ClinicalSessionLab.lab_code)
            )
            .scalars()
            .all()
        )

    assert len(labs) == 2
    assert [row.lab_code for row in labs] == ["alt", "ast"]


# -----------------------------------------------------------------------------
def test_livertox_data_keeps_internal_dataframe_copies_isolated() -> None:
    livertox_df = pd.DataFrame()
    master_list_df = pd.DataFrame(
        {
            "drug_name": pd.Series(["Acetaminophen"], dtype="string"),
            "brand_name": pd.Series(["Tylenol"], dtype="string"),
            "ingredient": pd.Series(["Acetaminophen"], dtype="string"),
        }
    )
    catalog_df = pd.DataFrame(
        {"drug_name": pd.Series(["Acetaminophen"], dtype="string")}
    )

    data = LiverToxData(
        lookup=LookupStub(),
        livertox_df=livertox_df,
        master_list_df=master_list_df,
        drugs_catalog_df=catalog_df,
        record_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert isinstance(data.master_list_df, pd.DataFrame)
    data.master_list_df.loc[0, "brand_name"] = "Updated"
    assert master_list_df.loc[0, "brand_name"] == "Tylenol"

    assert isinstance(data.drugs_catalog_df, pd.DataFrame)
    data.drugs_catalog_df.loc[0, "drug_name"] = "Updated"
    assert catalog_df.loc[0, "drug_name"] == "Acetaminophen"


# -----------------------------------------------------------------------------
def test_master_list_sanitization_handles_string_dtype_inputs() -> None:
    updater = LiverToxUpdater(sources_path=".", redownload=False)
    raw = pd.DataFrame(
        {
            "Count": pd.Series(["2", "0"], dtype="string"),
            "Ingredient": pd.Series(["Acetaminophen", "ingredient"], dtype="string"),
            "Brand Name": pd.Series(["Tylenol", "brand name"], dtype="string"),
            "Likelihood Score": pd.Series(["A", "B"], dtype="string"),
            "Chapter Title": pd.Series(["Acetaminophen", "Header"], dtype="string"),
            "Last Update": pd.Series(["2025-01-03", "bad-date"], dtype="string"),
            "Year Approved": pd.Series(["1955", "1900"], dtype="string"),
            "Type of Agent": pd.Series(["Drug", "Drug"], dtype="string"),
            "In LiverTox": pd.Series(["Yes", "Yes"], dtype="string"),
            "Primary Classification": pd.Series(["ClassA", "ClassB"], dtype="string"),
            "Secondary Classification": pd.Series(["ClassA2", "ClassB2"], dtype="string"),
        }
    )

    sanitized = updater.sanitize_livertox_master_list(raw)

    assert sanitized is not None
    assert len(sanitized.index) == 1
    assert sanitized.iloc[0]["chapter_title"] == "Acetaminophen"
    assert sanitized.iloc[0]["reference_count"] == 2
    assert pd.api.types.is_datetime64_any_dtype(sanitized["last_update"])
