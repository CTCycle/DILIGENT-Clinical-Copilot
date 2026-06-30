from __future__ import annotations

from typing import Any

from repositories.schemas.models import Base, ClinicalSessionLab
from repositories.serialization.data import DataSerializer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


def build_serializer() -> tuple[Any, Any]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    return serializer, engine


def test_save_clinical_session_persists_each_lab_observation() -> None:
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
                        "unit": "U/L",
                        "upper_limit_normal": "40",
                        "source": "laboratory_analysis",
                    },
                    {
                        "marker_name": "ALT",
                        "sample_date": "2025-01-03",
                        "value": "150",
                        "unit": "U/L",
                        "upper_limit_normal": "40",
                        "source": "laboratory_analysis",
                    },
                ],
            },
        }
    )

    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        labs = (
            db_session.execute(
                select(ClinicalSessionLab).order_by(
                    ClinicalSessionLab.observation_index
                )
            )
            .scalars()
            .all()
        )

    assert len(labs) == 2
    assert [row.lab_code for row in labs] == ["alt", "alt"]
    assert [row.sample_date_raw for row in labs] == ["2025-01-01", "2025-01-03"]
    assert [row.value_raw for row in labs] == ["120", "150"]
    assert [row.unit_raw for row in labs] == ["U/L", "U/L"]
    assert [row.upper_limit_raw for row in labs] == ["40", "40"]
    assert {row.source for row in labs} == {"laboratory_analysis"}
