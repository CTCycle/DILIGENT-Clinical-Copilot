from __future__ import annotations

from typing import Any

import pandas as pd

from DILIGENT.server.services.updater.dili_priors import DiliPriorUpdater


###############################################################################
class SerializerStub:
    def __init__(self) -> None:
        self.saved_frame: pd.DataFrame | None = None

    def stream_drugs_catalog(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "drug_id": 1,
                    "canonical_name_norm": "acetaminophen",
                    "aliases": [{"alias_norm": "paracetamol"}],
                },
                {
                    "drug_id": 2,
                    "canonical_name_norm": "ibuprofen",
                    "aliases": [{"alias_norm": "advil"}],
                },
                {
                    "drug_id": 3,
                    "canonical_name_norm": "aspirin",
                    "aliases": [{"alias_norm": "asa"}],
                },
                {
                    "drug_id": 4,
                    "canonical_name_norm": "aspirin",
                    "aliases": [{"alias_norm": "asa"}],
                },
            ]
        )

    def save_dili_annotations(self, frame: pd.DataFrame) -> None:
        self.saved_frame = frame.copy()

    @staticmethod
    def to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


# -----------------------------------------------------------------------------
def test_parse_dilirank() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = pd.DataFrame(
        [
            {
                "drug_name": "Acetaminophen",
                "dilirank": "Most-DILI-Concern",
                "severity": "high",
            }
        ]
    )

    parsed = updater.parse_dilirank(frame)

    assert len(parsed.index) == 1
    row = parsed.to_dict(orient="records")[0]
    assert row["source_dataset"] == "dilirank"
    assert row["source_name_norm"] == "acetaminophen"
    assert row["classification"] == "Most-DILI-Concern"


# -----------------------------------------------------------------------------
def test_parse_dilist() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = pd.DataFrame(
        [
            {
                "drug": "Ibuprofen",
                "classification": "Known",
                "concern_class": "moderate",
                "route": "oral",
            }
        ]
    )

    parsed = updater.parse_dilist(frame)

    assert len(parsed.index) == 1
    row = parsed.to_dict(orient="records")[0]
    assert row["source_dataset"] == "dilist"
    assert row["source_name_norm"] == "ibuprofen"
    assert row["concern_class"] == "moderate"
    assert row["routes"] == "oral"


# -----------------------------------------------------------------------------
def test_exact_canonical_match() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = pd.DataFrame(
        [
            {
                "source_dataset": "dilirank",
                "source_record_id": "x1",
                "source_name": "Acetaminophen",
                "source_name_norm": "acetaminophen",
            }
        ]
    )

    matched, summary = updater.match_rows_to_drugs(frame, source_dataset="dilirank")

    assert matched.iloc[0]["drug_id"] == 1
    assert summary["linked_rows"] == 1
    assert summary["ambiguous_rows"] == 0


# -----------------------------------------------------------------------------
def test_exact_alias_match() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = pd.DataFrame(
        [
            {
                "source_dataset": "dilirank",
                "source_record_id": "x2",
                "source_name": "Advil",
                "source_name_norm": "advil",
            }
        ]
    )

    matched, summary = updater.match_rows_to_drugs(frame, source_dataset="dilirank")

    assert matched.iloc[0]["drug_id"] == 2
    assert summary["linked_rows"] == 1


# -----------------------------------------------------------------------------
def test_ambiguous_match_is_stored_as_unlinked() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = pd.DataFrame(
        [
            {
                "source_dataset": "dilirank",
                "source_record_id": "x3",
                "source_name": "ASA",
                "source_name_norm": "asa",
            }
        ]
    )

    matched, summary = updater.match_rows_to_drugs(frame, source_dataset="dilirank")

    assert pd.isna(matched.iloc[0]["drug_id"])
    assert summary["ambiguous_rows"] == 1
    assert summary["linked_rows"] == 0
