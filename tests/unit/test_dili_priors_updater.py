from __future__ import annotations

from typing import Any

import pytest
import pandas as pd

from DILIGENT.server.services.updater.dili_priors import DiliPriorUpdater


###############################################################################
class SerializerStub:
    def __init__(self) -> None:
        self.saved_frame: pd.DataFrame | None = None

    def get_drugs_catalog(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "drug_id": 1,
                    "name": "Acetaminophen",
                    "raw_name": "Acetaminophen",
                    "brand_names": "Tylenol",
                    "synonyms": '["paracetamol"]',
                },
                {
                    "drug_id": 2,
                    "name": "Ibuprofen",
                    "raw_name": "Ibuprofen",
                    "brand_names": "Advil",
                    "synonyms": '["ibuprofen"]',
                },
                {
                    "drug_id": 3,
                    "name": "Aspirin",
                    "raw_name": "Aspirin",
                    "brand_names": "Bayer",
                    "synonyms": '["asa"]',
                },
                {
                    "drug_id": 4,
                    "name": "Aspirin",
                    "raw_name": "Aspirin",
                    "brand_names": "Bufferin",
                    "synonyms": '["asa"]',
                },
            ]
        )

    def stream_drugs_catalog(self) -> Any:
        raise AssertionError("match_rows_to_drugs should use get_drugs_catalog()")

    def save_dili_annotations(self, frame: pd.DataFrame) -> None:
        self.saved_frame = frame.copy()

    @staticmethod
    def session_factory() -> Any:
        class DummySession:
            def close(self) -> None:
                return None

        return DummySession()

    @staticmethod
    def resolve_drug_id(
        db_session: Any,
        *,
        matched_drug_name: str | None,
        rxcui: str | None,
        nbk_id: str | None,
    ) -> int | None:
        _ = db_session, rxcui, nbk_id
        if matched_drug_name is None:
            return None
        normalized = matched_drug_name.strip().casefold()
        return {"acetaminophen": 1, "advil": 2, "ibuprofen": 2}.get(normalized)

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
def test_parse_dilist_fda_columns() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = pd.DataFrame(
        [
            {
                "DILIST_ID": "1",
                "Compound Name": "Acetaminophen",
                "DILIst Classification": "1",
                "Routes of Administration": "Oral",
            }
        ]
    )

    parsed = updater.parse_dilist(frame)

    assert len(parsed.index) == 1
    row = parsed.to_dict(orient="records")[0]
    assert row["source_record_id"] == "1"
    assert row["source_name_norm"] == "acetaminophen"
    assert row["classification"] == "1"
    assert row["routes"] == "Oral"


# -----------------------------------------------------------------------------


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
                "source_name": "Unlisted compound",
                "source_name_norm": "unlisted compound",
            }
        ]
    )

    matched, summary = updater.match_rows_to_drugs(frame, source_dataset="dilirank")

    assert pd.isna(matched.iloc[0]["drug_id"])
    assert summary["unmatched_rows"] == 1
    assert summary["linked_rows"] == 0


# -----------------------------------------------------------------------------
def test_download_and_parse_fda_html_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from DILIGENT.server.services.updater import dili_priors as module

    html = """
    <html>
      <body>
        <table>
          <tbody>
            <tr>
              <td><strong>LTKBID</strong></td>
              <td><strong>CompoundName</strong></td>
              <td><strong>SeverityClass</strong></td>
              <td><strong>LabelSection</strong></td>
              <td><strong>vDILI-Concern</strong></td>
              <td><strong>Comment</strong></td>
            </tr>
            <tr>
              <td>LT00001</td>
              <td>Acetaminophen</td>
              <td>8</td>
              <td>Box warning</td>
              <td>Most-DILI-concern</td>
              <td>Example</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    class FakeResponse:
        headers = {"content-type": "text/html; charset=utf-8"}
        content = html.encode("utf-8")
        text = html

        @staticmethod
        def raise_for_status() -> None:
            return None

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args, kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = exc_type, exc, tb

        def get(self, url: str) -> FakeResponse:
            _ = url
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "Client", FakeClient)

    updater = DiliPriorUpdater(serializer=SerializerStub())
    frame = updater.download_dilirank()
    parsed = updater.parse_dilirank(frame)

    assert len(frame.index) == 1
    assert frame.iloc[0]["CompoundName"] == "Acetaminophen"
    assert len(parsed.index) == 1
    row = parsed.to_dict(orient="records")[0]
    assert row["source_record_id"] == "LT00001"
    assert row["classification"] == "Most-DILI-concern"
    assert row["severity_class"] == "8"
    assert row["label_section"] == "Box warning"
