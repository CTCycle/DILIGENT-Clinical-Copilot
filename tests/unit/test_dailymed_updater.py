from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from DILIGENT.server.services.updater.dailymed import DailyMedLabelUpdater


###############################################################################
class SerializerStub:
    def __init__(self) -> None:
        self.saved_records: list[dict[str, Any]] = []

    def stream_drugs_catalog(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"drug_id": 1, "canonical_name": "Acetaminophen", "rxcui": "161"},
                {"drug_id": 2, "canonical_name": "Ibuprofen", "rxcui": "5640"},
            ]
        )

    def replace_drug_label_documents(self, records: list[dict[str, Any]]) -> None:
        self.saved_records = records

    @staticmethod
    def to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def normalize_date(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def to_sortable_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).casefold()


###############################################################################
class UpdaterStub(DailyMedLabelUpdater):
    def __init__(self) -> None:
        super().__init__(serializer=SerializerStub(), section_max_length=80, max_sections_per_drug=2)

    async def download_label_xml(self, set_id: str) -> str:
        return f"""
        <document xmlns="urn:hl7-org:v3">
          <section>
            <title>Boxed Warning</title>
            <paragraph>Severe hepatotoxicity warning applies.</paragraph>
          </section>
          <section>
            <title>Adverse Reactions</title>
            <paragraph>Nausea and headache.</paragraph>
            <paragraph>Rare hepatitis and elevated ALT observed.</paragraph>
          </section>
          <section>
            <title>Contraindications</title>
            <paragraph>Contraindicated in severe hepatic impairment.</paragraph>
          </section>
        </document>
        """

    def load_rxnorm_to_setid_mapping(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "161": [
                {"set_id": "A", "effective_date": "2024-01-01", "spl_version": 1, "status": "active"},
                {"set_id": "B", "effective_date": "2025-01-01", "spl_version": 1, "status": "active"},
                {"set_id": "C", "effective_date": "2025-01-01", "spl_version": 2, "status": "active"},
            ]
        }


# -----------------------------------------------------------------------------
def test_rxnorm_to_label_selection_prefers_newest_then_version_then_set_id() -> None:
    updater = UpdaterStub()
    mapping = updater.load_rxnorm_to_setid_mapping()

    selected = updater.select_best_label_for_drug("161", mapping)

    assert selected is not None
    assert selected["set_id"] == "C"
    assert selected["spl_version"] == 2


# -----------------------------------------------------------------------------
def test_spl_section_extraction_and_hepatic_filtering() -> None:
    updater = UpdaterStub()

    xml_text = asyncio.run(updater.download_label_xml("C"))
    sections = updater.extract_relevant_sections(xml_text)

    assert any(item["section_key"] == "boxed_warning" for item in sections)
    assert any(item["section_key"] == "adverse_reactions" for item in sections)
    assert all(item["section_key"] in {"boxed_warning", "adverse_reactions", "contraindications"} for item in sections)
    assert any("hepatitis" in item["text"].lower() or "hepatic" in item["text"].lower() for item in sections)


# -----------------------------------------------------------------------------
def test_truncation_and_max_sections_cap() -> None:
    updater = UpdaterStub()
    long_xml = """
    <document xmlns="urn:hl7-org:v3">
      <section><title>Boxed Warning</title><paragraph>""" + ("hepatic risk " * 100) + """</paragraph></section>
      <section><title>Warnings and Precautions</title><paragraph>hepatic warning one</paragraph></section>
      <section><title>Adverse Reactions</title><paragraph>hepatic warning two</paragraph></section>
      <section><title>Contraindications</title><paragraph>hepatic warning three</paragraph></section>
    </document>
    """

    sections = updater.extract_relevant_sections(long_xml)

    assert len(sections) == 2
    assert all(len(item["text"]) <= updater.section_max_length for item in sections)


# -----------------------------------------------------------------------------
def test_update_labels_persists_documents() -> None:
    updater = UpdaterStub()

    summary = updater.update_labels()

    assert summary["documents_persisted"] >= 1
    assert updater.serializer.saved_records
