from __future__ import annotations

import asyncio
from typing import Any

from domain.clinical import DrugEntry
from services.clinical.parser import DrugsParser


class FailingStructuredClient:
    async def llm_structured_call(self, **kwargs: Any):
        _ = kwargs
        raise AssertionError("LLM should not be called for deterministic therapy lines")


def test_extract_drugs_from_therapy_parses_schedule_route_and_dates() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Acetaminophen 500 mg 1 - 0 - 0 - 0 po started from 01/01/2024
    Ceftriaxone 1 g iv sospesa dal 03.01.2024
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 2

    first = parsed.entries[0]
    assert first.name == "Acetaminophen"
    assert first.dosage == "500 mg"
    assert first.route == "oral"
    assert first.administration_pattern == "1-0-0-0"
    assert first.daytime_administration == [1.0, 0.0, 0.0, 0.0]
    assert first.therapy_start_status is True
    assert first.therapy_start_date == "2024-01-01"
    assert first.temporal_classification == "temporal_known"
    assert first.source == "therapy"
    assert first.historical_flag is False

    second = parsed.entries[1]
    assert second.name == "Ceftriaxone"
    assert second.route == "iv"
    assert second.suspension_status is True
    assert second.suspension_date == "2024-01-03"
    assert second.temporal_classification == "temporal_known"


def test_extract_drugs_from_therapy_missing_schedule_remains_parseable() -> None:
    parser = DrugsParser(client=object())
    therapy_text = "Pantoprazole 40 mg"

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Pantoprazole"
    assert entry.dosage == "40 mg"
    assert entry.administration_pattern is None
    assert entry.daytime_administration == []
    assert entry.temporal_classification == "temporal_uncertain"
    assert entry.source == "therapy"
    assert entry.historical_flag is False


def test_extract_drugs_from_therapy_supports_decimal_schedule_padding() -> None:
    parser = DrugsParser(client=object())
    therapy_text = "Prednisone 25 mg 0,5-0-0"

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Prednisone"
    assert entry.administration_pattern == "0.5-0-0"
    assert entry.daytime_administration == [0.5, 0.0, 0.0, 0.0]
    assert entry.temporal_classification == "temporal_known"


def test_extract_drugs_from_therapy_detects_ongoing_vs_suspended() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Ursodeoxycholic acid 250 mg sospesa dal 10/02/2024
    Metformin 500 mg started on 11-02-2024
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 2

    suspended = parsed.entries[0]
    ongoing = parsed.entries[1]

    assert suspended.suspension_status is True
    assert suspended.suspension_date == "2024-02-10"
    assert suspended.temporal_classification == "temporal_known"

    assert ongoing.suspension_status is None
    assert ongoing.therapy_start_status is True
    assert ongoing.therapy_start_date == "2024-02-11"
    assert ongoing.temporal_classification == "temporal_known"


def test_extract_drugs_from_therapy_strips_temporal_tail_from_name() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Nivolumab EV, ultima somministrazione 12 giorni prima del picco enzimatico
    Ipilimumab EV, ultima somministrazione 12 giorni prima del picco enzimatico
    Trastuzumab deruxtecan EV (linea precedente, sospeso 6 settimane fa)
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Nivolumab",
        "Ipilimumab",
        "Trastuzumab deruxtecan",
    ]
    assert [entry.route for entry in parsed.entries] == ["iv", "iv", "iv"]


def test_extract_drugs_from_therapy_does_not_parse_iso_dates_as_schedule() -> None:
    parser = DrugsParser(client=object())
    therapy_text = (
        "Piperacillina/tazobactam 4.5 g EV q8h, iniziata 2026-02-10, sospesa 2026-02-16"
    )

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Piperacillina/tazobactam"
    assert entry.dosage == "4.5 g EV q8h"
    assert entry.route == "iv"
    assert entry.administration_pattern is None
    assert entry.daytime_administration == []
    assert entry.therapy_start_status is True
    assert entry.therapy_start_date == "2026-02-10"
    assert entry.suspension_status is True
    assert entry.suspension_date == "2026-02-16"


def test_extract_drugs_from_therapy_skips_non_assumed_drug_line() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Esomeprazolo 40 mg PO 1 volta/die, terapia cronica (>12 mesi)
    Farmaci non assunti: paracetamolo ad alto dosaggio, antibiotici recenti
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == ["Esomeprazolo"]


def test_extract_drugs_from_therapy_keeps_continuation_lines_with_drug_blocks() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    ■Amlodipin axapharm cpr 5 mg 0-0-1-0 per os
    ■Prednison 20 mg cpr [cpr] 2-0-0-0 per os
     Dal 15.01.2025 40 mg (inizio terapia il 6-7 gennaio, alla dose di 60 mg/die) - Peso della paziente
    51.60 kg
    ■Diovan 80 mg cpr [cpr] 0-0-0-0 per os
     se PAS>o= 100 mmHg
    ■Domperidon axapharm lingual cpr orodisp 10 mg 0-0-0-0 per os
     In riserva: se nausea, vomito
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Amlodipin axapharm",
        "Prednison",
        "Diovan",
        "Domperidon axapharm lingual cpr orodisp",
    ]


def test_extract_drugs_from_therapy_uses_rules_before_llm_for_structured_blocks() -> (
    None
):
    parser = DrugsParser(client=FailingStructuredClient())
    therapy_text = """
    ■Fortecortin 4 mg cpr
    [cpr]
    1-0-0-0
    15.03 - 20.03

    ■De-Ursil 150 mg caps
    [caps]
    1-0-1-0
    per os
    dal 21.03

    ■Pantozol 40 mg cpr
    [cpr]
    1-0-0-0
    per os
    dal 06.02
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Fortecortin",
        "De-Ursil",
        "Pantozol",
    ]


def test_extract_drugs_from_therapy_splits_reserve_drugs_without_bullets() -> None:
    parser = DrugsParser(client=FailingStructuredClient())
    therapy_text = """
    Imodium lingual 2 mg cpr orodisp
    [cpr]
    0-0-0-0
    per os
    In riserva:

    Dafalgan 1 g cpr
    [cpr]
    0-0-0-0
    In riserva:

    Rivotril 2,5mg/ml gtt orali 10 ml
    [mg]
    0-0-0-0
    In riserva:
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Imodium lingual",
        "Dafalgan",
        "Rivotril",
    ]


def test_extract_drugs_from_therapy_empty_input_is_safe() -> None:
    parser = DrugsParser(client=object())

    parsed = asyncio.run(parser.extract_drugs_from_therapy(""))

    assert parsed.entries == []


def test_normalize_entry_filters_non_drug_fragments() -> None:
    parser = DrugsParser(client=object())

    assert (
        parser.normalize_entry(
            DrugEntry(name="In riserva"),
            source="therapy",
            historical_flag=False,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="Paziente femmina"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="Dopo"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="il lunedi"),
            source="therapy",
            historical_flag=False,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="ulteriore ciclo (originariamente previsto il"),
            source="therapy",
            historical_flag=False,
        )
        is None
    )
    kept = parser.normalize_entry(
        DrugEntry(name="Pemetrexed"),
        source="therapy",
        historical_flag=False,
    )
    assert kept is not None
    assert kept.name == "Pemetrexed"


def test_post_process_llm_entry_splits_dosage_from_temporal_details() -> None:
    parser = DrugsParser(client=object())
    raw_line = (
        "Boswellia serrata estratto secco 1 cps BID, iniziata circa 6 settimane "
        "prima dell'ittero, sospesa alla comparsa sintomi"
    )
    entry = DrugEntry(
        name="Boswellia serrata estratto secco",
        dosage=(
            "1 cps BID, iniziata circa 6 settimane prima dell'ittero, "
            "sospesa alla comparsa sintomi"
        ),
    )

    parsed = parser.post_process_llm_entry(
        entry,
        raw_line=raw_line,
        source="therapy",
        historical_flag=False,
    )

    assert parsed is not None
    assert parsed.dosage == "1 cps BID"
    assert parsed.therapy_start_status is True
    assert parsed.therapy_start_date == "circa 6 settimane prima dell'ittero"
    assert parsed.suspension_status is True
    assert parsed.suspension_date == "alla comparsa sintomi"
    assert parsed.temporal_classification == "temporal_known"
