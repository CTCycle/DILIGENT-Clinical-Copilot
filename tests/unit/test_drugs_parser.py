from __future__ import annotations

import asyncio

from DILIGENT.server.services.clinical.parser import DrugsParser


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
