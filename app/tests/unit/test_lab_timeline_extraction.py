from __future__ import annotations

import asyncio
from typing import Any

from domain.clinical import (
    ClinicalLabEntry,
    LiverInjuryOnsetContext,
    PatientData,
)
from services.clinical.labs import (
    ClinicalLabExtractor,
    LabExtractionPayload,
)


class FakeLabClient:
    def __init__(self, responses: list[LabExtractionPayload]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.prompts: list[str] = []

    async def llm_structured_call(self, **kwargs: Any) -> LabExtractionPayload:
        self.call_count += 1
        self.prompts.append(str(kwargs.get("user_prompt", "")))
        if self.responses:
            return self.responses.pop(0)
        return LabExtractionPayload(entries=[], onset_context=None)


def test_extracts_dated_alt_alp_and_bilirubin() -> None:
    extractor = ClinicalLabExtractor(
        client=FakeLabClient(
            [
                LabExtractionPayload(
                    entries=[
                        ClinicalLabEntry(
                            marker_name="ALT",
                            value=320.0,
                            upper_limit_normal=40.0,
                            sample_date="2025-01-10",
                            source="anamnesis",
                        ),
                        ClinicalLabEntry(
                            marker_name="ALP",
                            value=240.0,
                            upper_limit_normal=120.0,
                            sample_date="10/01/2025",
                            source="anamnesis",
                        ),
                        ClinicalLabEntry(
                            marker_name="total bilirubin",
                            value=2.1,
                            upper_limit_normal=1.2,
                            sample_date="2025-01-10",
                            source="anamnesis",
                        ),
                    ],
                    onset_context=None,
                )
            ]
        )
    )
    payload = PatientData(
        laboratory_analysis="ALT, ALP and bilirubin elevated.",
        drugs="Drug A",
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    markers = [entry.marker_name for entry in timeline.entries]
    assert "ALT" in markers
    assert "ALP" in markers
    assert "TBIL" in markers


def test_uses_ast_when_alt_absent() -> None:
    extractor = ClinicalLabExtractor(
        client=FakeLabClient(
            [
                LabExtractionPayload(
                    entries=[
                        ClinicalLabEntry(
                            marker_name="AST",
                            value=200.0,
                            upper_limit_normal=40.0,
                            sample_date="2025-01-10",
                            source="anamnesis",
                        )
                    ],
                    onset_context=None,
                )
            ]
        )
    )
    payload = PatientData(laboratory_analysis="AST 200 U/L.", drugs="Drug A")

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    assert len(timeline.entries) == 1
    assert timeline.entries[0].marker_name == "AST"


def test_merges_manual_labs_with_extracted_entries() -> None:
    extractor = ClinicalLabExtractor(
        client=FakeLabClient([LabExtractionPayload(entries=[], onset_context=None)])
    )
    payload = PatientData(
        anamnesis="No explicit labs in anamnesis.",
        drugs="Drug A",
        laboratory_analysis="ALT 180 U/L (ULN 45), ALP 200 U/L (ULN 120)",
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    assert len(timeline.entries) == 2
    assert {entry.marker_name for entry in timeline.entries} == {"ALT", "ALP"}


def test_preserves_relative_timing_without_absolute_dates() -> None:
    extractor = ClinicalLabExtractor(
        client=FakeLabClient(
            [
                LabExtractionPayload(
                    entries=[
                        ClinicalLabEntry(
                            marker_name="ALT",
                            value=300.0,
                            upper_limit_normal=40.0,
                            relative_time="2 weeks after starting therapy",
                            source="anamnesis",
                        )
                    ],
                    onset_context=None,
                )
            ]
        )
    )
    payload = PatientData(
        laboratory_analysis="ALT peak two weeks after therapy start.",
        drugs="Drug A",
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    assert len(timeline.entries) == 1
    assert timeline.entries[0].sample_date is None
    assert timeline.entries[0].relative_time == "2 weeks after starting therapy"


def test_deduplicates_near_identical_entries() -> None:
    extractor = ClinicalLabExtractor(
        client=FakeLabClient(
            [
                LabExtractionPayload(
                    entries=[
                        ClinicalLabEntry(
                            marker_name="ALT",
                            value=320.0,
                            upper_limit_normal=40.0,
                            sample_date="2025-01-10",
                            source="anamnesis",
                        ),
                        ClinicalLabEntry(
                            marker_name="ALAT",
                            value=320.0,
                            upper_limit_normal=40.0,
                            sample_date="2025-01-10",
                            source="anamnesis",
                        ),
                    ],
                    onset_context=None,
                )
            ]
        )
    )
    payload = PatientData(
        laboratory_analysis="Duplicate ALT mentions.",
        drugs="Drug A",
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    assert len(timeline.entries) == 1


def test_extracts_onset_clue_context() -> None:
    onset = LiverInjuryOnsetContext(
        onset_date="2025-01-11",
        onset_basis="first_symptom",
        evidence="Jaundice started on 11 Jan 2025.",
    )
    extractor = ClinicalLabExtractor(
        client=FakeLabClient([LabExtractionPayload(entries=[], onset_context=onset)])
    )
    payload = PatientData(
        laboratory_analysis="Jaundice started on 11 Jan 2025.",
        drugs="Drug A",
    )

    _, onset_context = asyncio.run(extractor.extract_from_payload(payload))

    assert onset_context is not None
    assert onset_context.onset_date == "2025-01-11"
    assert onset_context.onset_basis == "first_symptom"


def test_lab_llm_receives_full_text_without_chunk_markers() -> None:
    client = FakeLabClient([LabExtractionPayload(entries=[], onset_context=None)])
    extractor = ClinicalLabExtractor(client=client)
    payload = PatientData(
        laboratory_analysis="\n".join([f"ALT {index} U/L" for index in range(1, 80)]),
        drugs="Drug A",
    )

    asyncio.run(extractor.extract_from_payload(payload))

    assert client.call_count >= 1
    assert all("full clinical laboratory text" in prompt for prompt in client.prompts)
    assert all("[Chunk" not in prompt for prompt in client.prompts)


def test_extracts_explicit_pattern_and_rucam_score() -> None:
    extractor = ClinicalLabExtractor(client=FakeLabClient([]))
    text = "Hepatic pattern: mixed. RUCAM score: 7."
    assert extractor.extract_explicit_hepatic_pattern(text) == "mixed"
    assert extractor.extract_explicit_rucam_score(text) == 7


def test_calculates_pattern_from_alt_alp_with_uln() -> None:
    extractor = ClinicalLabExtractor(client=FakeLabClient([]))
    timeline = PatientData(
        laboratory_analysis="ALT 300 U/L (ULN 40), ALP 120 U/L (ULN 120)",
        drugs="Drug A",
    )
    parsed, _ = asyncio.run(extractor.extract_from_payload(timeline))
    assert (
        extractor.calculate_hepatic_pattern_from_lab_timeline(parsed)
        == "hepatocellular"
    )
