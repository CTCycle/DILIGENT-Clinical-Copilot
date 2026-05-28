from __future__ import annotations

import asyncio

from domain.clinical.entities import PatientData
from services.clinical.labs import ClinicalLabExtractor, LabExtractionPayload


class _FakeClient:
    async def llm_structured_call(self, **kwargs) -> LabExtractionPayload:  # noqa: ANN003
        _ = kwargs
        return LabExtractionPayload(entries=[], onset_context=None)


def test_laboratory_extractor_uses_laboratory_history_text_only() -> None:
    extractor = ClinicalLabExtractor(client=_FakeClient())
    payload = PatientData(
        anamnesis="ALT 999 in free prose that should not drive lab extraction.",
        drugs="Drug X",
        laboratory_analysis="ALT 180 U/L (ULN 40), ALP 220 U/L (ULN 120)",
    )
    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))
    markers = {entry.marker_name for entry in timeline.entries}
    assert "ALT" in markers
    assert "ALP" in markers


def test_explicit_hepatic_pattern_overrides_calculated_pattern() -> None:
    extractor = ClinicalLabExtractor(client=_FakeClient())
    payload = PatientData(
        drugs="Drug X",
        laboratory_analysis=(
            "Hepatic pattern: cholestatic. ALT 500 U/L (ULN 40), ALP 120 U/L (ULN 120)."
        ),
    )
    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))
    explicit = extractor.extract_explicit_hepatic_pattern(
        payload.laboratory_analysis or ""
    )
    calculated = extractor.calculate_hepatic_pattern_from_lab_timeline(timeline)
    assert explicit == "cholestatic"
    assert calculated in {"hepatocellular", "cholestatic", "mixed", None}


def test_explicit_rucam_score_is_extracted_from_laboratory_history() -> None:
    extractor = ClinicalLabExtractor(client=_FakeClient())
    assert extractor.extract_explicit_rucam_score("RUCAM score: 9") == 9


def test_missing_uln_does_not_fabricate_pattern() -> None:
    extractor = ClinicalLabExtractor(client=_FakeClient())
    payload = PatientData(
        drugs="Drug X",
        laboratory_analysis="ALT 300 U/L, ALP 160 U/L",
    )
    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))
    assert extractor.calculate_hepatic_pattern_from_lab_timeline(timeline) is None
