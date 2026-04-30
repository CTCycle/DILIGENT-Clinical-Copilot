from __future__ import annotations

import asyncio

import pytest

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSessionRequest,
)
from services.session.clinical_input_extractor import ClinicalInputExtractor
from services.session.session_service import (
    ClinicalSessionService,
    disease_extractor,
    drugs_parser,
    lab_extractor,
    pattern_analyzer,
    payload_sanitization_service,
    rucam_estimator,
    serializer,
)


def _build_service() -> ClinicalSessionService:
    return ClinicalSessionService(
        drugs_parser=drugs_parser,
        disease_extractor=disease_extractor,
        lab_extractor=lab_extractor,
        pattern_analyzer=pattern_analyzer,
        rucam_estimator=rucam_estimator,
        serializer=serializer,
        payload_sanitizer=payload_sanitization_service,
    )


def test_section_extraction_model_accepts_plain_sections_with_confidence() -> None:
    source_text = "## Anamnesis\nPatient with jaundice.\n## Therapy\nDrug A"
    result = ClinicalSectionExtractionResult(
        source_text=source_text,
        anamnesis="Patient with jaundice.",
        drugs="Drug A",
        laboratory_analysis=None,
        confidence=0.82,
    )
    assert result.source_text == source_text
    assert result.confidence == 0.82


def test_section_extraction_model_rejects_blank_section_text() -> None:
    source_text = "## Anamnesis\nPatient with jaundice."
    with pytest.raises(ValueError):
        ClinicalSectionExtractionResult(
            source_text=source_text,
            anamnesis="  ",
            drugs=None,
            laboratory_analysis=None,
            confidence=0.4,
        )


def test_preprocess_unified_input_requires_clinical_input() -> None:
    service = _build_service()
    request = ClinicalSessionRequest(
        name="Test",
        visit_date="2024-01-15",
        clinical_input=None,
        use_rag=False,
    )
    with pytest.raises(Exception, match="Clinical input is required"):
        asyncio.run(service.preprocess_unified_input(request))


def test_preprocess_unified_input_maps_sections() -> None:
    service = _build_service()
    source_text = (
        "## Anamnesis\nPatient with jaundice.\n\n"
        "## Therapy\nDrug A 10 mg started on 2024-01-01.\n\n"
        "## Laboratory Analysis\nALT 120 U/L (ULN 40)"
    )

    class StubExtractor:
        async def extract(self, *, clinical_input: str, progress_callback=None):
            assert clinical_input == source_text
            return ClinicalSectionExtractionResult(
                source_text=source_text,
                anamnesis="Patient with jaundice.",
                drugs="Drug A 10 mg started on 2024-01-01.",
                laboratory_analysis="ALT 120 U/L (ULN 40)",
                confidence=0.91,
            )

    service.clinical_input_extractor = StubExtractor()  # type: ignore[assignment]
    request = ClinicalSessionRequest(
        name="Test",
        visit_date="2024-01-15",
        clinical_input=source_text,
        use_rag=False,
    )
    preprocessed, extraction = asyncio.run(service.preprocess_unified_input(request))
    assert extraction is not None
    assert preprocessed.anamnesis == "Patient with jaundice."
    assert preprocessed.drugs == "Drug A 10 mg started on 2024-01-01."
    assert preprocessed.laboratory_analysis == "ALT 120 U/L (ULN 40)"


def test_deterministic_extractor_parses_markdown_sections() -> None:
    extractor = ClinicalInputExtractor()
    clinical_input = (
        "## Anamnesis\nPaziente con ittero.\n\n"
        "## Therapy\n- Drug A 10 mg PO\n\n"
        "## Laboratory Analysis\nALT 120 U/L"
    )
    result = asyncio.run(extractor.extract(clinical_input=clinical_input))
    assert result.source_text == clinical_input
    assert result.anamnesis == "Paziente con ittero."
    assert result.drugs == "- Drug A 10 mg PO"
    assert result.laboratory_analysis == "ALT 120 U/L"
    assert result.confidence >= 0.8


def test_deterministic_extractor_parses_indexed_multilingual_sections() -> None:
    extractor = ClinicalInputExtractor()
    clinical_input = (
        "1) Anamnesi:\nPaziente con nausea e prurito.\n\n"
        "2) Farmaci in uso:\nPantoprazolo 40 mg PO/die\n\n"
        "3) Esami di laboratorio:\nALT 390 U/L; AST 301 U/L"
    )
    result = asyncio.run(extractor.extract(clinical_input=clinical_input))
    assert result.anamnesis is not None and "nausea" in result.anamnesis.lower()
    assert result.drugs is not None and "pantoprazolo" in result.drugs.lower()
    assert result.laboratory_analysis is not None and "alt 390" in result.laboratory_analysis.lower()


def test_source_text_compatibility_allows_minor_whitespace_drift_only() -> None:
    extractor = ClinicalInputExtractor()
    source = "A\nB\n"
    assert extractor._is_source_text_compatible(source, "A\nB")
    assert extractor._is_source_text_compatible(source, "A\r\nB\r\n")
    assert not extractor._is_source_text_compatible(
        source,
        "Completely different content",
    )
