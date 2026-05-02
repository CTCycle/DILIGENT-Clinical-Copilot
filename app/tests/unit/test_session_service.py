from __future__ import annotations

import asyncio

import pytest

from common.exceptions import ServiceValidationError
from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionFragment,
    ClinicalSessionRequest,
)
from services.session.clinical_input_extractor import ClinicalInputExtractionError
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


def test_preprocess_unified_input_accepts_fragment_aggregated_sections() -> None:
    input_text = (
        "# Anamnesis\nA1\n\n# Current therapy\nD\n\n# Anamnesis\nA2\n\n# Laboratory analysis\nL"
    )
    extraction = ClinicalSectionExtractionResult(
        source_text=input_text,
        anamnesis="A1\n\n\n\nA2\n\n",
        drugs="D\n\n",
        laboratory_analysis="L",
        fragments=[
            ClinicalSectionFragment(section="anamnesis", start=12, end=16, text="A1\n\n"),
            ClinicalSectionFragment(section="drugs", start=35, end=38, text="D\n\n"),
            ClinicalSectionFragment(section="anamnesis", start=52, end=56, text="A2\n\n"),
            ClinicalSectionFragment(section="laboratory_analysis", start=78, end=79, text="L"),
        ],
        confidence=0.94,
    )

    class FakeExtractor:
        async def extract(self, *, clinical_input: str) -> ClinicalSectionExtractionResult:
            assert clinical_input == input_text
            return extraction

    service = _build_service()
    service.clinical_input_extractor = FakeExtractor()  # type: ignore[assignment]
    request = ClinicalSessionRequest(clinical_input=input_text)
    preprocessed, returned_extraction = asyncio.run(service.preprocess_unified_input(request))

    assert preprocessed.anamnesis == extraction.anamnesis
    assert preprocessed.drugs == extraction.drugs
    assert preprocessed.laboratory_analysis == extraction.laboratory_analysis
    assert returned_extraction == extraction


def test_preprocess_unified_input_converts_extraction_error_to_service_validation_error() -> None:
    input_text = "raw input"

    class FakeExtractor:
        async def extract(self, *, clinical_input: str) -> ClinicalSectionExtractionResult:
            assert clinical_input == input_text
            raise ClinicalInputExtractionError("boom")

    service = _build_service()
    service.clinical_input_extractor = FakeExtractor()  # type: ignore[assignment]
    request = ClinicalSessionRequest(clinical_input=input_text)

    with pytest.raises(ServiceValidationError, match="boom"):
        asyncio.run(service.preprocess_unified_input(request))
