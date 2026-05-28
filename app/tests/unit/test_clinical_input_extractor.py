from __future__ import annotations

import asyncio

import pytest
from services.session.clinical_input_extractor import (
    ClinicalInputExtractionError,
    ClinicalInputExtractor,
)


def test_deterministic_success_does_not_call_fallback() -> None:
    text = "# Anamnesis\nA\n# Therapy\nT\n# Laboratory history\nL"

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> object:
            raise AssertionError("LLM should not be called")

    extractor = ClinicalInputExtractor(client=FakeClient())
    result = asyncio.run(extractor.extract(clinical_input=text))
    assert result.anamnesis == "A"
    assert result.drugs == "T"
    assert result.laboratory_analysis == "L"


def test_deterministic_failure_raises_without_fallback() -> None:
    text = "no explicit sections"
    called = {"value": False}

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> object:
            called["value"] = True
            return {"anamnesis": "no", "therapy": "sections", "lab_analysis": "here"}

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=text))
    assert called["value"] is False


def test_untitled_prose_is_rejected() -> None:
    text = "Anamnesis text. Therapy text. Lab text."
    extractor = ClinicalInputExtractor()
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=text))


def test_fallback_summarized_is_rejected() -> None:
    text = "Anamnesis text. Therapy text. Lab text."

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> object:
            return {
                "anamnesis": "summary",
                "therapy": "summary",
                "lab_analysis": "summary",
            }

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=text))


def test_fallback_missing_section_is_rejected() -> None:
    text = "Anamnesis text. Therapy text. Lab text."

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> object:
            return {
                "anamnesis": "Anamnesis text.",
                "therapy": "Therapy text.",
                "lab_analysis": "",
            }

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=text))
