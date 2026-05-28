from __future__ import annotations

import asyncio
from datetime import date

import pytest
from common.exceptions import ServiceValidationError
from domain.clinical.entities import ClinicalSessionRequest
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.session_service import ClinicalSessionService
from services.session.session_workflow import start_clinical_job_workflow


def _build_service() -> ClinicalSessionService:
    return build_clinical_session_service(get_job_manager())


def test_preprocess_unified_input_accepts_fragment_aggregated_sections() -> None:
    input_text = (
        "# Anamnesis\nA1\nA2\n\n# Current therapy\nD\n\n# Laboratory analysis\nL"
    )
    service = _build_service()
    request = ClinicalSessionRequest(clinical_input=input_text)
    preprocessed, returned_extraction = asyncio.run(service.preprocess_unified_input(request))

    assert "A1" in (preprocessed.anamnesis or "")
    assert "A2" in (preprocessed.anamnesis or "")
    assert preprocessed.drugs == "D"
    assert preprocessed.laboratory_analysis == "L"
    assert returned_extraction is not None


def test_preprocess_unified_input_rejects_invalid_sections() -> None:
    service = _build_service()
    request = ClinicalSessionRequest(clinical_input="raw input")

    with pytest.raises(ServiceValidationError, match="Clinical input sections are invalid"):
        asyncio.run(service.preprocess_unified_input(request))


def test_prepare_structured_clinical_input_returns_patient_payload_and_metadata(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(service.serializer, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1))
    monkeypatch.setattr(service.serializer, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1))
    request = ClinicalSessionRequest(
        clinical_input=(
            "## Anamnesis\nHistory text\n\n"
            "## Therapy\nDrug 10 mg 1-0-0-0\n\n"
            "## Laboratory Analysis\nALT 120 U/L\n"
        ),
        visit_date=date(2025, 1, 15),
    )

    prepared = service.prepare_structured_clinical_input(request)

    assert prepared["section_extraction"].metadata["parser"] == "deterministic_initial_text_sections_v2"
    assert prepared["patient_payload"].anamnesis == "History text"
    assert prepared["patient_payload"].drugs == "Drug 10 mg 1-0-0-0"
    assert prepared["patient_payload"].laboratory_analysis == "ALT 120 U/L"


def test_start_clinical_job_requires_active_cloud_key_before_extraction(monkeypatch) -> None:
    service = _build_service()

    class FakeExtractor:
        async def extract(self, *, clinical_input: str) -> ClinicalSectionExtractionResult:
            raise AssertionError("extractor should not run without a cloud key")

    class FakeAccessKeyService:
        def list_access_keys(self, provider: str):
            assert provider == "gemini"
            return []

    service.clinical_input_extractor = FakeExtractor()  # type: ignore[assignment]
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.get_llm_provider",
        lambda: "gemini",
    )
    monkeypatch.setattr(
        "services.session.session_workflow.AccessKeyService",
        FakeAccessKeyService,
    )
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)

    request = ClinicalSessionRequest(clinical_input="raw input")
    with pytest.raises(
        ServiceValidationError,
        match="Configure an active Gemini access key before running cloud analysis.",
    ):
        start_clinical_job_workflow(service, request)
