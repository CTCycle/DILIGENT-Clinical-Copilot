from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest
from common.exceptions import ServiceValidationError
from domain.clinical.entities import ClinicalSessionRequest
from domain.clinical.extras import HepatoxPreparedInputs
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.session_service import ClinicalSessionService
from services.session.session_workflow import start_clinical_job_workflow


###############################################################################
def _build_service() -> ClinicalSessionService:
    return build_clinical_session_service(get_job_manager())


###############################################################################
def test_preprocess_unified_input_accepts_fragment_aggregated_sections() -> None:
    input_text = (
        "# Anamnesis\nA1\nA2\n\n# Current therapy\nD\n\n# Laboratory analysis\nL"
    )
    service = _build_service()
    request = ClinicalSessionRequest(clinical_input=input_text)
    preprocessed, returned_extraction = asyncio.run(
        service.preprocess_unified_input(request)
    )

    assert "A1" in (preprocessed.anamnesis or "")
    assert "A2" in (preprocessed.anamnesis or "")
    assert preprocessed.drugs == "D"
    assert preprocessed.laboratory_analysis == "L"
    assert returned_extraction is not None


###############################################################################
def test_preprocess_unified_input_rejects_invalid_sections() -> None:
    service = _build_service()
    request = ClinicalSessionRequest(clinical_input="raw input")

    with pytest.raises(
        ServiceValidationError, match="Clinical input sections are invalid"
    ):
        asyncio.run(service.preprocess_unified_input(request))


###############################################################################
def test_prepare_structured_clinical_input_returns_patient_payload_and_metadata(
    monkeypatch,
) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service.serializer, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        service.serializer, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    request = ClinicalSessionRequest(
        clinical_input=(
            "## Anamnesis\nHistory text\n\n"
            "## Therapy\nDrug 10 mg 1-0-0-0\n\n"
            "## Laboratory Analysis\nALT 120 U/L\n"
        ),
        visit_date=date(2025, 1, 15),
    )

    prepared = service.prepare_structured_clinical_input(request)

    assert (
        prepared["section_extraction"].metadata["parser"]
        == "deterministic_initial_text_sections_v2"
    )
    assert prepared["patient_payload"].anamnesis == "History text"
    assert prepared["patient_payload"].drugs == "Drug 10 mg 1-0-0-0"
    assert prepared["patient_payload"].laboratory_analysis == "ALT 120 U/L"


###############################################################################
def test_start_clinical_job_requires_active_cloud_key_before_extraction(
    monkeypatch,
) -> None:
    service = _build_service()

    ###############################################################################
    class FakeExtractor:

        # -------------------------------------------------------------------------
        async def extract(
            self, *, clinical_input: str
        ) -> ClinicalSectionExtractionResult:
            raise AssertionError("extractor should not run without a cloud key")

    ###############################################################################
    class FakeAccessKeyService:

        # -------------------------------------------------------------------------
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


###############################################################################
def test_resolve_runtime_timeout_does_not_apply_legacy_cloud_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.session_service.LLMRuntimeConfig.is_cloud_enabled",
        lambda: True,
    )

    resolved = ClinicalSessionService._resolve_runtime_timeout(base_timeout_s=7200.0)

    assert resolved == 7200.0


###############################################################################
def test_resolve_consultation_timeout_uses_runtime_configuration(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        runtime=SimpleNamespace(clinical_llm_timeout=5400.0),
    )
    for module in (
        "services.session.session_service",
        "services.session.consultation",
    ):
        monkeypatch.setattr(
            f"{module}.get_server_settings",
            lambda: fake_settings,
        )
    monkeypatch.setattr(
        "services.session.session_service.LLMRuntimeConfig.is_cloud_enabled",
        lambda: True,
    )

    resolved = ClinicalSessionService._resolve_consultation_timeout()

    assert resolved == 5400.0


###############################################################################
def test_run_revision_consultation_uses_revision_analysis_entrypoint(
    monkeypatch,
) -> None:
    service = _build_service()
    payload = SimpleNamespace(name="Revision Patient", visit_date=date(2025, 1, 15))
    analysis_drugs = SimpleNamespace(entries=[SimpleNamespace(name="Drug X")])
    prepared_inputs = HepatoxPreparedInputs(
        resolved_drugs={"drug-x": {"canonical_name": "Drug X"}},
        pattern_prompt="Pattern prompt",
        clinical_context="Revision context",
    )

    ###############################################################################
    class FakeConsultation:

        # -------------------------------------------------------------------------
        def __init__(self, drugs, *, patient_name=None):
            self.drugs = drugs
            self.patient_name = patient_name
            self.llm_model = "revision-model"
            self.pipeline_issues = []

        # -------------------------------------------------------------------------
        async def run_analysis(self, **kwargs):
            raise AssertionError(
                "Revision consultation should not call run_analysis"
            )

        # -------------------------------------------------------------------------
        async def run_revision_analysis(self, **kwargs):
            assert kwargs["prepared_inputs"].clinical_context == "Revision context"
            return {
                "final_report": "Revision synthesis report",
                "revision_consultation_metadata": {
                    "drug_analysis_entrypoint": "request_revision_drug_analysis",
                    "report_finalization_entrypoint": "finalize_revision_patient_report",
                    "conclusion_entrypoint": "generate_revision_conclusion",
                    "synthesis_mode": "revision_comparison_aware",
                },
            }

    monkeypatch.setattr(service, "hepatox_consultation_cls", FakeConsultation)

    clinical_session, final_report, payload_metadata = asyncio.run(
        service.run_revision_consultation(
            payload=payload,
            analysis_drugs=analysis_drugs,
            prepared_inputs=prepared_inputs,
            consultation_context="Revision context",
            consultation_context_metadata={
                "source_version_id": 11,
                "revision_version_id": 12,
                "pipeline_run_id": "pipe-123",
            },
            report_language="en",
            rag_query=None,
            rucam_bundle=None,
            issues=[],
            progress_callback=None,
            stop_check=None,
        )
    )

    assert clinical_session.llm_model == "revision-model"
    assert final_report == "Revision synthesis report"
    assert payload_metadata["analysis_entrypoint"] == "run_revision_analysis"
    assert payload_metadata["consultation_model"] == "revision-model"
    assert payload_metadata["drug_analysis_entrypoint"] == "request_revision_drug_analysis"
    assert (
        payload_metadata["report_finalization_entrypoint"]
        == "finalize_revision_patient_report"
    )
    assert payload_metadata["conclusion_entrypoint"] == "generate_revision_conclusion"
    assert payload_metadata["synthesis_mode"] == "revision_comparison_aware"
    assert payload_metadata["pipeline_run_id"] == "pipe-123"
