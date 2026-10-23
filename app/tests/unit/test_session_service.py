# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
import pytest
from common.exceptions import ServiceValidationError
from domain.clinical.entities import ClinicalSectionExtractionResult, ClinicalSessionRequest
from domain.clinical.extras import HepatoxPreparedInputs
from services.runtime.jobs import JobErrorSanitizer, get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.session_service import ClinicalSessionService
from services.session.session_workflow import start_clinical_job_workflow
from typing import Any
from domain.clinical import DrugEntry, PatientData, PatientDrugs
from domain.clinical.extras import LabExtractionPayload
from services.clinical.job_progress import ClinicalJobCancelled
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.parser import DrugsParser
from services.runtime.jobs import JobManager
from services.session import extraction_pipeline, factory

def _build_service() -> ClinicalSessionService:
    return build_clinical_session_service(get_job_manager())

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

def test_preprocess_unified_input_rejects_invalid_sections() -> None:
    service = _build_service()
    request = ClinicalSessionRequest(clinical_input="raw input")

    with pytest.raises(
        ServiceValidationError, match="Clinical input sections are invalid"
    ):
        asyncio.run(service.preprocess_unified_input(request))

def test_prepare_structured_clinical_input_returns_patient_payload_and_metadata(
    monkeypatch,
) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service.knowledge_repository,
        "list_livertox_catalog",
        lambda **kwargs: ([{"id": 1}], 1),
    )
    monkeypatch.setattr(
        service.drug_catalog_repository,
        "list_rxnav_catalog",
        lambda **kwargs: ([{"id": 1}], 1),
    )
    request = ClinicalSessionRequest(
        clinical_input=(
            "## Anamnesis\nHistory text\n\n"
            "## Therapy\nDrug 10 mg 1-0-0-0\n\n"
            "## Laboratory Analysis\nALT 120 U/L\n"
        ),
        visit_date=date(2025, 1, 15),
    )

    parse_result = service.validate_assessment_prerequisites_without_llm(request)
    monkeypatch.setattr(
        service,
        "validate_assessment_prerequisites_without_llm",
        lambda _: (_ for _ in ()).throw(
            AssertionError("structured preparation must reuse the validated parse result")
        ),
    )
    prepared = service.prepare_structured_clinical_input(
        request,
        parse_result=parse_result,
    )

    assert (
        prepared["section_extraction"].metadata["parser"]
        == "deterministic_initial_text_sections_v2"
    )
    assert prepared["patient_payload"].anamnesis == "History text"
    assert prepared["patient_payload"].drugs == "Drug 10 mg 1-0-0-0"
    assert prepared["patient_payload"].laboratory_analysis == "ALT 120 U/L"

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

def test_resolve_runtime_timeout_does_not_apply_legacy_cloud_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.session_service.LLMRuntimeConfig.is_cloud_enabled",
        lambda: True,
    )

    resolved = ClinicalSessionService._resolve_runtime_timeout(base_timeout_s=7200.0)

    assert resolved == 7200.0

def test_resolve_runtime_timeout_does_not_apply_local_cap(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        runtime=SimpleNamespace(
            minimum_llm_timeout=5.0,
        ),
    )
    monkeypatch.setattr(
        "services.session.session_service.get_server_settings",
        lambda: fake_settings,
    )
    monkeypatch.setattr(
        "services.session.session_service.LLMRuntimeConfig.is_cloud_enabled",
        lambda: False,
    )

    resolved = ClinicalSessionService._resolve_runtime_timeout(
        base_timeout_s=3600.0,
        cloud_cap_s=300.0,
        local_cap_s=45.0,
    )

    assert resolved == 3600.0

def test_job_error_sanitizer_reports_local_model_oom_safely() -> None:
    error = RuntimeError(
        "LLM call failed: Ollama HTTP 500: llama-server reported out-of-memory "
        "during startup: cudaMalloc failed: out of memory"
    )

    message = JobErrorSanitizer.build_safe_job_error_message(error)

    assert message == JobErrorSanitizer.LOCAL_MODEL_MEMORY_MESSAGE

def test_extraction_failure_classification_reports_schema_error() -> None:
    code, message = ClinicalSessionService.classify_extraction_failure(
        RuntimeError("Structured parsing failed: validation error"),
        fallback_code="therapy_extraction_fallback",
        fallback_message="fallback",
    )

    assert code == "structured_extraction_schema_invalid"
    assert "required schema" in message

def test_resolve_consultation_timeout_uses_runtime_configuration(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        runtime=SimpleNamespace(
            clinical_llm_timeout=5400.0,
            cloud_llm_timeout_cap=30.0,
            local_llm_timeout_cap=45.0,
        ),
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

    assert resolved == 30.0

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
            raise AssertionError("Revision consultation should not call run_analysis")

        # -------------------------------------------------------------------------
        async def run_revision_analysis(self, **kwargs):
            assert kwargs["prepared_inputs"].clinical_context == "Revision context"
            return {
                "final_report": "Revision synthesis report",
                "revision_consultation_metadata": {
                    "drug_analysis_entrypoint": "request_revision_drug_analysis",
                    "report_finalization_entrypoint": "finalize_report",
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
    assert (
        payload_metadata["drug_analysis_entrypoint"] == "request_revision_drug_analysis"
    )
    assert payload_metadata["report_finalization_entrypoint"] == "finalize_report"
    assert payload_metadata["conclusion_entrypoint"] == "generate_revision_conclusion"
    assert payload_metadata["synthesis_mode"] == "revision_comparison_aware"
    assert payload_metadata["pipeline_run_id"] == "pipe-123"

ITALIAN_LAB_TEXT = """
Labor 10.01.2025: ALAT 345 U/L, ALP 1055 U/L, Amilasi P 94 U/L, Lipasi 172 U/L.
Labor 16.01.2025: ALAT 822 U/L, ASAT 344 U/L, ALP 898 U/L, Amilasi P 119 U/L, Lipasi 215 U/L.
Labor 13.02.2025: ALAT 591 U/L, ASAT 205 U/L, ALP 666 U/L, Amilasi P 89 U/L, Lipasi 145 U/L.
"""

ITALIAN_THERAPY_TEXT = """
■Amlodipin axapharm cpr 5 mg 0-0-1-0 per os
■Bilol cpr rivestite 5 mg 1-0-0-0 per os
■Clopidogrel Spirig HC cpr rivestite 75 mg 1-0-0-0 per os
■Aspirin Cardio 100 mg cpr [cpr] 1-0-0-0 per os
■Pantoprazol Helvepharm cpr rivestite 20 mg 1-0-0-0 per os
■Prednison 20 mg cpr [cpr] 2-0-0-0 per os
■Seresta 15 mg cpr [cpr] 0-0-0-0.5 per os
■Venlafaxin ER Sandoz Ret caps 75 mg 1-0-0-0 per os
■Forxiga 10 mg cpr [cpr] 1-0-0-0 per os
■Diovan 80 mg cpr [cpr] 0-0-0-0 per os
■Domperidon axapharm lingual cpr orodisp 10 mg 0-0-0-0 per os
■ Invanz sol iniet [g] 0-0-0-0 i.
"""

class FailingLabClient:

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> LabExtractionPayload:
        _ = kwargs
        raise RuntimeError("simulated lab extraction failure")

class SlowLabClient:

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> LabExtractionPayload:
        _ = kwargs
        await asyncio.sleep(1.0)
        return LabExtractionPayload(entries=[], onset_context=None)

def test_clinical_session_factory_uses_configured_parser_timeouts(
    monkeypatch: Any,
) -> None:
    settings = SimpleNamespace(
        runtime=SimpleNamespace(
            parser_llm_timeout=120.0,
            disease_llm_timeout=150.0,
        )
    )
    monkeypatch.setattr(factory, "get_server_settings", lambda: settings)

    service = factory.build_clinical_session_service(JobManager())

    assert service.drugs_parser.timeout_s == 120.0
    assert service.lab_extractor.timeout_s == 120.0
    assert service.disease_extractor.timeout_s == 150.0

def test_italian_therapy_text_parses_multiple_drugs_without_llm() -> None:
    parser = DrugsParser(client=object())

    parsed = asyncio.run(parser.extract_drugs_from_therapy(ITALIAN_THERAPY_TEXT))

    names = {entry.name for entry in parsed.entries}
    assert len(names) >= 10
    assert "Amlodipin axapharm" in names
    assert "Clopidogrel Spirig HC" in names
    assert "Aspirin Cardio" in names

def test_italian_laboratory_text_uses_deterministic_fallback() -> None:
    extractor = ClinicalLabExtractor(client=FailingLabClient())
    payload = PatientData(
        anamnesis="",
        drugs="Invanz sol iniet",
        laboratory_analysis=ITALIAN_LAB_TEXT,
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    observed = {
        (entry.marker_name, entry.sample_date, entry.value)
        for entry in timeline.entries
    }
    assert ("ALT", "2025-01-10", 345.0) in observed
    assert ("ALP", "2025-01-10", 1055.0) in observed
    assert ("AST", "2025-01-16", 344.0) in observed
    assert ("ALT", "2025-02-13", 591.0) in observed

def test_lab_llm_chunk_timeout_uses_deterministic_fallback(monkeypatch: Any) -> None:
    extractor = ClinicalLabExtractor(client=SlowLabClient(), timeout_s=3600.0)
    monkeypatch.setattr(extractor, "LOCAL_LLM_CHUNK_TIMEOUT_CAP_S", 0.01)
    payload = PatientData(
        anamnesis="",
        drugs="Invanz sol iniet",
        laboratory_analysis="Labor 10.01.2025: ALAT 345 U/L, ALP 1055 U/L.",
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    observed = {
        (entry.marker_name, entry.sample_date, entry.value)
        for entry in timeline.entries
    }
    assert ("ALT", "2025-01-10", 345.0) in observed
    assert ("ALP", "2025-01-10", 1055.0) in observed

def test_runtime_timeout_resolution_does_not_apply_six_second_parser_cap() -> None:
    timeout = ClinicalSessionService._resolve_runtime_timeout(base_timeout_s=120.0)

    assert timeout > 6.0

def test_stage_stop_check_cancels_inflight_extraction_task() -> None:
    cancelled = False
    stop_calls = 0

    async def slow_stage() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled = True
            raise

    def stop_check() -> None:
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls >= 2:
            raise ClinicalJobCancelled("stop requested")

    with pytest.raises(ClinicalJobCancelled):
        asyncio.run(
            extraction_pipeline._await_with_stop_check(
                slow_stage(),
                stop_check=stop_check,
            )
        )

    assert cancelled is True

def test_livertox_timeout_does_not_claim_knowledge_base_is_unavailable(
    monkeypatch: Any,
) -> None:

    ###############################################################################
    class Owner:
        drugs_parser = SimpleNamespace(client=None, model=None, temperature=0.0)
        input_preparator = SimpleNamespace()

        # -------------------------------------------------------------------------
        @staticmethod
        def emit_progress(*args: Any, **kwargs: Any) -> None:
            _ = args, kwargs

        # -------------------------------------------------------------------------
        @staticmethod
        def build_stage_progress_callback(*args: Any, **kwargs: Any):
            _ = args, kwargs
            return lambda _value: None

        # -------------------------------------------------------------------------
        @staticmethod
        def run_stop_check(stop_check: Any) -> None:
            _ = stop_check

        # -------------------------------------------------------------------------
        @staticmethod
        def append_warning_issue(
            issues: list[Any], *, code: str, message: str, field: str | None = None
        ) -> None:
            issues.append(SimpleNamespace(code=code, message=message, field=field))

        # -------------------------------------------------------------------------
        @staticmethod
        def append_knowledge_base_unavailable_issue(issues: list[Any]) -> None:
            issues.append(SimpleNamespace(code="knowledge_base_unavailable"))

    async def slow_prepare_inputs(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        await asyncio.sleep(0.05)

    Owner.input_preparator.prepare_inputs = slow_prepare_inputs
    monkeypatch.setattr(
        extraction_pipeline,
        "get_server_settings",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(
                livertox_llm_timeout=0.01,
                cloud_llm_timeout_cap=0.01,
                local_llm_timeout_cap=0.01,
                minimum_llm_timeout=0.01,
            )
        ),
    )
    owner = Owner()
    issues: list[Any] = []

    prepared = asyncio.run(
        extraction_pipeline.ClinicalSessionExtractionPipelineMixin.run_livertox_lookup(
            owner,
            all_detected_drugs=PatientDrugs(entries=[DrugEntry(name="Pantozol")]),
            structured_context="",
            pattern_score=None,
            issues=issues,
            progress_callback=None,
            stop_check=None,
            use_rag=False,
        )
    )

    assert prepared is None
    assert [issue.code for issue in issues] == ["livertox_lookup_timeout"]
