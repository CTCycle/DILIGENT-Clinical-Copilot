from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from common.exceptions import ServiceValidationError
from domain.clinical.entities import ClinicalSessionRequest
from domain.clinical.robustness import RagReadiness
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.preflight import validate_clinical_input_preflight
from services.session.session_workflow import start_clinical_job_workflow


###############################################################################
def _build_service():
    return build_clinical_session_service(get_job_manager())

###############################################################################
def _valid_input() -> str:
    return (
        "ANAMNESIS\nhistory\n"
        "DRUGS\nacetaminophen 500 mg\n"
        "LABORATORY ANALYSIS\nALT 240 U/L\n"
    )

###############################################################################
def test_missing_visit_date_blocks_job_start_before_preprocess(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service,
        "validate_clinical_input",
        lambda req: type("P", (), {"ready": True, "blocking_issues": []})(),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        service,
        "preprocess_unified_input",
        lambda request_payload: (_ for _ in ()).throw(
            AssertionError("preprocess should not run")
        ),
    )
    request = ClinicalSessionRequest(clinical_input=_valid_input(), visit_date=None)
    with pytest.raises(ServiceValidationError, match="Visit date is required"):
        start_clinical_job_workflow(service, request)

###############################################################################
def test_empty_livertox_catalog_blocks_job_start_before_preprocess(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service,
        "validate_clinical_input",
        lambda req: type("P", (), {"ready": True, "blocking_issues": []})(),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        service,
        "preprocess_unified_input",
        lambda request_payload: (_ for _ in ()).throw(
            AssertionError("preprocess should not run")
        ),
    )
    monkeypatch.setattr(
        service.session_repository.knowledge_repository, "list_livertox_catalog", lambda **kwargs: ([], 0)
    )
    request = ClinicalSessionRequest(
        clinical_input=_valid_input(),
        visit_date=date(2025, 1, 15),
    )
    with pytest.raises(ServiceValidationError, match="LiverTox catalog is empty"):
        start_clinical_job_workflow(service, request)

###############################################################################
def test_empty_rxnav_catalog_blocks_job_start_before_preprocess(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service,
        "validate_clinical_input",
        lambda req: type("P", (), {"ready": True, "blocking_issues": []})(),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        service.knowledge_repository, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        service.session_repository.drug_catalog_repository, "list_rxnav_catalog", lambda **kwargs: ([], 0)
    )
    request = ClinicalSessionRequest(
        clinical_input=_valid_input(),
        visit_date=date(2025, 1, 15),
    )
    with pytest.raises(ServiceValidationError, match="RxNav catalog is empty"):
        start_clinical_job_workflow(service, request)

###############################################################################
def test_malformed_sections_block_job_start(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service,
        "validate_clinical_input",
        lambda req: type("P", (), {"ready": True, "blocking_issues": []})(),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        service.session_repository.knowledge_repository, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        service.drug_catalog_repository, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    request = ClinicalSessionRequest(
        clinical_input="ANAMNESIS\nonly anamnesis\n",
        visit_date=date(2025, 1, 15),
    )
    with pytest.raises(
        ServiceValidationError, match="Clinical input sections are invalid"
    ):
        start_clinical_job_workflow(service, request)

###############################################################################
def test_job_start_does_not_repeat_deep_preflight_after_ui_validation(
    monkeypatch,
) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service,
        "validate_clinical_input",
        lambda req: (_ for _ in ()).throw(
            AssertionError("deep preflight should not rerun during job start")
        ),
    )
    monkeypatch.setattr(
        service,
        "validate_assessment_prerequisites_without_llm",
        lambda req: object(),
    )
    monkeypatch.setattr(
        service,
        "prepare_structured_clinical_input",
        lambda req: {
            "normalized_document": object(),
            "section_extraction": object(),
            "patient_payload": object(),
        },
    )
    monkeypatch.setattr(
        service,
        "ensure_submission_requirements",
        lambda payload: None,
    )
    monkeypatch.setattr(
        "services.session.session_workflow.check_rag_readiness",
        lambda requested: RagReadiness(
            requested=requested,
            available=True,
            backend="ollama",
            model="dummy-model",
        ),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        service.job_manager, "is_job_running", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(service.job_manager, "start_job", lambda **kwargs: "job-123")
    monkeypatch.setattr(
        service.job_manager,
        "get_job_status",
        lambda job_id: {"job_id": job_id, "job_type": "clinical", "status": "pending"},
    )

    request = ClinicalSessionRequest(
        clinical_input=_valid_input(),
        visit_date=date(2025, 1, 15),
        selected_model_providers=["ollama"],
    )

    result = start_clinical_job_workflow(service, request)

    assert result.job_id == "job-123"
    assert result.status == "pending"

###############################################################################
def test_preflight_returns_deterministic_diagnostics_for_complex_input(
    monkeypatch,
) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service.session_repository.knowledge_repository, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        service.session_repository.drug_catalog_repository, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        "services.session.preflight._validate_provider_key", lambda blocking: None
    )
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda role: (
            ("ollama", "qwen3.5:2b") if role == "parser" else ("ollama", "gpt-oss:20b")
        ),
    )
    request = ClinicalSessionRequest(
        visit_date=date(2025, 3, 20),
        selected_model_providers=["ollama"],
        clinical_input=(
            "## Anamnesis\n"
            "High grade ovarian serous carcinoma con carcinosi peritoneale.\n"
            "Dal 17.03.2023 al 30.06.2023 Carboplatino e Paclitaxel, con aggiunta di Bevacizumab.\n"
            "Nozione di terapia antibiotica con Co-Amoxicillina 1-0-1 dal 18.02.\n\n"
            "## Terapia farmacologica\n"
            "Fortecortin 4 mg cpr 1-0-0-0 15.03 - 20.03\n"
            "De-Ursil 150 mg caps 1-0-1-0 per os dal 21.03\n\n"
            "## Laboratory Analysis\n"
            "Labor 20.03.2025: Bil tot 51.6 umol/L, ALP 1064 U/L, AST 385 U/L, ALT 730 U/L.\n"
        ),
    )

    result = validate_clinical_input_preflight(
        service,
        request,
        knowledge_repository=service.knowledge_repository,
        drug_catalog_repository=service.drug_catalog_repository,
    )

    assert result.ready is True
    assert result.deterministic_diagnostics["anamnesis"]["drug_count"] >= 3
    assert result.deterministic_diagnostics["diseases"]["disease_count"] >= 2
    assert result.extraction_quality["timed_drug_count"] >= 1

###############################################################################
def test_preflight_does_not_warn_when_deterministic_disease_matching_is_empty(
    monkeypatch,
) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service.session_repository.knowledge_repository, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        service.session_repository.drug_catalog_repository, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        "services.session.preflight._validate_provider_key", lambda blocking: None
    )
    monkeypatch.setattr(
        "services.session.preflight.extract_deterministic_diseases",
        lambda text: type(
            "DiseaseResult",
            (),
            {
                "context": type("Context", (), {"entries": []})(),
                "matched_lines": [],
                "unresolved_lines": [text],
            },
        )(),
    )

    result = validate_clinical_input_preflight(
        service,
        ClinicalSessionRequest(
            visit_date=date(2025, 3, 20),
            selected_model_providers=["ollama"],
            clinical_input=_valid_input(),
        ),
        knowledge_repository=service.knowledge_repository,
        drug_catalog_repository=service.drug_catalog_repository,
    )

    assert "anamnesis_disease_context_sparse" not in {
        issue.code for issue in result.non_blocking_issues
    }

###############################################################################
def test_job_start_uses_opencode_credential_scope_for_opencode_go(
    monkeypatch,
) -> None:
    service = _build_service()
    requested_scopes: list[str] = []
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.get_llm_provider",
        lambda: "opencode_go",
    )
    monkeypatch.setattr(
        "services.session.session_workflow.LLMRuntimeConfig.resolve_provider_and_model",
        lambda role: ("opencode_go", "minimax-m2.5-free"),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.AccessKeyService.list_access_keys",
        lambda _self, provider: requested_scopes.append(provider)
        or [SimpleNamespace(is_active=True)],
    )
    monkeypatch.setattr(
        service,
        "validate_assessment_prerequisites_without_llm",
        lambda request: (_ for _ in ()).throw(AssertionError("validation reached")),
    )

    with pytest.raises(AssertionError, match="validation reached"):
        start_clinical_job_workflow(
            service,
            ClinicalSessionRequest(
                clinical_input=_valid_input(),
                visit_date=date(2025, 3, 20),
                selected_model_providers=["opencode_go"],
            ),
        )

    assert requested_scopes == ["opencode"]

###############################################################################
def test_preflight_accepts_ollama_when_effective_clinical_runtime_is_local(
    monkeypatch,
) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service.session_repository.knowledge_repository, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        service.session_repository.drug_catalog_repository, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1)
    )
    monkeypatch.setattr(
        "services.session.preflight._validate_provider_key", lambda blocking: None
    )
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.get_llm_provider",
        lambda: "openai",
    )
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda role: (
            ("ollama", "qwen3.5:2b") if role == "parser" else ("ollama", "gpt-oss:20b")
        ),
    )

    request = ClinicalSessionRequest(
        visit_date=date(2025, 3, 20),
        selected_model_providers=["ollama"],
        clinical_input=(
            "## Anamnesis\n"
            "The patient reports fatigue, pruritus, dark urine, and poor appetite after medication exposure.\n\n"
            "## Therapy\n"
            "Amoxicillin 500 mg three times daily for seven days.\n"
            "Ibuprofen 400 mg as needed for fever.\n"
            "Atorvastatin 20 mg nightly as chronic therapy.\n\n"
            "## Laboratory Analysis\n"
            "ALT 220 U/L with ULN 50, ALP 180 U/L with ULN 120, total bilirubin 2.4 mg/dL, and INR 1.1.\n"
        ),
    )

    result = validate_clinical_input_preflight(
        service,
        request,
        knowledge_repository=service.knowledge_repository,
        drug_catalog_repository=service.drug_catalog_repository,
    )

    assert not any(
        issue.code == "requested_provider_mismatch" for issue in result.blocking_issues
    )
    assert result.runtime_settings["llm_provider"] == "openai"
    assert result.runtime_settings["clinical_provider"] == "ollama"

###############################################################################
def test_job_start_rechecks_rag_readiness_before_submission(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(
        service,
        "validate_assessment_prerequisites_without_llm",
        lambda req: object(),
    )
    monkeypatch.setattr(
        "services.session.session_workflow.check_rag_readiness",
        lambda requested: RagReadiness(
            requested=requested,
            available=False,
            backend="ollama",
            model="nomic-embed-text:v1.5",
            reason_code="rag_ollama_unavailable",
            message="Start Ollama and retry.",
        ),
    )
    request = ClinicalSessionRequest(
        clinical_input=_valid_input(),
        visit_date=date(2025, 1, 15),
        use_rag=True,
    )

    with pytest.raises(ServiceValidationError, match="Start Ollama and retry"):
        start_clinical_job_workflow(service, request)
