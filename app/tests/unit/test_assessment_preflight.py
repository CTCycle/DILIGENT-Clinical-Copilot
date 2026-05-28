from __future__ import annotations

from datetime import date

import pytest

from common.exceptions import ServiceValidationError
from domain.clinical.entities import ClinicalSessionRequest
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.preflight import validate_clinical_input_preflight
from services.session.session_workflow import start_clinical_job_workflow


def _build_service():
    return build_clinical_session_service(get_job_manager())


def _valid_input() -> str:
    return (
        "ANAMNESIS\nhistory\n"
        "DRUGS\nacetaminophen 500 mg\n"
        "LABORATORY ANALYSIS\nALT 240 U/L\n"
    )


def test_missing_visit_date_blocks_job_start_before_preprocess(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(service, "validate_clinical_input", lambda req: type("P", (), {"ready": True, "blocking_issues": []})())
    monkeypatch.setattr("services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled", lambda: False)
    monkeypatch.setattr(
        service,
        "preprocess_unified_input",
        lambda request_payload: (_ for _ in ()).throw(AssertionError("preprocess should not run")),
    )
    request = ClinicalSessionRequest(clinical_input=_valid_input(), visit_date=None)
    with pytest.raises(ServiceValidationError, match="Visit date is required"):
        start_clinical_job_workflow(service, request)


def test_empty_livertox_catalog_blocks_job_start_before_preprocess(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(service, "validate_clinical_input", lambda req: type("P", (), {"ready": True, "blocking_issues": []})())
    monkeypatch.setattr("services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled", lambda: False)
    monkeypatch.setattr(
        service,
        "preprocess_unified_input",
        lambda request_payload: (_ for _ in ()).throw(AssertionError("preprocess should not run")),
    )
    monkeypatch.setattr(service.serializer, "list_livertox_catalog", lambda **kwargs: ([], 0))
    request = ClinicalSessionRequest(
        clinical_input=_valid_input(),
        visit_date=date(2025, 1, 15),
    )
    with pytest.raises(ServiceValidationError, match="LiverTox catalog is empty"):
        start_clinical_job_workflow(service, request)


def test_empty_rxnav_catalog_blocks_job_start_before_preprocess(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(service, "validate_clinical_input", lambda req: type("P", (), {"ready": True, "blocking_issues": []})())
    monkeypatch.setattr("services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled", lambda: False)
    monkeypatch.setattr(service.serializer, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1))
    monkeypatch.setattr(service.serializer, "list_rxnav_catalog", lambda **kwargs: ([], 0))
    request = ClinicalSessionRequest(
        clinical_input=_valid_input(),
        visit_date=date(2025, 1, 15),
    )
    with pytest.raises(ServiceValidationError, match="RxNav catalog is empty"):
        start_clinical_job_workflow(service, request)


def test_malformed_sections_block_job_start(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(service, "validate_clinical_input", lambda req: type("P", (), {"ready": True, "blocking_issues": []})())
    monkeypatch.setattr("services.session.session_workflow.LLMRuntimeConfig.is_cloud_enabled", lambda: False)
    monkeypatch.setattr(service.serializer, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1))
    monkeypatch.setattr(service.serializer, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1))
    request = ClinicalSessionRequest(
        clinical_input="ANAMNESIS\nonly anamnesis\n",
        visit_date=date(2025, 1, 15),
    )
    with pytest.raises(ServiceValidationError, match="Clinical input sections are invalid"):
        start_clinical_job_workflow(service, request)


def test_preflight_returns_deterministic_diagnostics_for_complex_input(monkeypatch) -> None:
    service = _build_service()
    monkeypatch.setattr(service, "apply_persisted_runtime_configuration", lambda: None)
    monkeypatch.setattr(service.serializer, "list_livertox_catalog", lambda **kwargs: ([{"id": 1}], 1))
    monkeypatch.setattr(service.serializer, "list_rxnav_catalog", lambda **kwargs: ([{"id": 1}], 1))
    monkeypatch.setattr("services.session.preflight._validate_provider_key", lambda blocking: None)
    request = ClinicalSessionRequest(
        visit_date=date(2025, 3, 20),
        selected_model_providers=["openai"],
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

    result = validate_clinical_input_preflight(service, request)

    assert result.ready is True
    assert result.deterministic_diagnostics["anamnesis"]["drug_count"] >= 3
    assert result.deterministic_diagnostics["diseases"]["disease_count"] >= 2
    assert result.extraction_quality["timed_drug_count"] >= 1
