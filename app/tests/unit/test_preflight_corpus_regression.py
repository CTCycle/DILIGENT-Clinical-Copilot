from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from configurations.llm_configs import LLMRuntimeConfig
from domain.clinical.entities import ClinicalSessionRequest
from domain.clinical.robustness import FactGraph, ReportMetadata
from services.runtime.jobs import get_job_manager
from services.session import preflight as preflight_module
from services.session.document_normalizer import DocumentNormalizer
from services.session.factory import build_clinical_session_service
from services.session.robust_pipeline import (
    audit_report,
    build_extraction_artifact,
    validate_fact_graph,
)


def _load_corpus_payloads() -> list[dict[str, object]]:
    corpus_file = Path("tmp_dili_5run_results.json")
    if not corpus_file.exists():
        pytest.skip("Captured corpus file tmp_dili_5run_results.json is not available.")
    content = json.loads(corpus_file.read_text(encoding="utf-8"))
    if not isinstance(content, list) or not content:
        pytest.skip("Captured corpus file is empty or malformed.")
    return [item for item in content if isinstance(item, dict)]


def test_preflight_allows_captured_corpus_without_blocking_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _load_corpus_payloads()
    service = build_clinical_session_service(get_job_manager())

    # Preflight regression should not depend on cloud-key state.
    monkeypatch.setattr(preflight_module, "_validate_provider_key", lambda blocking: None)

    runtime_provider = (LLMRuntimeConfig.get_llm_provider() or "").strip()
    selected_model_providers = [runtime_provider] if runtime_provider else []

    blocking_results: list[tuple[str, list[str]]] = []
    for index, item in enumerate(payloads, start=1):
        request = ClinicalSessionRequest(
            name=str(item.get("patient_name") or f"patient-{index}"),
            visit_date=item.get("visit_date_iso"),
            clinical_input=str(item.get("physician_report") or ""),
            selected_model_providers=selected_model_providers,
        )
        result = preflight_module.validate_clinical_input_preflight(service, request)
        if result.blocking_issues:
            codes = [issue.code for issue in result.blocking_issues]
            document = str(item.get("document") or f"row_{index}")
            blocking_results.append((document, codes))

    assert not blocking_results, f"Unexpected preflight blocking issues: {blocking_results}"


def test_preprocess_unified_input_succeeds_for_all_captured_corpus_cases() -> None:
    payloads = _load_corpus_payloads()
    service = build_clinical_session_service(get_job_manager())

    runtime_provider = (LLMRuntimeConfig.get_llm_provider() or "").strip()
    selected_model_providers = [runtime_provider] if runtime_provider else []

    failures: list[tuple[str, str]] = []
    for index, item in enumerate(payloads, start=1):
        request = ClinicalSessionRequest(
            name=str(item.get("patient_name") or f"patient-{index}"),
            visit_date=item.get("visit_date_iso"),
            clinical_input=str(item.get("physician_report") or ""),
            selected_model_providers=selected_model_providers,
        )
        document = str(item.get("document") or f"row_{index}")
        try:
            preprocessed_request, extraction = asyncio.run(
                service.preprocess_unified_input(request)
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((document, str(exc)))
            continue

        assert (preprocessed_request.anamnesis or "").strip(), (
            f"Empty anamnesis after preprocessing for {document}"
        )
        assert (preprocessed_request.drugs or "").strip(), (
            f"Empty therapy section after preprocessing for {document}"
        )
        assert (preprocessed_request.laboratory_analysis or "").strip(), (
            f"Empty laboratory section after preprocessing for {document}"
        )
        assert extraction is not None, f"Missing extraction metadata for {document}"
        payload = service.build_patient_payload(preprocessed_request)
        normalized = DocumentNormalizer().normalize(str(item.get("physician_report") or ""))
        extraction_artifact = build_extraction_artifact(
            normalized_document=normalized,
            section_extraction=extraction,
            payload=payload,
        )
        audit = audit_report(
            extraction_artifact=extraction_artifact,
            fact_graph_validation=validate_fact_graph(FactGraph(nodes=[])),
            report_metadata=ReportMetadata(
                report_mode="faithful_only",
                claim_references={"claim_1": ["fact-1"]},
            ),
        )
        assert audit.manual_review_required is False, (
            f"Unexpected manual review requirement for {document}"
        )
        comparison = json.loads(audit.discrepancy_report)
        assert comparison.get("outcome") == "structured_agreement", (
            f"Unexpected comparison outcome for {document}: {comparison.get('outcome')}"
        )

    assert not failures, f"Preprocess failures for captured corpus cases: {failures}"
