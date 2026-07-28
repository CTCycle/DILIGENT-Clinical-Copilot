from __future__ import annotations

import json
from pathlib import Path

import pytest
from services.llm.runtime_config import LLMRuntimeConfig
from domain.clinical.entities import ClinicalSessionRequest
from services.runtime.jobs import get_job_manager
from services.session import preflight as preflight_module
from services.session.factory import build_clinical_session_service

###############################################################################
def _load_corpus_payloads() -> list[dict[str, object]]:
    corpus_file = Path("tmp_dili_5run_results.json")
    if not corpus_file.exists():
        pytest.skip("Captured corpus file tmp_dili_5run_results.json is not available.")
    content = json.loads(corpus_file.read_text(encoding="utf-8"))
    if not isinstance(content, list) or not content:
        pytest.skip("Captured corpus file is empty or malformed.")
    return [item for item in content if isinstance(item, dict)]

###############################################################################
def test_preflight_allows_captured_corpus_without_blocking_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _load_corpus_payloads()
    service = build_clinical_session_service(get_job_manager())

    # Preflight regression should not depend on cloud-key state.
    monkeypatch.setattr(
        preflight_module, "_validate_provider_key", lambda blocking: None
    )

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

    assert not blocking_results, (
        f"Unexpected preflight blocking issues: {blocking_results}"
    )
