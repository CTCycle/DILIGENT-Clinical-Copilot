from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from domain.clinical.entities import ClinicalSessionRequest
from domain.clinical.robustness import (
    ClinicalInputPreflightIssue,
    RagReadiness,
)
from services.session.preflight import (
    _present_preflight_issue,
    validate_clinical_input_preflight,
)


###############################################################################
def test_blocking_issue_is_enriched_with_modal_metadata() -> None:
    issue = _present_preflight_issue(
        ClinicalInputPreflightIssue(
            severity="blocking",
            code="clinical_input_missing",
            message="Clinical input is required.",
            field="clinical_input",
        )
    )

    assert issue.title == "Clinical input is empty"
    assert issue.description == "Clinical input is required."
    assert issue.affected_section == "Clinical input"
    assert issue.consequence
    assert issue.continuation_allowed is False


###############################################################################
def test_unknown_warning_receives_safe_fallback_metadata() -> None:
    issue = _present_preflight_issue(
        ClinicalInputPreflightIssue(
            severity="non_blocking",
            code="future_warning",
            message="A future warning.",
            field="future_section",
        )
    )

    assert issue.title == "Review required"
    assert issue.affected_section == "Future Section"
    assert issue.continuation_allowed is True


###############################################################################
def test_unavailable_requested_rag_is_returned_as_non_blocking_issue(
    monkeypatch,
) -> None:
    service = SimpleNamespace(
        apply_persisted_runtime_configuration=lambda: None,
        serializer=SimpleNamespace(
            list_livertox_catalog=lambda **kwargs: ([{"id": 1}], 1),
            list_rxnav_catalog=lambda **kwargs: ([{"id": 1}], 1),
        ),
    )
    monkeypatch.setattr(
        "services.session.preflight._validate_provider_key",
        lambda blocking: None,
    )
    monkeypatch.setattr(
        "services.session.preflight._runtime_settings",
        lambda: {"clinical_provider": "ollama", "llm_provider": "ollama"},
    )
    monkeypatch.setattr(
        "services.session.preflight.check_rag_readiness",
        lambda requested: RagReadiness(
            requested=requested,
            available=False,
            backend="ollama",
            model="nomic-embed-text:v1.5",
            reason_code="rag_ollama_unavailable",
            message="Start Ollama and retry.",
        ),
    )

    result = validate_clinical_input_preflight(
        service,
        ClinicalSessionRequest(
            visit_date=date(2026, 7, 17),
            clinical_input=None,
            selected_model_providers=["ollama"],
            use_rag=True,
        ),
    )

    rag_issue = next(
        issue
        for issue in result.non_blocking_issues
        if issue.code == "rag_ollama_unavailable"
    )
    assert rag_issue.field == "use_rag"
    assert rag_issue.continuation_allowed is True
    assert rag_issue.affected_section == "RAG evidence"
