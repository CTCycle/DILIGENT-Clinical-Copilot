from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
import services.session.session_workflow as session_workflow_module
from services.session.document_normalizer import DocumentNormalizer
from services.session.session_shared import build_failed_session_payload
from services.session.session_workflow import process_single_patient_workflow
from services.clinical.report_finalizer import ReportFinalizer
from services.session.session_service import ClinicalSessionService
from services.session.workflow_shared import ClinicalPersistenceError


###############################################################################
class FakePatternAnalyzer:
    # -------------------------------------------------------------------------
    def stringify_scores(
        self, pattern_score: HepatotoxicityPatternScore
    ) -> dict[str, str]:
        return {
            "r_score": f"{pattern_score.r_score:.2f}"
            if pattern_score.r_score
            else "Not available"
        }


###############################################################################
class FakeDrugsParser:
    model = "test-parser"

    # -------------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        return text


###############################################################################
class FakeSerializer:
    # -------------------------------------------------------------------------
    def save_clinical_session(self, payload: dict[str, Any]) -> int | None:
        self.saved_payload = payload
        return 101

    # -------------------------------------------------------------------------
    def upsert_session_result_payload(
        self, session_id: int, payload: dict[str, Any]
    ) -> None:
        self.upserted_session_id = session_id
        self.upserted_payload = payload


###############################################################################
class FakeClinicalService:
    JOB_TYPE = "clinical"

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.drugs_parser = FakeDrugsParser()
        self.pattern_analyzer = FakePatternAnalyzer()
        self.serializer = FakeSerializer()
        self.lab_extractor = SimpleNamespace(
            extract_explicit_hepatic_pattern=lambda text: (
                "cholestatic"
                if "Hepatic pattern: cholestatic" in (text or "")
                else None
            )
        )

    # -------------------------------------------------------------------------
    def run_stop_check(self, stop_check: Any) -> None:
        if stop_check is not None:
            stop_check()

    # -------------------------------------------------------------------------
    def emit_progress(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    # -------------------------------------------------------------------------
    def build_validation_bundle_for_payload(self, payload: PatientData) -> object:
        _ = payload
        return object()

    # -------------------------------------------------------------------------
    def ensure_submission_requirements(self, payload: PatientData) -> None:
        _ = payload

    # -------------------------------------------------------------------------
    async def extract_therapy_drugs(self, **kwargs: Any) -> PatientDrugs:
        _ = kwargs
        return PatientDrugs(entries=[DrugEntry(name="Paracetamolo", source="therapy")])

    # -------------------------------------------------------------------------
    async def extract_anamnesis_drugs(self, **kwargs: Any) -> PatientDrugs:
        _ = kwargs
        return PatientDrugs(entries=[])

    # -------------------------------------------------------------------------
    async def extract_disease_context(self, **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        return SimpleNamespace(entries=[])

    # -------------------------------------------------------------------------
    async def extract_lab_timeline(
        self, **kwargs: Any
    ) -> tuple[PatientLabTimeline, None]:
        _ = kwargs
        return (
            PatientLabTimeline(
                entries=[
                    ClinicalLabEntry(marker_name="ALT", value=100.0, unit="U/L"),
                    ClinicalLabEntry(marker_name="ALP", value=120.0, unit="U/L"),
                ]
            ),
            None,
        )

    # -------------------------------------------------------------------------
    def assess_pattern(self, **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        return SimpleNamespace(
            status="ok",
            score=HepatotoxicityPatternScore(
                classification="mixed",
                r_score=2.5,
            ),
        )

    # -------------------------------------------------------------------------
    def estimate_rucam(self, **kwargs: Any) -> PatientRucamAssessmentBundle:
        _ = kwargs
        return PatientRucamAssessmentBundle(entries=[])

    # -------------------------------------------------------------------------
    def build_structured_clinical_context(self, *args: Any, **kwargs: Any) -> str:
        _ = args, kwargs
        return "Contesto clinico strutturato."

    # -------------------------------------------------------------------------
    def build_rag_query(self, **kwargs: Any) -> dict[str, str]:
        _ = kwargs
        return {}

    # -------------------------------------------------------------------------
    async def run_livertox_lookup(self, **kwargs: Any) -> None:
        _ = kwargs
        return None

    # -------------------------------------------------------------------------
    def reestimate_rucam_with_livertox(
        self,
        *,
        rucam_bundle: PatientRucamAssessmentBundle,
        **kwargs: Any,
    ) -> PatientRucamAssessmentBundle:
        _ = kwargs
        return rucam_bundle

    # -------------------------------------------------------------------------
    async def run_consultation(self, **kwargs: Any) -> tuple[SimpleNamespace, str]:
        _ = kwargs
        return (
            SimpleNamespace(llm_model="test-clinical-model"),
            "Relazione narrativa con discussione farmacologica e sintesi finale.",
        )

    # -------------------------------------------------------------------------
    def _normalized_resolved_drug_map(self, prepared_inputs: Any) -> dict[str, Any]:
        _ = prepared_inputs
        return {}

    # -------------------------------------------------------------------------
    def _normalized_rucam_map(
        self,
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> dict[str, Any]:
        _ = rucam_bundle
        return {}

    # -------------------------------------------------------------------------
    def serialize_pipeline_issues(self, issues: list[Any]) -> list[dict[str, Any]]:
        serialized = []
        for issue in issues:
            if hasattr(issue, "model_dump"):
                serialized.append(issue.model_dump())
            else:
                serialized.append(dict(issue))
        return serialized


###############################################################################
def test_workflow_keeps_narrative_report_and_stores_audit_report() -> None:
    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2025, 5, 20),
        anamnesis="Paziente con ittero.",
        drugs="Paracetamolo 1-0-0-0",
        laboratory_analysis="Hepatic pattern: cholestatic. ALT 100 U/L, ALP 120 U/L.",
    )

    result = asyncio.run(
        process_single_patient_workflow(
            FakeClinicalService(),
            payload,
            normalized_document=DocumentNormalizer().normalize(
                "Paziente con ittero.\nParacetamolo 1-0-0-0\nALT 100 U/L, ALP 120 U/L."
            ),
            report_mode="faithful_only",
        ),
    )

    assert result["final_report"].startswith("## DILI adjudication summary")
    assert (
        "Relazione narrativa con discussione farmacologica e sintesi finale."
        in result["final_report"]
    )
    assert not result["final_report"].startswith("# Structured DILI causality dossier")
    assert "## DILI adjudication summary" in result["final_report"]
    assert result["pipeline_artifacts"]["structured_dili_report"].startswith(
        "# Structured DILI causality dossier"
    )
    assert (
        "## 14. Acceptance questions"
        in result["pipeline_artifacts"]["structured_dili_report"]
    )
    assert result["pipeline_artifacts"]["dili_user_summary"] in result["final_report"]
    assert result["dili_evidence_bundle"]["manual_review_required"] is True
    assert (
        result["llm_clinical_summary"]
        == "Relazione narrativa con discussione farmacologica e sintesi finale."
    )
    assert result["final_report"] in result["report"]
    assert "## Report Clinico" in result["pipeline_artifacts"]["generated_report"]
    assert (
        "### Esposizione ai Farmaci" in result["pipeline_artifacts"]["generated_report"]
    )
    assert result["pipeline_artifacts"]["generated_report"] != result["final_report"]
    assert (
        result["pipeline_artifacts"]["llm_clinical_summary"]
        == "Relazione narrativa con discussione farmacologica e sintesi finale."
    )
    assert result["extraction_metadata"]["hepatic_pattern"]["source"] == "provided"
    assert result["extraction_metadata"]["hepatic_pattern"]["value"] == "cholestatic"
    assert (
        result["extraction_metadata"]["rucam"]["source"]
        == "not_calculated_insufficient_data"
    )
    audit = result["pipeline_artifacts"]["rag_reference_audit"]
    assert {
        key: audit[key]
        for key in (
            "rag_retrieval_enabled",
            "rag_query_keys",
            "retrieved_references_by_drug",
            "retrieved_reference_count",
            "llm_clinical_summary_has_bibliography",
            "final_report_has_bibliography",
        )
    } == {
        "rag_retrieval_enabled": False,
        "rag_query_keys": [],
        "retrieved_references_by_drug": {},
        "retrieved_reference_count": 0,
        "llm_clinical_summary_has_bibliography": False,
        "final_report_has_bibliography": False,
    }


###############################################################################
def test_workflow_does_not_recreate_bibliography_outside_report_finalizer() -> None:

    ###############################################################################
    class FakeRagClinicalService(FakeClinicalService):
        # -------------------------------------------------------------------------
        def build_rag_query(self, **kwargs: Any) -> dict[str, str]:
            _ = kwargs
            return {
                "Paracetamolo": "paracetamol hepatocellular DILI LiverTox references"
            }

        # -------------------------------------------------------------------------
        async def run_consultation(self, **kwargs: Any) -> tuple[SimpleNamespace, str]:
            _ = kwargs
            clinical_session = SimpleNamespace(
                llm_model="gpt-4.1-mini",
                latest_drug_assessment_payload={
                    "entries": [
                        {
                            "drug_name": "Paracetamolo",
                            "match_status": "accepted_exact_livertox",
                            "paragraph": "Clinical RAG-supported paragraph.",
                            "rag_references": [
                                {
                                    "file_name": "paracetamol-dili.pdf",
                                    "page_start": 4,
                                    "page_end": 5,
                                }
                            ],
                        }
                    ],
                    "final_report": "Relazione narrativa senza bibliografia.",
                },
            )
            clinical_session.report_finalizer = ReportFinalizer(clinical_session)
            return clinical_session, "Relazione narrativa senza bibliografia."

    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2025, 5, 20),
        anamnesis="Paziente con ittero.",
        drugs="Paracetamolo 1-0-0-0",
        laboratory_analysis="ALT 100 U/L, ALP 120 U/L.",
        use_rag=True,
    )

    result = asyncio.run(
        process_single_patient_workflow(
            FakeRagClinicalService(),
            payload,
            normalized_document=DocumentNormalizer().normalize(
                "Paziente con ittero.\nParacetamolo 1-0-0-0\nALT 100 U/L, ALP 120 U/L."
            ),
            report_mode="faithful_only",
        ),
    )

    assert "## Bibliografia" not in result["final_report"]
    assert "## Bibliografia" not in result["llm_clinical_summary"]
    audit = result["pipeline_artifacts"]["rag_reference_audit"]
    assert audit["rag_retrieval_enabled"] is True
    assert audit["rag_query_keys"] == ["Paracetamolo"]
    assert audit["retrieved_reference_count"] == 1
    assert audit["llm_clinical_summary_has_bibliography"] is False
    assert audit["final_report_has_bibliography"] is False
    assert audit["contract_valid"] is False
    assert audit["retrieved_references_by_drug"] == {
        "Paracetamolo": [
            {
                "file_name": "paracetamol-dili.pdf",
                "page_start": 4,
                "page_end": 5,
                "line_start": None,
                "line_end": None,
                "document_title": None,
                "section_title": None,
                "chunk_id": None,
            }
        ]
    }


###############################################################################
def test_workflow_fails_when_persistence_returns_no_session_id() -> None:
    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2025, 5, 20),
        anamnesis="Paziente con ittero.",
        drugs="Paracetamolo 1-0-0-0",
        laboratory_analysis="ALT 100 U/L, ALP 120 U/L.",
    )
    service = FakeClinicalService()
    service.serializer.save_clinical_session = lambda payload: None

    with pytest.raises(ClinicalPersistenceError):
        asyncio.run(
            process_single_patient_workflow(
                service,
                payload,
                normalized_document=DocumentNormalizer().normalize(
                    "Paziente con ittero.\nParacetamolo 1-0-0-0\nALT 100 U/L, ALP 120 U/L."
                ),
                report_mode="faithful_only",
            ),
        )


###############################################################################
def test_failed_session_payload_omits_raw_clinical_text_and_base64_image() -> None:
    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2025, 5, 20),
        anamnesis="Sensitive anamnesis text.",
        drugs="Sensitive drug text.",
        laboratory_analysis="Sensitive lab text.",
    )

    failed_payload = build_failed_session_payload(
        payload=payload,
        patient_image_base64="base64-sensitive-image",
        issues=[],
        error_message="Sensitive error details",
        elapsed_seconds=1.2,
    )

    assert failed_payload["patient_image_base64"] is None
    assert failed_payload["anamnesis"] is None
    assert failed_payload["drugs"] is None
    assert failed_payload["laboratory_analysis"] is None
    result_payload = failed_payload["session_result_payload"]
    assert result_payload["section_extraction"] is None
    assert result_payload["error"] == "Clinical analysis failed before completion."
    assert result_payload["failure_metadata"]["has_patient_image"] is True
    assert result_payload["failure_metadata"]["input_character_counts"] == {
        "anamnesis": len(payload.anamnesis or ""),
        "drugs": len(payload.drugs or ""),
        "laboratory_analysis": len(payload.laboratory_analysis or ""),
    }


###############################################################################
def test_runtime_timeout_respects_provider_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.session_service.LLMRuntimeConfig.is_cloud_enabled",
        lambda: True,
    )

    timeout = ClinicalSessionService._resolve_runtime_timeout(
        base_timeout_s=180.0,
        cloud_cap_s=30.0,
        local_cap_s=45.0,
    )

    assert timeout == 30.0


###############################################################################
def test_workflow_marks_blocking_faithfulness_result_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2025, 5, 20),
        anamnesis="Paziente con ittero.",
        drugs="Paracetamolo 1-0-0-0",
        laboratory_analysis="ALT 100 U/L, ALP 120 U/L.",
    )
    service = FakeClinicalService()
    original_audit_report = session_workflow_module.audit_report

    def blocked_audit_report(*args: Any, **kwargs: Any) -> Any:
        audit = original_audit_report(*args, **kwargs)
        audit_payload = audit.model_dump()
        audit_payload["manual_review_required"] = True
        audit_payload["blocking_issues"] = [
            {
                "code": "faithfulness_gate_blocked",
                "message": "Source evidence did not support a final claim.",
            }
        ]
        return audit.__class__(**audit_payload)

    monkeypatch.setattr(session_workflow_module, "audit_report", blocked_audit_report)

    result = asyncio.run(
        process_single_patient_workflow(
            service,
            payload,
            normalized_document=DocumentNormalizer().normalize(
                "Paziente con ittero.\nParacetamolo 1-0-0-0\nALT 100 U/L, ALP 120 U/L."
            ),
            report_mode="faithful_only",
        ),
    )

    assert result["clinical_validity"] == "requires_human_review"
    assert result["manual_review_required"] is True
    assert any(
        issue["code"] == "faithfulness_gate_blocked" for issue in result["issues"]
    )
    assert service.serializer.saved_payload["session_status"] == "failed"
