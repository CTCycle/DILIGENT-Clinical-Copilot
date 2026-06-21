from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from domain.clinical.extras import HepatoxPreparedInputs
from domain.clinical.robustness import NormalizedDocument
from services.session.document_normalizer import DocumentNormalizer
import services.session.session_workflow as session_workflow_module
from services.session.revision_workflow import process_revision_patient_workflow

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
        return 202

    # -------------------------------------------------------------------------
    def upsert_session_result_payload(
        self, session_id: int, payload: dict[str, Any]
    ) -> None:
        self.upserted_session_id = session_id
        self.upserted_payload = payload

###############################################################################
class FakeRevisionClinicalService:
    JOB_TYPE = "clinical"

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.drugs_parser = FakeDrugsParser()
        self.pattern_analyzer = FakePatternAnalyzer()
        self.serializer = FakeSerializer()
        self.lab_extractor = None
        self.rag_context: str | None = None
        self.lookup_context: str | None = None
        self.lookup_drug_names: list[str] = []
        self.consultation_context: str | None = None
        self.consultation_drug_names: list[str] = []
        self.rucam_drug_names: list[str] = []

    # -------------------------------------------------------------------------
    def run_stop_check(self, stop_check: Any) -> None:
        if stop_check is not None:
            stop_check()

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
        return PatientDrugs(entries=[DrugEntry(name="Drug A", source="therapy")])

    # -------------------------------------------------------------------------
    async def extract_anamnesis_drugs(self, **kwargs: Any) -> PatientDrugs:
        _ = kwargs
        return PatientDrugs(entries=[DrugEntry(name="Drug B", source="anamnesis")])

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
                    ClinicalLabEntry(marker_name="ALT", value=180.0, unit="U/L"),
                    ClinicalLabEntry(marker_name="ALP", value=100.0, unit="U/L"),
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
                classification="hepatocellular",
                r_score=4.2,
            ),
        )

    # -------------------------------------------------------------------------
    def estimate_rucam(self, **kwargs: Any) -> PatientRucamAssessmentBundle:
        self.rucam_drug_names = [
            entry.name for entry in kwargs["analysis_drugs"].entries if entry.name
        ]
        return PatientRucamAssessmentBundle(entries=[])

    # -------------------------------------------------------------------------
    def build_structured_clinical_context(self, *args: Any, **kwargs: Any) -> str:
        _ = args, kwargs
        return "Base structured context"

    # -------------------------------------------------------------------------
    def build_rag_query(self, **kwargs: Any) -> dict[str, str]:
        self.rag_context = str(kwargs["structured_context"])
        return {"drug-a": "Base structured context"}

    # -------------------------------------------------------------------------
    async def run_livertox_lookup(self, **kwargs: Any) -> HepatoxPreparedInputs:
        self.lookup_context = str(kwargs["structured_context"])
        self.lookup_drug_names = [
            entry.name for entry in kwargs["all_detected_drugs"].entries if entry.name
        ]
        return HepatoxPreparedInputs(
            resolved_drugs={"drug-a": {"canonical_name": "Drug A"}},
            pattern_prompt="Pattern prompt",
            clinical_context=str(kwargs["structured_context"]),
        )

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
        raise AssertionError("Revision workflow should not call run_consultation")

    # -------------------------------------------------------------------------
    async def run_revision_consultation(
        self,
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, str, dict[str, Any]]:
        self.consultation_context = kwargs["consultation_context"]
        self.consultation_drug_names = [
            entry.name for entry in kwargs["analysis_drugs"].entries if entry.name
        ]
        return (
            SimpleNamespace(llm_model="test-clinical-model"),
            "Revision-focused final report",
            {
                "revision_kind": "llm_assisted_revision",
                "analysis_entrypoint": "run_revision_analysis",
                "used_fallback_report": False,
                "consultation_model": "test-clinical-model",
                "drug_analysis_entrypoint": "request_revision_drug_analysis",
                "report_finalization_entrypoint": "finalize_revision_patient_report",
                "conclusion_entrypoint": "generate_revision_conclusion",
                "synthesis_mode": "revision_comparison_aware",
                "analysis_drug_names": self.consultation_drug_names,
                "consultation_context_length": len(self.consultation_context or ""),
                "source_version_id": kwargs["consultation_context_metadata"].get(
                    "source_version_id"
                ),
                "revision_version_id": kwargs["consultation_context_metadata"].get(
                    "revision_version_id"
                ),
                "pipeline_run_id": kwargs["consultation_context_metadata"].get(
                    "pipeline_run_id"
                ),
            },
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
def test_revision_focus_context_isolated_from_retrieval_and_lookup(
    monkeypatch,
) -> None:
    payload = PatientData(
        name="Revision Patient",
        visit_date=date(2025, 5, 20),
        anamnesis="Baseline clinical history.",
        drugs="Drug A",
        laboratory_analysis="ALT 180 U/L, ALP 100 U/L.",
        use_rag=True,
    )
    service = FakeRevisionClinicalService()
    normalized_document: NormalizedDocument = DocumentNormalizer().normalize(
        "Baseline clinical history.\nDrug A\nALT 180 U/L, ALP 100 U/L."
    )

    async def fake_check_parser_batch_capacity(task_count: int) -> SimpleNamespace:
        _ = task_count
        return SimpleNamespace(
            concurrency_allowed=False,
            provider="test",
            model="test-model",
            reason="test",
        )

    monkeypatch.setattr(
        session_workflow_module,
        "check_parser_batch_capacity",
        fake_check_parser_batch_capacity,
    )
    result = asyncio.run(
        process_revision_patient_workflow(
            service,
            payload,
            normalized_document=normalized_document,
            report_mode="faithful_only",
            session_metadata={
                "source_version_id": 11,
                "target_revision_version_id": 12,
                "pipeline_run_id": "pipe-test-001",
                "source_official_report_text": "Prior report content",
                "source_rucam_assessments": [
                    {
                        "drug_name": "Drug A",
                        "causality_category": "possible",
                    }
                ],
            },
            revision_focus_context="Rewrite the chronology section only.",
        ),
    )

    assert service.rag_context == "Base structured context"
    assert service.lookup_context == "Base structured context"
    assert service.lookup_drug_names == ["Drug B"]
    assert service.rucam_drug_names == ["Drug B"]
    assert service.consultation_context == (
        "Base structured context\n\n"
        "Revision entity snapshot:\n"
        "Revision anamnesis additions:\nDrug B\n\n"
        "Revision supplemental anamnesis drugs:\nDrug B\n\n"
        "Revision lab markers:\nALT, ALP\n\n"
        "Revision analysis drugs:\nDrug B\n\n"
        "Revision focus context:\n"
        "Rewrite the chronology section only.\n\n"
        "Revision metadata:\n"
        "Source version: 11\n"
        "Revision version: 12\n"
        "Pipeline run: pipe-test-001\n\n"
        "Previous report for comparison only:\n"
        "Prior report content\n\n"
        "Previous per-drug assessments for comparison only:\n"
        "Drug A: possible\n\n"
        "Revision evidence handling:\n"
        "Use source evidence and revised structured artifacts as primary support.\n"
        "Treat previous report content as comparison context only."
    )
    assert service.consultation_drug_names == ["Drug B"]
    assert result["revision"]["focus_context"] == "Rewrite the chronology section only."
    assert result["revision"]["revision_kind"] == "llm_assisted_revision"
    assert "execution_mode" not in result["revision"]
    assert result["revision"]["source_session_id"] is None
    assert result["revision"]["source_version_id"] == 11
    assert result["revision"]["revision_version_id"] == 12
    assert result["revision"]["pipeline_run_id"] == "pipe-test-001"
    assert result["revision"]["extraction_bundle"] == {
        "status": "resolved",
        "therapy_source": "recomputed_current_revision",
        "anamnesis_source": "recomputed_current_revision",
        "disease_source": "recomputed_current_revision",
        "therapy_deterministic_names": [],
        "anamnesis_deterministic_names": [],
        "disease_deterministic_names": [],
        "therapy_structured_names": ["Drug A"],
        "anamnesis_structured_names": ["Drug B"],
        "therapy_unresolved_lines": [],
        "anamnesis_unresolved_lines": [],
        "anamnesis_regimen_lines": [],
        "disease_unresolved_lines": [],
    }
    assert result["revision"]["entity_snapshot_context"] == (
        "Revision anamnesis additions:\nDrug B\n\n"
        "Revision supplemental anamnesis drugs:\nDrug B\n\n"
        "Revision lab markers:\nALT, ALP\n\n"
        "Revision analysis drugs:\nDrug B"
    )
    assert result["revision"]["consultation_drug_names"] == ["Drug B"]
    assert result["revision"]["consultation_context_metadata"] == {
        "source_version_id": 11,
        "revision_version_id": 12,
        "pipeline_run_id": "pipe-test-001",
        "has_previous_report": True,
        "previous_assessment_count": 1,
        "previous_assessment_summaries": ["Drug A: possible"],
    }
    assert result["revision"]["consultation_execution"] == {
        "revision_kind": "llm_assisted_revision",
        "analysis_entrypoint": "run_revision_analysis",
        "used_fallback_report": False,
        "consultation_model": "test-clinical-model",
        "drug_analysis_entrypoint": "request_revision_drug_analysis",
        "report_finalization_entrypoint": "finalize_revision_patient_report",
        "conclusion_entrypoint": "generate_revision_conclusion",
        "synthesis_mode": "revision_comparison_aware",
        "analysis_drug_names": ["Drug B"],
        "snapshot_context_present": True,
        "consultation_context_length": len(service.consultation_context or ""),
        "source_version_id": 11,
        "revision_version_id": 12,
        "pipeline_run_id": "pipe-test-001",
        "has_previous_report_context": True,
        "previous_assessment_count": 1,
    }
    assert result["revision"]["finalization_execution"]["final_report_present"] is True
    assert (
        result["revision"]["finalization_execution"]["consultation_model"]
        == "test-clinical-model"
    )
    assert result["revision"]["entity_pipeline"]["validate_anamnesis_drugs"] == {
        "status": "supplemented",
        "deterministic_detected_names": [],
        "revised_detected_names": ["Drug B"],
        "overlapping_names": [],
        "deterministic_only_names": [],
        "revised_only_names": ["Drug B"],
        "unresolved_lines": [],
    }
    assert result["revision"]["entity_pipeline"]["resolve_revision_extraction"] == {
        "status": "resolved",
        "therapy_source": "recomputed_current_revision",
        "anamnesis_source": "recomputed_current_revision",
        "disease_source": "recomputed_current_revision",
        "therapy_deterministic_names": [],
        "anamnesis_deterministic_names": [],
        "disease_deterministic_names": [],
        "therapy_structured_names": ["Drug A"],
        "anamnesis_structured_names": ["Drug B"],
        "therapy_unresolved_lines": [],
        "anamnesis_unresolved_lines": [],
        "anamnesis_regimen_lines": [],
        "disease_unresolved_lines": [],
    }
    assert result["revision"]["entity_pipeline"]["revise_labs_timeline"] == {
        "status": "revised",
        "lab_entry_count": 2,
        "source_counts": {"anamnesis": 2},
        "marker_names": ["ALT", "ALP"],
        "onset_context_present": False,
        "pattern_classification": "hepatocellular",
    }
    assert result["revision"]["entity_pipeline"]["reconcile_revision_candidates"] == {
        "status": "reconciled",
        "analysis_drug_names": ["Drug B"],
        "relevant_drug_names": ["Drug A", "Drug B"],
        "excluded_drug_names": [],
        "unresolved_drug_names": [],
    }
    assert result["revision"]["entity_pipeline"]["merge_revision_snapshot"] == {
        "status": "merged",
        "therapy_drug_names": ["Drug A"],
        "anamnesis_drug_names": ["Drug B"],
        "disease_names": [],
        "lab_marker_names": ["ALT", "ALP"],
        "analysis_drug_names": ["Drug B"],
        "relevant_drug_names": ["Drug A", "Drug B"],
        "excluded_drug_names": [],
        "unresolved_drug_names": [],
        "rucam_assessment_count": 0,
    }
    assert result["relevant_drugs"] == [
        {
            "drug": "Drug A",
            "reason": "Active or plausibly timed exposure with aligned relevance.",
        },
        {
            "drug": "Drug B",
            "reason": (
                "Revision pipeline promoted this drug from staged anamnesis additions "
                "for targeted reassessment."
            ),
        },
    ]
    assert result["unresolved_drugs"] == []

###############################################################################
def test_revision_reuses_persisted_deterministic_extraction(
    monkeypatch,
) -> None:
    payload = PatientData(
        name="Persisted Deterministic Patient",
        visit_date=date(2025, 5, 20),
        anamnesis="Baseline clinical history.",
        drugs="Drug A",
        laboratory_analysis="ALT 180 U/L, ALP 100 U/L.",
        use_rag=True,
    )
    service = FakeRevisionClinicalService()
    normalized_document: NormalizedDocument = DocumentNormalizer().normalize(
        "Baseline clinical history.\nDrug A\nALT 180 U/L, ALP 100 U/L."
    )

    async def fake_check_parser_batch_capacity(task_count: int) -> SimpleNamespace:
        _ = task_count
        return SimpleNamespace(
            concurrency_allowed=False,
            provider="test",
            model="test-model",
            reason="test",
        )

    def fail_deterministic_drugs(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError("Deterministic drug extraction should reuse persisted source data")

    def fail_deterministic_diseases(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError(
            "Deterministic disease extraction should reuse persisted source data"
        )

    monkeypatch.setattr(
        session_workflow_module,
        "check_parser_batch_capacity",
        fake_check_parser_batch_capacity,
    )
    monkeypatch.setattr(
        session_workflow_module,
        "_extract_deterministic_drugs",
        fail_deterministic_drugs,
    )
    monkeypatch.setattr(
        session_workflow_module,
        "extract_deterministic_diseases",
        fail_deterministic_diseases,
    )
    result = asyncio.run(
        process_revision_patient_workflow(
            service,
            payload,
            normalized_document=normalized_document,
            report_mode="faithful_only",
            session_metadata={
                "source_deterministic_extraction": {
                    "therapy": {
                        "entries": [{"name": "Persisted Therapy Drug", "source": "therapy"}],
                        "unresolved_lines": [],
                    },
                    "anamnesis": {
                        "entries": [
                            {
                                "name": "Persisted Anamnesis Drug",
                                "source": "anamnesis",
                                "historical_flag": True,
                            }
                        ],
                        "regimen_lines": [],
                        "unresolved_lines": [],
                    },
                    "diseases": {
                        "entries": [
                            {
                                "name": "Persisted Disease",
                                "hepatic_related": False,
                                "evidence": "Persisted source artifact",
                            }
                        ],
                        "matched_lines": ["Persisted Disease line"],
                        "unresolved_lines": [],
                    },
                }
            },
            revision_focus_context="Rewrite the chronology section only.",
        ),
    )

    assert result["revision"]["extraction_bundle"] == {
        "status": "resolved",
        "therapy_source": "persisted_source_version",
        "anamnesis_source": "persisted_source_version",
        "disease_source": "persisted_source_version",
        "therapy_deterministic_names": ["Persisted Therapy Drug"],
        "anamnesis_deterministic_names": ["Persisted Anamnesis Drug"],
        "disease_deterministic_names": ["Persisted Disease"],
        "therapy_structured_names": ["Drug A"],
        "anamnesis_structured_names": ["Drug B"],
        "therapy_unresolved_lines": [],
        "anamnesis_unresolved_lines": [],
        "anamnesis_regimen_lines": [],
        "disease_unresolved_lines": [],
    }
    assert result["revision"]["entity_pipeline"]["validate_anamnesis_drugs"] == {
        "status": "supplemented",
        "deterministic_detected_names": ["Persisted Anamnesis Drug"],
        "revised_detected_names": ["Drug B"],
        "overlapping_names": [],
        "deterministic_only_names": ["Persisted Anamnesis Drug"],
        "revised_only_names": ["Drug B"],
        "unresolved_lines": [],
    }

###############################################################################
def test_revision_reuses_persisted_disease_and_lab_artifacts(
    monkeypatch,
) -> None:
    payload = PatientData(
        name="Persisted Structured Patient",
        visit_date=date(2025, 5, 20),
        anamnesis="Baseline clinical history.",
        drugs="Drug A",
        laboratory_analysis="ALT 180 U/L, ALP 100 U/L.",
        use_rag=True,
    )
    service = FakeRevisionClinicalService()
    normalized_document: NormalizedDocument = DocumentNormalizer().normalize(
        "Baseline clinical history.\nDrug A\nALT 180 U/L, ALP 100 U/L."
    )

    async def fake_check_parser_batch_capacity(task_count: int) -> SimpleNamespace:
        _ = task_count
        return SimpleNamespace(
            concurrency_allowed=False,
            provider="test",
            model="test-model",
            reason="test",
        )

    async def fail_disease_context(**kwargs: Any) -> Any:
        _ = kwargs
        raise AssertionError("Disease context extraction should reuse persisted source data")

    async def fail_lab_timeline(**kwargs: Any) -> Any:
        _ = kwargs
        raise AssertionError("Lab timeline extraction should reuse persisted source data")

    service.extract_disease_context = fail_disease_context  # type: ignore[method-assign]
    service.extract_lab_timeline = fail_lab_timeline  # type: ignore[method-assign]
    monkeypatch.setattr(
        session_workflow_module,
        "check_parser_batch_capacity",
        fake_check_parser_batch_capacity,
    )
    result = asyncio.run(
        process_revision_patient_workflow(
            service,
            payload,
            normalized_document=normalized_document,
            report_mode="faithful_only",
            session_metadata={
                "source_structured_case": {
                    "anamnesis_diseases": [
                        {
                            "name": "Persisted Disease Context",
                            "hepatic_related": False,
                            "evidence": "Persisted structured case",
                        }
                    ]
                },
                "source_lab_timeline": [
                    {
                        "marker_name": "ALT",
                        "value": 210.0,
                        "unit": "U/L",
                        "source": "laboratory_analysis",
                    },
                    {
                        "marker_name": "ALP",
                        "value": 95.0,
                        "unit": "U/L",
                        "source": "laboratory_analysis",
                    },
                ],
                "source_onset_context": {
                    "onset_date": "2025-05-10",
                    "onset_basis": "first_abnormal_lab",
                    "evidence": "Persisted onset context",
                },
            },
            revision_focus_context="Rewrite the chronology section only.",
        ),
    )

    assert result["revision"]["source_artifact_reuse"] == {
        "therapy_deterministic": "recomputed_current_revision",
        "anamnesis_deterministic": "recomputed_current_revision",
        "disease_deterministic": "recomputed_current_revision",
        "disease_context": "persisted_source_version",
        "lab_timeline": "persisted_source_version",
        "onset_context": "persisted_source_version",
    }
    assert result["revision"]["entity_pipeline"]["revise_labs_timeline"] == {
        "status": "revised",
        "lab_entry_count": 2,
        "source_counts": {"laboratory_analysis": 2},
        "marker_names": ["ALT", "ALP"],
        "onset_context_present": True,
        "pattern_classification": "hepatocellular",
    }
    assert result["revision"]["entity_pipeline"]["merge_revision_snapshot"] == {
        "status": "merged",
        "therapy_drug_names": ["Drug A"],
        "anamnesis_drug_names": ["Drug B"],
        "disease_names": ["Persisted Disease Context"],
        "lab_marker_names": ["ALT", "ALP"],
        "analysis_drug_names": ["Drug B"],
        "relevant_drug_names": ["Drug A", "Drug B"],
        "excluded_drug_names": [],
        "unresolved_drug_names": [],
        "rucam_assessment_count": 0,
    }
