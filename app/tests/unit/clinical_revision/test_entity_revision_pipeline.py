from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from domain.clinical.entities import (
    ClinicalLabEntry,
    DiseaseContextEntry,
    DrugEntry,
    DrugRucamAssessment,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from services.session import revision_workflow as revision_workflow_module


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "dili_revision"
    / "entity_revision_cases.json"
)


def _load_cases() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_entity_revision_pipeline_fixture_case_produces_expected_stages() -> None:
    case = _load_cases()[0]
    therapy_drugs = PatientDrugs(
        entries=[DrugEntry.model_validate(item) for item in case["therapy_drugs"]]
    )
    anamnesis_drugs = PatientDrugs(
        entries=[DrugEntry.model_validate(item) for item in case["anamnesis_drugs"]]
    )
    anamnesis_deterministic = SimpleNamespace(
        entries=[DrugEntry.model_validate(item) for item in case["anamnesis_deterministic"]["entries"]],
        regimen_lines=list(case["anamnesis_deterministic"]["regimen_lines"]),
        unresolved_lines=list(case["anamnesis_deterministic"]["unresolved_lines"]),
    )
    lab_timeline = PatientLabTimeline(
        entries=[ClinicalLabEntry.model_validate(item) for item in case["lab_timeline"]]
    )
    disease_context = PatientDiseaseContext(
        entries=[DiseaseContextEntry.model_validate(item) for item in case["disease_entries"]]
    )
    rucam_bundle = PatientRucamAssessmentBundle(
        entries=[DrugRucamAssessment.model_validate(item) for item in case["rucam_assessments"]]
    )
    pattern_score = SimpleNamespace(classification=case["pattern_classification"])

    resolution = revision_workflow_module._select_revision_candidates(
        extraction_bundle=dict(case["extraction_bundle"]),
        anamnesis_deterministic=anamnesis_deterministic,
        anamnesis_drugs=anamnesis_drugs,
        therapy_drugs=therapy_drugs,
        lab_timeline=lab_timeline,
        onset_context=None,
        pattern_score=pattern_score,
        visit_date=None,
    )
    merge_stage = revision_workflow_module._build_revision_snapshot_merge_stage(
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        disease_context=disease_context,
        lab_timeline=lab_timeline,
        analysis_drugs=resolution.analysis_drugs,
        candidate_selection=resolution.candidate_selection,
        rucam_bundle=rucam_bundle,
    )

    expected = case["expected"]
    assert resolution.entity_pipeline["validate_anamnesis_drugs"]["status"] == expected["validate_status"]
    assert (
        resolution.entity_pipeline["extract_missing_anamnesis_drugs"]["supplemental_drug_names"]
        == expected["supplemental_drug_names"]
    )
    assert resolution.entity_pipeline["reconcile_revision_candidates"]["analysis_drug_names"] == expected["analysis_drug_names"]
    assert resolution.entity_pipeline["reconcile_revision_candidates"]["relevant_drug_names"] == expected["relevant_drug_names"]
    assert resolution.entity_pipeline["revise_labs_timeline"]["marker_names"] == expected["lab_marker_names"]
    assert merge_stage["analysis_drug_names"] == expected["analysis_drug_names"]
    assert merge_stage["disease_names"] == expected["disease_names"]
    assert merge_stage["lab_marker_names"] == expected["lab_marker_names"]
    assert merge_stage["rucam_assessment_count"] == expected["rucam_assessment_count"]
