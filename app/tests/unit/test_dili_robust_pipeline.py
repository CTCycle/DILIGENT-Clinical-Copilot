from __future__ import annotations

from datetime import date

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from domain.clinical.robustness import FactGraph, FactGraphNode, ReportMetadata
from services.session.document_normalizer import DocumentNormalizer
from services.session.robust_pipeline import (
    audit_report,
    build_extraction_artifact,
    build_fact_graph,
    build_run_bundle_index,
    render_fact_graph_report,
    validate_fact_graph,
)


def test_document_normalizer_keeps_raw_text_and_labels_bibliography() -> None:
    raw_text = (
        "Patient: UI metadata\n\nClinical note text\n\nReferences\nDoe et al. doi:123"
    )

    normalized = DocumentNormalizer().normalize(raw_text)

    assert normalized.raw_text == raw_text
    assert "Clinical note text" in normalized.clean_text
    assert any(block.block_type == "bibliography" for block in normalized.blocks)
    assert normalized.span_mappings[0].raw_end == len(raw_text)


def test_extraction_artifact_uses_ui_metadata_outside_document_sections() -> None:
    raw_text = (
        "## Anamnesis\nPatient reports jaundice.\n\n"
        "## Therapy\nZetamycin 10 mg 1-0-0-0\n\n"
        "## Laboratory Analysis\nALT 100 U/L.\n\n"
        "## Assessment\nPossible DILI."
    )
    payload = PatientData(
        name="UI Patient",
        visit_date=date(2024, 1, 15),
        anamnesis="Patient reports jaundice.",
        drugs="Zetamycin 10 mg 1-0-0-0",
        laboratory_analysis="ALT 100 U/L.",
    )

    artifact = build_extraction_artifact(
        normalized_document=DocumentNormalizer().normalize(raw_text),
        section_extraction=None,
        payload=payload,
    )

    assert "patient_name" not in artifact.sections
    assert "report_date" not in artifact.sections
    assert artifact.sections["therapy"].text == "Zetamycin 10 mg 1-0-0-0"
    assert artifact.timed_drugs[0].drug == "Zetamycin 10 mg 1-0-0-0"


def test_fact_graph_validation_blocks_source_verbatim_nodes_without_spans() -> None:
    graph = FactGraph(
        nodes=[
            FactGraphNode(
                node_id="fact-1",
                family="drug_exposure",
                value={"name": "Zetamycin"},
                source_spans=[],
                confidence=0.9,
                origin="source_verbatim",
            )
        ]
    )

    validation = validate_fact_graph(graph)

    assert validation.hard_issues[0]["code"] == "source_span_missing"


def test_report_metadata_links_claims_to_fact_nodes() -> None:
    payload = PatientData(
        name="UI Patient",
        visit_date=date(2024, 1, 15),
        anamnesis="Patient reports jaundice.",
        drugs="Zetamycin 10 mg 1-0-0-0",
        laboratory_analysis="ALT 100 U/L.",
    )
    extraction = build_extraction_artifact(
        normalized_document=DocumentNormalizer().normalize(
            "Patient reports jaundice.\nZetamycin 10 mg 1-0-0-0\nALT 100 U/L."
        ),
        section_extraction=None,
        payload=payload,
    )
    graph = build_fact_graph(
        extraction_artifact=extraction,
        payload=payload,
        therapy_drugs=PatientDrugs(
            entries=[DrugEntry(name="Zetamycin", source="therapy")]
        ),
        anamnesis_drugs=PatientDrugs(entries=[]),
        lab_timeline=PatientLabTimeline(
            entries=[ClinicalLabEntry(marker_name="ALT", value=100.0, unit="U/L")]
        ),
        pattern_score=type("Pattern", (), {"classification": "hepatocellular"})(),
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    report, metadata = render_fact_graph_report(
        fact_graph=graph,
        patient_name=payload.name,
        visit_date=payload.visit_date,
        report_mode="faithful_only",
    )

    assert "UI Patient" in report
    assert metadata.claim_references
    assert (
        build_run_bundle_index(run_id="1", session_id=1).storage
        == "database_session_result_payload"
    )


def test_fact_graph_report_localizes_italian_audit_labels() -> None:
    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2024, 1, 15),
        anamnesis="Paziente con ittero.",
        drugs="Paracetamolo 1-0-0-0",
        laboratory_analysis="ALT 100 U/L.",
    )
    extraction = build_extraction_artifact(
        normalized_document=DocumentNormalizer().normalize(
            "Paziente con ittero.\nParacetamolo 1-0-0-0\nALT 100 U/L."
        ),
        section_extraction=None,
        payload=payload,
    )
    graph = build_fact_graph(
        extraction_artifact=extraction,
        payload=payload,
        therapy_drugs=PatientDrugs(
            entries=[DrugEntry(name="Paracetamolo", source="therapy")]
        ),
        anamnesis_drugs=PatientDrugs(entries=[]),
        lab_timeline=PatientLabTimeline(
            entries=[ClinicalLabEntry(marker_name="ALT", value=100.0, unit="U/L")]
        ),
        pattern_score=type("Pattern", (), {"classification": "mixed"})(),
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    report, metadata = render_fact_graph_report(
        fact_graph=graph,
        patient_name=payload.name,
        visit_date=payload.visit_date,
        report_mode="faithful_only",
        report_language="it",
    )

    assert "## Report Clinico" in report
    assert "Paziente: Mario Rossi" in report
    assert "### Esposizione ai Farmaci" in report
    assert "### Evidenze di Laboratorio" in report
    assert "### Pattern DILI" in report
    assert "Drug Exposure" not in report
    assert "Laboratory Evidence" not in report
    assert "Clinical Report" not in report
    assert metadata.claim_references


def test_audit_blocks_report_without_claim_references() -> None:
    audit = audit_report(
        extraction_artifact=build_extraction_artifact(
            normalized_document=DocumentNormalizer().normalize("No timed drug."),
            section_extraction=None,
            payload=PatientData(
                name="UI Patient",
                visit_date=date(2024, 1, 15),
                anamnesis="No timed drug.",
                drugs="Drug without timing",
                laboratory_analysis="ALT 100 U/L.",
            ),
        ),
        fact_graph_validation=validate_fact_graph(FactGraph(nodes=[])),
        report_metadata=ReportMetadata(
            report_mode="faithful_only", claim_references={}
        ),
    )

    assert audit.blocking_issues
    assert audit.outcome == "partially_faithful_with_major_issues"
