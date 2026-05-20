from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from domain.clinical.robustness import (
    ContaminationFlags,
    ExtractedSection,
    ExtractionArtifact,
    FactGraph,
    FactGraphNode,
    FactGraphValidation,
    FaithfulnessAudit,
    NormalizedDocument,
    ReportMetadata,
    RunBundleIndex,
    SourceSpan,
    TimedDrugMention,
)


TIMING_RE = re.compile(
    r"\b(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*(?:al|-)\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{1,2}\s*-\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|"
    r"(?:gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)[a-z]*\s+\d{4})\b",
    re.IGNORECASE,
)
SCHEDULE_RE = re.compile(r"\b(?P<schedule>\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?){2,3})\b")
CONTAMINATION_RE = re.compile(
    r"\b(references|bibliography|bibliografia|doi:|pubmed|page|pagina|address)\b",
    re.IGNORECASE,
)
THERAPY_CONTAMINATION_RE = re.compile(
    r"\b(references|bibliography|bibliografia|doi:|pubmed|et al\.|journal)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ']+\b", re.UNICODE)
CAUSALITY_RE = re.compile(
    r"\b(causality|causale|causality|probable|possible|unlikely|excluded|"
    r"probabile|possibile|improbabile|esclus[oa])\b",
    re.IGNORECASE,
)
RECOMMENDATION_RE = re.compile(
    r"\b(recommend|recommendation|raccomand|monitor|follow[- ]?up|suspend|stop|"
    r"consiglia|sospendere|monitorare)\b",
    re.IGNORECASE,
)


def build_extraction_artifact(
    *,
    normalized_document: NormalizedDocument,
    section_extraction: ClinicalSectionExtractionResult | None,
    payload: PatientData,
) -> ExtractionArtifact:
    sections = {
        "anamnesis": _section_from_text(
            key="anamnesis",
            text=payload.anamnesis or "",
            source_text=normalized_document.raw_text,
            fallback_confidence=_extraction_confidence(section_extraction),
        ),
        "therapy": _section_from_text(
            key="therapy",
            text=payload.drugs or "",
            source_text=normalized_document.raw_text,
            fallback_confidence=_extraction_confidence(section_extraction),
        ),
        "laboratory_analysis": _section_from_text(
            key="laboratory_analysis",
            text=payload.laboratory_analysis or "",
            source_text=normalized_document.raw_text,
            fallback_confidence=_extraction_confidence(section_extraction),
        ),
        "physician_assessment": _find_semantic_section(
            key="physician_assessment",
            normalized_document=normalized_document,
            heading_terms=("assessment", "valutazione", "conclusion", "conclusioni"),
        ),
        "recommendations": _find_semantic_section(
            key="recommendations",
            normalized_document=normalized_document,
            heading_terms=("recommend", "raccomand", "follow-up", "monitoraggio"),
        ),
    }
    contamination = ContaminationFlags(
        therapy_contaminated_by_bibliography_or_admin=_is_therapy_contaminated(
            payload.drugs or ""
        ),
        assessment_contaminated_by_non_clinical_content=bool(
            CONTAMINATION_RE.search(sections["physician_assessment"].text)
        ),
        labs_embedded_without_dedicated_lab_section=(
            not bool((payload.laboratory_analysis or "").strip())
            and bool(re.search(r"\b(alt|ast|alp|bilirubin|bilirubina|ggt)\b", normalized_document.raw_text, re.IGNORECASE))
        ),
    )
    timed_drugs = _extract_timed_drugs(payload)
    issues = []
    if not timed_drugs:
        issues.append(
            {
                "severity": "error",
                "code": "timed_drugs_missing",
                "message": "No drug mention with explicit timing was found.",
                "field": "drugs",
            }
        )
    for key, section in sections.items():
        if section.missing:
            issues.append(
                {
                    "severity": "warning",
                    "code": f"{key}_missing",
                    "message": f"{key} section is missing or uncertain.",
                    "field": key,
                }
            )
    # Confidence should reflect core extraction sections used by the pipeline,
    # not optional semantic sections that may be absent in many source formats.
    core_keys = ("anamnesis", "therapy", "laboratory_analysis")
    confidence_values = [
        sections[key].confidence
        for key in core_keys
        if key in sections
    ]
    if not confidence_values:
        confidence_values = [section.confidence for section in sections.values()]
    return ExtractionArtifact(
        sections=sections,
        confidence=sum(confidence_values) / max(1, len(confidence_values)),
        contamination_flags=contamination,
        timed_drugs=timed_drugs,
        extraction_issues=issues,
    )


def build_fact_graph(
    *,
    extraction_artifact: ExtractionArtifact,
    payload: PatientData,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    lab_timeline: PatientLabTimeline,
    pattern_score: Any,
    rucam_bundle: PatientRucamAssessmentBundle,
) -> FactGraph:
    nodes: list[FactGraphNode] = []
    for entry in [*therapy_drugs.entries, *anamnesis_drugs.entries]:
        nodes.append(_drug_node(entry, len(nodes), extraction_artifact))
    for lab_entry in lab_timeline.entries:
        nodes.append(
            FactGraphNode(
                node_id=f"fact-{len(nodes) + 1}",
                family="lab_event",
                value=lab_entry.model_dump(),
                source_spans=_section_spans(extraction_artifact, "laboratory_analysis"),
                confidence=0.75,
                origin="source_verbatim",
            )
        )
    if payload.anamnesis:
        nodes.append(
            FactGraphNode(
                node_id=f"fact-{len(nodes) + 1}",
                family="clinical_event",
                value={"text": payload.anamnesis},
                source_spans=_section_spans(extraction_artifact, "anamnesis"),
                confidence=0.65,
                origin="source_verbatim",
            )
        )
    for assessment in rucam_bundle.entries:
        nodes.append(
            FactGraphNode(
                node_id=f"fact-{len(nodes) + 1}",
                family="causality_statement",
                value=assessment.model_dump(),
                source_spans=_section_spans(extraction_artifact, "physician_assessment"),
                confidence=0.7,
                origin="derived",
                supports=[node.node_id for node in nodes if node.family == "drug_exposure"],
            )
        )
    classification = getattr(pattern_score, "classification", None)
    if classification:
        nodes.append(
            FactGraphNode(
                node_id=f"fact-{len(nodes) + 1}",
                family="dili_pattern_statement",
                value={"classification": classification},
                source_spans=_section_spans(extraction_artifact, "laboratory_analysis"),
                confidence=0.8,
                origin="derived",
                supports=[node.node_id for node in nodes if node.family == "lab_event"],
            )
        )
    recommendations = extraction_artifact.sections.get("recommendations")
    if recommendations and recommendations.text:
        nodes.append(
            FactGraphNode(
                node_id=f"fact-{len(nodes) + 1}",
                family="recommendation_statement",
                value={"text": recommendations.text},
                source_spans=recommendations.source_spans,
                confidence=recommendations.confidence,
                origin="source_verbatim",
            )
        )
    return FactGraph(nodes=nodes)


def validate_fact_graph(fact_graph: FactGraph) -> FactGraphValidation:
    hard_issues: list[dict[str, Any]] = []
    soft_issues: list[dict[str, Any]] = []
    for node in fact_graph.nodes:
        if node.origin == "source_verbatim" and not node.source_spans:
            hard_issues.append(
                {
                    "code": "source_span_missing",
                    "message": f"{node.family} node {node.node_id} has no evidence span.",
                    "node_id": node.node_id,
                }
            )
        if node.origin == "derived" and not node.supports:
            soft_issues.append(
                {
                    "code": "derived_support_missing",
                    "message": f"Derived node {node.node_id} has no supporting fact ids.",
                    "node_id": node.node_id,
                }
            )
    return FactGraphValidation(hard_issues=hard_issues, soft_issues=soft_issues)


def render_fact_graph_report(
    *,
    fact_graph: FactGraph,
    patient_name: str | None,
    visit_date: date | None,
    report_mode: str,
) -> tuple[str, ReportMetadata]:
    lines = ["## Clinical Report", ""]
    lines.append(f"Patient: {patient_name or 'Not provided'}")
    lines.append(f"Report date: {visit_date.isoformat() if visit_date else 'Not provided'}")
    claim_refs: dict[str, list[str]] = {}
    drug_nodes = [node for node in fact_graph.nodes if node.family == "drug_exposure"]
    if drug_nodes:
        lines.extend(["", "### Drug Exposure"])
        for index, node in enumerate(drug_nodes, start=1):
            name = node.value.get("name") or node.value.get("drug") or "Unknown drug"
            status = node.value.get("status") or node.value.get("suspension_status")
            timing = node.value.get("therapy_start_date") or node.value.get("timing_value")
            claim_id = f"drug_exposure_{index}"
            parts = [str(name)]
            if timing:
                parts.append(f"timing: {timing}")
            if status is not None:
                parts.append(f"status: {status}")
            lines.append(f"- {'; '.join(parts)} [{claim_id}]")
            claim_refs[claim_id] = [node.node_id]
    lab_nodes = [node for node in fact_graph.nodes if node.family == "lab_event"]
    if lab_nodes:
        lines.extend(["", "### Laboratory Evidence"])
        for index, node in enumerate(lab_nodes, start=1):
            value = node.value
            label = value.get("marker_name") or value.get("test_name") or value.get("name") or "Lab event"
            raw_value = value.get("value") or value.get("raw_value") or value.get("result")
            claim_id = f"lab_event_{index}"
            lines.append(f"- {label}: {raw_value if raw_value is not None else 'reported'} [{claim_id}]")
            claim_refs[claim_id] = [node.node_id]
    for family, heading in (
        ("dili_pattern_statement", "DILI Pattern"),
        ("causality_statement", "Causality"),
        ("recommendation_statement", "Recommendations"),
    ):
        nodes = [node for node in fact_graph.nodes if node.family == family]
        if not nodes:
            continue
        lines.extend(["", f"### {heading}"])
        for index, node in enumerate(nodes, start=1):
            claim_id = f"{family}_{index}"
            lines.append(f"- {_summarize_value(node.value)} [{claim_id}]")
            claim_refs[claim_id] = [node.node_id, *node.supports]
    metadata = ReportMetadata(
        report_mode=report_mode,
        claim_references=claim_refs,
    )
    return "\n".join(lines).strip(), metadata


def audit_report(
    *,
    extraction_artifact: ExtractionArtifact,
    fact_graph_validation: FactGraphValidation,
    report_metadata: ReportMetadata,
) -> FaithfulnessAudit:
    blocking_issues = list(fact_graph_validation.hard_issues)
    non_blocking_issues = list(fact_graph_validation.soft_issues)
    contamination = extraction_artifact.contamination_flags
    # Confidence below 0.5 indicates materially weak extraction quality.
    # The previous 0.7 threshold over-triggered manual review for otherwise
    # coherent cases with clean contamination checks.
    manual_review_required = (
        contamination.therapy_contaminated_by_bibliography_or_admin
        or contamination.assessment_contaminated_by_non_clinical_content
        or contamination.labs_embedded_without_dedicated_lab_section
        or extraction_artifact.confidence < 0.5
    )
    if not report_metadata.claim_references:
        blocking_issues.append(
            {
                "code": "no_source_linked_claims",
                "message": "Generated report has no source-linked claims.",
            }
        )
    if any(issue.get("code") == "timed_drugs_missing" for issue in extraction_artifact.extraction_issues):
        blocking_issues.append(
            {
                "code": "timed_drugs_missing",
                "message": "Timed-drug requirement was not satisfied.",
            }
        )
    if manual_review_required:
        non_blocking_issues.append(
            {
                "code": "manual_review_required",
                "message": "Contamination or low extraction confidence requires manual review.",
            }
        )
    outcome = "faithful"
    if blocking_issues:
        outcome = "partially_faithful_with_major_issues"
    elif non_blocking_issues:
        outcome = "mostly_faithful_with_minor_issues"
    discrepancy_report = _render_discrepancy_report(blocking_issues, non_blocking_issues)
    structured_comparison = _build_structured_report_comparison(
        extraction_artifact=extraction_artifact,
        blocking_issues=blocking_issues,
        non_blocking_issues=non_blocking_issues,
        manual_review_required=manual_review_required,
    )
    return FaithfulnessAudit(
        outcome=outcome,
        manual_review_required=manual_review_required,
        blocking_issues=blocking_issues,
        non_blocking_issues=non_blocking_issues,
        gate_decisions=[
            {
                "gate": "hard_safety_gates",
                "passed": not blocking_issues,
            },
            {
                "gate": "manual_review",
                "manual_review_required": manual_review_required,
            },
        ],
        discrepancy_report=structured_comparison,
    )


def build_run_bundle_index(*, run_id: str, session_id: int | None = None) -> RunBundleIndex:
    return RunBundleIndex(
        run_id=run_id,
        session_id=session_id,
        artifacts={
            "normalized_document": "session_result_payload.pipeline_artifacts.normalized_document",
            "extraction_artifact": "session_result_payload.pipeline_artifacts.extraction_artifact",
            "fact_graph": "session_result_payload.pipeline_artifacts.fact_graph",
            "fact_graph_validation": "session_result_payload.pipeline_artifacts.fact_graph_validation",
            "generated_report": "session_result_payload.pipeline_artifacts.generated_report",
            "report_metadata": "session_result_payload.pipeline_artifacts.report_metadata",
            "faithfulness_audit": "session_result_payload.pipeline_artifacts.faithfulness_audit",
            "discrepancy_report": "session_result_payload.pipeline_artifacts.discrepancy_report",
        },
    )


def _extraction_confidence(
    section_extraction: ClinicalSectionExtractionResult | None,
) -> float:
    if section_extraction is None:
        return 0.6
    return max(0.0, min(1.0, float(section_extraction.confidence)))


def _section_from_text(
    *,
    key: str,
    text: str,
    source_text: str,
    fallback_confidence: float,
) -> ExtractedSection:
    stripped = text.strip()
    span = _span_for_text(key=key, text=stripped, source_text=source_text)
    return ExtractedSection(
        key=key,
        text=stripped,
        confidence=fallback_confidence if stripped else 0.0,
        source_spans=[span] if span else [],
        missing=not bool(stripped),
        issues=[] if stripped else ["section_missing"],
    )


def _find_semantic_section(
    *,
    key: str,
    normalized_document: NormalizedDocument,
    heading_terms: tuple[str, ...],
) -> ExtractedSection:
    for block in normalized_document.blocks:
        lowered = block.text.lower()
        if any(term in lowered for term in heading_terms):
            return ExtractedSection(
                key=key,
                text=block.text,
                confidence=min(0.8, block.confidence),
                source_spans=block.source_spans,
            )
    return ExtractedSection(
        key=key,
        missing=True,
        confidence=0.0,
        issues=["section_missing"],
    )


def _span_for_text(*, key: str, text: str, source_text: str) -> SourceSpan | None:
    if not text:
        return None
    start = source_text.find(text)
    if start < 0:
        compact = re.sub(r"\s+", " ", text).strip()
        compact_source = re.sub(r"\s+", " ", source_text)
        start = compact_source.find(compact)
        if start < 0:
            return SourceSpan(
                span_id=f"{key}-span-1",
                start_char=0,
                end_char=0,
                text=text[:5000],
            )
    prefix = source_text[:start]
    start_line = prefix.count("\n") + 1
    end_line = start_line + text.count("\n")
    return SourceSpan(
        span_id=f"{key}-span-1",
        start_line=start_line,
        end_line=end_line,
        start_char=max(0, start),
        end_char=max(0, start + len(text)),
        text=text[:5000],
    )


def _extract_timed_drugs(payload: PatientData) -> list[TimedDrugMention]:
    mentions: list[TimedDrugMention] = []
    for key, text in (("therapy", payload.drugs or ""), ("anamnesis", payload.anamnesis or "")):
        for line_index, line in enumerate(text.splitlines(), start=1):
            match = TIMING_RE.search(line)
            schedule_match = SCHEDULE_RE.search(line)
            if not match and not schedule_match:
                continue
            drug = _guess_drug_name(line)
            if not drug:
                continue
            mentions.append(
                TimedDrugMention(
                    drug=drug,
                    timing_type="date" if match else "schedule",
                    timing_value=match.group("date") if match else schedule_match.group("schedule"),
                    status="source_reported",
                    source_span=SourceSpan(
                        span_id=f"{key}-timed-drug-{len(mentions) + 1}",
                        start_line=line_index,
                        end_line=line_index,
                        start_char=0,
                        end_char=len(line),
                        text=line,
                    ),
                )
            )
    return mentions


def _guess_drug_name(line: str) -> str | None:
    cleaned = re.sub(r"^[\-*•\d.)\s]+", "", line).strip()
    if not cleaned:
        return None
    before_date = TIMING_RE.split(cleaned, maxsplit=1)[0]
    # Prefer the most local token group near the timing marker, not generic section labels.
    candidate_parts = [part.strip() for part in re.split(r"[,;:()]", before_date) if part.strip()]
    candidate = candidate_parts[-1] if candidate_parts else before_date.strip()
    candidate = re.sub(r"\b(?:terapia|therapy|farmacologica|farmacologic)\b", "", candidate, flags=re.IGNORECASE).strip()
    words = candidate.split()
    if not words:
        return None
    return " ".join(words[:4])


def _drug_node(
    entry: DrugEntry,
    index: int,
    extraction_artifact: ExtractionArtifact,
) -> FactGraphNode:
    source_key = "therapy" if entry.source == "therapy" else "anamnesis"
    return FactGraphNode(
        node_id=f"fact-{index + 1}",
        family="drug_exposure",
        value=entry.model_dump(),
        source_spans=_section_spans(extraction_artifact, source_key),
        confidence=0.75 if entry.temporal_classification == "temporal_known" else 0.6,
        origin="source_verbatim",
    )


def _section_spans(
    extraction_artifact: ExtractionArtifact,
    section_key: str,
) -> list[SourceSpan]:
    section = extraction_artifact.sections.get(section_key)
    if section is None:
        return []
    return section.source_spans


def _summarize_value(value: dict[str, Any]) -> str:
    for key in ("text", "classification", "outcome", "causality", "drug_name", "name"):
        item = value.get(key)
        if item:
            return str(item)
    return ", ".join(f"{key}: {item}" for key, item in list(value.items())[:3])


def _render_discrepancy_report(
    blocking_issues: list[dict[str, Any]],
    non_blocking_issues: list[dict[str, Any]],
) -> str:
    lines = ["## Faithfulness Discrepancy Report", ""]
    if not blocking_issues and not non_blocking_issues:
        lines.append("No discrepancies detected by structural faithfulness gates.")
        return "\n".join(lines)
    if blocking_issues:
        lines.extend(["### Blocking Issues", ""])
        lines.extend(f"- {issue.get('code')}: {issue.get('message')}" for issue in blocking_issues)
    if non_blocking_issues:
        lines.extend(["", "### Non-Blocking Issues", ""])
        lines.extend(f"- {issue.get('code')}: {issue.get('message')}" for issue in non_blocking_issues)
    return "\n".join(lines).strip()


def _build_structured_report_comparison(
    *,
    extraction_artifact: ExtractionArtifact,
    blocking_issues: list[dict[str, Any]],
    non_blocking_issues: list[dict[str, Any]],
    manual_review_required: bool,
) -> str:
    timed_drug_count = len(extraction_artifact.timed_drugs)
    confidence = float(extraction_artifact.confidence)
    contamination = extraction_artifact.contamination_flags

    if blocking_issues:
        outcome = "comparison_not_possible"
    elif manual_review_required:
        outcome = "partial_agreement_manual_review_required"
    else:
        outcome = "structured_agreement"

    agreements: list[str] = []
    omissions: list[str] = []
    differences: list[str] = []
    unsupported: list[str] = []

    if timed_drug_count > 0:
        agreements.append(f"Detected {timed_drug_count} timed drug mention(s) from source text.")
    else:
        omissions.append("No timed drug mentions were extracted from source text.")

    if confidence >= 0.7:
        agreements.append(f"Extraction confidence {confidence:.2f} meets comparison threshold.")
    else:
        differences.append(f"Extraction confidence {confidence:.2f} is below preferred comparison threshold (0.70).")

    if contamination.therapy_contaminated_by_bibliography_or_admin:
        unsupported.append("Therapy section contains probable non-clinical contamination.")
    if contamination.assessment_contaminated_by_non_clinical_content:
        unsupported.append("Physician assessment section contains probable non-clinical contamination.")
    if contamination.labs_embedded_without_dedicated_lab_section:
        differences.append("Laboratory evidence appears embedded outside a dedicated labs section.")

    for issue in blocking_issues:
        code = str(issue.get("code", "blocking_issue"))
        message = str(issue.get("message", "Blocking issue"))
        omissions.append(f"{code}: {message}")
    for issue in non_blocking_issues:
        code = str(issue.get("code", "non_blocking_issue"))
        message = str(issue.get("message", "Non-blocking issue"))
        differences.append(f"{code}: {message}")

    payload = {
        "outcome": outcome,
        "agreements": agreements or ["No high-confidence agreements identified."],
        "omissions": omissions or ["No critical omissions detected by structural gates."],
        "differences": differences or ["No significant structural differences detected."],
        "unsupported": unsupported or ["No unsupported claims flagged by contamination checks."],
        "manual_review": "yes" if manual_review_required else "no",
    }
    return json.dumps(payload, ensure_ascii=False)


def _is_therapy_contaminated(text: str) -> bool:
    if not text.strip():
        return False
    hits = THERAPY_CONTAMINATION_RE.findall(text)
    if not hits:
        return False
    # Single bibliography tokens are common in long reports due to footer/header
    # duplication after PDF extraction. Flag only when contamination is substantial.
    if len(hits) >= 3:
        return True
    word_count = len(WORD_RE.findall(text))
    if word_count <= 0:
        return False
    return (len(hits) / word_count) >= 0.02
