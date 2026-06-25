from __future__ import annotations

from domain.clinical.dili import ClinicalDataCompleteness, DiliEvidenceBundle
from domain.clinical.entities import (
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from services.clinical.dili_causality import DiliCausalityEngine
from services.clinical.dili_differential import DiliDifferentialEngine
from services.clinical.dili_hys_law import HysLawDetector
from services.clinical.dili_pattern import DiliPatternEngine
from services.clinical.dili_phenotype import DiliPhenotypeClassifier
from services.clinical.dili_severity import DiliSeverityGrader
from services.clinical.dili_timeline import DiliTimelineEngine
from services.text.normalization import normalize_drug_query_name


class DiliEvidenceBuilder:
    def build(
        self,
        *,
        payload: PatientData,
        drugs: PatientDrugs,
        labs: PatientLabTimeline,
        resolved_drugs: dict[str, dict] | None,
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> DiliEvidenceBundle:
        source_text = "\n".join(
            item for item in (payload.anamnesis, payload.drugs, payload.laboratory_analysis) if item
        )
        timeline = DiliTimelineEngine().build(drugs.entries, labs)
        patterns = DiliPatternEngine().assess(labs)
        differential = DiliDifferentialEngine().assess(source_text)
        phenotype = DiliPhenotypeClassifier().assess(patterns, source_text)
        hys_law = HysLawDetector().assess(labs, differential)
        severity = DiliSeverityGrader().assess(labs, source_text)
        resolved_map = {
            normalize_drug_query_name(key): value
            for key, value in (resolved_drugs or {}).items()
        }
        rucam_map = {
            normalize_drug_query_name(item.drug_name): item for item in rucam_bundle.entries
        }
        causality = DiliCausalityEngine()
        exposures = [
            causality.exposure(
                drug,
                resolved_map.get(normalize_drug_query_name(drug.name), {}),
                rucam_map.get(normalize_drug_query_name(drug.name)),
                differential,
                timeline.dechallenge_status,
            )
            for drug in drugs.entries
        ]
        missing = list(timeline.missing_fields)
        if patterns[0].pattern == "indeterminate":
            missing.append("paired ALT and ALP values with ULN")
        missing.extend(differential.unresolved_causes)
        return DiliEvidenceBundle(
            completeness=ClinicalDataCompleteness(
                complete_fields=["drug_exposures", "laboratory_timeline"]
                if drugs.entries and labs.entries
                else [],
                missing_fields=sorted(set(missing)),
                manual_review_required=True,
                reasons=["DILI is a diagnosis of exclusion.", "Clinical hepatology review required."],
            ),
            timeline=timeline,
            patterns=patterns,
            phenotype=phenotype,
            differential=differential,
            exposures=exposures,
            hys_law=hys_law,
            severity=severity,
            manual_review_required=True,
        )

    @staticmethod
    def render(bundle: DiliEvidenceBundle) -> str:
        pattern = bundle.patterns[0]
        lines = [
            "# Structured DILI causality dossier",
            "",
            "## 1. Case completeness and missing data",
            f"- Missing: {', '.join(bundle.completeness.missing_fields) or 'none documented'}",
            "## 2. Liver injury pattern and severity",
            f"- Pattern: {pattern.pattern}; R={pattern.r_ratio if pattern.r_ratio is not None else 'not assessable'} "
            f"(ALT {pattern.alt}/{pattern.alt_uln} ULN; ALP {pattern.alp}/{pattern.alp_uln} ULN)",
            f"- Severity: {bundle.severity.grade} ({bundle.severity.symptom_flag})",
            "## 3. Timeline summary",
            f"- First abnormal test: {bundle.timeline.first_abnormal_liver_test_date or 'missing'}",
            f"- Dechallenge: {bundle.timeline.dechallenge_status}",
            "## 4. Competing-cause assessment",
        ]
        lines.extend(f"- {item.cause}: {item.status}" for item in bundle.differential.causes)
        lines.extend(["## 5. Drug exposure table", ""])
        for exposure in bundle.exposures:
            lines.append(
                f"- {exposure.drug_name}: identity={exposure.identity.accepted_identity or 'unresolved'}; "
                f"start={exposure.start_date or 'missing'}; stop={exposure.stop_date or 'missing'}"
            )
        lines.extend(["## 6. Per-drug identity resolution", ""])
        lines.extend(
            f"- {item.drug_name}: {item.identity.identity_reason or 'no accepted identity rationale'}"
            for item in bundle.exposures
        )
        lines.extend(["## 7. Per-drug causality assessment", ""])
        lines.extend(
            f"- {item.drug_name}: {item.causality.category if item.causality else 'unassessable'}"
            for item in bundle.exposures
        )
        lines.extend(["## 8. RUCAM components", ""])
        for item in bundle.exposures:
            lines.append(f"### {item.drug_name}")
            if item.rucam is None:
                lines.append("- Not assessable")
            else:
                lines.append(f"- Total: {item.rucam.total_score}; category: {item.rucam.category}")
                lines.extend(
                    f"- {component.component}: {component.status}; score={component.score}; "
                    f"evidence={component.evidence_quote or 'missing'}"
                    for component in item.rucam.components
                )
        lines.extend(
            [
                "## 9. DILIN-like causality category",
                *[
                    f"- {item.drug_name}: {item.causality.category if item.causality else 'unassessable'}"
                    for item in bundle.exposures
                ],
                "## 10. Hy's Law status",
                f"- {bundle.hys_law.status}. This is a risk signal, not a diagnosis.",
                "## 11. Dechallenge/rechallenge",
                f"- Dechallenge: {bundle.timeline.dechallenge_status}",
                *[
                    f"- {item.drug_name} rechallenge: {item.rechallenge_status}; rechallenge is never recommended."
                    for item in bundle.exposures
                ],
                "## 12. Knowledge base and RAG evidence",
                "- Source hierarchy: AASLD, LiverTox, FDA, DILIN/RUCAM.",
                "## 13. Clinical limitations",
                "- RUCAM is structured support and is not dispositive.",
                "- Missing or unresolved competing causes prevent definitive attribution.",
                "## 14. Manual review requirements",
                "- Manual hepatology review required.",
            ]
        )
        return "\n".join(lines)
