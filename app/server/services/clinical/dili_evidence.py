from __future__ import annotations

from domain.clinical.dili import (
    ClinicalDataCompleteness,
    ClinicalEvidenceQuote,
    DiliAcceptanceQuestion,
    DiliEvidenceBundle,
    DrugExposureAssessment,
)
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
            item
            for item in (payload.anamnesis, payload.drugs, payload.laboratory_analysis)
            if item
        )
        timeline = DiliTimelineEngine().build(drugs.entries, labs)
        patterns = DiliPatternEngine().assess(labs)
        primary_pattern = patterns[0].pattern if patterns else "indeterminate"
        differential = DiliDifferentialEngine().assess(source_text)
        phenotype = DiliPhenotypeClassifier().assess(patterns, source_text)
        hys_law = HysLawDetector().assess(
            labs=labs,
            differential=differential,
            timeline=timeline,
            drugs=drugs.entries,
            source_text=source_text,
        )
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
                primary_pattern,
                timeline.first_abnormal_liver_test_date,
            )
            for drug in drugs.entries
        ]
        for exposure in exposures:
            exposure.dose_changes = [
                event
                for event in timeline.events
                if event.drug_name == exposure.drug_name and event.event_type == "dose_change"
            ]

        missing = list(timeline.missing_fields)
        if primary_pattern == "indeterminate":
            missing.append("paired ALT and ALP values with ULN")
        missing.extend(differential.unresolved_causes)
        acceptance_questions = self._acceptance_questions(
            timeline=timeline,
            patterns=patterns,
            phenotype=phenotype,
            differential=differential,
            hys_law=hys_law,
            severity=severity,
            exposures=exposures,
        )
        evidence = [quote for question in acceptance_questions for quote in question.supporting_evidence][:12]
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
            evidence=evidence,
            acceptance_questions=acceptance_questions,
            manual_review_required=True,
        )

    def _acceptance_questions(
        self,
        *,
        timeline,
        patterns,
        phenotype,
        differential,
        hys_law,
        severity,
        exposures,
    ) -> list[DiliAcceptanceQuestion]:
        first_pattern = patterns[0] if patterns else None
        top_exposure = self._primary_suspect(exposures, differential.all_major_causes_excluded)
        questions: list[DiliAcceptanceQuestion] = [
            self._question(
                "What is the latency from first compatible exposure to first liver injury signal?",
                timeline.first_abnormal_liver_test_date or "missing",
                [event.evidence for event in timeline.events if event.event_type in {"drug_start", "abnormal_liver_test"} and event.evidence][:3],
                "Latency remains uncertain because exposure start or first abnormal liver-test timing is missing." if not timeline.first_abnormal_liver_test_date else None,
            ),
            self._question(
                "Did the injury improve, persist, or worsen after discontinuation?",
                timeline.dechallenge_status,
                [event.evidence for event in timeline.events if event.event_type in {"drug_stop", "abnormal_liver_test"} and event.evidence][:3],
                "Dechallenge cannot be graded honestly when follow-up after discontinuation is incomplete." if timeline.dechallenge_status in {"no_follow_up", "insufficient_interval"} else None,
            ),
            self._question(
                "What is the liver injury pattern at the first qualifying episode?",
                first_pattern.pattern if first_pattern is not None else "indeterminate",
                list(first_pattern.evidence[:2]) if first_pattern is not None else [],
                "Pattern is indeterminate when paired ALT and ALP values with ULN are unavailable." if first_pattern is None or first_pattern.pattern == "indeterminate" else None,
            ),
            self._question(
                "Which clinically conservative phenotype candidates are supported?",
                ", ".join(phenotype.candidates) or "none identified",
                list((first_pattern.evidence[:1] if first_pattern is not None else [])),
                "Phenotype remains limited by missing biopsy, imaging, autoimmune markers, or long follow-up." if phenotype.missing_data else None,
            ),
            self._question(
                "Are mandatory alternative causes excluded?",
                "yes" if differential.all_major_causes_excluded else "no",
                [item.evidence[0] for item in differential.causes if item.evidence][:4],
                "One or more mandatory competing causes are unresolved or not excluded." if not differential.all_major_causes_excluded else None,
            ),
            self._question(
                "Does the episode satisfy Hy's Law requirements?",
                hys_law.status,
                list(hys_law.evidence[:4]),
                "Hy's Law is not assessable or only possible when same-episode timing, cholestasis exclusion, alternatives, or exposure compatibility remain incomplete." if hys_law.status != "meets_criteria" else None,
            ),
            self._question(
                "What is the severity grade?",
                f"{severity.grade} ({severity.symptom_flag})",
                list(severity.evidence[:3]),
                "Severity is unassessable when the laboratory burden or severe clinical outcomes are not documented." if severity.grade == "unassessable" else None,
            ),
            self._question(
                "Is any rechallenge documented, and was it positive?",
                top_exposure.rechallenge_status if top_exposure is not None else "unknown",
                [ClinicalEvidenceQuote(
                    claim="rechallenge status",
                    quote=top_exposure.identity.evidence_quote,
                    source_section=top_exposure.identity.source_section,
                    source_kind="patient_record" if top_exposure.identity.evidence_quote else "missing",
                )] if top_exposure is not None else [],
                "Absent evidence is treated as unknown, never as negative, and rechallenge is never recommended." if top_exposure is None or top_exposure.rechallenge_status == "unknown" else None,
            ),
            self._question(
                "Is the suspect-drug identity reliable enough for adjudication?",
                (top_exposure.identity.accepted_identity or "unresolved") if top_exposure is not None else "unresolved",
                [ClinicalEvidenceQuote(
                    claim="identity resolution",
                    quote=top_exposure.identity.identity_reason,
                    source_section=top_exposure.identity.source_section,
                    source_kind="patient_record" if top_exposure and top_exposure.identity.identity_reason else "missing",
                )] if top_exposure is not None else [],
                "Brand, salt, combination, historical, negated, allergy, or family-history mentions remain unresolved unless locally validated." if top_exposure is None or top_exposure.identity.accepted_identity is None else None,
            ),
            self._question(
                "What prior LiverTox likelihood supports the exposure?",
                (top_exposure.livertox_likelihood or "unknown") if top_exposure is not None else "unknown",
                [ClinicalEvidenceQuote(
                    claim="LiverTox likelihood",
                    quote=top_exposure.livertox_likelihood,
                    source_kind="livertox" if top_exposure and top_exposure.livertox_likelihood else "missing",
                )] if top_exposure is not None else [],
                "Sparse, unknown, or E-class likelihood grades are treated conservatively and do not upgrade patient-level causality." if top_exposure is None or not top_exposure.livertox_likelihood else None,
            ),
            self._question(
                "What is the supportive RUCAM conclusion?",
                top_exposure.rucam.category if top_exposure and top_exposure.rucam else "not_assessable",
                [
                    ClinicalEvidenceQuote(
                        claim=component.component,
                        quote=component.evidence_quote,
                        event_date=component.evidence_date,
                        source_kind="patient_record" if component.evidence_quote else "missing",
                    )
                    for component in (top_exposure.rucam.components if top_exposure and top_exposure.rucam else [])
                ][:4],
                "RUCAM remains supportive only and is not assessable when criteria-level evidence is missing." if top_exposure is None or top_exposure.rucam is None or top_exposure.rucam.total_score is None else None,
            ),
            self._question(
                "What is the overall DILIN-like causality category?",
                top_exposure.causality.category if top_exposure and top_exposure.causality else "unassessable",
                [ClinicalEvidenceQuote(
                    claim="overall causality rationale",
                    quote="; ".join(top_exposure.causality.rationale),
                    source_kind="calculated",
                )] if top_exposure and top_exposure.causality else [],
                "Overall causality stays limited when timing, identity, phenotype match, source quality, or competing-cause exclusion are incomplete." if top_exposure is None or not top_exposure.causality or top_exposure.causality.category in {"possible", "unlikely", "unassessable"} else None,
            ),
        ]
        return questions

    @staticmethod
    def _question(
        question: str,
        answer: str,
        evidence: list[ClinicalEvidenceQuote | None],
        missing_data_statement: str | None,
    ) -> DiliAcceptanceQuestion:
        return DiliAcceptanceQuestion(
            question=question,
            answer=answer,
            supporting_evidence=[item for item in evidence if item is not None],
            missing_data_statement=missing_data_statement,
        )

    @classmethod
    def _primary_suspect(
        cls,
        exposures: list[DrugExposureAssessment],
        competing_causes_complete: bool,
    ) -> DrugExposureAssessment | None:
        if not exposures:
            return None
        return max(
            exposures,
            key=lambda exposure: cls._suspect_rank(exposure, competing_causes_complete),
        )

    @staticmethod
    def _suspect_rank(
        exposure: DrugExposureAssessment,
        competing_causes_complete: bool,
    ) -> tuple[int, int, int, int, int, int, str]:
        identity_score = 1 if exposure.identity.accepted_identity else 0
        temporal_score = (
            1
            if exposure.causality
            and exposure.causality.temporal_compatibility == "compatible"
            else 0
        )
        rucam_score = DiliEvidenceBuilder._rucam_rank(exposure)
        livertox_score = DiliEvidenceBuilder._livertox_rank(exposure.livertox_likelihood)
        dechallenge_score = DiliEvidenceBuilder._dechallenge_rank(exposure)
        competing_score = 1 if competing_causes_complete else 0
        stable_name = exposure.drug_name.lower()
        return (
            identity_score,
            temporal_score,
            rucam_score,
            livertox_score,
            dechallenge_score,
            competing_score,
            stable_name,
        )

    @staticmethod
    def _rucam_rank(exposure: DrugExposureAssessment) -> int:
        if exposure.rucam is None:
            return 0
        if exposure.rucam.total_score is not None:
            return exposure.rucam.total_score
        category = (exposure.rucam.category or "").lower()
        return {
            "excluded": -2,
            "unlikely": -1,
            "possible": 2,
            "probable": 4,
            "highly_probable": 6,
        }.get(category, 0)

    @staticmethod
    def _livertox_rank(likelihood: str | None) -> int:
        normalized = (likelihood or "").upper()
        return {
            "A": 5,
            "B": 4,
            "C": 3,
            "D": 2,
            "E": 0,
            "E*": 0,
            "T": 1,
            "T*": 1,
        }.get(normalized, 0)

    @staticmethod
    def _dechallenge_rank(exposure: DrugExposureAssessment) -> int:
        if exposure.rechallenge_status == "positive":
            return 3
        detail = exposure.causality.dechallenge_rechallenge if exposure.causality else ""
        if "resolved_to_baseline" in detail:
            return 2
        if "improving_after_stop" in detail:
            return 1
        return 0

    @staticmethod
    def render(bundle: DiliEvidenceBundle) -> str:
        pattern = bundle.patterns[0] if bundle.patterns else None
        lines = [
            "# Structured DILI causality dossier",
            "",
            "## 1. Case completeness and missing data",
            f"- Missing: {', '.join(bundle.completeness.missing_fields) or 'none documented'}",
            "## 2. Liver injury pattern and severity",
            (
                f"- Pattern: {pattern.pattern}; R={pattern.r_ratio if pattern and pattern.r_ratio is not None else 'not assessable'} "
                f"(ALT {pattern.alt if pattern else 'NA'}/{pattern.alt_uln if pattern else 'NA'} ULN; "
                f"ALP {pattern.alp if pattern else 'NA'}/{pattern.alp_uln if pattern else 'NA'} ULN)"
            ),
            f"- Severity: {bundle.severity.grade} ({bundle.severity.symptom_flag})",
            "## 3. Timeline summary",
            f"- First abnormal test: {bundle.timeline.first_abnormal_liver_test_date or 'missing'}",
            f"- First symptom: {bundle.timeline.first_symptom_date or 'missing'}",
            f"- Jaundice/bilirubin timing: {bundle.timeline.jaundice_or_bilirubin_rise_date or 'missing'}",
            f"- Dechallenge: {bundle.timeline.dechallenge_status}",
            "## 4. Competing-cause assessment",
        ]
        lines.extend(f"- {item.cause}: {item.status}" for item in bundle.differential.causes)
        lines.extend(["## 5. Drug exposure table", ""])
        for exposure in bundle.exposures:
            lines.append(
                f"- {exposure.drug_name}: identity={exposure.identity.accepted_identity or 'unresolved'}; "
                f"start={exposure.start_date or 'missing'}; stop={exposure.stop_date or 'missing'}; "
                f"rechallenge={exposure.rechallenge_status}"
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
                    f"evidence={component.evidence_quote or 'missing'}; date={component.evidence_date or 'missing'}"
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
                "## 14. Acceptance questions",
                "- Manual hepatology review required.",
            ]
        )
        for question in bundle.acceptance_questions:
            lines.append(f"- {question.question} -> {question.answer}")
            if question.missing_data_statement:
                lines.append(f"  Missing-data note: {question.missing_data_statement}")
        return "\n".join(lines)
