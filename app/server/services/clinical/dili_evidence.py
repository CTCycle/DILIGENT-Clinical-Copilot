from __future__ import annotations

import re
from collections.abc import Sequence

from domain.clinical.dili import (
    ClinicalDataCompleteness,
    ClinicalEvidenceQuote,
    DiliAcceptanceQuestion,
    DiliEvidenceBundle,
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


###############################################################################
class DiliEvidenceBuilder:
    _COMPETING_CAUSES_EXCLUDED_PHRASES = (
        "no competing causes",
        "competing causes have been excluded",
        "competing causes were excluded",
        "competing causes have been effectively excluded",
        "competing causes were effectively excluded",
        "all competing causes were excluded",
        "alternative causes have been excluded",
        "alternative causes were excluded",
        "viral hepatitis ruled out",
        "viral hepatitis was ruled out",
    )
    _HYS_LAW_ASSERTION_PHRASES = (
        "hy's law pattern",
        "meets hy's law",
        "meets the criteria for hy's law",
        "hy's law criteria are met",
    )
    _DEFINITIVE_CAUSALITY_PHRASES = (
        "definitively caused",
        "causal relationship is certain",
        "absolutely contraindicated",
        "contraindicated for life",
        "lifelong avoidance",
        "strict lifelong",
    )

    # -------------------------------------------------------------------------
    @classmethod
    def audit_generated_narrative(
        cls,
        *,
        clinical_narrative: str | None,
        bundle: DiliEvidenceBundle,
    ) -> list[dict[str, str]]:
        """Return blocking issues for prose that contradicts structured evidence."""

        text = re.sub(r"\s+", " ", str(clinical_narrative or "")).strip().casefold()
        text = text.replace("’", "'").replace("‘", "'")
        if not text:
            return []

        issues: list[dict[str, str]] = []
        if not bundle.differential.all_major_causes_excluded and cls._contains_any(
            text, cls._COMPETING_CAUSES_EXCLUDED_PHRASES
        ):
            issues.append(
                {
                    "code": "clinical_narrative_contradicts_competing_causes",
                    "message": (
                        "Generated narrative states that competing causes were excluded "
                        "although the structured DILI differential remains unresolved."
                    ),
                }
            )

        if bundle.hys_law.status != "meets_criteria" and cls._contains_any(
            text, cls._HYS_LAW_ASSERTION_PHRASES
        ):
            issues.append(
                {
                    "code": "clinical_narrative_overstates_hys_law",
                    "message": (
                        "Generated narrative asserts a Hy's Law pattern although the "
                        f"structured status is {bundle.hys_law.status}."
                    ),
                }
            )

        causality_is_limited = any(
            exposure.causality is None
            or exposure.causality.category in {"possible", "unlikely", "unassessable"}
            or exposure.rucam is None
            or exposure.rucam.total_score is None
            for exposure in bundle.exposures
        )
        if causality_is_limited and cls._contains_unsupported_definitive_language(text):
            issues.append(
                {
                    "code": "clinical_narrative_overstates_causality",
                    "message": (
                        "Generated narrative uses definitive diagnosis or absolute "
                        "avoidance language while structured causality remains limited."
                    ),
                }
            )

        if causality_is_limited and re.search(
            r"\bthis patient\b.{0,100}\blikelihood score\s+[a-e]\b", text
        ):
            issues.append(
                {
                    "code": "clinical_narrative_conflates_livertox_likelihood",
                    "message": (
                        "Generated narrative applies a drug-level LiverTox likelihood "
                        "grade as if it were patient-level causality."
                    ),
                }
            )

        return issues

    # -------------------------------------------------------------------------
    @staticmethod
    def _contains_any(text: str, phrases: Sequence[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    # -------------------------------------------------------------------------
    @classmethod
    def _contains_unsupported_definitive_language(cls, text: str) -> bool:
        if cls._contains_any(text, cls._DEFINITIVE_CAUSALITY_PHRASES):
            return True
        for match in re.finditer(r"\b(?:confident|definitive) diagnosis\b", text):
            preceding_text = text[max(0, match.start() - 80) : match.start()]
            following_text = text[match.end() : match.end() + 80]
            if re.search(
                r"\b(?:not|cannot|can't|no|without|unable|uncertain|unassessable)\b",
                f"{preceding_text} {following_text}",
            ):
                continue
            return True
        return False

    # -------------------------------------------------------------------------
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
            normalize_drug_query_name(item.drug_name): item
            for item in rucam_bundle.entries
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
                if event.drug_name == exposure.drug_name
                and event.event_type == "dose_change"
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
        evidence = [
            quote
            for question in acceptance_questions
            for quote in question.supporting_evidence
        ][:12]
        return DiliEvidenceBundle(
            completeness=ClinicalDataCompleteness(
                complete_fields=["drug_exposures", "laboratory_timeline"]
                if drugs.entries and labs.entries
                else [],
                missing_fields=sorted(set(missing)),
                manual_review_required=True,
                reasons=[
                    "DILI is a diagnosis of exclusion.",
                    "Clinical hepatology review required.",
                ],
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

    # -------------------------------------------------------------------------
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
        top_exposure = exposures[0] if exposures else None
        questions: list[DiliAcceptanceQuestion] = [
            self._question(
                "What is the latency from first compatible exposure to first liver injury signal?",
                timeline.first_abnormal_liver_test_date or "missing",
                [
                    event.evidence
                    for event in timeline.events
                    if event.event_type in {"drug_start", "abnormal_liver_test"}
                    and event.evidence
                ][:3],
                "Latency remains uncertain because exposure start or first abnormal liver-test timing is missing."
                if not timeline.first_abnormal_liver_test_date
                else None,
            ),
            self._question(
                "Did the injury improve, persist, or worsen after discontinuation?",
                timeline.dechallenge_status,
                [
                    event.evidence
                    for event in timeline.events
                    if event.event_type in {"drug_stop", "abnormal_liver_test"}
                    and event.evidence
                ][:3],
                "Dechallenge cannot be graded honestly when follow-up after discontinuation is incomplete."
                if timeline.dechallenge_status
                in {"no_follow_up", "insufficient_interval"}
                else None,
            ),
            self._question(
                "What is the liver injury pattern at the first qualifying episode?",
                first_pattern.pattern if first_pattern is not None else "indeterminate",
                list(first_pattern.evidence[:2]) if first_pattern is not None else [],
                "Pattern is indeterminate when paired ALT and ALP values with ULN are unavailable."
                if first_pattern is None or first_pattern.pattern == "indeterminate"
                else None,
            ),
            self._question(
                "Which clinically conservative phenotype candidates are supported?",
                ", ".join(phenotype.candidates) or "none identified",
                list((first_pattern.evidence[:1] if first_pattern is not None else [])),
                "Phenotype remains limited by missing biopsy, imaging, autoimmune markers, or long follow-up."
                if phenotype.missing_data
                else None,
            ),
            self._question(
                "Are mandatory alternative causes excluded?",
                "yes" if differential.all_major_causes_excluded else "no",
                [item.evidence[0] for item in differential.causes if item.evidence][:4],
                "One or more mandatory competing causes are unresolved or not excluded."
                if not differential.all_major_causes_excluded
                else None,
            ),
            self._question(
                "Does the episode satisfy Hy's Law requirements?",
                hys_law.status,
                list(hys_law.evidence[:4]),
                "Hy's Law is not assessable or only possible when same-episode timing, cholestasis exclusion, alternatives, or exposure compatibility remain incomplete."
                if hys_law.status != "meets_criteria"
                else None,
            ),
            self._question(
                "What is the severity grade?",
                f"{severity.grade} ({severity.symptom_flag})",
                list(severity.evidence[:3]),
                "Severity is unassessable when the laboratory burden or severe clinical outcomes are not documented."
                if severity.grade == "unassessable"
                else None,
            ),
            self._question(
                "Is any rechallenge documented, and was it positive?",
                top_exposure.rechallenge_status
                if top_exposure is not None
                else "unknown",
                [
                    ClinicalEvidenceQuote(
                        claim="rechallenge status",
                        quote=top_exposure.identity.evidence_quote,
                        source_section=top_exposure.identity.source_section,
                        source_kind="patient_record"
                        if top_exposure.identity.evidence_quote
                        else "missing",
                    )
                ]
                if top_exposure is not None
                else [],
                "Absent evidence is treated as unknown, never as negative, and rechallenge is never recommended."
                if top_exposure is None or top_exposure.rechallenge_status == "unknown"
                else None,
            ),
            self._question(
                "Is the suspect-drug identity reliable enough for adjudication?",
                (top_exposure.identity.accepted_identity or "unresolved")
                if top_exposure is not None
                else "unresolved",
                [
                    ClinicalEvidenceQuote(
                        claim="identity resolution",
                        quote=top_exposure.identity.identity_reason,
                        source_section=top_exposure.identity.source_section,
                        source_kind="patient_record"
                        if top_exposure and top_exposure.identity.identity_reason
                        else "missing",
                    )
                ]
                if top_exposure is not None
                else [],
                "Brand, salt, combination, historical, negated, allergy, or family-history mentions remain unresolved unless locally validated."
                if top_exposure is None
                or top_exposure.identity.accepted_identity is None
                else None,
            ),
            self._question(
                "What prior LiverTox likelihood supports the exposure?",
                (top_exposure.livertox_likelihood or "unknown")
                if top_exposure is not None
                else "unknown",
                [
                    ClinicalEvidenceQuote(
                        claim="LiverTox likelihood",
                        quote=top_exposure.livertox_likelihood,
                        source_kind="livertox"
                        if top_exposure and top_exposure.livertox_likelihood
                        else "missing",
                    )
                ]
                if top_exposure is not None
                else [],
                "Sparse, unknown, or E-class likelihood grades are treated conservatively and do not upgrade patient-level causality."
                if top_exposure is None or not top_exposure.livertox_likelihood
                else None,
            ),
            self._question(
                "What is the supportive RUCAM conclusion?",
                top_exposure.rucam.category
                if top_exposure and top_exposure.rucam
                else "not_assessable",
                [
                    ClinicalEvidenceQuote(
                        claim=component.component,
                        quote=component.evidence_quote,
                        event_date=component.evidence_date,
                        source_kind="patient_record"
                        if component.evidence_quote
                        else "missing",
                    )
                    for component in (
                        top_exposure.rucam.components
                        if top_exposure and top_exposure.rucam
                        else []
                    )
                ][:4],
                "RUCAM remains supportive only and is not assessable when criteria-level evidence is missing."
                if top_exposure is None
                or top_exposure.rucam is None
                or top_exposure.rucam.total_score is None
                else None,
            ),
            self._question(
                "What is the overall DILIN-like causality category?",
                top_exposure.causality.category
                if top_exposure and top_exposure.causality
                else "unassessable",
                [
                    ClinicalEvidenceQuote(
                        claim="overall causality rationale",
                        quote="; ".join(top_exposure.causality.rationale),
                        source_kind="calculated",
                    )
                ]
                if top_exposure and top_exposure.causality
                else [],
                "Overall causality stays limited when timing, identity, phenotype match, source quality, or competing-cause exclusion are incomplete."
                if top_exposure is None
                or not top_exposure.causality
                or top_exposure.causality.category
                in {"possible", "unlikely", "unassessable"}
                else None,
            ),
        ]
        return questions

    # -------------------------------------------------------------------------
    @staticmethod
    def _question(
        question: str,
        answer: str,
        evidence: Sequence[ClinicalEvidenceQuote | None],
        missing_data_statement: str | None,
    ) -> DiliAcceptanceQuestion:
        return DiliAcceptanceQuestion(
            question=question,
            answer=answer,
            supporting_evidence=[item for item in evidence if item is not None],
            missing_data_statement=missing_data_statement,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def compose_clinical_report(
        *,
        clinical_narrative: str | None,
        generated_report: str | None,
        bundle: DiliEvidenceBundle,
        fallback_text: str,
    ) -> str:
        body = (clinical_narrative or "").strip()
        if not body:
            body = (generated_report or "").strip()
        if not body:
            body = fallback_text.strip()
        user_summary = DiliEvidenceBuilder.render_user_summary(bundle)
        if user_summary:
            return f"{user_summary}\n\n---\n\n{body}".strip()
        return body

    # -------------------------------------------------------------------------
    @staticmethod
    def render_user_summary(bundle: DiliEvidenceBundle) -> str:
        pattern = bundle.patterns[0] if bundle.patterns else None
        lines = [
            "## DILI adjudication summary",
            "",
            "- DILI remains an exclusion-based assessment; structured adjudication data are retained for audit and hepatology review.",
            (
                f"- Liver injury pattern: {pattern.pattern if pattern else 'indeterminate'}; "
                f"R ratio: {pattern.r_ratio if pattern and pattern.r_ratio is not None else 'not assessable'}."
            ),
            f"- Severity: {bundle.severity.grade} ({bundle.severity.symptom_flag}).",
            f"- Hy's Law status: {bundle.hys_law.status}; this is a risk signal, not a diagnosis.",
        ]
        if bundle.exposures:
            lines.extend(["", "### Per-drug causality"])
            for exposure in bundle.exposures:
                category = (
                    exposure.causality.category
                    if exposure.causality is not None
                    else "unassessable"
                )
                identity = exposure.identity.accepted_identity or "identity unresolved"
                rucam = (
                    exposure.rucam.category
                    if exposure.rucam is not None
                    else "RUCAM not assessable"
                )
                lines.append(
                    f"- {exposure.drug_name}: {category}; {identity}; supportive RUCAM: {rucam}."
                )
        missing_groups = DiliEvidenceBuilder._group_missing_fields(
            bundle.completeness.missing_fields
        )
        if missing_groups:
            lines.extend(["", "### Clinically relevant missing data"])
            for group, items in missing_groups.items():
                lines.append(f"- {group}: {'; '.join(items)}.")
        lines.extend(
            [
                "",
                "### Clinical limitations",
                "- RUCAM is structured support and is not dispositive.",
                "- Missing or unresolved competing causes prevent definitive attribution.",
                "- Manual hepatology review is required before clinical reuse.",
            ]
        )
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    @staticmethod
    def _group_missing_fields(fields: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for raw_field in fields:
            group, label = DiliEvidenceBuilder._format_missing_field(raw_field)
            if not label:
                continue
            grouped.setdefault(group, [])
            if label not in grouped[group]:
                grouped[group].append(label)
        return grouped

    # -------------------------------------------------------------------------
    @staticmethod
    def _format_missing_field(raw_field: str) -> tuple[str, str]:
        field = str(raw_field or "").strip()
        if not field:
            return "", ""
        lower_field = field.casefold()
        if ":" in field:
            drug_name, key = [part.strip() for part in field.split(":", 1)]
            normalized_key = key.replace(" ", "_").replace("-", "_").casefold()
            compact_key = normalized_key.replace("_", "")
            if "drug_start_date" in normalized_key or "drugstartdate" in compact_key:
                return "Exposure timing", f"{drug_name}: start date not documented"
            if "drug_stop_date" in normalized_key or "drugstopdate" in compact_key:
                return "Exposure timing", f"{drug_name}: stop date not documented"
            return "Exposure timing", (
                f"{drug_name}: {DiliEvidenceBuilder._humanize_key(key)} not documented"
            )
        if "paired alt and alp" in lower_field:
            return (
                "Liver chemistry timing",
                "paired ALT and ALP values with ULN are unavailable",
            )
        timeline_labels = {
            "first_abnormal_liver_test_date": "first abnormal liver-test date not documented",
            "first_symptom_date": "first symptom date not documented",
            "jaundice_or_bilirubin_timing": "jaundice or bilirubin timing not documented",
            "jaundice_or_bilirubin_rise_date": "jaundice or bilirubin rise date not documented",
        }
        if lower_field in timeline_labels:
            return "Liver chemistry timing", timeline_labels[lower_field]
        competing_causes = {
            "viral_hepatitis_a_b_c_d_e": "viral hepatitis A-E not excluded",
            "viralhepatitisabcde": "viral hepatitis A-E not excluded",
            "ebv_cmv_hsv": "EBV, CMV, or HSV not excluded",
            "ebvcmvhsv": "EBV, CMV, or HSV not excluded",
            "autoimmune_hepatitis": "autoimmune hepatitis not excluded",
            "autoimmunehepatitis": "autoimmune hepatitis not excluded",
            "alcoholic_hepatitis": "alcoholic hepatitis not excluded",
            "alcoholichepatitis": "alcoholic hepatitis not excluded",
            "masld_mash_nash": "MASLD, MASH, or NASH not excluded",
            "masldmashnash": "MASLD, MASH, or NASH not excluded",
            "biliary_obstruction_gallstones": "biliary obstruction or gallstones not excluded",
            "biliaryobstructiongallstones": "biliary obstruction or gallstones not excluded",
            "ischemic_hypoxic": "ischemic or hypoxic hepatitis not excluded",
            "ischemichypoxic": "ischemic or hypoxic hepatitis not excluded",
            "sepsis_shock_cardiac_failure": "sepsis, shock, or cardiac failure not excluded",
            "sepsisshockcardiacfailure": "sepsis, shock, or cardiac failure not excluded",
            "overdose_or_toxin": "overdose or toxin exposure not excluded",
            "overdoseortoxin": "overdose or toxin exposure not excluded",
            "supplement_otc_recreational_occupational": "supplement, OTC, recreational, or occupational exposure not excluded",
            "supplementotcrecreationaloccupational": "supplement, OTC, recreational, or occupational exposure not excluded",
            "preexisting_chronic_liver_disease": "pre-existing chronic liver disease not excluded",
            "preexistingchronicliverdisease": "pre-existing chronic liver disease not excluded",
        }
        if lower_field in competing_causes:
            return "Alternative cause exclusion", competing_causes[lower_field]
        return "Clinical context and follow-up", (
            f"{DiliEvidenceBuilder._humanize_key(field)} not documented"
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _humanize_key(value: str) -> str:
        compact = str(value or "").strip().replace("_", " ").replace("-", " ")
        compact = " ".join(compact.split())
        replacements = {
            "drug start date": "start date",
            "drug stop date": "stop date",
            "uln": "ULN",
            "alt": "ALT",
            "alp": "ALP",
            "ebv": "EBV",
            "cmv": "CMV",
            "hsv": "HSV",
            "otc": "OTC",
        }
        words = []
        for word in compact.split(" "):
            words.append(replacements.get(word.casefold(), word))
        return " ".join(words)

    # -------------------------------------------------------------------------
    @staticmethod
    def render(bundle: DiliEvidenceBundle) -> str:
        pattern = bundle.patterns[0] if bundle.patterns else None
        pattern_name = pattern.pattern if pattern is not None else "indeterminate"
        lines = [
            "# Structured DILI causality dossier",
            "",
            "## 1. Case completeness and missing data",
            "## 2. Liver injury pattern and severity",
            (
                f"- Pattern: {pattern_name}; R={pattern.r_ratio if pattern and pattern.r_ratio is not None else 'not assessable'} "
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
        missing_groups = DiliEvidenceBuilder._group_missing_fields(
            bundle.completeness.missing_fields
        )
        if missing_groups:
            insertion_index = lines.index("## 2. Liver injury pattern and severity")
            missing_lines = [
                f"- {group}: {'; '.join(items)}."
                for group, items in missing_groups.items()
            ]
            lines[insertion_index:insertion_index] = missing_lines
        else:
            insertion_index = lines.index("## 2. Liver injury pattern and severity")
            lines.insert(insertion_index, "- Missing data: none documented")
        lines.extend(
            f"- {item.cause}: {item.status}" for item in bundle.differential.causes
        )
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
                lines.append(
                    f"- Total: {item.rucam.total_score}; category: {item.rucam.category}"
                )
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
