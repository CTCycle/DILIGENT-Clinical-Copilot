from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal, cast

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    DrugRucamAssessment,
    HepatotoxicityPatternScore,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    RucamComponentAssessment,
)
from domain.clinical.rucam import (
    RucamAnchor,
    RucamDataSufficiency,
    RucamSourceReportedScore,
)
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.clinical.report_language import phrase, resolve_report_language
from services.text.normalization import normalize_drug_query_name


###############################################################################
def _compile_terms_regex(category: str) -> re.Pattern[str]:
    values = get_reference_catalog_snapshot().values(
        "dili_assessment",
        category,
        key="default",
    )
    terms = [re.escape(value.strip()) for value in values if value.strip()]
    if terms:
        return re.compile(r"\b(" + "|".join(terms) + r")\b", re.IGNORECASE)
    return re.compile(r"$^")


###############################################################################
def _alcohol_re() -> re.Pattern[str]:
    return _compile_terms_regex("rucam_alcohol_terms")


###############################################################################
def _exclusion_re() -> re.Pattern[str]:
    return _compile_terms_regex("rucam_exclusion_terms")


RUCAM_SCORE_RE = re.compile(
    r"\brucam\b\s*(?:score)?\s*[:=]?\s*(-?\d{1,2})", re.IGNORECASE
)
NON_PATIENT_RUCAM_CONTEXT_TERMS = (
    "livertox",
    "monograph",
    "representative case",
    "literature",
    "publication",
    "paper",
    "study",
    "trial",
)

RucamInjuryType = Literal[
    "hepatocellular",
    "cholestatic",
    "mixed",
    "indeterminate",
]
RucamCausalityCategory = Literal[
    "excluded",
    "unlikely",
    "indeterminate",
    "possible",
    "probable",
    "highly probable",
    "not assessable",
]


###############################################################################
class RucamScoreEstimator:
    """Preserve patient-reported RUCAM and build a non-scoring evidence checklist.

    DILIGENT does not synthesize a formal updated-RUCAM total from partially
    structured data. A numerical RUCAM value is accepted only when it is
    explicitly present in the current patient record and can be associated with
    the assessed drug. Literature and LiverTox material are never patient-score
    sources.
    """

    # -------------------------------------------------------------------------
    def resolve_provided_rucam_score(
        self,
        laboratory_history_text: str,
        *,
        drug_name: str | None = None,
        require_drug_attribution: bool = False,
    ) -> RucamSourceReportedScore | None:
        if not laboratory_history_text:
            return None
        for match in RUCAM_SCORE_RE.finditer(laboratory_history_text):
            start = max(0, match.start() - 220)
            end = min(len(laboratory_history_text), match.end() + 220)
            window = laboratory_history_text[start:end]
            lowered = window.casefold()
            if any(term in lowered for term in NON_PATIENT_RUCAM_CONTEXT_TERMS):
                continue
            if require_drug_attribution:
                if not drug_name or drug_name.casefold() not in window.casefold():
                    continue
            score = max(-10, min(14, int(match.group(1))))
            category = None
            for candidate in (
                "highly probable",
                "probable",
                "possible",
                "unlikely",
                "excluded",
            ):
                if candidate in lowered:
                    category = candidate
                    break
            return RucamSourceReportedScore(
                score=score,
                causality_category=category,
                source_name="patient_laboratory_history",
                evidence=window.strip(),
            )
        return None

    # -------------------------------------------------------------------------
    def has_sufficient_rucam_inputs(
        self,
        *,
        injury_type: str,
        anchor: RucamAnchor,
        drug: DrugEntry,
        lab_timeline: PatientLabTimeline,
        payload: PatientData,
        disease_context: PatientDiseaseContext,
    ) -> bool:
        sufficiency = self.evaluate_data_sufficiency(
            injury_type=injury_type,
            anchor=anchor,
            drug=drug,
            lab_timeline=lab_timeline,
            payload=payload,
            disease_context=disease_context,
        )
        return sufficiency.sufficient

    # -------------------------------------------------------------------------
    def estimate(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score: HepatotoxicityPatternScore,
        resolved_drugs: dict[str, dict[str, Any]] | None = None,
        report_language: str = "en",
    ) -> PatientRucamAssessmentBundle:
        resolved_mapping = resolved_drugs or {}
        all_drugs = [*analysis_drugs.entries, *anamnesis_drugs.entries]
        anchor = self.select_pattern_anchor(payload=payload, lab_timeline=lab_timeline)
        injury_type = self.resolve_injury_type(
            pattern_score=pattern_score,
            anchor=anchor,
        )
        language = resolve_report_language(report_language)
        unique_names = {
            normalize_drug_query_name(drug.name)
            for drug in all_drugs
            if normalize_drug_query_name(drug.name)
        }
        require_drug_attribution = len(unique_names) > 1

        entries: list[DrugRucamAssessment] = []
        seen: set[str] = set()
        for drug in all_drugs:
            name = (drug.name or "").strip()
            if not name:
                continue
            key = normalize_drug_query_name(name)
            if not key or key in seen:
                continue
            seen.add(key)
            resolved = resolved_mapping.get(key, {})
            entries.append(
                self.estimate_for_drug(
                    payload=payload,
                    drug=drug,
                    all_drugs=all_drugs,
                    disease_context=disease_context,
                    lab_timeline=lab_timeline,
                    onset_context=onset_context,
                    injury_type=injury_type,
                    anchor=anchor,
                    resolved_item=resolved if isinstance(resolved, dict) else {},
                    report_language=language,
                    require_drug_attribution=require_drug_attribution,
                )
            )
        return PatientRucamAssessmentBundle(entries=entries)

    # -------------------------------------------------------------------------
    @staticmethod
    def try_parse_date(value: str | None) -> date | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        if not cleaned:
            return None
        normalized = cleaned.replace("/", "-").replace(".", "-")
        try:
            return date.fromisoformat(normalized)
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(normalized, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def marker_multiple(entry: ClinicalLabEntry | None) -> float | None:
        if entry is None or entry.value is None:
            return None
        if entry.upper_limit_normal and entry.upper_limit_normal > 0:
            return entry.value / entry.upper_limit_normal
        return None

    # -------------------------------------------------------------------------
    def select_pattern_anchor(
        self, *, payload: PatientData, lab_timeline: PatientLabTimeline
    ) -> RucamAnchor:
        grouped: dict[date, dict[str, ClinicalLabEntry]] = {}
        for entry in lab_timeline.entries:
            marker = entry.marker_name.upper()
            if marker not in {"ALT", "ALP"}:
                continue
            parsed_date = self.try_parse_date(entry.sample_date)
            if parsed_date is None:
                continue
            bucket = grouped.setdefault(parsed_date, {})
            current = bucket.get(marker)
            current_multiple = self.marker_multiple(current)
            candidate_multiple = self.marker_multiple(entry)
            if current is None or (
                candidate_multiple is not None
                and (current_multiple is None or candidate_multiple > current_multiple)
            ):
                bucket[marker] = entry

        for sample_date in sorted(grouped.keys()):
            bucket = grouped[sample_date]
            alt = bucket.get("ALT")
            alp = bucket.get("ALP")
            alt_mult = self.marker_multiple(alt)
            alp_mult = self.marker_multiple(alp)
            qualifies = bool(
                (alt_mult is not None and alt_mult >= 5.0)
                or (alp_mult is not None and alp_mult >= 2.0)
            )
            if qualifies:
                return RucamAnchor(
                    onset_date=sample_date,
                    used_alt=alt.value if alt else None,
                    used_alt_uln=alt.upper_limit_normal if alt else None,
                    used_alp=alp.value if alp else None,
                    used_alp_uln=alp.upper_limit_normal if alp else None,
                    rationale=(
                        "Earliest ALT >=5x ULN or ALP >=2x ULN laboratory anchor "
                        f"selected on {sample_date.isoformat()}."
                    ),
                    source="qualifying_lab",
                    is_score_eligible=True,
                )

        return RucamAnchor(
            onset_date=payload.visit_date,
            used_alt=None,
            used_alt_uln=None,
            used_alp=None,
            used_alp_uln=None,
            rationale="No qualifying RUCAM liver-injury anchor; visit-date proxy used for context only.",
            source="visit_proxy",
            is_score_eligible=False,
        )

    # -------------------------------------------------------------------------
    def resolve_injury_type(
        self, *, pattern_score: HepatotoxicityPatternScore, anchor: RucamAnchor
    ) -> RucamInjuryType:
        _ = anchor
        classification = (
            (pattern_score.classification or "indeterminate").strip().lower()
        )
        if classification == "mixed":
            return "mixed"
        if classification in {"hepatocellular", "cholestatic"}:
            return cast(RucamInjuryType, classification)
        return "indeterminate"

    # -------------------------------------------------------------------------
    def evaluate_data_sufficiency(
        self,
        *,
        injury_type: str,
        anchor: RucamAnchor,
        drug: DrugEntry,
        lab_timeline: PatientLabTimeline,
        payload: PatientData,
        disease_context: PatientDiseaseContext,
    ) -> RucamDataSufficiency:
        reasons: list[str] = []
        if injury_type == "indeterminate":
            reasons.append("injury pattern indeterminate")
        if not anchor.is_score_eligible:
            reasons.append("qualifying liver-injury anchor unavailable")
        if drug.therapy_start_date is None:
            reasons.append("drug start timing unavailable")
        if not lab_timeline.entries:
            reasons.append("no laboratory timeline entries")
        has_alternative_context = bool((payload.anamnesis or "").strip()) and (
            len(_exclusion_re().findall(payload.anamnesis or "")) > 0
            or len(disease_context.entries) > 0
        )
        if not has_alternative_context:
            reasons.append("alternative-cause assessment evidence unavailable")
        return RucamDataSufficiency(sufficient=not reasons, blocking_reasons=reasons)

    # -------------------------------------------------------------------------
    def build_not_calculated_assessment(
        self,
        *,
        drug: DrugEntry,
        injury_type: str,
        reasons: list[str],
        report_language: str,
        components: list[RucamComponentAssessment] | None = None,
    ) -> DrugRucamAssessment:
        limitations = reasons or [phrase("rucam_insufficient_data", report_language)]
        return DrugRucamAssessment(
            drug_name=drug.name,
            injury_type_for_rucam=cast(RucamInjuryType, injury_type),
            total_score=None,
            causality_category="not assessable",
            confidence="low",
            estimated=False,
            components=components
            or [
                RucamComponentAssessment(
                    component_key="rucam",
                    label="Updated RUCAM evidence checklist",
                    score=0,
                    status="not_assessable",
                    rationale="; ".join(limitations),
                )
            ],
            limitations=limitations,
            summary=(
                "Updated RUCAM numerical scoring is not automated from partially "
                "structured evidence. Review the captured components and score the "
                "validated instrument clinically if required."
            ),
            calculation_method="not_calculated",
            score_source=None,
            data_sufficient=False,
        )

    # -------------------------------------------------------------------------
    def build_source_reported_assessment(
        self,
        *,
        drug: DrugEntry,
        injury_type: str,
        source: RucamSourceReportedScore,
        report_language: str,
    ) -> DrugRucamAssessment:
        category = source.causality_category or self.resolve_causality_bucket(
            source.score
        )
        return DrugRucamAssessment(
            drug_name=drug.name,
            injury_type_for_rucam=cast(RucamInjuryType, injury_type),
            total_score=source.score,
            causality_category=cast(RucamCausalityCategory, category),
            confidence="moderate",
            estimated=False,
            components=[
                RucamComponentAssessment(
                    component_key="source_reported",
                    label="Patient-record RUCAM",
                    score=source.score,
                    status="scored",
                    evidence=source.evidence,
                    evidence_date=drug.suspension_date or drug.therapy_start_date,
                    rationale=phrase("rucam_source_reported", report_language),
                )
            ],
            limitations=[
                "The score is preserved from the current patient record and was not independently recalculated by DILIGENT."
            ],
            summary=phrase("rucam_source_reported", report_language),
            calculation_method="source_reported",
            score_source=source.source_name,
            data_sufficient=True,
        )

    # -------------------------------------------------------------------------
    def estimate_for_drug(
        self,
        *,
        payload: PatientData,
        drug: DrugEntry,
        all_drugs: list[DrugEntry],
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        injury_type: str,
        anchor: RucamAnchor,
        resolved_item: dict[str, Any],
        report_language: str = "en",
        require_drug_attribution: bool = False,
    ) -> DrugRucamAssessment:
        provided = self.resolve_provided_rucam_score(
            payload.laboratory_analysis or "",
            drug_name=drug.name,
            require_drug_attribution=require_drug_attribution,
        )
        if provided is not None:
            return self.build_source_reported_assessment(
                drug=drug,
                injury_type=injury_type,
                source=provided,
                report_language=report_language,
            )

        checklist = self._build_evidence_checklist(
            payload=payload,
            drug=drug,
            all_drugs=all_drugs,
            disease_context=disease_context,
            lab_timeline=lab_timeline,
            onset_context=onset_context,
            injury_type=injury_type,
            anchor=anchor,
            resolved_item=resolved_item,
        )
        sufficiency = self.evaluate_data_sufficiency(
            injury_type=injury_type,
            anchor=anchor,
            drug=drug,
            lab_timeline=lab_timeline,
            payload=payload,
            disease_context=disease_context,
        )
        limitations = [
            "Automatic updated RUCAM total disabled because all validated criteria are not represented with sufficient structured precision.",
            "LiverTox and RAG literature are never accepted as sources of the current patient's RUCAM score.",
            *sufficiency.blocking_reasons,
        ]
        return self.build_not_calculated_assessment(
            drug=drug,
            injury_type=injury_type,
            reasons=list(dict.fromkeys(limitations)),
            report_language=report_language,
            components=checklist,
        )

    # -------------------------------------------------------------------------
    def _build_evidence_checklist(
        self,
        *,
        payload: PatientData,
        drug: DrugEntry,
        all_drugs: list[DrugEntry],
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        injury_type: str,
        anchor: RucamAnchor,
        resolved_item: dict[str, Any],
    ) -> list[RucamComponentAssessment]:
        onset_component, _ = self.score_time_to_onset(
            payload=payload,
            drug=drug,
            onset_context=onset_context,
            anchor=anchor,
            injury_type=injury_type,
            resolved_item=resolved_item,
        )
        components = [
            onset_component,
            self.score_course(
                injury_type=injury_type,
                lab_timeline=lab_timeline,
                onset_date=anchor.onset_date,
                suspension_status=drug.suspension_status,
            ),
            self.score_risk_factors(payload=payload, injury_type=injury_type),
            self.score_concomitant_drugs(target_drug=drug, all_drugs=all_drugs),
            self.score_non_drug_causes(
                payload=payload,
                disease_context=disease_context,
            ),
            self.score_previous_hepatotoxicity(resolved_item=resolved_item),
            self.score_rechallenge(payload=payload, drug=drug),
        ]
        return [self._as_checklist_component(component) for component in components]

    # -------------------------------------------------------------------------
    @staticmethod
    def _as_checklist_component(
        component: RucamComponentAssessment,
    ) -> RucamComponentAssessment:
        captured = component.rationale or "Criterion evidence captured for clinical review."
        return component.model_copy(
            update={
                "score": 0,
                "status": "not_assessable",
                "rationale": (
                    f"{captured} Numerical scoring is intentionally not automated; "
                    "apply the validated updated RUCAM worksheet clinically if needed."
                ),
            }
        )

    # -------------------------------------------------------------------------
    def score_time_to_onset(
        self,
        *,
        payload: PatientData,
        drug: DrugEntry,
        onset_context: LiverInjuryOnsetContext | None,
        anchor: RucamAnchor,
        injury_type: str,
        resolved_item: dict[str, Any] | None = None,
    ) -> tuple[RucamComponentAssessment, date | None]:
        _ = (payload, injury_type)
        start_date = self.try_parse_date(drug.therapy_start_date)
        onset_date = (
            self.try_parse_date(onset_context.onset_date) if onset_context else None
        )
        if onset_date is None and anchor.is_score_eligible:
            onset_date = anchor.onset_date
        if onset_date is None or start_date is None:
            metadata = (
                resolved_item.get("matched_livertox_row")
                if isinstance(resolved_item, dict)
                else None
            )
            likelihood = (
                str(metadata.get("likelihood_score") or "").strip().upper()
                if isinstance(metadata, dict)
                else ""
            )
            suspension_available = bool(
                drug.suspension_date or drug.suspension_status is not None
            )
            rationale = "Missing start and/or qualifying onset date."
            if start_date is None and suspension_available and likelihood in {"A", "B"}:
                rationale = (
                    "Drug start timing unavailable; suspension timing and a "
                    "high-evidence LiverTox likelihood do not establish latency."
                )
            return RucamComponentAssessment(
                component_key="time_to_onset",
                label="Time to onset",
                score=0,
                status="not_assessable",
                evidence_date=(
                    anchor.onset_date.isoformat() if anchor.onset_date else None
                ),
                rationale=rationale,
            ), onset_date
        delta_days = (onset_date - start_date).days
        if delta_days < 0:
            score = 0
            status = "not_assessable"
        else:
            score = 2 if 5 <= delta_days <= 90 else 1
            status = "scored"
        return RucamComponentAssessment(
            component_key="time_to_onset",
            label="Time to onset",
            score=score,
            status=status,
            evidence_date=onset_date.isoformat(),
            evidence=f"{drug.therapy_start_date} -> {onset_date.isoformat()}",
            rationale=f"Documented latency: {delta_days} days.",
        ), onset_date

    # -------------------------------------------------------------------------
    def score_course(
        self,
        *,
        injury_type: str,
        lab_timeline: PatientLabTimeline,
        onset_date: date | None,
        suspension_status: bool | None,
    ) -> RucamComponentAssessment:
        marker_names = {"ALT"} if injury_type == "hepatocellular" else {"ALP"}
        if onset_date is None or suspension_status is None:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                evidence_date=onset_date.isoformat() if onset_date else None,
                rationale="Withdrawal chronology is incomplete.",
            )
        dated = [
            (
                parsed,
                entry,
            )
            for entry in lab_timeline.entries
            if entry.marker_name.upper() in marker_names
            and entry.value is not None
            and (parsed := self.try_parse_date(entry.sample_date)) is not None
            and parsed >= onset_date
        ]
        if len(dated) < 2:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                evidence_date=onset_date.isoformat(),
                rationale="Insufficient marker-specific follow-up for dechallenge scoring.",
            )
        return RucamComponentAssessment(
            component_key="course",
            label="Course after withdrawal",
            score=0,
            status="not_assessable",
            evidence_date=dated[-1][0].isoformat(),
            rationale="Marker-specific post-onset follow-up is available for clinical dechallenge review.",
        )

    # -------------------------------------------------------------------------
    def score_risk_factors(
        self, *, payload: PatientData, injury_type: str
    ) -> RucamComponentAssessment:
        _ = injury_type
        text = (payload.anamnesis or "").strip()
        return RucamComponentAssessment(
            component_key="risk_factors",
            label="Risk factors",
            score=0,
            status="not_assessable",
            evidence=text[:300] or None,
            rationale=(
                "Age, quantified alcohol exposure, sex-specific thresholds, and pregnancy status are not all available as validated structured RUCAM fields."
            ),
        )

    # -------------------------------------------------------------------------
    def score_concomitant_drugs(
        self, *, target_drug: DrugEntry, all_drugs: list[DrugEntry]
    ) -> RucamComponentAssessment:
        target_key = normalize_drug_query_name(target_drug.name or "")
        other = [
            drug
            for drug in all_drugs
            if normalize_drug_query_name(drug.name or "") != target_key
        ]
        return RucamComponentAssessment(
            component_key="concomitant_drugs",
            label="Concomitant drugs",
            score=0,
            status="not_assessable",
            evidence=", ".join(drug.name for drug in other[:5]) or None,
            rationale=(
                "Concomitant drugs are recorded, but formal RUCAM penalties require drug-specific timing and hepatotoxicity evidence for each competing exposure."
            ),
        )

    # -------------------------------------------------------------------------
    def score_non_drug_causes(
        self, *, payload: PatientData, disease_context: PatientDiseaseContext
    ) -> RucamComponentAssessment:
        text = (payload.anamnesis or "").strip()
        hepatic_entries = [
            entry
            for entry in disease_context.entries
            if bool(entry.hepatic_related) or "hepat" in (entry.name or "").lower()
        ]
        evidence = (
            hepatic_entries[0].evidence or hepatic_entries[0].name
            if hepatic_entries
            else text[:500] or None
        )
        clues = len(_exclusion_re().findall(text))
        return RucamComponentAssessment(
            component_key="non_drug_causes",
            label="Non-drug causes",
            score=0,
            status="not_assessable",
            evidence=evidence,
            rationale=(
                f"Alternative-cause evidence contains {clues} explicit exclusion clue(s); the validated RUCAM cause groups must be reviewed individually."
            ),
        )

    # -------------------------------------------------------------------------
    def score_previous_hepatotoxicity(
        self, *, resolved_item: dict[str, Any]
    ) -> RucamComponentAssessment:
        metadata = (
            resolved_item.get("matched_livertox_row")
            if isinstance(resolved_item, dict)
            else None
        )
        token = ""
        if isinstance(metadata, dict):
            token = str(metadata.get("likelihood_score") or "").strip().upper()
        return RucamComponentAssessment(
            component_key="previous_hepatotoxicity",
            label="Previous hepatotoxicity of the drug",
            score=0,
            status="not_assessable",
            evidence=token or None,
            rationale=(
                "LiverTox likelihood is retained as drug-level evidence only and is not converted directly into a RUCAM previous-hepatotoxicity score."
            ),
        )

    # -------------------------------------------------------------------------
    def score_rechallenge(
        self, *, payload: PatientData, drug: DrugEntry
    ) -> RucamComponentAssessment:
        text = " ".join(
            item.strip()
            for item in (
                payload.anamnesis or "",
                payload.drugs or "",
                drug.evidence or "",
            )
            if item
        )
        lowered = text.lower()
        if "rechallenge positive" in lowered or "recurred after restart" in lowered:
            return RucamComponentAssessment(
                component_key="rechallenge",
                label="Rechallenge",
                score=3,
                status="scored",
                evidence=text[:500],
                rationale="Positive re-exposure language is present and requires biochemical verification; rechallenge is never recommended.",
            )
        if "rechallenge" in lowered or "restarted" in lowered or "resumed" in lowered:
            return RucamComponentAssessment(
                component_key="rechallenge",
                label="Rechallenge",
                score=0,
                status="not_assessable",
                evidence=text[:500],
                rationale="Re-exposure language is present but the biochemical response is unclear.",
            )
        return RucamComponentAssessment(
            component_key="rechallenge",
            label="Rechallenge",
            score=0,
            status="not_assessable",
            rationale="No reliable rechallenge evidence; absence is not treated as a negative rechallenge.",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_causality_bucket(total_score: int) -> str:
        if total_score <= 0:
            return "excluded"
        if total_score <= 2:
            return "unlikely"
        if total_score <= 5:
            return "possible"
        if total_score <= 8:
            return "probable"
        return "highly probable"
