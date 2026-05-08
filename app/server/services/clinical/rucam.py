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
from services.clinical.report_language import phrase, resolve_report_language
from services.text.normalization import normalize_drug_query_name

ALCOHOL_RE = re.compile(r"\b(alcohol|ethanol|wine|beer|abuse)\b", re.IGNORECASE)
PREGNANCY_RE = re.compile(r"\b(pregnan|gestation|gravida)\b", re.IGNORECASE)
EXCLUSION_RE = re.compile(r"(viral|hepatitis|serolog|autoimmune|imaging|ultrasound|mr[ci]|ct)\s+(negative|excluded|normal|without)", re.IGNORECASE)
RUCAM_SCORE_RE = re.compile(r"\brucam\b\s*(?:score)?\s*[:=]?\s*(-?\d{1,2})", re.IGNORECASE)

RucamInjuryType = Literal[
    "hepatocellular",
    "cholestatic",
    "mixed",
    "indeterminate",
]
RucamCausalityCategory = Literal[
    "excluded",
    "unlikely",
    "possible",
    "probable",
    "highly probable",
    "not assessable",
]


class RucamScoreEstimator:
    def estimate(self, *, payload: PatientData, analysis_drugs: PatientDrugs, anamnesis_drugs: PatientDrugs, disease_context: PatientDiseaseContext, lab_timeline: PatientLabTimeline, onset_context: LiverInjuryOnsetContext | None, pattern_score: HepatotoxicityPatternScore, resolved_drugs: dict[str, dict[str, Any]] | None = None, report_language: str = "en") -> PatientRucamAssessmentBundle:
        resolved_mapping = resolved_drugs or {}
        all_drugs = [*analysis_drugs.entries, *anamnesis_drugs.entries]
        anchor = self.select_pattern_anchor(payload=payload, lab_timeline=lab_timeline)
        injury_type = self.resolve_injury_type(pattern_score=pattern_score, anchor=anchor)
        language = resolve_report_language(report_language)
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
            entries.append(self.estimate_for_drug(payload=payload, drug=drug, all_drugs=all_drugs, disease_context=disease_context, lab_timeline=lab_timeline, onset_context=onset_context, injury_type=injury_type, anchor=anchor, resolved_item=resolved if isinstance(resolved, dict) else {}, report_language=language))
        return PatientRucamAssessmentBundle(entries=entries)

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
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def marker_multiple(entry: ClinicalLabEntry | None) -> float | None:
        if entry is None or entry.value is None:
            return None
        if entry.upper_limit_normal and entry.upper_limit_normal > 0:
            return entry.value / entry.upper_limit_normal
        return None

    def collect_trusted_source_text(self, resolved_item: dict[str, Any] | None) -> list[str]:
        if not isinstance(resolved_item, dict):
            return []
        texts: list[str] = []
        for key in ("matched_livertox_row", "extracted_excerpts", "match_notes"):
            raw = resolved_item.get(key)
            if isinstance(raw, dict):
                texts.extend(str(v) for v in raw.values() if isinstance(v, str))
            elif isinstance(raw, list):
                texts.extend(str(v) for v in raw if isinstance(v, str))
            elif isinstance(raw, str):
                texts.append(raw)
        return [item.strip() for item in texts if item and item.strip()]

    def extract_source_reported_rucam(self, resolved_item: dict[str, Any] | None) -> RucamSourceReportedScore | None:
        for text in self.collect_trusted_source_text(resolved_item):
            match = RUCAM_SCORE_RE.search(text)
            if not match:
                continue
            score = max(-10, min(14, int(match.group(1))))
            category = None
            lowered = text.lower()
            for candidate in ("highly probable", "probable", "possible", "unlikely", "excluded"):
                if candidate in lowered:
                    category = candidate
                    break
            return RucamSourceReportedScore(score=score, causality_category=category, source_name="LiverTox", evidence=text)
        return None

    def select_pattern_anchor(self, *, payload: PatientData, lab_timeline: PatientLabTimeline) -> RucamAnchor:
        grouped: dict[date, dict[str, ClinicalLabEntry]] = {}
        for entry in lab_timeline.entries:
            marker = entry.marker_name.upper()
            if marker not in {"ALT", "AST", "ALP", "TBIL"}:
                continue
            parsed_date = self.try_parse_date(entry.sample_date)
            if parsed_date is None:
                continue
            bucket = grouped.setdefault(parsed_date, {})
            current = bucket.get(marker)
            if current is None or (entry.value or float("-inf")) > (current.value or float("-inf")):
                bucket[marker] = entry

        for sample_date in sorted(grouped.keys()):
            bucket = grouped[sample_date]
            alt_like = bucket.get("ALT") or bucket.get("AST")
            alp = bucket.get("ALP")
            tbil = bucket.get("TBIL")
            alt_mult = self.marker_multiple(alt_like)
            alp_mult = self.marker_multiple(alp)
            tbil_mult = self.marker_multiple(tbil)
            qualifies = bool((alt_mult is not None and alt_mult >= 2.0) or (alp_mult is not None and alp_mult >= 2.0) or (tbil is not None and ((tbil_mult is not None and tbil_mult > 1.0) or ((tbil.value or 0.0) >= 2.0)) and (alt_like is not None or alp is not None)))
            if qualifies:
                return RucamAnchor(onset_date=sample_date, used_alt=alt_like.value if alt_like else None, used_alt_uln=alt_like.upper_limit_normal if alt_like else None, used_alp=alp.value if alp else None, used_alp_uln=alp.upper_limit_normal if alp else None, rationale=f"Earliest qualifying timeline anchor selected on {sample_date.isoformat()}.", source="qualifying_lab", is_score_eligible=True)

        return RucamAnchor(onset_date=payload.visit_date, used_alt=None, used_alt_uln=None, used_alp=None, used_alp_uln=None, rationale="No qualifying timeline anchor; visit-date proxy used for context only.", source="visit_proxy", is_score_eligible=False)

    def resolve_injury_type(self, *, pattern_score: HepatotoxicityPatternScore, anchor: RucamAnchor) -> RucamInjuryType:
        classification = (pattern_score.classification or "indeterminate").strip().lower()
        if classification == "mixed":
            return "cholestatic"
        if classification in {"hepatocellular", "cholestatic"}:
            return cast(RucamInjuryType, classification)
        return "indeterminate"

    def evaluate_data_sufficiency(self, *, injury_type: str, anchor: RucamAnchor, drug: DrugEntry, lab_timeline: PatientLabTimeline, payload: PatientData, disease_context: PatientDiseaseContext) -> RucamDataSufficiency:
        reasons: list[str] = []
        if injury_type == "indeterminate":
            reasons.append("injury pattern indeterminate")
        if not anchor.is_score_eligible:
            reasons.append("onset anchor not score-eligible")
        if not (drug.therapy_start_date or drug.therapy_start_status):
            reasons.append("drug start timing unavailable")
        if not (drug.suspension_status is True or drug.suspension_status is False):
            reasons.append("withdrawal status unavailable")
        if len(lab_timeline.entries) < 2:
            reasons.append("insufficient follow-up labs")
        has_alt = bool((payload.anamnesis or "").strip()) and (len(EXCLUSION_RE.findall(payload.anamnesis or "")) > 0 or len(disease_context.entries) > 0)
        if not has_alt:
            reasons.append("alternative-cause assessment evidence unavailable")
        return RucamDataSufficiency(sufficient=not reasons, blocking_reasons=reasons)

    def build_not_calculated_assessment(self, *, drug: DrugEntry, injury_type: str, reasons: list[str], report_language: str) -> DrugRucamAssessment:
        limitations = reasons or [phrase("rucam_insufficient_data", report_language)]
        return DrugRucamAssessment(drug_name=drug.name, injury_type_for_rucam=cast(RucamInjuryType, injury_type), total_score=None, causality_category="not assessable", confidence="low", estimated=False, components=[RucamComponentAssessment(component_key="rucam", label="RUCAM", score=0, status="not_assessable", rationale="; ".join(limitations))], limitations=limitations, summary=phrase("rucam_not_calculated", report_language), calculation_method="not_calculated", score_source=None, data_sufficient=False)

    def build_source_reported_assessment(self, *, drug: DrugEntry, injury_type: str, source: RucamSourceReportedScore, report_language: str) -> DrugRucamAssessment:
        category = source.causality_category or self.resolve_causality_bucket(source.score)
        return DrugRucamAssessment(drug_name=drug.name, injury_type_for_rucam=cast(RucamInjuryType, injury_type), total_score=source.score, causality_category=cast(RucamCausalityCategory, category), confidence="moderate", estimated=False, components=[RucamComponentAssessment(component_key="source_reported", label="Source-reported RUCAM", score=source.score, status="scored", evidence=source.evidence, rationale=phrase("rucam_source_reported", report_language))], limitations=[], summary=phrase("rucam_source_reported", report_language), calculation_method="source_reported", score_source=source.source_name, data_sufficient=True)

    def estimate_for_drug(self, *, payload: PatientData, drug: DrugEntry, all_drugs: list[DrugEntry], disease_context: PatientDiseaseContext, lab_timeline: PatientLabTimeline, onset_context: LiverInjuryOnsetContext | None, injury_type: str, anchor: RucamAnchor, resolved_item: dict[str, Any], report_language: str = "en") -> DrugRucamAssessment:
        source_reported = self.extract_source_reported_rucam(resolved_item)
        if source_reported is not None:
            return self.build_source_reported_assessment(drug=drug, injury_type=injury_type, source=source_reported, report_language=report_language)

        sufficiency = self.evaluate_data_sufficiency(injury_type=injury_type, anchor=anchor, drug=drug, lab_timeline=lab_timeline, payload=payload, disease_context=disease_context)
        if not sufficiency.sufficient:
            return self.build_not_calculated_assessment(drug=drug, injury_type=injury_type, reasons=sufficiency.blocking_reasons, report_language=report_language)

        components: list[RucamComponentAssessment] = []
        onset_component, onset_date = self.score_time_to_onset(payload=payload, drug=drug, onset_context=onset_context, anchor=anchor, injury_type=injury_type)
        components.append(onset_component)
        components.append(self.score_course(injury_type=injury_type, lab_timeline=lab_timeline, onset_date=onset_date, suspension_status=drug.suspension_status))
        components.append(self.score_risk_factors(payload=payload, injury_type=injury_type))
        components.append(self.score_concomitant_drugs(target_drug=drug, all_drugs=all_drugs))
        components.append(self.score_non_drug_causes(payload=payload, disease_context=disease_context))
        components.append(self.score_previous_hepatotoxicity(resolved_item=resolved_item))
        components.append(self.score_rechallenge(payload=payload, drug=drug))

        total = int(sum(component.score for component in components if component.status == "scored"))
        category = self.resolve_causality_bucket(total)
        summary = phrase("rucam_structured_score", report_language, score=total, category=category)
        return DrugRucamAssessment(drug_name=drug.name, injury_type_for_rucam=cast(RucamInjuryType, injury_type), total_score=total, causality_category=cast(RucamCausalityCategory, category), confidence="moderate", estimated=True, components=components, limitations=[], summary=summary, calculation_method="structured_rucam", score_source=None, data_sufficient=True)

    def score_time_to_onset(self, *, payload: PatientData, drug: DrugEntry, onset_context: LiverInjuryOnsetContext | None, anchor: RucamAnchor, injury_type: str) -> tuple[RucamComponentAssessment, date | None]:
        _ = injury_type
        start_date = self.try_parse_date(drug.therapy_start_date)
        onset_date = self.try_parse_date(onset_context.onset_date) if onset_context else None
        if onset_date is None and anchor.is_score_eligible:
            onset_date = anchor.onset_date
        if onset_date is None or start_date is None:
            return RucamComponentAssessment(component_key="time_to_onset", label="Time to onset", score=0, status="not_assessable", rationale="Missing start and/or score-eligible onset date."), onset_date
        delta_days = (onset_date - start_date).days
        score = 2 if 5 <= delta_days <= 90 else 1 if 1 <= delta_days < 5 or 91 <= delta_days <= 365 else 0
        return RucamComponentAssessment(component_key="time_to_onset", label="Time to onset", score=score, status="scored", rationale=f"Latency: {delta_days} days."), onset_date

    def score_course(self, *, injury_type: str, lab_timeline: PatientLabTimeline, onset_date: date | None, suspension_status: bool | None) -> RucamComponentAssessment:
        if onset_date is None or suspension_status is None:
            return RucamComponentAssessment(component_key="course", label="Course after withdrawal", score=0, status="not_assessable", rationale="No onset date or withdrawal status available.")
        _ = injury_type
        dated = []
        for entry in lab_timeline.entries:
            d = self.try_parse_date(entry.sample_date)
            if d is not None and d > onset_date and entry.value is not None:
                dated.append((d, entry.value))
        if not dated:
            return RucamComponentAssessment(component_key="course", label="Course after withdrawal", score=0, status="not_assessable", rationale="No follow-up labs after onset/withdrawal.")
        return RucamComponentAssessment(component_key="course", label="Course after withdrawal", score=1, status="scored", rationale="Follow-up labs available after withdrawal context.")

    def score_risk_factors(self, *, payload: PatientData, injury_type: str) -> RucamComponentAssessment:
        _ = injury_type
        text = (payload.anamnesis or "").strip()
        score = 1 if ALCOHOL_RE.search(text) else 0
        return RucamComponentAssessment(component_key="risk_factors", label="Risk factors", score=score, status="scored")

    def score_concomitant_drugs(self, *, target_drug: DrugEntry, all_drugs: list[DrugEntry]) -> RucamComponentAssessment:
        target_key = normalize_drug_query_name(target_drug.name or "")
        other = [d for d in all_drugs if normalize_drug_query_name(d.name or "") != target_key]
        return RucamComponentAssessment(component_key="concomitant_drugs", label="Concomitant drugs", score=-1 if other else 0, status="scored")

    def score_non_drug_causes(self, *, payload: PatientData, disease_context: PatientDiseaseContext) -> RucamComponentAssessment:
        text = (payload.anamnesis or "").strip()
        hepatic_entries = [entry for entry in disease_context.entries if bool(entry.hepatic_related) or "hepat" in (entry.name or "").lower()]
        if hepatic_entries:
            return RucamComponentAssessment(component_key="non_drug_causes", label="Non-drug causes", score=-3, status="scored")
        clues = len(EXCLUSION_RE.findall(text))
        if clues == 0:
            return RucamComponentAssessment(component_key="non_drug_causes", label="Non-drug causes", score=0, status="not_assessable", rationale="No explicit exclusion workup evidence.")
        return RucamComponentAssessment(component_key="non_drug_causes", label="Non-drug causes", score=2 if clues >= 2 else 1, status="scored")

    def score_previous_hepatotoxicity(self, *, resolved_item: dict[str, Any]) -> RucamComponentAssessment:
        metadata = resolved_item.get("matched_livertox_row") if isinstance(resolved_item, dict) else None
        token = ""
        if isinstance(metadata, dict):
            token = str(metadata.get("likelihood_score") or "").strip().upper()
        score = 2 if token in {"A", "B"} else 1 if token in {"C", "D", "E"} else 0
        return RucamComponentAssessment(component_key="previous_hepatotoxicity", label="Previous hepatotoxicity of the drug", score=score, status="scored")

    def score_rechallenge(self, *, payload: PatientData, drug: DrugEntry) -> RucamComponentAssessment:
        _ = payload
        _ = drug
        return RucamComponentAssessment(component_key="rechallenge", label="Rechallenge", score=0, status="not_assessable")

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

