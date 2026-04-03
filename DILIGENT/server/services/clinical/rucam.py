from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from DILIGENT.server.domain.clinical import (
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
from DILIGENT.server.domain.rucam import RucamAnchor
from DILIGENT.server.services.text.normalization import normalize_drug_query_name


ALCOHOL_RE = re.compile(r"\b(alcohol|ethanol|wine|beer|abuse)\b", re.IGNORECASE)
PREGNANCY_RE = re.compile(r"\b(pregnan|gestation|gravida)\b", re.IGNORECASE)
EXCLUSION_RE = re.compile(
    r"(viral|hepatitis|serolog|autoimmune|imaging|ultrasound|mr[ci]|ct)\s+(negative|excluded|normal|without)",
    re.IGNORECASE,
)
RECHALLENGE_RE = re.compile(
    r"\b(rechallenge|re-?expos|reintroduc|restart|ripres[ao])\b",
    re.IGNORECASE,
)
RECURRENCE_RE = re.compile(r"\b(recur|rise|worsen|flare|peggior)\b", re.IGNORECASE)
AGE_RE = re.compile(r"\b(\d{2})\s*(?:years?|yo|anni)\b", re.IGNORECASE)

class RucamScoreEstimator:
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
    ) -> PatientRucamAssessmentBundle:
        resolved_mapping = resolved_drugs or {}
        all_drugs = [*analysis_drugs.entries, *anamnesis_drugs.entries]
        anchor = self.select_pattern_anchor(payload=payload, lab_timeline=lab_timeline)
        injury_type = self.resolve_injury_type(pattern_score=pattern_score, anchor=anchor)

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
                )
            )
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

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "."))
        except ValueError:
            return None

    def select_pattern_anchor(
        self,
        *,
        payload: PatientData,
        lab_timeline: PatientLabTimeline,
    ) -> RucamAnchor:
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
            qualifies = bool(
                (alt_mult is not None and alt_mult >= 2.0)
                or (alp_mult is not None and alp_mult >= 2.0)
                or (
                    tbil is not None
                    and ((tbil_mult is not None and tbil_mult > 1.0) or ((tbil.value or 0.0) >= 2.0))
                    and (alt_like is not None or alp is not None)
                )
            )
            if not qualifies:
                continue
                return RucamAnchor(
                onset_date=sample_date,
                used_alt=alt_like.value if alt_like else None,
                used_alt_uln=alt_like.upper_limit_normal if alt_like else None,
                used_alp=alp.value if alp else None,
                used_alp_uln=alp.upper_limit_normal if alp else None,
                rationale=(
                    "Earliest qualifying timeline anchor selected on "
                    f"{sample_date.isoformat()} (ALT/AST={alt_like.value if alt_like else 'n/a'}, "
                    f"ALP={alp.value if alp else 'n/a'})."
                ),
            )

        return RucamAnchor(
            onset_date=payload.visit_date,
            used_alt=None,
            used_alt_uln=None,
            used_alp=None,
            used_alp_uln=None,
            rationale="No qualifying timeline anchor; visit-date proxy used.",
        )

    def resolve_injury_type(
        self,
        *,
        pattern_score: HepatotoxicityPatternScore,
        anchor: RucamAnchor,
    ) -> str:
        classification = (pattern_score.classification or "indeterminate").strip().lower()
        if classification == "mixed":
            return "cholestatic"
        if classification in {"hepatocellular", "cholestatic"}:
            return classification
        if (
            anchor.used_alt
            and anchor.used_alt_uln
            and anchor.used_alp
            and anchor.used_alp_uln
            and anchor.used_alt_uln > 0
            and anchor.used_alp_uln > 0
        ):
            alt_mult = anchor.used_alt / anchor.used_alt_uln
            alp_mult = anchor.used_alp / anchor.used_alp_uln
            if alp_mult > 0:
                ratio = alt_mult / alp_mult
                if ratio > 5:
                    return "hepatocellular"
                if ratio < 2:
                    return "cholestatic"
                return "cholestatic"
        return "indeterminate"

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
    ) -> DrugRucamAssessment:
        limitations: list[str] = []
        components: list[RucamComponentAssessment] = []

        onset_component, onset_date = self.score_time_to_onset(
            payload=payload,
            drug=drug,
            onset_context=onset_context,
            anchor=anchor,
            injury_type=injury_type,
        )
        components.append(onset_component)
        if onset_component.status != "scored":
            limitations.append("Time-to-onset was not fully assessable.")

        course_component = self.score_course(
            injury_type=injury_type,
            lab_timeline=lab_timeline,
            onset_date=onset_date,
        )
        components.append(course_component)
        if course_component.status != "scored":
            limitations.append("Serial follow-up liver labs were insufficient for course scoring.")

        risk_component = self.score_risk_factors(payload=payload, injury_type=injury_type)
        components.append(risk_component)
        components.append(self.score_concomitant_drugs(target_drug=drug, all_drugs=all_drugs))
        non_drug = self.score_non_drug_causes(payload=payload, disease_context=disease_context)
        components.append(non_drug)
        components.append(self.score_previous_hepatotoxicity(resolved_item=resolved_item))
        components.append(self.score_rechallenge(payload=payload, drug=drug))

        if non_drug.score == 0:
            limitations.append("Competing non-drug causes were only partially represented.")
        if injury_type == "indeterminate":
            limitations.append("Injury type remained indeterminate; conservative scoring applied.")

        total = int(sum(component.score for component in components))
        category = self.resolve_causality_bucket(total)
        confidence = self.resolve_confidence(
            components=components,
            onset_date=onset_date,
            drug=drug,
        )
        unique_limitations = list(dict.fromkeys(limitations))
        condensed_limitations = unique_limitations[:3]
        if condensed_limitations:
            limitations_sentence = "; ".join(condensed_limitations)
            summary = (
                f"Estimated RUCAM score {total} ({category}). "
                f"The score is estimated due to incomplete clinical data. "
                f"Key limitations: {limitations_sentence}."
            )
        else:
            summary = (
                f"Estimated RUCAM score {total} ({category}). "
                "The score is estimated due to potential incompleteness in available clinical data."
            )
        return DrugRucamAssessment(
            drug_name=drug.name,
            injury_type_for_rucam=injury_type,  # type: ignore[arg-type]
            total_score=total,
            causality_category=category,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            estimated=True,
            components=components,
            limitations=unique_limitations,
            summary=summary,
        )

    def score_time_to_onset(
        self,
        *,
        payload: PatientData,
        drug: DrugEntry,
        onset_context: LiverInjuryOnsetContext | None,
        anchor: RucamAnchor,
        injury_type: str,
    ) -> tuple[RucamComponentAssessment, date | None]:
        start_date = self.try_parse_date(drug.therapy_start_date)
        onset_date = self.try_parse_date(onset_context.onset_date) if onset_context else None
        rationale: list[str] = []
        evidence = onset_context.evidence if onset_context else None

        if start_date:
            rationale.append(f"Therapy start: {start_date.isoformat()}.")
        if onset_date:
            rationale.append(f"Onset from extraction: {onset_date.isoformat()}.")
        elif anchor.onset_date is not None:
            onset_date = anchor.onset_date
            rationale.append(f"Onset fallback used lab anchor: {anchor.onset_date.isoformat()}.")
        elif payload.visit_date is not None:
            onset_date = payload.visit_date
            rationale.append("Onset fallback used visit-date proxy.")

        if onset_date is None or start_date is None:
            return (
                RucamComponentAssessment(
                    component_key="time_to_onset",
                    label="Time to onset",
                    score=0,
                    status="not_assessable",
                    evidence=evidence,
                    rationale=" ".join(rationale) or "Missing start and/or onset date.",
                ),
                onset_date,
            )

        delta_days = (onset_date - start_date).days
        if 5 <= delta_days <= 90:
            score = 2
        elif 1 <= delta_days < 5 or 91 <= delta_days <= 365:
            score = 1
        else:
            score = 0
        if injury_type == "cholestatic":
            rationale.append("Conservative cholestatic/mixed timing bucket applied.")
        rationale.append(f"Latency: {delta_days} days.")
        return (
            RucamComponentAssessment(
                component_key="time_to_onset",
                label="Time to onset",
                score=score,
                status="scored",
                evidence=evidence,
                rationale=" ".join(rationale),
            ),
            onset_date,
        )

    def score_course(
        self,
        *,
        injury_type: str,
        lab_timeline: PatientLabTimeline,
        onset_date: date | None,
    ) -> RucamComponentAssessment:
        if onset_date is None:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                rationale="No onset anchor date available.",
            )
        if injury_type == "hepatocellular":
            return self._score_hepatocellular_course(lab_timeline=lab_timeline, onset_date=onset_date)
        return self._score_cholestatic_course(lab_timeline=lab_timeline, onset_date=onset_date)

    def _dated_entries(
        self,
        *,
        entries: list[ClinicalLabEntry],
        markers: set[str],
        onset_date: date,
    ) -> list[tuple[date, ClinicalLabEntry]]:
        selected: list[tuple[date, ClinicalLabEntry]] = []
        for entry in entries:
            if entry.marker_name.upper() not in markers:
                continue
            parsed_date = self.try_parse_date(entry.sample_date)
            if parsed_date is None or parsed_date < onset_date:
                continue
            selected.append((parsed_date, entry))
        selected.sort(key=lambda pair: pair[0])
        return selected

    def _score_hepatocellular_course(
        self,
        *,
        lab_timeline: PatientLabTimeline,
        onset_date: date,
    ) -> RucamComponentAssessment:
        series = self._dated_entries(
            entries=lab_timeline.entries,
            markers={"ALT", "AST"},
            onset_date=onset_date,
        )
        if len(series) < 2:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                rationale="ALT/AST serial trend after onset was insufficient.",
            )
        peak_date, peak_entry = max(series, key=lambda pair: pair[1].value or float("-inf"))
        if peak_entry.value is None or peak_entry.value <= 0:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                rationale="Peak ALT/AST value was unavailable.",
            )
        target = peak_entry.value * 0.5
        drop: tuple[date, ClinicalLabEntry] | None = None
        for point in series:
            if point[0] <= peak_date:
                continue
            if (point[1].value or float("inf")) <= target:
                drop = point
                break
        if drop is None:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="scored",
                rationale="No >50% ALT/AST decline in available follow-up.",
            )
        days = (drop[0] - peak_date).days
        score = 3 if days <= 8 else 2 if days <= 30 else 1
        return RucamComponentAssessment(
            component_key="course",
            label="Course after withdrawal",
            score=score,
            status="scored",
            rationale=(
                f"Peak {peak_entry.marker_name} {peak_entry.value} on {peak_date.isoformat()} "
                f"fell to {drop[1].value} by {drop[0].isoformat()} ({days} days)."
            ),
        )

    def _score_cholestatic_course(
        self,
        *,
        lab_timeline: PatientLabTimeline,
        onset_date: date,
    ) -> RucamComponentAssessment:
        series = self._dated_entries(
            entries=lab_timeline.entries,
            markers={"ALP"},
            onset_date=onset_date,
        )
        marker_name = "ALP"
        if len(series) < 2:
            series = self._dated_entries(
                entries=lab_timeline.entries,
                markers={"TBIL"},
                onset_date=onset_date,
            )
            marker_name = "TBIL"
        if len(series) < 2:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                rationale="ALP/TBIL serial trend after onset was insufficient.",
            )
        peak_date, peak_entry = max(series, key=lambda pair: pair[1].value or float("-inf"))
        latest_date, latest_entry = series[-1]
        if peak_entry.value is None or latest_entry.value is None or peak_entry.value <= 0:
            return RucamComponentAssessment(
                component_key="course",
                label="Course after withdrawal",
                score=0,
                status="not_assessable",
                rationale=f"{marker_name} follow-up values were not usable.",
            )
        decline = ((peak_entry.value - latest_entry.value) / peak_entry.value) * 100.0
        days = (latest_date - peak_date).days
        if decline >= 50.0 and days <= 180:
            score = 2
        elif 20.0 <= decline < 50.0:
            score = 1
        else:
            score = 0
        return RucamComponentAssessment(
            component_key="course",
            label="Course after withdrawal",
            score=score,
            status="scored",
            rationale=(
                f"Peak {marker_name} {peak_entry.value} on {peak_date.isoformat()} changed to "
                f"{latest_entry.value} on {latest_date.isoformat()} ({decline:.1f}% fall over {days} days)."
            ),
        )

    def score_risk_factors(self, *, payload: PatientData, injury_type: str) -> RucamComponentAssessment:
        text = (payload.anamnesis or "").strip()
        age_score = 0
        age_match = AGE_RE.search(text)
        if age_match:
            try:
                age_score = 1 if int(age_match.group(1)) >= 55 else 0
            except ValueError:
                age_score = 0
        alcohol = bool(ALCOHOL_RE.search(text))
        pregnancy = bool(PREGNANCY_RE.search(text))
        if injury_type == "hepatocellular":
            score = age_score + (1 if alcohol else 0)
        else:
            score = age_score + (1 if (alcohol or pregnancy) else 0)
        return RucamComponentAssessment(
            component_key="risk_factors",
            label="Risk factors",
            score=score,
            status="scored",
            rationale=(
                f"Age>=55: {'yes' if age_score else 'no'}; "
                f"alcohol clue: {'yes' if alcohol else 'no'}; "
                f"pregnancy clue: {'yes' if pregnancy else 'no'}."
            ),
        )

    def score_concomitant_drugs(
        self,
        *,
        target_drug: DrugEntry,
        all_drugs: list[DrugEntry],
    ) -> RucamComponentAssessment:
        target_key = normalize_drug_query_name(target_drug.name or "")
        other = [d for d in all_drugs if normalize_drug_query_name(d.name or "") != target_key]
        if not other:
            return RucamComponentAssessment(
                component_key="concomitant_drugs",
                label="Concomitant drugs",
                score=0,
                status="scored",
                rationale="No compatible concomitant drug detected.",
            )
        strong = sum(1 for item in other if item.suspension_status and item.therapy_start_status)
        weak = sum(1 for item in other if item.therapy_start_date or item.suspension_date)
        if strong > 0:
            score = -2
            rationale = "At least one concomitant drug had stronger compatible temporal metadata."
        elif weak > 0:
            score = -1
            rationale = "Concomitant drug exposure was temporally compatible but weakly documented."
        else:
            score = 0
            rationale = "Concomitant drugs lacked usable temporal metadata."
        return RucamComponentAssessment(
            component_key="concomitant_drugs",
            label="Concomitant drugs",
            score=score,
            status="scored",
            rationale=rationale,
        )

    def score_non_drug_causes(
        self,
        *,
        payload: PatientData,
        disease_context: PatientDiseaseContext,
    ) -> RucamComponentAssessment:
        text = (payload.anamnesis or "").strip()
        hepatic_entries = [
            entry
            for entry in disease_context.entries
            if bool(entry.hepatic_related) or "hepat" in (entry.name or "").lower()
        ]
        if payload.has_hepatic_diseases and hepatic_entries:
            return RucamComponentAssessment(
                component_key="non_drug_causes",
                label="Non-drug causes",
                score=-3,
                status="scored",
                evidence="; ".join((item.evidence or item.name) for item in hepatic_entries),
                rationale="Another hepatic diagnosis is explicitly supported.",
            )
        clues = len(EXCLUSION_RE.findall(text))
        score = 2 if clues >= 2 else 1 if clues == 1 else 0
        rationale = (
            "Multiple exclusion clues documented."
            if score == 2
            else "Partial exclusion clue documented."
            if score == 1
            else "No robust exclusion workup documented."
        )
        return RucamComponentAssessment(
            component_key="non_drug_causes",
            label="Non-drug causes",
            score=score,
            status="scored",
            rationale=rationale,
        )

    def score_previous_hepatotoxicity(self, *, resolved_item: dict[str, Any]) -> RucamComponentAssessment:
        metadata = resolved_item.get("matched_livertox_row")
        token = ""
        if isinstance(metadata, dict):
            raw = metadata.get("likelihood_score")
            token = str(raw).strip().upper() if raw is not None else ""
        if token in {"A", "B"}:
            score = 2
        elif token in {"C", "D", "E"}:
            score = 1
        else:
            score = 0
        rationale = (
            f"LiverTox likelihood score {token} used as proxy."
            if token
            else "No local prior hepatotoxicity proxy available."
        )
        return RucamComponentAssessment(
            component_key="previous_hepatotoxicity",
            label="Previous hepatotoxicity of the drug",
            score=score,
            status="scored",
            rationale=rationale,
        )

    def score_rechallenge(self, *, payload: PatientData, drug: DrugEntry) -> RucamComponentAssessment:
        text = (payload.anamnesis or "").strip()
        drug_name = (drug.name or "").strip()
        if not text or not drug_name:
            return RucamComponentAssessment(
                component_key="rechallenge",
                label="Rechallenge",
                score=0,
                status="not_assessable",
                rationale="Insufficient detail for rechallenge assessment.",
            )
        if drug_name.lower() in text.lower() and RECHALLENGE_RE.search(text) and RECURRENCE_RE.search(text):
            return RucamComponentAssessment(
                component_key="rechallenge",
                label="Rechallenge",
                score=3,
                status="scored",
                rationale="Explicit rechallenge with recurrence clue detected.",
            )
        return RucamComponentAssessment(
            component_key="rechallenge",
            label="Rechallenge",
            score=0,
            status="scored",
            rationale="No explicit positive rechallenge evidence.",
        )

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

    @staticmethod
    def resolve_confidence(
        *,
        components: list[RucamComponentAssessment],
        onset_date: date | None,
        drug: DrugEntry,
    ) -> str:
        assessable = sum(1 for component in components if component.status == "scored")
        not_assessable = sum(1 for component in components if component.status == "not_assessable")
        course_component = next(
            (component for component in components if component.component_key == "course"),
            None,
        )
        has_exposure_dates = bool(drug.therapy_start_date or drug.suspension_date)
        if course_component is None or course_component.status != "scored":
            return "low"
        if assessable >= 6 and onset_date is not None and has_exposure_dates and not_assessable <= 1:
            return "high"
        if assessable >= 4 and (onset_date is not None or has_exposure_dates):
            return "moderate"
        return "low"
