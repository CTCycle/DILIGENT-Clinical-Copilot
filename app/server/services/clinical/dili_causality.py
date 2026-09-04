from __future__ import annotations

from datetime import date
from typing import Literal

from common.utils.clinical_safety import contains_explicitly_negative_rechallenge
from domain.clinical.dili import (
    DiliDifferentialAssessment,
    DiliRucamAssessment,
    DiliRucamComponent,
    DrugExposureAssessment,
    DrugIdentityResolution,
    StructuredCausalityAssessment,
)
from domain.clinical.entities import (
    DrugEntry,
    DrugRucamAssessment,
    PatientLabTimeline,
)
from services.clinical.dili_timeline import DiliTimelineEngine

LOW_CONFIDENCE_LIVERTOX = {"", "U", "E", "E*", "X", "UNKNOWN"}
DIRECT_TOXIN_LIVERTOX = {"T", "T*"}
SUPPORTIVE_DECHALLENGE = {"improving_after_stop", "resolved_to_baseline"}


###############################################################################
class DiliCausalityEngine:
    # -------------------------------------------------------------------------
    @staticmethod
    def rucam(
        source: DrugRucamAssessment | None, drug_name: str
    ) -> DiliRucamAssessment | None:
        if source is None:
            return None
        return DiliRucamAssessment(
            drug_name=drug_name,
            total_score=source.total_score,
            category=source.causality_category,
            components=[
                DiliRucamComponent(
                    component=item.component_key,
                    score=item.score if item.status == "scored" else None,
                    status=item.status,
                    evidence_quote=item.evidence,
                    evidence_date=item.evidence_date,
                    rationale=item.rationale,
                )
                for item in source.components
            ],
            limitations=source.limitations,
        )

    # -------------------------------------------------------------------------
    def exposure(
        self,
        drug: DrugEntry,
        resolved: dict,
        rucam: DrugRucamAssessment | None,
        differential: DiliDifferentialAssessment,
        labs: PatientLabTimeline,
        primary_pattern: str,
        first_injury_date: str | None,
    ) -> DrugExposureAssessment:
        status = str(
            resolved.get("decision_status") or resolved.get("match_status") or ""
        )
        accepted = status.startswith("accepted_")
        row = resolved.get("matched_livertox_row")
        likelihood = (
            str(row.get("likelihood_score") or "").strip().upper()
            if isinstance(row, dict)
            else ""
        )
        identity = self._identity(drug, resolved, accepted)
        temporal = self._temporal_compatibility(drug, first_injury_date)
        rechallenge_status = self._rechallenge_status(drug)
        drug_dechallenge_status = self._dechallenge_for_drug(
            drug=drug,
            labs=labs,
            primary_pattern=primary_pattern,
        )
        signature = self._signature_concordance(primary_pattern, likelihood)
        competing = (
            "complete" if differential.all_major_causes_excluded else "incomplete"
        )
        source_quality = "quoted" if drug.evidence else "limited"
        category = self._synthesis_category(
            accepted=accepted,
            attribution=drug.attribution,
            temporal=temporal,
            dechallenge=drug_dechallenge_status,
            rechallenge=rechallenge_status,
            competing=competing,
        )

        rationale = [
            "Structured causality synthesis is an evidence summary, not a calibrated DILIN probability or autonomous diagnosis.",
            "Drug-level LiverTox likelihood is retained as prior hepatotoxicity evidence and does not cap patient-specific causality.",
        ]
        if drug_dechallenge_status in {"not_assessable", "insufficient_interval"}:
            rationale.append(
                "This drug lacks sufficient drug-specific post-discontinuation laboratory follow-up for dechallenge interpretation."
            )
        if temporal == "incompatible_pre_exposure":
            rationale.append(
                "The first documented injury signal predates the documented drug start."
            )
        elif temporal == "long_latency_requires_drug_specific_review":
            rationale.append(
                "Long latency is not automatically excluded because some agents can cause delayed or prolonged-latency DILI."
            )
        if competing != "complete":
            rationale.append(
                "One or more major competing causes remain unresolved or not excluded."
            )
        if likelihood in LOW_CONFIDENCE_LIVERTOX:
            rationale.append(
                "Sparse or unknown LiverTox evidence reduces drug-level prior support but does not rule out a novel patient-specific DILI event."
            )
        if rucam is not None and not rucam.estimated and rucam.total_score is not None:
            rationale.append(
                "A patient-record RUCAM score is retained as supportive evidence but is not dispositive."
            )

        return DrugExposureAssessment(
            drug_name=drug.name,
            identity=identity,
            start_date=drug.therapy_start_date,
            dose_changes=[],
            stop_date=drug.suspension_date,
            rechallenge_date=None,
            rechallenge_status=rechallenge_status,
            livertox_likelihood=likelihood or None,
            direct_toxin_or_dose_dependent=likelihood in DIRECT_TOXIN_LIVERTOX,
            causality=StructuredCausalityAssessment(
                drug_name=drug.name,
                category=category,
                temporal_compatibility=temporal,
                dechallenge_rechallenge=(
                    f"{drug_dechallenge_status}; rechallenge={rechallenge_status}"
                ),
                drug_signature_concordance=signature,
                known_hepatotoxic_potential=likelihood or "unknown",
                competing_cause_exclusion=competing,
                drug_identity_quality="accepted" if accepted else "unresolved",
                source_evidence_quality=source_quality,
                rationale=rationale,
            ),
            rucam=self.rucam(rucam, drug.name),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _synthesis_category(
        *,
        accepted: bool,
        attribution: str | None,
        temporal: str,
        dechallenge: str,
        rechallenge: str,
        competing: str,
    ) -> Literal["supportive", "limited", "argues_against", "unassessable"]:
        if not accepted or attribution in {"negated", "allergy", "family_history"}:
            return "unassessable"
        if temporal == "incompatible_pre_exposure" and rechallenge != "positive":
            return "argues_against"
        if rechallenge == "positive":
            return "supportive"
        if (
            temporal == "compatible"
            and dechallenge in SUPPORTIVE_DECHALLENGE
            and competing == "complete"
        ):
            return "supportive"
        if temporal in {
            "compatible",
            "long_latency_requires_drug_specific_review",
        } or dechallenge in SUPPORTIVE_DECHALLENGE:
            return "limited"
        return "unassessable"

    # -------------------------------------------------------------------------
    def _identity(
        self, drug: DrugEntry, resolved: dict, accepted: bool
    ) -> DrugIdentityResolution:
        components = resolved.get("regimen_components") or []
        return DrugIdentityResolution(
            raw_mention=drug.name,
            source_section=drug.source,
            evidence_quote=drug.evidence,
            normalized_name=resolved.get("normalized_name")
            or resolved.get("lookup_key"),
            rxnav_candidates=resolved.get("rxnav_candidates") or [],
            livertox_candidates=resolved.get("livertox_candidates") or [],
            accepted_identity=resolved.get("accepted_livertox_name")
            if accepted
            else None,
            identity_confidence=resolved.get("match_confidence"),
            identity_reason=resolved.get("match_reason"),
            rejected_candidates=resolved.get("rejected_candidates") or [],
            combination_components=list(components)
            if isinstance(components, list)
            else [],
            is_current_exposure=drug.current_status == "current"
            or drug.historical_flag is False,
            is_historical_exposure=drug.current_status == "past"
            or bool(drug.historical_flag),
            is_negated=drug.attribution == "negated",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _temporal_compatibility(drug: DrugEntry, first_injury_date: str | None) -> str:
        if not drug.therapy_start_date or not first_injury_date:
            return "unknown"
        start = DiliTimelineEngine.parse_date(drug.therapy_start_date)
        injury = DiliTimelineEngine.parse_date(first_injury_date)
        if start is None or injury is None:
            return "unknown"
        delta = (injury - start).days
        if delta < 0:
            return "incompatible_pre_exposure"
        if delta <= 365:
            return "compatible"
        return "long_latency_requires_drug_specific_review"

    # -------------------------------------------------------------------------
    def _dechallenge_for_drug(
        self,
        *,
        drug: DrugEntry,
        labs: PatientLabTimeline,
        primary_pattern: str,
    ) -> str:
        stop_date = DiliTimelineEngine.parse_date(drug.suspension_date)
        if stop_date is None:
            return "not_assessable"
        marker_names = {"ALT"} if primary_pattern == "hepatocellular" else {"ALP"}
        dated = [
            entry
            for entry in labs.entries
            if entry.marker_name.upper() in marker_names
            and entry.value is not None
            and entry.upper_limit_normal
            and float(entry.upper_limit_normal) > 0
            and DiliTimelineEngine.parse_date(entry.sample_date) is not None
        ]
        dated.sort(
            key=lambda item: DiliTimelineEngine.parse_date(item.sample_date) or date.max
        )
        after_stop = [
            entry
            for entry in dated
            if (DiliTimelineEngine.parse_date(entry.sample_date) or date.min) >= stop_date
        ]
        if len(after_stop) < 2:
            return "insufficient_interval"
        first_multiple = float(after_stop[0].value or 0.0) / float(
            after_stop[0].upper_limit_normal or 1.0
        )
        last_multiple = float(after_stop[-1].value or 0.0) / float(
            after_stop[-1].upper_limit_normal or 1.0
        )
        last_date = DiliTimelineEngine.parse_date(after_stop[-1].sample_date) or stop_date
        if last_multiple <= 1.0:
            return "resolved_to_baseline"
        if first_multiple > 0 and last_multiple <= first_multiple * 0.5:
            return "improving_after_stop"
        if first_multiple > 0 and last_multiple > first_multiple * 1.2:
            return "worsening_after_stop"
        if (last_date - stop_date).days >= 180 and last_multiple > 1.0:
            return "chronic_or_persistent"
        return "stable_abnormality"

    # -------------------------------------------------------------------------
    @staticmethod
    def _rechallenge_status(
        drug: DrugEntry,
    ) -> Literal["positive", "present_unclear", "absent", "unknown"]:
        evidence = (drug.evidence or "").lower()
        if (
            "rechallenge positive" in evidence
            or "re-exposure with recurrence" in evidence
            or "recurred after restart" in evidence
        ):
            return "positive"
        if contains_explicitly_negative_rechallenge(evidence):
            return "absent"
        if (
            "rechallenge" in evidence
            or "restarted" in evidence
            or "resumed" in evidence
            or "re-exposure" in evidence
        ):
            return "present_unclear"
        return "unknown"

    # -------------------------------------------------------------------------
    @staticmethod
    def _signature_concordance(primary_pattern: str, likelihood: str) -> str:
        if primary_pattern == "indeterminate":
            return "not_assessable"
        if likelihood in LOW_CONFIDENCE_LIVERTOX:
            return "reference_evidence_sparse"
        return "not_assessed_from_livertox_likelihood_grade"
