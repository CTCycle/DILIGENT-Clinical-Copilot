from __future__ import annotations

from domain.clinical.dili import (
    DiliDifferentialAssessment,
    DiliRucamAssessment,
    DiliRucamComponent,
    DilinLikeCausalityAssessment,
    DrugExposureAssessment,
    DrugIdentityResolution,
)
from domain.clinical.entities import DrugEntry, DrugRucamAssessment
from services.clinical.dili_timeline import DiliTimelineEngine

LOW_CONFIDENCE_LIVERTOX = {"", "U", "E", "E*", "X", "UNKNOWN"}
DIRECT_TOXIN_LIVERTOX = {"T", "T*"}


###############################################################################
class DiliCausalityEngine:

    # -------------------------------------------------------------------------
    @staticmethod
    def rucam(source: DrugRucamAssessment | None, drug_name: str) -> DiliRucamAssessment | None:
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
        dechallenge_status: str,
        primary_pattern: str,
        first_injury_date: str | None,
    ) -> DrugExposureAssessment:
        status = str(resolved.get("decision_status") or resolved.get("match_status") or "")
        accepted = status.startswith("accepted_")
        row = resolved.get("matched_livertox_row")
        likelihood = str(row.get("likelihood_score") or "").strip().upper() if isinstance(row, dict) else ""
        identity = self._identity(drug, resolved, accepted)
        temporal = self._temporal_compatibility(drug, first_injury_date)
        rechallenge_status = self._rechallenge_status(drug)
        signature = self._signature_match(primary_pattern, likelihood)
        competing = "complete" if differential.all_major_causes_excluded else "incomplete"
        source_quality = "quoted" if drug.evidence else "limited"

        total = 0
        if accepted:
            total += 1
        if temporal == "compatible":
            total += 1
        if dechallenge_status in {"improving_after_stop", "resolved_to_baseline"}:
            total += 1
        if rechallenge_status == "positive":
            total += 2
        if signature == "compatible":
            total += 1
        if competing == "complete":
            total += 1
        if source_quality == "quoted":
            total += 1
        if not accepted or drug.attribution in {"negated", "allergy", "family_history"}:
            total = 0
        if likelihood in LOW_CONFIDENCE_LIVERTOX:
            total = min(total, 2)

        if total >= 6 and likelihood not in LOW_CONFIDENCE_LIVERTOX:
            category = "very_likely"
        elif total >= 4:
            category = "probable"
        elif total >= 2:
            category = "possible"
        elif accepted:
            category = "unlikely"
        else:
            category = "unassessable"

        return DrugExposureAssessment(
            drug_name=drug.name,
            identity=identity,
            start_date=drug.therapy_start_date,
            dose_changes=[],
            stop_date=drug.suspension_date,
            rechallenge_date=drug.suspension_date if rechallenge_status != "unknown" else None,
            rechallenge_status=rechallenge_status,
            livertox_likelihood=likelihood or None,
            direct_toxin_or_dose_dependent=likelihood in DIRECT_TOXIN_LIVERTOX,
            causality=DilinLikeCausalityAssessment(
                drug_name=drug.name,
                category=category,
                temporal_compatibility=temporal,
                dechallenge_rechallenge=f"{dechallenge_status}; rechallenge={rechallenge_status}",
                phenotype_match=signature,
                known_drug_signature=likelihood or "unknown",
                competing_cause_exclusion=competing,
                drug_identity_quality="accepted" if accepted else "unresolved",
                source_evidence_quality=source_quality,
                rationale=[
                    "DILIN-like category integrates temporal fit, drug identity, phenotype fit, dechallenge/rechallenge, and competing causes.",
                    "Absence of an alternative cause alone does not justify upgrading causality.",
                ],
            ),
            rucam=self.rucam(rucam, drug.name),
        )

    # -------------------------------------------------------------------------
    def _identity(self, drug: DrugEntry, resolved: dict, accepted: bool) -> DrugIdentityResolution:
        components = resolved.get("regimen_components") or []
        return DrugIdentityResolution(
            raw_mention=drug.name,
            source_section=drug.source,
            evidence_quote=drug.evidence,
            normalized_name=resolved.get("normalized_name") or resolved.get("lookup_key"),
            rxnav_candidates=resolved.get("rxnav_candidates") or [],
            livertox_candidates=resolved.get("livertox_candidates") or [],
            accepted_identity=resolved.get("accepted_livertox_name") if accepted else None,
            identity_confidence=resolved.get("match_confidence"),
            identity_reason=resolved.get("match_reason"),
            rejected_candidates=resolved.get("rejected_candidates") or [],
            combination_components=list(components) if isinstance(components, list) else [],
            is_current_exposure=drug.current_status == "current" or drug.historical_flag is False,
            is_historical_exposure=drug.current_status == "past" or bool(drug.historical_flag),
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
        if 1 <= delta <= 365:
            return "compatible"
        return "incompatible"

    # -------------------------------------------------------------------------
    @staticmethod
    def _rechallenge_status(drug: DrugEntry) -> str:
        evidence = (drug.evidence or "").lower()
        if "rechallenge positive" in evidence or "re-exposure with recurrence" in evidence:
            return "positive"
        if "rechallenge" in evidence or "restarted" in evidence or "resumed" in evidence:
            return "present_unclear"
        return "unknown"

    # -------------------------------------------------------------------------
    @staticmethod
    def _signature_match(primary_pattern: str, likelihood: str) -> str:
        if primary_pattern == "indeterminate":
            return "unknown"
        if likelihood in LOW_CONFIDENCE_LIVERTOX:
            return "limited_reference_support"
        return "compatible"
