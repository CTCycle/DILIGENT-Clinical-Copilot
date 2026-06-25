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


class DiliCausalityEngine:
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
                    rationale=item.rationale,
                )
                for item in source.components
            ],
            limitations=source.limitations,
        )

    def exposure(
        self,
        drug: DrugEntry,
        resolved: dict,
        rucam: DrugRucamAssessment | None,
        differential: DiliDifferentialAssessment,
        dechallenge_status: str,
    ) -> DrugExposureAssessment:
        status = str(resolved.get("decision_status") or resolved.get("match_status") or "")
        accepted = status.startswith("accepted_")
        likelihood = None
        row = resolved.get("matched_livertox_row")
        if isinstance(row, dict):
            likelihood = str(row.get("likelihood_score") or "").strip() or None
        if not accepted or not drug.therapy_start_date:
            category = "unassessable"
        elif not differential.all_major_causes_excluded:
            category = "possible"
        elif dechallenge_status == "improving_after_stop" and likelihood in {"A", "B"}:
            category = "probable"
        else:
            category = "possible"
        identity = DrugIdentityResolution(
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
            combination_components=resolved.get("regimen_components") or [],
            is_current_exposure=drug.current_status == "current" or drug.historical_flag is False,
            is_historical_exposure=drug.current_status == "past" or bool(drug.historical_flag),
            is_negated=drug.attribution == "negated",
        )
        return DrugExposureAssessment(
            drug_name=drug.name,
            identity=identity,
            start_date=drug.therapy_start_date,
            stop_date=drug.suspension_date,
            rechallenge_status="unknown",
            livertox_likelihood=likelihood,
            causality=DilinLikeCausalityAssessment(
                drug_name=drug.name,
                category=category,
                temporal_compatibility=(
                    "documented" if drug.therapy_start_date else "missing start date"
                ),
                dechallenge_rechallenge=dechallenge_status,
                phenotype_match="requires LiverTox signature comparison",
                known_drug_signature=likelihood or "unknown",
                competing_cause_exclusion=(
                    "complete" if differential.all_major_causes_excluded else "incomplete"
                ),
                drug_identity_quality="accepted" if accepted else "unresolved",
                source_evidence_quality="quoted" if drug.evidence else "limited",
                rationale=[
                    "DILIN-like category is separate from RUCAM.",
                    "Absence of another explanation alone does not establish probable causality.",
                ],
            ),
            rucam=self.rucam(rucam, drug.name),
        )
