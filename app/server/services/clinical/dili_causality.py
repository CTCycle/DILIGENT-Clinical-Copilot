# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
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
from services.clinical.dili_differential import NEGATION_CUE_RE
from services.clinical.dili_timeline import DiliTimelineEngine
from services.clinical.livertox_likelihood import normalize_livertox_likelihood

LOW_CONFIDENCE_LIVERTOX = {"", "U", "E", "E*", "X", "UNKNOWN"}
DIRECT_TOXIN_LIVERTOX = {"T", "T*"}
SUPPORTIVE_DECHALLENGE = {"improving_after_stop", "resolved_to_baseline"}
SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)\.(?!\d)|[;\n]")
LAB_TERMS = (
    r"(?:ALT|AST|ALP|GGT|bilirubin|aminotransferases?|transaminases?|"
    r"liver (?:enzymes|tests)|LFTs?)"
)
RECURRENCE_TERMS = r"(?:recur\w*|re-?elevat\w*|rose again|rising again|increased again|elevated again)"
POSITIVE_RECHALLENGE_RE = re.compile(
    r"\bpositive\s+re-?challenge\b|\bre-?challenge\b\s*(?:was|is|:|-|=)?\s*positive\b",
    re.IGNORECASE,
)
REEXPOSURE_RE = re.compile(
    r"\b(?:re-?challeng\w*|re-?expos\w*|re-?start\w*|resum\w*|reintroduc\w*)",
    re.IGNORECASE,
)
LAB_RECURRENCE_RE = re.compile(
    rf"\b{LAB_TERMS}\b[^.;\n]{{0,60}}?\b{RECURRENCE_TERMS}"
    rf"|\b{RECURRENCE_TERMS}[^.;\n]{{0,40}}?\b{LAB_TERMS}\b",
    re.IGNORECASE,
)

###############################################################################
class DiliCausalityEngine:

    # -------------------------------------------------------------------------
    @staticmethod
    def rucam(
        source: DrugRucamAssessment | None, drug_name: str
    ) -> DiliRucamAssessment | None:
        if source is None:
            return None
        score_available = (
            source.calculation_method in {"source_reported", "structured_rucam"}
            and not source.estimated
            and source.total_score is not None
        )
        limitations = list(source.limitations)
        if source.total_score is not None and not score_available:
            limitations.append(
                "A numerical RUCAM value was not retained because its provenance was not an explicit current patient-record report."
            )
        return DiliRucamAssessment(
            drug_name=drug_name,
            total_score=source.total_score if score_available else None,
            category=source.causality_category if score_available else "not assessable",
            calculation_method=source.calculation_method,
            score_source=source.score_source,
            estimated=source.estimated,
            components=[
                DiliRucamComponent(
                    component=item.component_key,
                    score=item.score if score_available and item.status == "scored" else None,
                    status=(
                        item.status
                        if score_available or item.status != "scored"
                        else "not_assessable"
                    ),
                    evidence_quote=item.evidence,
                    evidence_date=item.evidence_date,
                    rationale=item.rationale,
                )
                for item in source.components
            ],
            limitations=limitations,
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
            normalize_livertox_likelihood(row.get("likelihood_score")) or ""
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
            # Unresolved competing causes cap synthesis even after recurrence.
            return "supportive" if competing == "complete" else "limited"
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
        terminology = resolved.get("terminology")
        terminology = terminology if isinstance(terminology, dict) else {}
        return DrugIdentityResolution(
            raw_mention=drug.name,
            source_section=drug.source,
            evidence_quote=drug.evidence,
            normalized_name=resolved.get("normalized_name")
            or resolved.get("lookup_key"),
            terminology_candidates=terminology.get("candidates") or [],
            livertox_candidates=resolved.get("livertox_candidates") or [],
            accepted_identity=resolved.get("accepted_livertox_name")
            if accepted
            else None,
            identity_status=terminology.get("status"),
            identity_reason=resolved.get("match_reason"),
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
        status, _ = DiliTimelineEngine.assess_dechallenge(
            DiliTimelineEngine.dechallenge_series(labs.entries, primary_pattern),
            stop_date=stop_date,
            exposure_start=DiliTimelineEngine.parse_date(drug.therapy_start_date),
            cholestatic=primary_pattern in {"cholestatic", "mixed"},
        )
        return status

    # -------------------------------------------------------------------------
    @staticmethod
    def _rechallenge_status(
        drug: DrugEntry,
    ) -> Literal["positive", "present_unclear", "absent", "unknown"]:
        """Positive only with documented recurrence (RUCAM 2016).

        Explicit "positive rechallenge" wording, or re-exposure with a
        laboratory re-elevation in the same sentence, is required; a bare
        restart stays ``present_unclear``.
        """
        evidence = drug.evidence or ""
        for sentence in SENTENCE_SPLIT_RE.split(evidence):
            explicit = POSITIVE_RECHALLENGE_RE.search(sentence)
            if explicit and not NEGATION_CUE_RE.search(sentence[: explicit.start()]):
                return "positive"
            recurrence = LAB_RECURRENCE_RE.search(sentence)
            if (
                recurrence
                and REEXPOSURE_RE.search(sentence)
                and not NEGATION_CUE_RE.search(sentence)
            ):
                return "positive"
        if contains_explicitly_negative_rechallenge(evidence):
            return "absent"
        if REEXPOSURE_RE.search(evidence):
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
