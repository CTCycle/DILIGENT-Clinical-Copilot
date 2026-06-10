from __future__ import annotations

from typing import Any

from services.text.normalization import normalize_drug_query_name


###############################################################################
class InspectionRevisionDecisionsMixin:

    # -------------------------------------------------------------------------
    @staticmethod
    def build_revision_livertox_decisions(
        *,
        matched_drugs: list[Any],
        source_matched_drugs: list[Any] | None = None,
        instruction_profile: Any | None,
    ) -> list[dict[str, Any]]:
        challenged_matching = bool(
            instruction_profile
            and "matching_errors" in instruction_profile.target_entities
        )
        source_match_lookup: dict[str, dict[str, Any]] = {}
        for item in source_matched_drugs or []:
            if not isinstance(item, dict):
                continue
            drug_name = str(
                item.get("matched_drug_name")
                or item.get("raw_drug_name")
                or item.get("drug_name")
                or ""
            ).strip()
            normalized = normalize_drug_query_name(drug_name)
            if normalized:
                source_match_lookup[normalized] = item
        decisions: list[dict[str, Any]] = []
        for index, item in enumerate(matched_drugs):
            if not isinstance(item, dict):
                decisions.append(
                    {
                        "decision_id": f"livertox:{index}",
                        "drug_name": str(item).strip() or f"drug-{index + 1}",
                        "decision": "requires_human_review",
                        "decision_reason": "Matched-drug payload is not structured.",
                        "match_status": "unknown",
                        "match_confidence": None,
                        "requires_human_review": True,
                        "source": "none",
                        "previous_match_found": False,
                        "provenance": {"source_version_match": None},
                    }
                )
                continue
            drug_name = str(
                item.get("matched_drug_name")
                or item.get("raw_drug_name")
                or item.get("drug_name")
                or f"drug-{index + 1}"
            ).strip()
            match_status = str(item.get("match_status") or "unknown").strip().lower()
            raw_confidence = item.get("match_confidence")
            try:
                match_confidence = (
                    float(raw_confidence) if raw_confidence is not None else None
                )
            except (TypeError, ValueError):
                match_confidence = None
            normalized_drug_name = normalize_drug_query_name(drug_name)
            previous_match = (
                source_match_lookup.get(normalized_drug_name or "")
                if normalized_drug_name
                else None
            )
            previous_match_found = isinstance(previous_match, dict)
            previous_match_name = str(
                (previous_match or {}).get("matched_drug_name")
                or (previous_match or {}).get("raw_drug_name")
                or (previous_match or {}).get("drug_name")
                or ""
            ).strip()
            previous_match_confidence = None
            try:
                if previous_match is not None:
                    previous_confidence_raw = previous_match.get("match_confidence")
                    previous_match_confidence = (
                        float(previous_confidence_raw)
                        if previous_confidence_raw is not None
                        else None
                    )
            except (TypeError, ValueError):
                previous_match_confidence = None
            same_match_name = bool(
                previous_match_name
                and previous_match_name.casefold() == drug_name.casefold()
            )
            if challenged_matching and previous_match_found:
                decision = "llm_assisted_resolved_match"
                reason = (
                    "Reviewer instruction challenged the previous source-version match."
                )
                requires_human_review = False
                decision_source = "llm_fallback"
            elif (
                previous_match_found
                and previous_match_confidence is not None
                and previous_match_confidence >= 0.95
                and same_match_name
            ):
                decision = "reused_high_confidence_previous_match"
                reason = "High-confidence previous source-version match remains valid."
                requires_human_review = False
                decision_source = "previous_version"
            elif match_status in {"matched_with_excerpt", "matched"} and (
                match_confidence is not None and match_confidence >= 0.95
            ):
                decision = "deterministic_new_match"
                reason = (
                    "Revision produced a high-confidence structured LiverTox match."
                )
                requires_human_review = False
                decision_source = "deterministic"
            elif match_status in {"missing_match", "ambiguous_match", "missing"}:
                decision = "no_reliable_match_found"
                reason = "No reliable prior LiverTox match is available."
                requires_human_review = True
                decision_source = "none"
            else:
                decision = "llm_assisted_resolved_match"
                reason = "Revision required a refreshed LiverTox decision."
                requires_human_review = False
                decision_source = "llm_fallback"
            decisions.append(
                {
                    "decision_id": f"livertox:{index}",
                    "drug_name": drug_name,
                    "normalized_drug_name": normalized_drug_name,
                    "decision": decision,
                    "decision_reason": reason,
                    "match_status": match_status,
                    "match_confidence": match_confidence,
                    "requires_human_review": requires_human_review,
                    "reviewer_challenged": challenged_matching,
                    "source": decision_source,
                    "previous_match_found": previous_match_found,
                    "previous_match_confidence": previous_match_confidence,
                    "payload": item,
                    "provenance": {
                        "source_version_match": previous_match,
                        "current_revision_match": item,
                    },
                }
            )
        return decisions

    # -------------------------------------------------------------------------
    @staticmethod
    def build_revised_dili_assessments(
        *,
        rucam_assessments: list[Any],
        matched_drugs: list[Any],
        source_rucam_assessments: list[Any] | None = None,
        revision_version_id: int,
        source_version_id: int,
        instruction_profile: Any | None,
    ) -> list[dict[str, Any]]:
        matched_lookup: dict[str, dict[str, Any]] = {}
        for item in matched_drugs:
            if not isinstance(item, dict):
                continue
            drug_name = str(
                item.get("matched_drug_name")
                or item.get("raw_drug_name")
                or item.get("drug_name")
                or ""
            ).strip()
            normalized = normalize_drug_query_name(drug_name)
            if normalized:
                matched_lookup[normalized] = item
        previous_assessment_lookup: dict[str, dict[str, Any]] = {}
        for item in source_rucam_assessments or []:
            if not isinstance(item, dict):
                continue
            drug_name = str(item.get("drug_name") or "").strip()
            normalized = normalize_drug_query_name(drug_name)
            if normalized:
                previous_assessment_lookup[normalized] = item
        assessments: list[dict[str, Any]] = []
        for index, item in enumerate(rucam_assessments):
            if not isinstance(item, dict):
                continue
            drug_name = str(item.get("drug_name") or f"drug-{index + 1}").strip()
            normalized = normalize_drug_query_name(drug_name)
            matched_row = matched_lookup.get(normalized or "")
            previous_assessment = previous_assessment_lookup.get(normalized or "")
            total_score = item.get("total_score")
            confidence = "moderate"
            if isinstance(total_score, (int, float)):
                if float(total_score) >= 9:
                    confidence = "high"
                elif float(total_score) <= 3:
                    confidence = "low"
            unresolved_questions = []
            if matched_row is None:
                unresolved_questions.append(
                    "No reliable LiverTox match is available for this revised drug."
                )
            if (
                instruction_profile
                and "causality_reasoning" in instruction_profile.target_entities
            ):
                unresolved_questions.append(
                    "Reviewer explicitly requested reassessment of causality reasoning."
                )
            changes_from_previous_version: list[str] = []
            previous_causality = str(
                (previous_assessment or {}).get("causality_category")
                or (previous_assessment or {}).get("causality_assessment")
                or ""
            ).strip()
            current_causality = str(
                item.get("causality_category")
                or item.get("causality_assessment")
                or "unresolved"
            )
            if previous_causality and previous_causality != current_causality:
                changes_from_previous_version.append(
                    f"Causality changed from {previous_causality} to {current_causality}."
                )
            previous_score = (previous_assessment or {}).get("total_score")
            if (
                isinstance(previous_score, (int, float))
                and isinstance(total_score, (int, float))
                and float(previous_score) != float(total_score)
            ):
                changes_from_previous_version.append(
                    f"Total score changed from {float(previous_score):g} to {float(total_score):g}."
                )
            if previous_assessment and not changes_from_previous_version:
                changes_from_previous_version.append(
                    "Previous source-version assessment was reviewed and retained."
                )
            assessments.append(
                {
                    "drug_id": item.get("drug_id"),
                    "revised_drug_entry_id": f"revised-drug:{index}",
                    "revision_version_id": revision_version_id,
                    "source_version_id": source_version_id,
                    "assessment_version": "1",
                    "drug_name": drug_name,
                    "causality_assessment": str(
                        item.get("causality_category")
                        or item.get("causality_assessment")
                        or "unresolved"
                    ),
                    "confidence": confidence,
                    "evidence_for": [],
                    "evidence_against": [],
                    "lab_support": [],
                    "temporal_support": [],
                    "alternative_causes": [],
                    "livertox_support": [str(matched_row.get("matched_drug_name"))]
                    if isinstance(matched_row, dict)
                    and str(matched_row.get("matched_drug_name") or "").strip()
                    else [],
                    "changes_from_previous_version": changes_from_previous_version,
                    "unresolved_questions": unresolved_questions,
                    "requires_human_review": bool(unresolved_questions),
                    "previous_assessment_present": bool(previous_assessment),
                    "provenance": {
                        "source_version_assessment": previous_assessment,
                        "current_revision_assessment": item,
                    },
                }
            )
        return assessments

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_revision_runtime_overrides(
        *,
        effective_overrides: dict[str, Any],
    ) -> dict[str, object]:
        runtime_overrides: dict[str, object] = {}
        if "clinical_model" in effective_overrides:
            runtime_overrides["clinical_model"] = effective_overrides["clinical_model"]
        if "text_extraction_model" in effective_overrides:
            runtime_overrides["text_extraction_model"] = effective_overrides[
                "text_extraction_model"
            ]
        if "use_cloud_services" in effective_overrides:
            runtime_overrides["use_cloud_models"] = effective_overrides[
                "use_cloud_services"
            ]
        if "provider" in effective_overrides:
            runtime_overrides["cloud_provider"] = effective_overrides["provider"]
        if "cloud_model" in effective_overrides:
            runtime_overrides["cloud_model"] = effective_overrides["cloud_model"]
        if "ollama_temperature" in effective_overrides:
            runtime_overrides["ollama_temperature"] = effective_overrides[
                "ollama_temperature"
            ]
        if "cloud_temperature" in effective_overrides:
            runtime_overrides["cloud_temperature"] = effective_overrides[
                "cloud_temperature"
            ]
        if "ollama_reasoning" in effective_overrides:
            runtime_overrides["ollama_reasoning"] = effective_overrides[
                "ollama_reasoning"
            ]
        return runtime_overrides
