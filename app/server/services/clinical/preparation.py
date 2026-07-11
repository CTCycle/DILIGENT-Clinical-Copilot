from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.clinical.drug_resolution import DrugIdentityProposalBatch
from domain.clinical.entities import (
    DrugEntry,
    HepatotoxicityPatternScore,
    PatientDrugs,
    PipelineIssue,
)
from domain.clinical.extras import HepatoxPreparedInputs
from repositories.serialization.data import DataSerializer
from services.clinical.knowledge import ClinicalKnowledgeComposer
from services.clinical.drug_resolution import DrugResolutionService
from services.clinical.matches_core import (
    LiverToxMatcher,
)
from services.text.normalization import normalize_drug_query_name


###############################################################################
class ClinicalKnowledgePreparation:
    IDENTITY_FALLBACK_SYSTEM_PROMPT = """
You normalize medication product labels to generic drug identities.
Return one proposal per input mention. Use only pharmacologic identity knowledge.
Do not assess hepatotoxicity, causality, dosing, or the patient.
For combination products, list each active ingredient separately.
If the identity is uncertain, keep confidence low and do not invent an ingredient.
The application will independently validate every proposed name against local
RxNav and LiverTox evidence before accepting it.
""".strip()

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.serializer = DataSerializer()
        self.knowledge_composer = ClinicalKnowledgeComposer(serializer=self.serializer)
        self.livertox_matcher: LiverToxMatcher | None = None

    # -------------------------------------------------------------------------
    async def prepare_inputs(
        self,
        drugs: PatientDrugs,
        *,
        clinical_context: str | None,
        pattern_score: HepatotoxicityPatternScore | None,
        progress_callback: Callable[[float], None] | None = None,
        identity_resolution_client: Any | None = None,
        identity_resolution_model: str | None = None,
        identity_resolution_temperature: float = 0.0,
    ) -> HepatoxPreparedInputs | None:
        self.emit_progress(progress_callback, 0.0)
        self.emit_progress(progress_callback, 0.2)
        if not await self.ensure_livertox_matcher() or self.livertox_matcher is None:
            return None
        resolver = DrugResolutionService(
            self.livertox_matcher,
            cache_lookup=lambda key: self.serializer.load_livertox_match_from_db_cache(
                normalized_drug_key=key,
            ),
        )
        resolved_drugs = await asyncio.to_thread(resolver.resolve, drugs)
        resolved_drugs = await self.apply_validated_identity_fallback(
            resolver=resolver,
            drugs=drugs,
            resolved_drugs=resolved_drugs,
            llm_client=identity_resolution_client,
            model=identity_resolution_model,
            temperature=identity_resolution_temperature,
        )
        if not resolved_drugs:
            logger.info("No drugs detected for input preparation")
            return None
        self.emit_progress(progress_callback, 0.35)
        self.emit_progress(progress_callback, 0.65)
        self.emit_progress(progress_callback, 0.9)
        self.knowledge_composer.enrich_resolved_drugs(resolved_drugs)
        pattern_prompt = self.build_pattern_prompt(pattern_score)
        normalized_context = (clinical_context or "").strip()

        return HepatoxPreparedInputs(
            resolved_drugs=resolved_drugs,
            pattern_prompt=pattern_prompt,
            clinical_context=normalized_context,
        )

    # -------------------------------------------------------------------------
    async def apply_validated_identity_fallback(
        self,
        *,
        resolver: DrugResolutionService,
        drugs: PatientDrugs,
        resolved_drugs: dict[str, dict[str, Any]],
        llm_client: Any | None,
        model: str | None,
        temperature: float,
    ) -> dict[str, dict[str, Any]]:
        unresolved = self.collect_identity_fallback_mentions(drugs, resolved_drugs)
        if (
            not unresolved
            or llm_client is None
            or not hasattr(llm_client, "llm_structured_call")
            or not (model or "").strip()
        ):
            return resolved_drugs

        mention_list = [entry.name.strip() for entry, _payload in unresolved.values()]
        user_prompt = (
            "Normalize these unresolved medication labels. Return exactly one proposal "
            "for each label, preserving `original_mention` exactly:\n"
            + "\n".join(f"- {name}" for name in mention_list)
        )
        try:
            parsed = await asyncio.wait_for(
                llm_client.llm_structured_call(
                    model=str(model).strip(),
                    system_prompt=self.IDENTITY_FALLBACK_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=DrugIdentityProposalBatch,
                    temperature=float(temperature),
                    use_json_mode=True,
                    max_repair_attempts=1,
                ),
                timeout=max(
                    float(get_server_settings().runtime.minimum_llm_timeout),
                    float(get_server_settings().runtime.parser_llm_timeout),
                ),
            )
            proposals = DrugIdentityProposalBatch.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Drug identity LLM fallback failed: %s", exc)
            return resolved_drugs

        for proposal in proposals.proposals:
            original_key = normalize_drug_query_name(proposal.original_mention)
            unresolved_item = unresolved.get(original_key)
            if unresolved_item is None:
                logger.warning(
                    "Ignoring identity proposal for unknown mention '%s'",
                    proposal.original_mention,
                )
                continue
            original_entry, original_payload = unresolved_item
            candidate_names = self.proposal_candidate_names(proposal)
            accepted_payloads: dict[str, dict[str, Any]] = {}
            attempted_payloads: list[dict[str, Any]] = []
            for candidate_name in candidate_names:
                candidate_drugs = PatientDrugs(
                    entries=[
                        DrugEntry(
                            name=candidate_name,
                            dosage=None,
                            administration_mode=None,
                            route=None,
                            administration_pattern=None,
                            suspension_status=None,
                            suspension_date=None,
                            therapy_start_status=None,
                            therapy_start_date=None,
                            source=original_entry.source,
                        )
                    ]
                )
                candidate_results = await asyncio.to_thread(
                    resolver.resolve,
                    candidate_drugs,
                )
                attempted_payloads.extend(candidate_results.values())
                for candidate_payload in candidate_results.values():
                    if not self.is_locally_accepted(candidate_payload):
                        continue
                    accepted_key = str(candidate_payload["lookup_key"])
                    accepted_payloads[accepted_key] = candidate_payload

            identity_audit = {
                "source": "llm_candidate_local_validation",
                "original_mention": proposal.original_mention,
                "proposed_canonical_name": proposal.proposed_canonical_name,
                "alternate_names": proposal.alternate_names,
                "ingredients": proposal.ingredients,
                "confidence": proposal.confidence,
                "rationale": proposal.rationale,
                "candidate_names": candidate_names,
                "accepted_local_names": [
                    payload.get("accepted_livertox_name")
                    for payload in accepted_payloads.values()
                ],
            }
            if len(accepted_payloads) == 1:
                self.remove_unresolved_payload(resolved_drugs, original_payload)
                for accepted_key, accepted_payload in accepted_payloads.items():
                    self.attach_validated_identity_provenance(
                        accepted_payload,
                        original_payload=original_payload,
                        identity_audit=identity_audit,
                    )
                    existing = resolved_drugs.get(accepted_key)
                    resolved_drugs[accepted_key] = (
                        resolver._merge_payload(existing, accepted_payload)
                        if existing is not None
                        else accepted_payload
                    )
                continue
            self.attach_unresolved_identity_audit(
                original_payload,
                identity_audit=identity_audit,
                attempted_payloads=attempted_payloads,
                multiple_accepted_candidates=len(accepted_payloads) > 1,
            )
        return resolved_drugs

    # -------------------------------------------------------------------------
    @staticmethod
    def collect_identity_fallback_mentions(
        drugs: PatientDrugs,
        resolved_drugs: dict[str, dict[str, Any]],
    ) -> dict[str, tuple[DrugEntry, dict[str, Any]]]:
        entries_by_name = {
            normalize_drug_query_name(entry.name): entry
            for entry in drugs.entries
            if (entry.name or "").strip()
        }
        unresolved: dict[str, tuple[DrugEntry, dict[str, Any]]] = {}
        for payload in resolved_drugs.values():
            decision_status = str(payload.get("decision_status") or "")
            confidence = payload.get("match_confidence")
            low_confidence = (
                isinstance(confidence, int | float) and float(confidence) < 0.75
            )
            if (
                decision_status
                not in {
                    "missing_livertox",
                    "missing_rxnav",
                    "ambiguous_requires_review",
                }
                and not low_confidence
            ):
                continue
            raw_mentions = payload.get("raw_mentions") or []
            for raw_mention in raw_mentions:
                normalized = normalize_drug_query_name(str(raw_mention))
                entry = entries_by_name.get(normalized)
                if entry is not None:
                    unresolved[normalized] = (entry, payload)
        return unresolved

    # -------------------------------------------------------------------------
    @staticmethod
    def proposal_candidate_names(proposal: Any) -> list[str]:
        values = [
            proposal.proposed_canonical_name,
            *proposal.alternate_names,
            *proposal.ingredients,
        ]
        candidates: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            normalized = normalize_drug_query_name(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(text)
        return candidates

    # -------------------------------------------------------------------------
    @staticmethod
    def is_locally_accepted(payload: dict[str, Any]) -> bool:
        return bool(
            str(payload.get("decision_status") or "").startswith("accepted_")
            and payload.get("accepted_livertox_name")
            and payload.get("matched_livertox_row")
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def remove_unresolved_payload(
        resolved_drugs: dict[str, dict[str, Any]],
        original_payload: dict[str, Any],
    ) -> None:
        for key, payload in list(resolved_drugs.items()):
            if payload is original_payload:
                resolved_drugs.pop(key, None)

    # -------------------------------------------------------------------------
    @staticmethod
    def attach_validated_identity_provenance(
        payload: dict[str, Any],
        *,
        original_payload: dict[str, Any],
        identity_audit: dict[str, Any],
    ) -> None:
        payload["raw_mentions"] = list(
            dict.fromkeys(
                [
                    *(original_payload.get("raw_mentions") or []),
                    *(payload.get("raw_mentions") or []),
                ]
            )
        )
        payload["origins"] = list(
            dict.fromkeys(
                [
                    *(original_payload.get("origins") or []),
                    *(payload.get("origins") or []),
                ]
            )
        )
        payload["extraction_metadata"] = [
            *(original_payload.get("extraction_metadata") or []),
            *(payload.get("extraction_metadata") or []),
        ]
        payload.setdefault("identity_candidates", []).append(identity_audit)
        reasons = list(payload.get("match_notes") or [])
        reasons.extend(
            [
                "identity proposed by configured LLM",
                "identity accepted only after unique local evidence resolution",
            ]
        )
        payload["match_notes"] = list(dict.fromkeys(reasons))
        decision = payload.get("resolution_decision")
        if isinstance(decision, dict):
            decision_reasons = list(decision.get("reasons") or [])
            decision_reasons.extend(payload["match_notes"])
            decision["reasons"] = list(dict.fromkeys(decision_reasons))
            payload["match_reason"] = next(
                (
                    str(reason).strip()
                    for reason in decision["reasons"]
                    if str(reason).strip()
                ),
                None,
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def attach_unresolved_identity_audit(
        payload: dict[str, Any],
        *,
        identity_audit: dict[str, Any],
        attempted_payloads: list[dict[str, Any]],
        multiple_accepted_candidates: bool,
    ) -> None:
        payload.setdefault("identity_candidates", []).append(identity_audit)
        notes = list(payload.get("match_notes") or [])
        notes.append(
            "LLM identity candidates did not produce a unique local evidence match"
        )
        payload["match_notes"] = list(dict.fromkeys(notes))
        if multiple_accepted_candidates:
            candidate_names = [
                str(name)
                for name in identity_audit.get("accepted_local_names") or []
                if name
            ]
            payload["ambiguous_match"] = True
            payload["requires_human_review"] = True
            payload["decision_status"] = "ambiguous_requires_review"
            payload["match_status"] = "ambiguous_requires_review"
            payload["match_candidates"] = list(dict.fromkeys(candidate_names))
        decision = payload.get("resolution_decision")
        if isinstance(decision, dict):
            reasons = list(decision.get("reasons") or [])
            reasons.extend(payload["match_notes"])
            decision["reasons"] = list(dict.fromkeys(reasons))
            if multiple_accepted_candidates:
                decision["decision_status"] = "ambiguous_requires_review"
                decision["requires_human_review"] = True
        for attempted in attempted_payloads:
            for field_name in ("rxnav_candidates", "livertox_candidates"):
                existing = payload.setdefault(field_name, [])
                for candidate in attempted.get(field_name) or []:
                    if candidate not in existing:
                        existing.append(candidate)
                if isinstance(decision, dict):
                    decision_existing = decision.setdefault(field_name, [])
                    for candidate in attempted.get(field_name) or []:
                        if candidate not in decision_existing:
                            decision_existing.append(candidate)

    # -------------------------------------------------------------------------
    def build_match_audit_issues(
        self,
        resolved_drugs: dict[str, dict[str, Any]] | None,
    ) -> list[PipelineIssue]:
        issues: list[PipelineIssue] = []
        if not resolved_drugs:
            return issues
        for payload in resolved_drugs.values():
            raw_mentions = payload.get("raw_mentions") or []
            raw_label = ", ".join(str(item) for item in raw_mentions if item) or str(
                payload.get("drug_name") or payload.get("canonical_name") or "unknown"
            )
            status = str(payload.get("match_status") or "").lower()
            confidence = payload.get("match_confidence")
            if payload.get("missing_livertox") or status in {
                "missing",
                "missing_match",
                "no_match",
            }:
                issues.append(
                    PipelineIssue(
                        severity="warning",
                        code="livertox_match_missing",
                        field="matched_drugs",
                        message=f"No LiverTox match was validated for {raw_label}.",
                    )
                )
            if payload.get("ambiguous_match") or status == "ambiguous":
                issues.append(
                    PipelineIssue(
                        severity="warning",
                        code="livertox_match_ambiguous",
                        field="matched_drugs",
                        message=f"LiverTox match is ambiguous for {raw_label}.",
                    )
                )
            if isinstance(confidence, int | float) and float(confidence) < 0.75:
                issues.append(
                    PipelineIssue(
                        severity="warning",
                        code="livertox_match_low_confidence",
                        field="matched_drugs",
                        message=f"LiverTox match confidence is low for {raw_label}.",
                    )
                )
            if (
                not payload.get("rxnav_validated")
                and payload.get("rxnav_rxcui") is None
            ):
                issues.append(
                    PipelineIssue(
                        severity="warning",
                        code="rxnav_alias_not_validated",
                        field="matched_drugs",
                        message=f"RxNav alias was not validated for {raw_label}.",
                    )
                )
        return issues

    # -------------------------------------------------------------------------
    @staticmethod
    def emit_progress(
        progress_callback: Callable[[float], None] | None,
        fraction: float,
    ) -> None:
        if progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        progress_callback(bounded_fraction)

    # -------------------------------------------------------------------------
    async def ensure_livertox_matcher(self) -> bool:
        if self.livertox_matcher is not None:
            return True
        try:
            dataset = await asyncio.to_thread(self.serializer.get_livertox_records)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading LiverTox monographs from database: %s", exc)
            self.livertox_matcher = None
            return False
        if dataset is None or dataset.empty:
            logger.warning(
                "LiverTox monograph table is empty; toxicity essay cannot run"
            )
            self.livertox_matcher = None
            return False
        catalog_stream = self.serializer.stream_drugs_catalog()
        try:
            self.livertox_matcher = await asyncio.to_thread(
                LiverToxMatcher,
                dataset,
                drugs_catalog_df=catalog_stream,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed preparing LiverTox matcher: %s", exc)
            self.livertox_matcher = None
            return False
        return True

    # -------------------------------------------------------------------------
    @staticmethod
    def build_pattern_prompt(
        pattern_score: HepatotoxicityPatternScore | None,
    ) -> str:
        if pattern_score is None:
            return "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."
        classification = pattern_score.classification.replace("_", " ")
        segments: list[str] = [
            f"Observed liver injury pattern: {classification.capitalize()}.",
        ]
        if pattern_score.r_score is not None:
            segments.append(f"R ratio ≈ {pattern_score.r_score:.2f}.")
        if pattern_score.alt_multiple is not None:
            segments.append(
                f"ALT is about {pattern_score.alt_multiple:.2f} × the upper reference limit."
            )
        if pattern_score.alp_multiple is not None:
            segments.append(
                f"ALP is about {pattern_score.alp_multiple:.2f} × the upper reference limit."
            )
        segments.append(
            "Treat drugs whose known hepatotoxicity pattern matches this classification as stronger causal candidates, and downgrade mismatches."
        )
        return " ".join(segments)
