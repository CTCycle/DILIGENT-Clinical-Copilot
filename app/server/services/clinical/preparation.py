# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from common.prompts.clinical_context import build_hepatotoxicity_pattern_context
from common.prompts.identity import (
    MEDICATION_IDENTITY_SYSTEM_PROMPT,
    build_medication_identity_user_prompt,
)
from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.clinical.drug_resolution import (
    ACCEPTED_DECISION_STATUSES,
    DrugIdentityProposalBatch,
)
from domain.clinical.entities import (
    DrugEntry,
    EvidenceSelection,
    HepatotoxicityPatternScore,
    PatientDrugs,
    PipelineIssue,
)
from domain.clinical.extras import HepatoxPreparedInputs
from repositories import values as repository_values
from repositories.knowledge_repository import KnowledgeRepository
from repositories.serialization.catalogs import ReferenceCatalogSerializer
from repositories.terminology_lookup_repository import TerminologyLookupRepository
from services.clinical.knowledge import ClinicalKnowledgeComposer
from services.clinical.drug_resolution import DrugResolutionService
from services.llm.generation_policy import GenerationPurpose
from services.terminology.factory import build_terminology_resolver
from services.text.normalization import normalize_drug_query_name

FALLBACK_STATUSES = frozenset({"missing_livertox", "ambiguous_requires_review", "unresolved"})

###############################################################################
class ClinicalKnowledgePreparation:

    # -------------------------------------------------------------------------
    def __init__(self, *, knowledge_repository: KnowledgeRepository) -> None:
        self.knowledge_repository = knowledge_repository
        self.knowledge_composer = ClinicalKnowledgeComposer(
            knowledge_repository=knowledge_repository
        )
        self.terminology_lookup = TerminologyLookupRepository(knowledge_repository.context)

    # -------------------------------------------------------------------------
    def build_resolver(self) -> DrugResolutionService:
        return DrugResolutionService(
            build_terminology_resolver(self.terminology_lookup),
            self.terminology_lookup,
            self.knowledge_repository,
        )

    # -------------------------------------------------------------------------
    def record_drug_mention_observations(self, matched_drugs: list[dict[str, Any]]) -> bool:
        """Record mentions without an accepted identity for vocabulary review.

        Terminology aliases are never learned from sessions: only verified
        terminology relations and explicit user confirmations create aliases.
        """
        catalog_serializer = ReferenceCatalogSerializer(self.knowledge_repository.session_factory)
        changed = False
        with self.knowledge_repository.session_factory() as db_session:
            for item in matched_drugs:
                raw_name = repository_values.normalize_string(
                    item.get("raw_drug_name") or item.get("name")
                )
                terminology = item.get("terminology")
                terminology = terminology if isinstance(terminology, dict) else {}
                if raw_name is None or terminology.get("accepted"):
                    continue
                catalog_serializer.upsert_runtime_observation(
                    term=raw_name,
                    category="observed_unresolved_query",
                    source="session",
                    is_active=True,
                    db_session=db_session,
                )
                changed = True
            db_session.commit()
        return changed

    # -------------------------------------------------------------------------
    async def prepare_inputs(
        self,
        drugs: PatientDrugs,
        *,
        clinical_context: str | None,
        pattern_score: HepatotoxicityPatternScore | None,
        evidence: EvidenceSelection,
        progress_callback: Callable[[float], None] | None = None,
        identity_resolution_client: Any | None = None,
        identity_resolution_model: str | None = None,
        identity_resolution_temperature: float = 0.0,
    ) -> HepatoxPreparedInputs | None:
        self.emit_progress(progress_callback, 0.0)
        self.emit_progress(progress_callback, 0.2)
        if await asyncio.to_thread(self.knowledge_repository.count_livertox_monographs) == 0:
            logger.warning("LiverTox monograph table is empty; toxicity essay cannot run")
            return None
        resolver = self.build_resolver()
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
        self.knowledge_composer.enrich_resolved_drugs(resolved_drugs, evidence)
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
        user_prompt = build_medication_identity_user_prompt(mention_list)
        try:
            parsed = await asyncio.wait_for(
                llm_client.llm_structured_call(
                    model=str(model).strip(),
                    system_prompt=MEDICATION_IDENTITY_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=DrugIdentityProposalBatch,
                    purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
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
                        resolver.merge_payload(existing, accepted_payload)
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
            if str(payload.get("decision_status") or "") not in FALLBACK_STATUSES:
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
        """LLM proposals count only when they resolve to verified terminology."""
        terminology = payload.get("terminology")
        terminology = terminology if isinstance(terminology, dict) else {}
        return bool(
            str(payload.get("decision_status") or "") in ACCEPTED_DECISION_STATUSES
            and terminology.get("status") == "verified"
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
            for field_name in ("livertox_candidates",):
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
        *,
        detected_drug_names: list[str] | None = None,
    ) -> list[PipelineIssue]:
        issues: list[PipelineIssue] = []
        if not resolved_drugs:
            return issues
        detected_keys = (
            {
                normalize_drug_query_name(name)
                for name in detected_drug_names
                if normalize_drug_query_name(name)
            }
            if detected_drug_names is not None
            else None
        )
        direct_keys = {normalize_drug_query_name(key) for key in resolved_drugs}
        for key, payload in resolved_drugs.items():
            # A payload re-keyed by the identity fallback is still audited for
            # its mention unless another payload is keyed by that mention.
            mention_keys = {
                normalize_drug_query_name(str(name))
                for name in payload.get("raw_mentions") or []
            }
            if (
                detected_keys is not None
                and normalize_drug_query_name(key) not in detected_keys
                and not (mention_keys & detected_keys) - direct_keys
            ):
                continue
            raw_mentions = payload.get("raw_mentions") or []
            raw_label = ", ".join(str(item) for item in raw_mentions if item) or str(
                payload.get("drug_name") or payload.get("canonical_name") or "unknown"
            )
            issues.extend(self.match_audit_issues_for(payload, raw_label))
        return issues

    # -------------------------------------------------------------------------
    @staticmethod
    def match_audit_issues_for(payload: dict[str, Any], raw_label: str) -> list[PipelineIssue]:
        terminology = payload.get("terminology")
        terminology = terminology if isinstance(terminology, dict) else {}
        status = str(payload.get("decision_status") or "")
        resolved = terminology.get("preferred_name") or payload.get("drug_name") or "another name"

        def issue(code: str, message: str) -> PipelineIssue:
            return PipelineIssue(severity="warning", code=code, field="matched_drugs", message=message)

        if status == "unresolved":
            if terminology.get("method") == "product_unverified":
                return [
                    issue(
                        "drug_composition_unverified",
                        f"{raw_label} matches a product whose composition could not be verified.",
                    )
                ]
            return [issue("drug_identity_unresolved", f"No terminology identity was verified for {raw_label}.")]
        if status == "ambiguous_requires_review":
            candidates = ", ".join(
                str(item.get("name")) for item in terminology.get("candidates") or [] if item.get("name")
            )
            suffix = f" Candidates: {candidates}." if candidates else ""
            return [issue("drug_identity_ambiguous", f"Drug identity is ambiguous for {raw_label}.{suffix}")]
        issues: list[PipelineIssue] = []
        if terminology.get("status") == "probable":
            issues.append(
                issue(
                    "drug_identity_probable",
                    f"{raw_label} was matched to {resolved} by spelling correction; confirm the drug identity.",
                )
            )
        if status == "missing_livertox":
            issues.append(
                issue("livertox_link_missing", f"No LiverTox record is linked to {resolved} ({raw_label}).")
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
    @staticmethod
    def build_pattern_prompt(
        pattern_score: HepatotoxicityPatternScore | None,
    ) -> str:
        if pattern_score is None:
            return build_hepatotoxicity_pattern_context(
                classification=None,
                r_score=None,
                alt_multiple=None,
                alp_multiple=None,
            )
        return build_hepatotoxicity_pattern_context(
            classification=pattern_score.classification,
            r_score=pattern_score.r_score,
            alt_multiple=pattern_score.alt_multiple,
            alp_multiple=pattern_score.alp_multiple,
        )
