from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Callable
from typing import Any, Literal, cast

from common.prompts.medication_extraction import (
    ANAMNESIS_DRUG_EXTRACTION_SYSTEM_PROMPT,
    DRUG_EXTRACTION_SYSTEM_PROMPT,
    LOCAL_DRUG_EXTRACTION_SYSTEM_PROMPT,
    build_medication_extraction_retry_prompt,
    build_medication_extraction_user_prompt,
)
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    DrugEntry,
    PatientDrugs,
)
from services.clinical.deterministic_extraction import (
    DeterministicDrugExtractionResult,
)
from services.clinical.drug_blocks import isolate_drug_blocks
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider
from services.clinical.extraction_strategy import decide_extraction_strategy
from common.utils.text_utils import normalize_token
from services.clinical.parser_host import ParserHost
from domain.clinical.extractor_contracts import (
    LocalDrugEntryDraft,
    LocalPatientDrugs,
)


###############################################################################
class DrugLlmExtractionMixin(ParserHost):
    # -------------------------------------------------------------------------
    def active_provider_name(self) -> str | None:
        provider = self.forced_provider or self.client_provider
        if provider == "injected":
            return self.forced_provider
        return provider

    # -------------------------------------------------------------------------
    @staticmethod
    def is_local_runtime(provider: str | None) -> bool:
        return (provider or "").strip().casefold() == "ollama"

    # -------------------------------------------------------------------------
    def normalize_local_drug_entry(self, entry: LocalDrugEntryDraft) -> DrugEntry:
        current_status = (entry.current_status or "").strip().casefold() or None
        if current_status not in {
            "current",
            "past",
            "suspected",
            "ruled_out",
            "unclear",
        }:
            current_status = None
        return DrugEntry(
            name=entry.name,
            dosage=entry.dosage,
            administration_mode=entry.administration_mode,
            route=entry.route,
            administration_pattern=entry.administration_pattern,
            suspension_status=entry.suspension_status,
            suspension_date=entry.suspension_date,
            therapy_start_status=entry.therapy_start_status,
            therapy_start_date=entry.therapy_start_date,
            evidence=entry.evidence,
            current_status=cast(
                Literal["current", "past", "suspected", "ruled_out", "unclear"] | None,
                current_status,
            ),
        )

    # -------------------------------------------------------------------------
    async def ensure_client(self) -> None:
        revision = LLMRuntimeConfig.get_revision()
        resolved_provider, resolved_model = LLMRuntimeConfig.resolve_provider_and_model(
            "parser"
        )
        provider = self.forced_provider or resolved_provider
        model = self.forced_model or resolved_model
        await ensure_runtime_client(
            self,
            provider=provider,
            model=model,
            revision=revision,
            client_factory=lambda selected_provider, selected_model: (
                select_llm_provider(
                    provider=selected_provider,
                    default_model=selected_model,
                    timeout_s=self.timeout_s,
                    max_retries=0,
                )
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def minimum_timeout_s() -> float:
        return float(get_server_settings().runtime.minimum_llm_timeout)

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
    def clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        for raw_line in normalized.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = self.BULLET_RE.sub("", stripped)
            if not stripped:
                continue
            lines.append(stripped)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def conservative_prepare_drug_section_text(self, text: str | None) -> str:
        if not text:
            return ""
        prepared = unicodedata.normalize("NFKC", text)
        prepared = prepared.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join(
            line.rstrip() for line in prepared.split("\n") if line.rstrip()
        )

    # -------------------------------------------------------------------------
    def parse_drug_list(self, text: str | None) -> PatientDrugs:
        cleaned_text = self.clean_text(text)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.extract_drugs_from_therapy(cleaned_text))
        raise RuntimeError(
            "parse_drug_list cannot be used inside a running event loop; use"
            " 'await extract_drugs_from_therapy(...)' instead."
        )

    # -------------------------------------------------------------------------
    async def extract_drugs_from_therapy(
        self,
        text: str | None,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDrugs:
        cleaned = (
            (text or "")
            if already_cleaned
            else self.conservative_prepare_drug_section_text(text)
        )
        if not cleaned:
            return PatientDrugs(entries=[])
        self.emit_progress(progress_callback, 0.0)
        try:
            structured = await self.llm_extract_drugs_from_section(
                cleaned,
                source="therapy",
                historical_flag=False,
                progress_callback=progress_callback,
            )
            combined = self._attach_therapy_evidence_spans(
                structured.entries,
                cleaned,
            )
            logger.info(
                "Therapy LLM extraction succeeded with %s normalized entries",
                len(combined),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Therapy LLM extraction failed; using deterministic fallback: %s",
                exc,
            )
            combined = await self.extract_drugs_from_therapy_hybrid(
                cleaned,
                progress_callback=progress_callback,
            )
            combined = self._attach_therapy_evidence_spans(combined, cleaned)
        self.emit_progress(progress_callback, 1.0)
        return PatientDrugs(entries=self.deduplicate_drug_entries(combined))

    # -------------------------------------------------------------------------
    async def extract_drugs_from_therapy_with_audit(
        self,
        text: str | None,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        cleaned = (
            (text or "")
            if already_cleaned
            else self.conservative_prepare_drug_section_text(text)
        )
        if not cleaned:
            decision = decide_extraction_strategy(
                section="therapy",
                meaningful_line_count=0,
                parsed_line_count=0,
                unresolved_line_count=0,
                evidence_span_count=0,
            )
            return {
                "patient_drugs": PatientDrugs(entries=[]),
                "strategy": decision.strategy,
                "decision": decision.model_dump(),
                "unresolved_lines": [],
                "warnings": [],
            }
        deterministic = self.extract_drugs_from_therapy_deterministic(cleaned)
        meaningful_line_count = len(
            [
                block.text
                for block in isolate_drug_blocks(cleaned)
                if block.text.strip()
                and not self.is_non_therapy_line(block.text.strip())
            ]
        )
        structured_succeeded = False
        try:
            structured = await self.llm_extract_drugs_from_section(
                cleaned,
                source="therapy",
                historical_flag=False,
                progress_callback=progress_callback,
            )
            combined = structured.entries
            structured_succeeded = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Therapy audit LLM extraction failed; using deterministic fallback: %s",
                exc,
            )
            combined = await self.extract_drugs_from_therapy_hybrid(
                cleaned,
                progress_callback=progress_callback,
            )
        entries = self._attach_therapy_evidence_spans(combined, cleaned)
        decision = decide_extraction_strategy(
            section="therapy",
            meaningful_line_count=meaningful_line_count,
            parsed_line_count=max(
                0, meaningful_line_count - len(deterministic.unresolved_lines)
            ),
            unresolved_line_count=len(deterministic.unresolved_lines),
            evidence_span_count=sum(1 for entry in entries if entry.source_span),
        )
        if structured_succeeded:
            decision = decision.model_copy(
                update={
                    "strategy": "llm",
                    "reasons": [
                        "complete therapy corpus processed by structured LLM extraction"
                    ],
                }
            )
        return {
            "patient_drugs": PatientDrugs(
                entries=self.deduplicate_drug_entries(entries)
            ),
            "strategy": decision.strategy,
            "decision": decision.model_dump(),
            "unresolved_lines": deterministic.unresolved_lines,
            "warnings": [
                {"code": "therapy_unresolved_line", "raw_line": line}
                for line in deterministic.unresolved_lines
            ],
        }

    # -------------------------------------------------------------------------
    def _attach_therapy_evidence_spans(
        self,
        entries: list[DrugEntry],
        source_text: str,
    ) -> list[DrugEntry]:
        updated: list[DrugEntry] = []
        for entry in entries:
            if entry.source_span and entry.evidence:
                updated.append(entry)
                continue
            raw_name = (entry.name or "").strip()
            start, end = self.find_grounded_name_span(raw_name, source_text)
            if start < 0:
                updated.append(
                    entry.model_copy(
                        update={
                            "source": entry.source or "therapy",
                            "confidence": entry.confidence or "moderate",
                        }
                    )
                )
                continue
            updated.append(
                entry.model_copy(
                    update={
                        "source": entry.source or "therapy",
                        "evidence": source_text[start:end],
                        "source_span": [start, end],
                        "confidence": entry.confidence or "high",
                        "attribution": entry.attribution or "patient",
                        "current_status": entry.current_status or "unclear",
                    }
                )
            )
        return updated

    # -------------------------------------------------------------------------
    async def extract_drugs_from_therapy_hybrid(
        self,
        cleaned: str,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[DrugEntry]:
        deterministic = self.extract_drugs_from_therapy_deterministic(cleaned)
        deterministic_entries = deterministic.entries
        fallback = list(enumerate(deterministic.unresolved_lines))
        if not fallback:
            self.emit_progress(progress_callback, 1.0)
            return self.deduplicate_drug_entries(deterministic_entries)

        if self.client is None or not hasattr(self.client, "llm_structured_call"):
            self.emit_progress(progress_callback, 1.0)
            return self.deduplicate_drug_entries(deterministic_entries)

        fallback_lines = [line for _, line in fallback]
        try:
            structured = await self.llm_extract_drugs_from_section(
                cleaned,
                source="therapy",
                historical_flag=False,
                progress_callback=progress_callback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Contextual therapy LLM extraction failed for %s unresolved spans; using deterministic entries only: %s",
                len(fallback_lines),
                exc,
            )
            structured = PatientDrugs(entries=[])

        llm_entries: list[DrugEntry] = []
        for entry in structured.entries:
            grounded = self.validate_llm_drug_entry_grounding(
                entry,
                source_text=cleaned,
                source="therapy",
                historical_flag=False,
            )
            if grounded is not None:
                llm_entries.append(grounded)
        return self.deduplicate_drug_entries([*deterministic_entries, *llm_entries])

    # -------------------------------------------------------------------------
    def extract_drugs_from_therapy_deterministic(
        self,
        cleaned: str,
    ) -> DeterministicDrugExtractionResult:
        lines = [
            block.text.strip()
            for block in isolate_drug_blocks(cleaned)
            if block.text.strip() and not self.is_non_therapy_line(block.text.strip())
        ]
        parsed, fallback = self.rule_based_parse(lines)
        return DeterministicDrugExtractionResult(
            entries=self.deduplicate_drug_entries([entry for _, entry in parsed]),
            unresolved_lines=[line for _, line in fallback],
            regimen_lines=[],
        )

    # -------------------------------------------------------------------------
    async def llm_extract_drugs_from_section(
        self,
        text: str,
        *,
        source: Literal["anamnesis", "therapy"],
        historical_flag: bool,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDrugs:
        await self.ensure_client()
        if self.client is None:
            raise RuntimeError(self.LLM_CLIENT_NOT_INITIALIZED_ERROR)
        if not text.strip():
            return PatientDrugs(entries=[])
        if not hasattr(self.client, "llm_structured_call"):
            logger.warning(
                "LLM structured call unavailable; falling back to deterministic parsing for %s section.",
                source,
            )
            entries: list[DrugEntry]
            if source == "anamnesis":
                entries = self.extract_drugs_from_anamnesis_rule_based(text)
            else:
                lines = [
                    block.text.strip()
                    for block in isolate_drug_blocks(text)
                    if block.text.strip()
                    and not self.is_non_therapy_line(block.text.strip())
                ]
                entries = []
                for line in lines:
                    candidate = self.parse_line(line)
                    normalized_entry = self.normalize_entry(
                        candidate,
                        source=source,
                        historical_flag=historical_flag,
                    )
                    if normalized_entry is not None:
                        entries.append(normalized_entry)
            self.emit_progress(progress_callback, 1.0)
            return PatientDrugs(entries=entries)

        source_text = self.clean_text(text)
        use_local_schema = self.is_local_runtime(self.active_provider_name())
        schema_model = LocalPatientDrugs if use_local_schema else PatientDrugs
        system_prompt = (
            LOCAL_DRUG_EXTRACTION_SYSTEM_PROMPT
            if use_local_schema
            else (
                ANAMNESIS_DRUG_EXTRACTION_SYSTEM_PROMPT
                if source == "anamnesis"
                else DRUG_EXTRACTION_SYSTEM_PROMPT
            )
        )
        user_prompt = build_medication_extraction_user_prompt(
            source_text=source_text,
            source=source,
        )
        last_wrong_output = ""
        last_errors: list[str] = []
        max_attempts = 2
        deterministic_reference = (
            self.extract_drugs_from_anamnesis_deterministic(source_text)
            if source == "anamnesis"
            else self.extract_drugs_from_therapy_deterministic(source_text)
        )
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                user_prompt = build_medication_extraction_retry_prompt(
                    source_text=source_text,
                    validation_errors=last_errors,
                    previous_output=last_wrong_output,
                )
            raw_parsed = await asyncio.wait_for(
                self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema_model,
                    purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
                    use_json_mode=True,
                    max_repair_attempts=1,
                ),
                timeout=max(self.minimum_timeout_s(), float(self.timeout_s)),
            )
            parsed = (
                PatientDrugs(
                    entries=[
                        self.normalize_local_drug_entry(entry)
                        for entry in raw_parsed.entries
                    ]
                )
                if use_local_schema
                else raw_parsed
            )
            last_wrong_output = parsed.model_dump_json()
            normalized_candidates = [
                self.normalize_entry(
                    entry,
                    source=source,
                    historical_flag=historical_flag,
                )
                for entry in parsed.entries
            ]
            grounded_candidates: list[DrugEntry] = []
            for entry in normalized_candidates:
                if entry is None:
                    continue
                grounded = self.validate_llm_drug_entry_grounding(
                    entry,
                    source_text=source_text,
                    source=source,
                    historical_flag=historical_flag,
                )
                if grounded is not None:
                    grounded_candidates.append(grounded)
            filtered_candidates = self.deduplicate_drug_entries(grounded_candidates)
            source_has_medication_signal = (
                any(
                    self.is_likely_medication_line(line)
                    for line in source_text.splitlines()
                    if line.strip()
                )
                or self.has_medication_context_signal(source_text)
                or bool(deterministic_reference.entries)
            )
            missing_grounded = self.find_missing_grounded_reference_entries(
                filtered_candidates,
                deterministic_reference.entries,
                source_text=source_text,
            )
            if filtered_candidates or not source_has_medication_signal:
                if missing_grounded and attempt < max_attempts:
                    last_errors = self.describe_missing_reference_entries(
                        missing_grounded
                    )
                    continue
                self.emit_progress(progress_callback, 1.0)
                merged_candidates = (
                    [*filtered_candidates, *missing_grounded]
                    if missing_grounded
                    else filtered_candidates
                )
                return PatientDrugs(
                    entries=self.deduplicate_drug_entries(merged_candidates)
                )
            last_errors = [
                "The model returned no valid medication entries despite medication-like source evidence."
            ]
        raise RuntimeError("LLM drug extraction produced no semantically valid entries")

    # -------------------------------------------------------------------------
    def find_missing_grounded_reference_entries(
        self,
        llm_entries: list[DrugEntry],
        reference_entries: list[DrugEntry],
        *,
        source_text: str,
    ) -> list[DrugEntry]:
        missing: list[DrugEntry] = []
        llm_names = {normalize_token(entry.name) for entry in llm_entries}
        llm_evidence = {
            self.normalize_filter_key(entry.evidence or "")
            for entry in llm_entries
            if entry.evidence
        }
        for reference in reference_entries:
            grounded = self.attach_source_grounding(
                reference,
                source_text=source_text,
                historical_flag=bool(reference.historical_flag),
                require_medication_syntax=False,
            )
            if grounded is None:
                continue
            normalized_name = normalize_token(grounded.name)
            normalized_evidence = self.normalize_filter_key(grounded.evidence or "")
            if normalized_name in llm_names:
                continue
            if any(
                normalized_name and llm_name and llm_name in normalized_name
                for llm_name in llm_names
            ):
                continue
            if normalized_evidence and normalized_evidence in llm_evidence:
                continue
            if any(
                normalized_evidence and evidence and evidence in normalized_evidence
                for evidence in llm_evidence
            ):
                continue
            missing.append(grounded)
        return missing

    # -------------------------------------------------------------------------
    def describe_missing_reference_entries(self, entries: list[DrugEntry]) -> list[str]:
        messages: list[str] = []
        for entry in entries[:10]:
            evidence = (entry.evidence or entry.name or "").strip()
            if evidence:
                messages.append(f"Missing explicit medication evidence: {evidence}")
        if len(entries) > 10:
            messages.append(
                f"Missing {len(entries) - 10} additional medication entries."
            )
        return messages or ["Medication-like source evidence was omitted."]

    # -------------------------------------------------------------------------
