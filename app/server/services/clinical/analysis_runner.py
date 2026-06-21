from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any

from common.utils.logger import logger
from domain.clinical.claims import ClinicalClaim, DrugClinicalNarrative
from domain.clinical.entities import (
    DrugClinicalAssessment,
    DrugEntry,
    DrugRucamAssessment,
    PatientDrugClinicalReport,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from services.clinical.match_quality import classify_match_evidence
from services.clinical.preparation import HepatoxPreparedInputs
from services.text.normalization import normalize_drug_query_name

###############################################################################
class AnalysisRunner:
    """Orchestrates the top-level analysis workflow — runs the full drug assessment pipeline."""

    # -------------------------------------------------------------------------
    def __init__(self, consultation: Any) -> None:
        self.consultation = consultation

    # -------------------------------------------------------------------------
    @staticmethod
    def build_clinical_narrative(
        *,
        drug_name: str,
        excerpts: list[str],
        rucam: DrugRucamAssessment | None,
        missing_livertox: bool,
        evidence_warnings: list[str],
    ) -> DrugClinicalNarrative:
        claims: list[ClinicalClaim] = []
        if excerpts:
            claims.append(
                ClinicalClaim(
                    claim=f"{drug_name} has source-text evidence in the clinical record.",
                    source="source_text",
                    evidence_quote=excerpts[0],
                    confidence="high",
                    requires_review=False,
                )
            )
        else:
            claims.append(
                ClinicalClaim(
                    claim=f"{drug_name} lacks a direct source-text evidence quote in the generated assessment context.",
                    source="unknown",
                    evidence_quote=None,
                    confidence="low",
                    requires_review=True,
                )
            )
        if rucam is not None:
            confidence = "moderate" if rucam.data_sufficient else "low"
            claims.append(
                ClinicalClaim(
                    claim=(
                        f"{drug_name} RUCAM causality is {rucam.causality_category}."
                    ),
                    source="rucam",
                    evidence_quote=None,
                    confidence=confidence,
                    requires_review=not rucam.data_sufficient,
                )
            )
        limitations = list(evidence_warnings)
        if missing_livertox:
            limitations.append("No matched LiverTox monograph was available.")
        if rucam is not None:
            limitations.extend(rucam.limitations)
        return DrugClinicalNarrative(
            drug_name=drug_name,
            summary=f"Auditable claim envelope for {drug_name}.",
            claims=claims,
            limitations=limitations,
        )

    # -------------------------------------------------------------------------
    async def run_analysis(
        self,
        *,
        prepared_inputs: HepatoxPreparedInputs | None,
        visit_date: date | None = None,
        report_language: str = "en",
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any] | None:
        if prepared_inputs is None:
            logger.info("No prepared inputs provided; skipping hepatotoxicity consultation")
            return None
        resolved_mapping = prepared_inputs.resolved_drugs
        if not resolved_mapping:
            logger.info("No matched drugs available for hepatotoxicity consultation")
            return None
        logger.info("Running clinical hepatotoxicity assessment for matched drugs")
        report = await self.compile_clinical_assessment(
            resolved_mapping,
            clinical_context=prepared_inputs.clinical_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_prompt=prepared_inputs.pattern_prompt,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )
        return report.model_dump()

    # -------------------------------------------------------------------------
    async def run_revision_analysis(
        self,
        *,
        prepared_inputs: HepatoxPreparedInputs | None,
        visit_date: date | None = None,
        report_language: str = "en",
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any] | None:
        if prepared_inputs is None:
            logger.info("No prepared inputs provided; skipping revision hepatotoxicity consultation")
            return None
        resolved_mapping = prepared_inputs.resolved_drugs
        if not resolved_mapping:
            logger.info("No matched drugs available for revision hepatotoxicity consultation")
            return None
        logger.info("Running revision clinical hepatotoxicity assessment for matched drugs")
        report = await self.compile_revision_clinical_assessment(
            resolved_mapping,
            clinical_context=prepared_inputs.clinical_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_prompt=prepared_inputs.pattern_prompt,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )
        payload = report.model_dump()
        payload["revision_consultation_metadata"] = {
            "drug_analysis_entrypoint": "request_revision_drug_analysis",
            "report_finalization_entrypoint": "finalize_revision_patient_report",
            "conclusion_entrypoint": "generate_revision_conclusion",
            "synthesis_mode": "revision_comparison_aware",
        }
        return payload

    # -------------------------------------------------------------------------
    async def compile_clinical_assessment(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        *,
        clinical_context: str | None,
        visit_date: date | None,
        report_language: str,
        pattern_prompt: str,
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PatientDrugClinicalReport:
        normalized_context = clinical_context.strip() if clinical_context else ""
        pattern_summary = (
            pattern_prompt.strip()
            or "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."
        )
        return await self._build_clinical_assessment(
            resolved_drugs,
            normalized_context=normalized_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_summary=pattern_summary,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
            prepare_fn=self.prepare_drug_assessment,
            finalize_fn=self.consultation.report_finalizer.finalize_patient_report,
        )

    # -------------------------------------------------------------------------
    async def compile_revision_clinical_assessment(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        *,
        clinical_context: str | None,
        visit_date: date | None,
        report_language: str,
        pattern_prompt: str,
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PatientDrugClinicalReport:
        normalized_context = clinical_context.strip() if clinical_context else ""
        if normalized_context:
            normalized_context = (
                normalized_context
                + "\n\nRevision synthesis mode:\n"
                + "Prior report content is comparison-only context. Prefer revised structured evidence and current source evidence."
            )
        pattern_summary = (
            pattern_prompt.strip()
            or "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."
        )
        return await self._build_clinical_assessment(
            resolved_drugs,
            normalized_context=normalized_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_summary=pattern_summary,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
            prepare_fn=self.prepare_revision_drug_assessment,
            finalize_fn=self.consultation.report_finalizer.finalize_revision_patient_report,
        )

    # -------------------------------------------------------------------------
    async def _build_clinical_assessment(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        *,
        normalized_context: str,
        visit_date: date | None,
        report_language: str,
        pattern_summary: str,
        rag_query: dict[str, str] | None,
        rucam_bundle: PatientRucamAssessmentBundle | None,
        progress_callback: Callable[[str, float], None] | None,
        prepare_fn,
        finalize_fn,
    ) -> PatientDrugClinicalReport:
        entries: list[DrugClinicalAssessment] = []
        llm_jobs: list[tuple[int, Any]] = []
        rucam_by_key: dict[str, DrugRucamAssessment] = {}
        if rucam_bundle is not None:
            for item in rucam_bundle.entries:
                normalized_key = normalize_drug_query_name(item.drug_name)
                if normalized_key:
                    rucam_by_key[normalized_key] = item

        consultation = self.consultation
        for idx, drug_entry in enumerate(consultation.drugs.entries):
            entry, job = await prepare_fn(
                idx=idx,
                drug_entry=drug_entry,
                resolved_drugs=resolved_drugs,
                visit_date=visit_date,
                report_language=report_language,
                normalized_context=normalized_context,
                pattern_summary=pattern_summary,
                rag_query=rag_query,
                rucam_by_key=rucam_by_key,
            )
            entries.append(entry)
            if job:
                llm_jobs.append(job)

        consultation.emit_progress(progress_callback, stage="llm_analysis", fraction=0.0)
        if llm_jobs:
            semaphore = asyncio.Semaphore(consultation.max_parallel_analyses)
            pending_tasks = [
                asyncio.create_task(self.execute_bounded_job(idx, task, semaphore))
                for idx, task in llm_jobs
            ]
            completed = 0
            total = len(pending_tasks)
            for task in asyncio.as_completed(pending_tasks):
                idx, outcome = await task
                entry = entries[idx]
                if isinstance(outcome, Exception):
                    logger.error(
                        "Clinical analysis for drug '%s' failed: %s",
                        entry.drug_name,
                        outcome,
                    )
                    entry.paragraph = consultation.build_error_paragraph(entry)
                else:
                    normalized_outcome = (
                        outcome.strip()
                        if isinstance(outcome, str)
                        else str(outcome).strip()
                    )
                    normalized_outcome = consultation.remove_redundant_report_sentence(
                        normalized_outcome
                    )
                    entry.paragraph = (
                        normalized_outcome
                        if normalized_outcome
                        else consultation.build_error_paragraph(entry)
                    )
                completed += 1
                consultation.emit_progress(
                    progress_callback,
                    stage="llm_analysis",
                    fraction=completed / total if total else 1.0,
                )
        else:
            consultation.emit_progress(
                progress_callback, stage="llm_analysis", fraction=1.0
            )

        logger.info("Composing final clinical report for current patient")
        consultation.emit_progress(
            progress_callback, stage="report_composition", fraction=0.0
        )
        final_report = await finalize_fn(
            entries,
            clinical_context=normalized_context,
            report_language=report_language,
        )
        consultation.emit_progress(
            progress_callback, stage="report_composition", fraction=1.0
        )
        return PatientDrugClinicalReport(entries=entries, final_report=final_report)

    # -------------------------------------------------------------------------
    @staticmethod
    async def execute_indexed_job(index: int, coroutine: Any) -> tuple[int, Any]:
        try:
            return index, await coroutine
        except Exception as exc:
            return index, exc

    # -------------------------------------------------------------------------
    async def execute_bounded_job(
        self,
        index: int,
        coroutine: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, Any]:
        async with semaphore:
            return await self.execute_indexed_job(index, coroutine)

    # -------------------------------------------------------------------------
    @staticmethod
    async def build_drug_assessment_base(
        *,
        drug_entry: DrugEntry,
        resolved_drugs: dict[str, dict[str, Any]],
        visit_date: date | None,
        pattern_summary: str,
        rucam_by_key: dict[str, DrugRucamAssessment],
        consultation: Any,
    ) -> DrugClinicalAssessment:
        raw_name = drug_entry.name or ""
        normalized_drug_key = normalize_drug_query_name(raw_name)
        livertox_data = consultation.resolve_livertox_data_for_entry(
            raw_name=raw_name,
            normalized_key=normalized_drug_key,
            resolved_drugs=resolved_drugs,
        )
        matched_row = livertox_data.get("matched_livertox_row", None)
        excerpts_list = livertox_data.get("extracted_excerpts", [])
        canonical_name = (
            str(livertox_data.get("canonical_name") or raw_name).strip() or raw_name
        )
        origins = [
            origin
            for origin in livertox_data.get("origins", [])
            if isinstance(origin, str) and origin.strip()
        ]
        if not origins and drug_entry.source in {"therapy", "anamnesis"}:
            origins = [drug_entry.source]
        extraction_metadata = livertox_data.get("extraction_metadata", [])
        if not isinstance(extraction_metadata, list):
            extraction_metadata = []
        missing_livertox = bool(livertox_data.get("missing_livertox"))
        ambiguous_match = bool(livertox_data.get("ambiguous_match"))
        raw_match_status = livertox_data.get("match_status")
        match_status = (
            str(raw_match_status).strip().lower()
            if raw_match_status is not None
            else None
        )
        match_candidates = [
            str(candidate).strip()
            for candidate in livertox_data.get("match_candidates", [])
            if str(candidate).strip()
        ]
        match_notes = [
            str(note).strip()
            for note in livertox_data.get("match_notes", [])
            if str(note).strip()
        ]
        match_confidence = livertox_data.get("match_confidence")
        if match_confidence is not None:
            try:
                match_confidence = float(match_confidence)
            except (TypeError, ValueError):
                match_confidence = None
        match_reason = livertox_data.get("match_reason")
        match_quality = classify_match_evidence(
            match_status=match_status,
            match_reason=str(match_reason) if match_reason is not None else None,
            match_confidence=match_confidence,
            match_notes=match_notes,
            missing_livertox=missing_livertox,
            ambiguous_match=ambiguous_match,
        )
        suspension = consultation.evaluate_suspension(drug_entry, visit_date)
        matched_lvt_row = matched_row if isinstance(matched_row, dict) else None
        rucam = rucam_by_key.get(normalized_drug_key)
        source_context_summary = summarize_drug_source_context(drug_entry)
        temporal_plausibility = assess_temporal_plausibility(drug_entry, None)
        pattern_compatibility = assess_pattern_compatibility(
            drug_entry,
            pattern_summary,
            consultation.rag_support.select_excerpt(excerpts_list),
        )
        extraction_metadata = [
            *extraction_metadata,
            {
                "source_context": source_context_summary,
                "temporal_plausibility": temporal_plausibility,
                "pattern_compatibility": pattern_compatibility,
                "historical_flag": bool(getattr(drug_entry, "historical_flag", False)),
            },
        ]
        knowledge_prompt = str(livertox_data.get("knowledge_prompt") or "").strip()
        entry = DrugClinicalAssessment(
            drug_name=drug_entry.name,
            canonical_name=canonical_name,
            origins=origins,
            extraction_metadata=extraction_metadata,
            matched_livertox_row=matched_lvt_row,
            extracted_excerpts=excerpts_list,
            missing_livertox=missing_livertox,
            ambiguous_match=ambiguous_match,
            match_status=match_status,
            match_confidence=match_confidence,
            match_reason=str(match_reason).strip()
            if match_reason is not None
            else None,
            match_notes=match_notes,
            evidence_quality=match_quality["evidence_quality"],
            evidence_warnings=match_quality["evidence_warnings"],
            match_candidates=match_candidates,
            suspension=suspension,
            rucam=rucam,
        )
        entry.narrative = self.build_clinical_narrative(
            drug_name=drug_entry.name,
            excerpts=excerpts_list,
            rucam=rucam,
            missing_livertox=missing_livertox,
            evidence_warnings=match_quality["evidence_warnings"],
        )
        entry.claims = entry.narrative.claims
        return entry, knowledge_prompt, excerpts_list

    # -------------------------------------------------------------------------
    async def prepare_drug_assessment(
        self,
        *,
        idx: int,
        drug_entry: DrugEntry,
        resolved_drugs: dict[str, dict[str, Any]],
        visit_date: date | None,
        report_language: str,
        normalized_context: str,
        pattern_summary: str,
        rag_query: dict[str, str] | None,
        rucam_by_key: dict[str, DrugRucamAssessment],
    ) -> tuple[DrugClinicalAssessment, tuple[int, Any] | None]:
        consultation = self.consultation
        entry, knowledge_prompt, excerpts_list = await self.build_drug_assessment_base(
            drug_entry=drug_entry,
            resolved_drugs=resolved_drugs,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            rucam_by_key=rucam_by_key,
            consultation=consultation,
        )
        if entry.paragraph:
            return entry, None
        excerpt = consultation.rag_support.select_excerpt(excerpts_list)
        if excerpt is None or entry.missing_livertox:
            entry.missing_livertox = True
            entry.paragraph = consultation.build_missing_excerpt_paragraph(entry)
            return entry, None
        rag_documents = await consultation.rag_support.fetch_rag_documents(
            rag_query, drug_entry.name or ""
        )
        job = consultation.drug_analysis.request_drug_analysis(
            drug_name=drug_entry.name,
            canonical_name=entry.canonical_name or drug_entry.name,
            origins=entry.origins,
            extraction_metadata=entry.extraction_metadata,
            livertox_status="matched",
            excerpt=excerpt,
            rag_documents=rag_documents or None,
            clinical_context=normalized_context,
            suspension=entry.suspension,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            metadata=entry.matched_livertox_row,
            rucam=entry.rucam,
            knowledge_prompt=knowledge_prompt,
            report_language=report_language,
        )
        return entry, (idx, job)

    # -------------------------------------------------------------------------
    async def prepare_revision_drug_assessment(
        self,
        *,
        idx: int,
        drug_entry: DrugEntry,
        resolved_drugs: dict[str, dict[str, Any]],
        visit_date: date | None,
        report_language: str,
        normalized_context: str,
        pattern_summary: str,
        rag_query: dict[str, str] | None,
        rucam_by_key: dict[str, DrugRucamAssessment],
    ) -> tuple[DrugClinicalAssessment, tuple[int, Any] | None]:
        consultation = self.consultation
        entry, knowledge_prompt, excerpts_list = await self.build_drug_assessment_base(
            drug_entry=drug_entry,
            resolved_drugs=resolved_drugs,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            rucam_by_key=rucam_by_key,
            consultation=consultation,
        )
        if entry.paragraph:
            return entry, None
        excerpt = consultation.rag_support.select_excerpt(excerpts_list)
        if excerpt is None or entry.missing_livertox:
            entry.missing_livertox = True
            entry.paragraph = consultation.build_missing_excerpt_paragraph(entry)
            return entry, None
        rag_documents = await consultation.rag_support.fetch_rag_documents(
            rag_query, drug_entry.name or ""
        )
        job = consultation.drug_analysis.request_revision_drug_analysis(
            drug_name=drug_entry.name,
            canonical_name=entry.canonical_name or drug_entry.name,
            origins=entry.origins,
            extraction_metadata=entry.extraction_metadata,
            livertox_status="matched",
            excerpt=excerpt,
            rag_documents=rag_documents or None,
            clinical_context=normalized_context,
            suspension=entry.suspension,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            metadata=entry.matched_livertox_row,
            rucam=entry.rucam,
            knowledge_prompt=knowledge_prompt,
            report_language=report_language,
        )
        return entry, (idx, job)

    # -------------------------------------------------------------------------
    def resolve_livertox_data_for_entry(
        self,
        *,
        raw_name: str,
        normalized_key: str,
        resolved_drugs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        exact = resolved_drugs.get(normalized_key)
        if exact is not None and self.livertox_payload_rank(exact) >= 3:
            return exact

        raw_name_normalized = raw_name.strip().casefold()
        grouped: list[dict[str, Any]] = []
        for payload in resolved_drugs.values():
            raw_mentions = payload.get("raw_mentions", [])
            if not isinstance(raw_mentions, list):
                continue
            if any(
                isinstance(mention, str)
                and mention.strip().casefold() == raw_name_normalized
                for mention in raw_mentions
            ):
                grouped.append(payload)
        if not grouped:
            return exact or {}
        grouped.sort(
            key=lambda payload: (
                self.livertox_payload_rank(payload),
                len(str(payload.get("normalized_name") or "").split()),
            ),
            reverse=True,
        )
        if exact is not None and self.livertox_payload_rank(exact) >= self.livertox_payload_rank(grouped[0]):
            return exact
        return grouped[0]

    # -------------------------------------------------------------------------
    @staticmethod
    def livertox_payload_rank(payload: dict[str, Any]) -> int:
        status = str(payload.get("match_status") or "").strip().lower()
        if status == "matched_with_excerpt":
            return 4
        if status == "matched_no_excerpt":
            return 3
        if status in {"matched", "match"}:
            return 3
        if status in {"ambiguous", "ambiguous_match"} or payload.get("ambiguous_match"):
            return 2
        if status in {"missing", "missing_match"} or payload.get("missing_livertox"):
            return 1
        return 0

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(
        self, attempt: int, *, exc: Exception | None = None
    ) -> float:
        if exc is not None:
            hinted_wait = self.consultation.rag_support.extract_rate_limit_wait_hint_seconds(exc)
            if hinted_wait is not None:
                return hinted_wait
        normalized_attempt = max(int(attempt), 1)
        return min(8.0, 0.75 * (2 ** (normalized_attempt - 1)))

###############################################################################
def summarize_drug_source_context(entry: DrugEntry) -> str:
    source = (
        (entry.source or "unknown").strip()
        if isinstance(entry.source, str)
        else "unknown"
    )
    if source == "therapy":
        return "Current/past therapy section entry."
    if source == "anamnesis":
        return "Historical anamnesis section entry."
    return "Source section unavailable."

###############################################################################
def assess_temporal_plausibility(
    entry: DrugEntry,
    lab_timeline: PatientLabTimeline | None,
) -> str:
    _ = lab_timeline
    if entry.therapy_start_date and (
        entry.suspension_status is not None or entry.suspension_date
    ):
        return "Temporal sequence available for plausibility assessment."
    if entry.therapy_start_date:
        return "Therapy start is available; temporal assessment is partially supported."
    return "Temporal evidence is limited."

###############################################################################
def assess_pattern_compatibility(
    entry: DrugEntry,
    hepatic_pattern: str | None,
    livertox_excerpt: str | None,
) -> str:
    _ = entry
    pattern_value = (hepatic_pattern or "").strip().lower()
    excerpt_text = (livertox_excerpt or "").strip()
    if not pattern_value:
        return "Hepatic pattern unavailable for compatibility assessment."
    if not excerpt_text:
        return f"Pattern '{pattern_value}' available; LiverTox excerpt unavailable."
    return f"Pattern '{pattern_value}' can be compared against LiverTox evidence."
