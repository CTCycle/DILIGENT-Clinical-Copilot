from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any, TypeAlias

from common.utils.logger import logger
from domain.clinical.claims import ClinicalClaim, DrugClinicalNarrative
from domain.clinical.entities import (
    DrugClinicalAssessment,
    DrugEntry,
    DrugRucamAssessment,
    PatientDrugClinicalReport,
    PatientLabTimeline,
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.match_quality import classify_match_evidence
from services.clinical.drug_analysis import DrugAnalysisService
from services.clinical.exposure_timeline import ExposureTimelineService
from services.clinical.rag_support import RagSupportService
from services.clinical.report_finalizer import ReportFinalizer
from services.text.normalization import normalize_drug_query_name

CLAIM_EVIDENCE_QUOTE_MAX_LENGTH = 1000
CLAIM_EVIDENCE_TRUNCATION_MARKER = " [truncated]"
MATCH_REASON_MAX_LENGTH = 100
DrugAssessmentBase: TypeAlias = tuple[DrugClinicalAssessment, str, list[str]]

###############################################################################
def emit_progress(
    progress_callback: Callable[[str, float], None] | None,
    *,
    stage: str,
    fraction: float,
) -> None:
    if progress_callback is None:
        return
    bounded_fraction = min(1.0, max(0.0, float(fraction)))
    progress_callback(stage, bounded_fraction)

###############################################################################
def livertox_payload_rank(payload: dict[str, Any]) -> int:
    status = str(payload.get("match_status") or "").strip().lower()
    if status in {
        "matched_with_excerpt",
        "accepted_exact_livertox",
        "accepted_rxnav_validated",
        "accepted_livertox_without_rxnav",
    }:
        return 4
    if status in {"matched_no_excerpt", "matched", "match"}:
        return 3
    if status in {
        "ambiguous",
        "ambiguous_match",
        "ambiguous_requires_review",
    } or payload.get("ambiguous_match"):
        return 2
    if status in {
        "missing",
        "missing_match",
        "missing_livertox",
        "rejected_false_positive",
    } or payload.get("missing_livertox"):
        return 1
    return 0

###############################################################################
def resolve_livertox_data_for_entry(
    *,
    raw_name: str,
    normalized_key: str,
    resolved_drugs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exact = resolved_drugs.get(normalized_key)
    if exact is not None and livertox_payload_rank(exact) >= 3:
        return exact
    raw_name_normalized = raw_name.strip().casefold()
    grouped = [
        payload
        for payload in resolved_drugs.values()
        if isinstance(payload.get("raw_mentions"), list)
        and any(
            isinstance(mention, str)
            and mention.strip().casefold() == raw_name_normalized
            for mention in payload["raw_mentions"]
        )
    ]
    if not grouped:
        return exact or {}
    grouped.sort(
        key=lambda payload: (
            livertox_payload_rank(payload),
            len(str(payload.get("normalized_name") or "").split()),
        ),
        reverse=True,
    )
    if exact is not None and livertox_payload_rank(exact) >= livertox_payload_rank(grouped[0]):
        return exact
    return grouped[0]

###############################################################################
def claim_safe_evidence_quote(value: str | None) -> str | None:
    stripped = str(value).strip() if value is not None else ""
    if not stripped:
        return None
    if len(stripped) <= CLAIM_EVIDENCE_QUOTE_MAX_LENGTH:
        return stripped

    marker = CLAIM_EVIDENCE_TRUNCATION_MARKER
    target_length = CLAIM_EVIDENCE_QUOTE_MAX_LENGTH - len(marker)
    truncated = stripped[:target_length].rstrip()
    boundary = max(
        truncated.rfind("\n"),
        truncated.rfind(". "),
        truncated.rfind("; "),
        truncated.rfind(", "),
        truncated.rfind(" "),
    )
    if boundary >= 200:
        truncated = truncated[:boundary].rstrip(" .,;\n")
    return f"{truncated}{marker}"

###############################################################################
def normalize_match_reason(
    value: Any,
    notes: list[str],
) -> tuple[str | None, list[str]]:
    raw_reason = str(value).strip() if value is not None else ""
    normalized_notes = list(notes)
    if not raw_reason:
        return None, normalized_notes

    reason_parts = [part.strip() for part in raw_reason.split(";") if part.strip()]
    for part in reason_parts:
        if part not in normalized_notes:
            normalized_notes.append(part)

    primary_reason = reason_parts[0] if reason_parts else raw_reason
    if len(primary_reason) > MATCH_REASON_MAX_LENGTH:
        primary_reason = primary_reason[:MATCH_REASON_MAX_LENGTH].rstrip()
    return primary_reason or None, normalized_notes

###############################################################################
class AnalysisRunner:
    """Orchestrates the top-level analysis workflow — runs the full drug assessment pipeline."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        drugs: PatientDrugs,
        exposure_timeline: ExposureTimelineService,
        drug_analysis: DrugAnalysisService,
        rag_support: RagSupportService,
        report_finalizer: ReportFinalizer,
        max_parallel_analyses: int,
        pipeline_issues: list[PipelineIssue],
        resolve_livertox_data_for_entry: Callable[..., dict[str, Any]],
        emit_progress: Callable[..., None],
    ) -> None:
        self.drugs = drugs
        self.exposure_timeline = exposure_timeline
        self.drug_analysis = drug_analysis
        self.rag_support = rag_support
        self.report_finalizer = report_finalizer
        self.max_parallel_analyses = max(int(max_parallel_analyses), 1)
        self.pipeline_issues = pipeline_issues
        self.resolve_livertox_data_for_entry = resolve_livertox_data_for_entry
        self.emit_progress = emit_progress

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
        evidence_quote = claim_safe_evidence_quote(excerpts[0] if excerpts else None)
        if evidence_quote is not None:
            claims.append(
                ClinicalClaim(
                    claim=f"{drug_name} has source-text evidence in the clinical record.",
                    source="source_text",
                    evidence_quote=evidence_quote,
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
            finalize_fn=self.report_finalizer.finalize_report,
            generate_conclusion=self.drug_analysis.generate_conclusion,
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
            finalize_fn=self.report_finalizer.finalize_report,
            generate_conclusion=self.drug_analysis.generate_revision_conclusion,
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
        generate_conclusion,
    ) -> PatientDrugClinicalReport:
        entries: list[DrugClinicalAssessment] = []
        llm_jobs: list[tuple[int, Any]] = []
        rucam_by_key: dict[str, DrugRucamAssessment] = {}
        if rucam_bundle is not None:
            for item in rucam_bundle.entries:
                normalized_key = normalize_drug_query_name(item.drug_name)
                if normalized_key:
                    rucam_by_key[normalized_key] = item

        for idx, drug_entry in enumerate(self.drugs.entries):
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

        self.emit_progress(
            progress_callback, stage="llm_analysis", fraction=0.0
        )
        if llm_jobs:
            semaphore = asyncio.Semaphore(self.max_parallel_analyses)
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
                    entry.paragraph = self.report_finalizer.build_error_paragraph(entry)
                else:
                    normalized_outcome = (
                        outcome.strip()
                        if isinstance(outcome, str)
                        else str(outcome).strip()
                    )
                    normalized_outcome = self.report_finalizer.remove_redundant_report_sentence(
                        normalized_outcome
                    )
                    entry.paragraph = (
                        normalized_outcome
                        if normalized_outcome
                        else self.report_finalizer.build_error_paragraph(entry)
                    )
                completed += 1
                self.emit_progress(
                    progress_callback,
                    stage="llm_analysis",
                    fraction=completed / total if total else 1.0,
                )
        else:
            self.emit_progress(
                progress_callback, stage="llm_analysis", fraction=1.0
            )

        logger.info("Composing final clinical report for current patient")
        self.emit_progress(
            progress_callback, stage="report_composition", fraction=0.0
        )
        final_report = await finalize_fn(
            entries,
            clinical_context=normalized_context,
            report_language=report_language,
            generate_conclusion=generate_conclusion,
        )
        self.emit_progress(
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
        resolve_livertox_data_for_entry: Callable[..., dict[str, Any]],
        exposure_timeline: ExposureTimelineService,
        rag_support: RagSupportService,
    ) -> DrugAssessmentBase:
        raw_name = drug_entry.name or ""
        normalized_drug_key = normalize_drug_query_name(raw_name)
        livertox_data = resolve_livertox_data_for_entry(
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
        match_candidates = []
        for candidate in livertox_data.get("match_candidates", []):
            if isinstance(candidate, dict):
                candidate_name = str(candidate.get("drug_name") or "").strip()
            else:
                candidate_name = str(candidate).strip()
            if candidate_name and candidate_name not in match_candidates:
                match_candidates.append(candidate_name)
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
        match_reason, match_notes = normalize_match_reason(
            livertox_data.get("match_reason"),
            match_notes,
        )
        match_quality = classify_match_evidence(
            match_status=match_status,
            match_reason=match_reason,
            match_confidence=match_confidence,
            match_notes=match_notes,
            missing_livertox=missing_livertox,
            ambiguous_match=ambiguous_match,
        )
        suspension = exposure_timeline.evaluate_suspension(drug_entry, visit_date)
        matched_lvt_row = matched_row if isinstance(matched_row, dict) else None
        rucam = rucam_by_key.get(normalized_drug_key)
        source_context_summary = summarize_drug_source_context(drug_entry)
        temporal_plausibility = assess_temporal_plausibility(drug_entry, None)
        pattern_compatibility = assess_pattern_compatibility(
            drug_entry,
            pattern_summary,
            rag_support.select_excerpt(excerpts_list),
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
            match_reason=match_reason,
            match_notes=match_notes,
            evidence_quality=match_quality["evidence_quality"],
            evidence_warnings=match_quality["evidence_warnings"],
            match_candidates=match_candidates,
            suspension=suspension,
            rucam=rucam,
        )
        entry.narrative = AnalysisRunner.build_clinical_narrative(
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
        entry, knowledge_prompt, excerpts_list = await self.build_drug_assessment_base(
            drug_entry=drug_entry,
            resolved_drugs=resolved_drugs,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            rucam_by_key=rucam_by_key,
            resolve_livertox_data_for_entry=self.resolve_livertox_data_for_entry,
            exposure_timeline=self.exposure_timeline,
            rag_support=self.rag_support,
        )
        if entry.paragraph:
            return entry, None
        excerpt = self.rag_support.select_excerpt(excerpts_list)
        if excerpt is None or entry.missing_livertox:
            entry.missing_livertox = True
            entry.paragraph = self.report_finalizer.build_missing_excerpt_paragraph(entry)
            return entry, None
        rag_bundle = await self.rag_support.fetch_rag_documents(
            rag_query, drug_entry.name or ""
        )
        rag_context = rag_bundle.context_text if rag_bundle is not None else None
        entry.rag_references = (
            list(rag_bundle.references) if rag_bundle is not None else []
        )
        job = self.drug_analysis.request_drug_analysis(
            drug_name=drug_entry.name,
            canonical_name=entry.canonical_name or drug_entry.name,
            origins=entry.origins,
            extraction_metadata=entry.extraction_metadata,
            livertox_status="matched",
            excerpt=excerpt,
            rag_context=rag_context,
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
        entry, knowledge_prompt, excerpts_list = await self.build_drug_assessment_base(
            drug_entry=drug_entry,
            resolved_drugs=resolved_drugs,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            rucam_by_key=rucam_by_key,
            resolve_livertox_data_for_entry=self.resolve_livertox_data_for_entry,
            exposure_timeline=self.exposure_timeline,
            rag_support=self.rag_support,
        )
        if entry.paragraph:
            return entry, None
        excerpt = self.rag_support.select_excerpt(excerpts_list)
        if excerpt is None or entry.missing_livertox:
            entry.missing_livertox = True
            entry.paragraph = self.report_finalizer.build_missing_excerpt_paragraph(entry)
            return entry, None
        rag_bundle = await self.rag_support.fetch_rag_documents(
            rag_query, drug_entry.name or ""
        )
        rag_context = rag_bundle.context_text if rag_bundle is not None else None
        entry.rag_references = (
            list(rag_bundle.references) if rag_bundle is not None else []
        )
        job = self.drug_analysis.request_revision_drug_analysis(
            drug_name=drug_entry.name,
            canonical_name=entry.canonical_name or drug_entry.name,
            origins=entry.origins,
            extraction_metadata=entry.extraction_metadata,
            livertox_status="matched",
            excerpt=excerpt,
            rag_context=rag_context,
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
