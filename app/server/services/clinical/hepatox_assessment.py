from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import date
from typing import Any

from common.constants import (
    DEFAULT_DILI_CLASSIFICATION,
    R_SCORE_CHOLESTATIC_THRESHOLD,
    R_SCORE_HEPATOCELLULAR_THRESHOLD,
)
from common.utils.logger import logger
from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugClinicalAssessment,
    DrugEntry,
    DrugRucamAssessment,
    DrugSuspensionContext,
    HepatotoxicityPatternAssessment,
    HepatotoxicityPatternScore,
    PatientDrugClinicalReport,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.match_quality import classify_match_evidence
from services.clinical.preparation import HepatoxPreparedInputs
from services.clinical.report_language import (
    report_heading,
)
from services.llm.prompts import (
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
)
from services.retrieval.embeddings import (
    EmbeddingModelMismatchError,
    SimilaritySearch,
)
from services.text.normalization import normalize_drug_query_name
from services.text.vocabulary import get_text_normalization_snapshot

###############################################################################
NOT_AVAILABLE_TEXT = "Not available"
REDUNDANT_REPORT_LINE_RE = re.compile(
    r"generated\s+report.*?(drug[- ]induced\s+liver\s+injury|\bdili\b)",
    re.IGNORECASE,
)
LIVERTOX_TITLE_LINE_RE = re.compile(
    r"^\s*\*{0,2}[^*\n]+?\s*-\s*LiverTox score\b.*\*{0,2}\s*$",
    re.IGNORECASE,
)
REPORT_LABEL_LINE_RE = re.compile(r"^\s*\*{0,2}\s*Report\s*\*{0,2}\s*$", re.IGNORECASE)
BIBLIOGRAPHY_LINE_RE = re.compile(
    r"^\s*\*{0,2}\s*Bibliography source\s*\*{0,2}\s*:\s*LiverTox\s*$",
    re.IGNORECASE,
)
DRIFT_SECTION_LINE_RE = re.compile(
    r"^\s*(medication|assessment|plan)\s*$", re.IGNORECASE
)
STRUCTURED_DILI_SECTION_LINE_RE = re.compile(
    r"^\s*#{0,6}\s*\*{0,2}\s*Structured\s+DILI\s+Assessment\s+Report\s*\*{0,2}\s*$",
    re.IGNORECASE,
)
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)


###############################################################################
class HepatotoxicityPatternCalculator:
    
    # -------------------------------------------------------------------------
    def calculate(
        self,
        *,
        alt_value: float | None,
        alt_uln: float | None,
        alp_value: float | None,
        alp_uln: float | None,
    ) -> HepatotoxicityPatternScore:
        alt_multiple = self.safe_ratio(alt_value, alt_uln)
        alp_multiple = self.safe_ratio(alp_value, alp_uln)

        r_score = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = DEFAULT_DILI_CLASSIFICATION
        if r_score is not None:
            if r_score > R_SCORE_HEPATOCELLULAR_THRESHOLD:
                classification = "hepatocellular"
            elif r_score < R_SCORE_CHOLESTATIC_THRESHOLD:
                classification = "cholestatic"
            else:
                classification = "mixed"

        return HepatotoxicityPatternScore(
            alt_multiple=alt_multiple,
            alp_multiple=alp_multiple,
            r_score=r_score,
            classification=classification,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def safe_ratio(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class HepatotoxicityPatternAnalyzer:
    def __init__(self) -> None:
        self.r_score: float | None = None
        self.calculator = HepatotoxicityPatternCalculator()

    # -------------------------------------------------------------------------
    def calculate_hepatotoxicity_pattern(
        self, lab_timeline: PatientLabTimeline
    ) -> HepatotoxicityPatternScore:
        anchor = self.select_anchor_pair(lab_timeline)
        if anchor is None:
            score = HepatotoxicityPatternScore(
                alt_multiple=None,
                alp_multiple=None,
                r_score=None,
                classification=DEFAULT_DILI_CLASSIFICATION,
            )
            self.r_score = None
            return score
        score = self.calculator.calculate(
            alt_value=anchor["alt_value"],
            alt_uln=anchor["alt_uln"],
            alp_value=anchor["alp_value"],
            alp_uln=anchor["alp_uln"],
        )
        self.r_score = score.r_score
        return score

    # -------------------------------------------------------------------------
    def assess_payload(
        self,
        lab_timeline: PatientLabTimeline,
    ) -> HepatotoxicityPatternAssessment:
        score = self.calculate_hepatotoxicity_pattern(lab_timeline)
        if score.r_score is None:
            issue = PipelineIssue(
                severity="warning",
                code="missing_hepatotoxicity_inputs",
                message=(
                    "Laboratory data are insufficient for a numeric R ratio "
                    "(ideally ALT/AST, ALP, and bilirubin). Continuing with "
                    "indeterminate pattern and reduced confidence."
                ),
                field="laboratory_analysis",
            )
            self.r_score = None
            return HepatotoxicityPatternAssessment(
                score=score,
                status="undetermined_due_to_missing_labs",
                issues=[issue],
            )
        self.r_score = score.r_score
        return HepatotoxicityPatternAssessment(
            score=score,
            status="ok",
            issues=[],
        )

    # -------------------------------------------------------------------------
    def select_anchor_pair(
        self, lab_timeline: PatientLabTimeline
    ) -> dict[str, float] | None:
        dated_candidates = self.group_entries_by_date(lab_timeline.entries)
        for sample_date in sorted(dated_candidates):
            bucket = dated_candidates[sample_date]
            pair = self.build_anchor_from_bucket(bucket)
            if pair is not None:
                return pair
        undated = self.build_anchor_from_bucket(lab_timeline.entries)
        return undated

    # -------------------------------------------------------------------------
    def group_entries_by_date(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, list[ClinicalLabEntry]]:
        grouped: dict[str, list[ClinicalLabEntry]] = {}
        for entry in entries:
            if not entry.sample_date:
                continue
            grouped.setdefault(entry.sample_date, []).append(entry)
        return grouped

    # -------------------------------------------------------------------------
    def build_anchor_from_bucket(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, float] | None:
        alt_like = self.pick_best_entry(entries, {"ALT", "AST"})
        alp = self.pick_best_entry(entries, {"ALP"})
        if alt_like is None or alp is None:
            return None
        alt_value = self.parse_entry_value(alt_like)
        alp_value = self.parse_entry_value(alp)
        if alt_value is None or alp_value is None:
            return None
        alt_uln = self.resolve_uln(alt_like, fallback=40.0)
        alp_uln = self.resolve_uln(alp, fallback=120.0)
        if alt_uln <= 0 or alp_uln <= 0:
            return None
        return {
            "alt_value": alt_value,
            "alt_uln": alt_uln,
            "alp_value": alp_value,
            "alp_uln": alp_uln,
        }

    # -------------------------------------------------------------------------
    def pick_best_entry(
        self,
        entries: list[ClinicalLabEntry],
        marker_names: set[str],
    ) -> ClinicalLabEntry | None:
        selected: ClinicalLabEntry | None = None
        for entry in entries:
            if entry.marker_name.upper() not in marker_names:
                continue
            if selected is None:
                selected = entry
                continue
            selected_value = self.parse_entry_value(selected)
            current_value = self.parse_entry_value(entry)
            if selected_value is None and current_value is not None:
                selected = entry
            elif (
                current_value is not None
                and selected_value is not None
                and current_value > selected_value
            ):
                selected = entry
        return selected

    # -------------------------------------------------------------------------
    def parse_entry_value(self, entry: ClinicalLabEntry) -> float | None:
        if entry.value is not None:
            return float(entry.value)
        return self.parse_marker_value(entry.value_text)

    # -------------------------------------------------------------------------
    def resolve_uln(self, entry: ClinicalLabEntry, *, fallback: float) -> float:
        if entry.upper_limit_normal is not None and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        parsed = self.parse_marker_value(entry.upper_limit_text)
        if parsed is not None and parsed > 0:
            return parsed
        return fallback

    # -------------------------------------------------------------------------
    def parse_marker_value(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        normalized = raw.replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", normalized)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        return self.calculator.safe_ratio(value, reference)

    # -------------------------------------------------------------------------
    def stringify_scores(
        self, pattern_score: HepatotoxicityPatternScore | None
    ) -> dict[str, str]:
        if not pattern_score:
            return {}

        mapping = {
            "alt_multiple": (pattern_score.alt_multiple, "{:.2f}x ULN"),
            "alp_multiple": (pattern_score.alp_multiple, "{:.2f}x ULN"),
            "r_score": (pattern_score.r_score, "{:.2f}"),
        }

        return {
            key: fmt.format(val) if val is not None else NOT_AVAILABLE_TEXT
            for key, (val, fmt) in mapping.items()
        }


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.

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
        logger.info(
            "No prepared inputs provided; skipping hepatotoxicity consultation"
        )
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
    entries: list[DrugClinicalAssessment] = []
    llm_jobs: list[tuple[int, Any]] = []
    rucam_by_key: dict[str, DrugRucamAssessment] = {}
    if rucam_bundle is not None:
        for item in rucam_bundle.entries:
            normalized_key = normalize_drug_query_name(item.drug_name)
            if normalized_key:
                rucam_by_key[normalized_key] = item

    # iterate over all drugs to identify those with LiverTox excerpts and those without
    for idx, drug_entry in enumerate(self.drugs.entries):
        entry, job = await self.prepare_drug_assessment(
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

    self.emit_progress(progress_callback, stage="llm_analysis", fraction=0.0)
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
                entry.paragraph = self.build_error_paragraph(entry)
            else:
                normalized_outcome = (
                    outcome.strip()
                    if isinstance(outcome, str)
                    else str(outcome).strip()
                )
                normalized_outcome = self.remove_redundant_report_sentence(
                    normalized_outcome
                )
                entry.paragraph = (
                    normalized_outcome
                    if normalized_outcome
                    else self.build_error_paragraph(entry)
                )
            completed += 1
            self.emit_progress(
                progress_callback,
                stage="llm_analysis",
                fraction=completed / total if total else 1.0,
            )
    else:
        self.emit_progress(progress_callback, stage="llm_analysis", fraction=1.0)

    logger.info("Composing final clinical report for current patient")
    self.emit_progress(progress_callback, stage="report_composition", fraction=0.0)
    final_report = await self.finalize_patient_report(
        entries,
        clinical_context=normalized_context,
        report_language=report_language,
    )
    self.emit_progress(progress_callback, stage="report_composition", fraction=1.0)

    return PatientDrugClinicalReport(entries=entries, final_report=final_report)

async def execute_indexed_job(index: int, coroutine: Any) -> tuple[int, Any]:
    try:
        return index, await coroutine
    except Exception as exc:  # noqa: BLE001
        return index, exc

async def execute_bounded_job(
    self,
    index: int,
    coroutine: Any,
    semaphore: asyncio.Semaphore,
) -> tuple[int, Any]:
    async with semaphore:
        return await self.execute_indexed_job(index, coroutine)

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
    raw_name = drug_entry.name or ""
    normalized_drug_key = normalize_drug_query_name(raw_name)

    livertox_data = self.resolve_livertox_data_for_entry(
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
    knowledge_prompt = str(livertox_data.get("knowledge_prompt") or "").strip()
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

    suspension = self.evaluate_suspension(drug_entry, visit_date)
    matched_lvt_row = matched_row if isinstance(matched_row, dict) else None
    rucam = rucam_by_key.get(normalized_drug_key)
    source_context_summary = summarize_drug_source_context(drug_entry)
    temporal_plausibility = assess_temporal_plausibility(
        drug_entry,
        None,
    )
    pattern_compatibility = assess_pattern_compatibility(
        drug_entry,
        pattern_summary,
        self.select_excerpt(excerpts_list),
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

    if suspension.excluded:
        entry.paragraph = self.build_excluded_paragraph(entry)
        return entry, None

    if entry.ambiguous_match:
        entry.paragraph = self.build_ambiguous_match_paragraph(entry)
        return entry, None

    excerpt = self.select_excerpt(excerpts_list)
    if excerpt is None or entry.missing_livertox:
        entry.missing_livertox = True
        entry.paragraph = self.build_missing_excerpt_paragraph(entry)
        return entry, None

    rag_documents = await self.fetch_rag_documents(rag_query, raw_name)
    job = self.request_drug_analysis(
        drug_name=drug_entry.name,
        canonical_name=entry.canonical_name or drug_entry.name,
        origins=entry.origins,
        extraction_metadata=entry.extraction_metadata,
        livertox_status=(
            "missing_livertox"
            if entry.missing_livertox
            else "ambiguous_match"
            if entry.ambiguous_match
            else "matched"
        ),
        excerpt=excerpt,
        rag_documents=rag_documents or None,
        clinical_context=normalized_context,
        suspension=suspension,
        visit_date=visit_date,
        pattern_summary=pattern_summary,
        metadata=entry.matched_livertox_row,
        rucam=entry.rucam,
        knowledge_prompt=knowledge_prompt,
        report_language=report_language,
    )
    return entry, (idx, job)

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
    if exact is not None and self.livertox_payload_rank(
        exact
    ) >= self.livertox_payload_rank(grouped[0]):
        return exact
    return grouped[0]

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

async def fetch_rag_documents(
    self, rag_query: dict[str, str] | None, drug_name: str
) -> str | None:
    if not rag_query:
        return None
    normalized_key = normalize_drug_query_name(drug_name)
    drug_rag_query = rag_query.get(drug_name) or rag_query.get(normalized_key)
    if drug_rag_query is None:
        for key, value in rag_query.items():
            if normalize_drug_query_name(key) == normalized_key:
                drug_rag_query = value
                break
    if not drug_rag_query:
        return None
    try:
        return await asyncio.to_thread(
            self.search_supporting_documents,
            drug_rag_query,
        )
    except EmbeddingModelMismatchError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RAG retrieval unavailable for drug '%s'; continuing without supporting documents: %s",
            drug_name,
            exc,
        )
        self.record_rag_retrieval_issue(drug_name=drug_name, error=exc)
        return f"No additional documents provided (reason: RAG retrieval unavailable: {exc})."

def record_rag_retrieval_issue(self, *, drug_name: str, error: Exception) -> None:
    issue = PipelineIssue(
        severity="warning",
        code="rag_retrieval_unavailable",
        message=(
            "Internal RAG retrieval was unavailable for "
            f"{drug_name}; analysis continued without supporting documents."
        ),
        field="rag",
        raw_line=f"{drug_name}: {error}",
    )
    if not hasattr(self, "pipeline_issues"):
        self.pipeline_issues = []
    self.pipeline_issues.append(issue)

def ensure_similarity_search(self) -> bool:
    if self.similarity_search is not None:
        return True
    try:
        self.similarity_search = SimilaritySearch()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize similarity search: %s", exc)
        self.similarity_search = None
        return False
    return True

def select_excerpt(self, excerpts: list[str]) -> str | None:
    excerpts = [chunk.strip() for chunk in excerpts if chunk.strip()]
    if not excerpts:
        return None
    combined = "\n\n".join(excerpts)
    if len(combined) <= self.MAX_EXCERPT_LENGTH:
        return combined
    # Keep the most informative text while respecting the token budget
    truncated = combined[: self.MAX_EXCERPT_LENGTH]
    cutoff = truncated.rfind("\n")
    if cutoff > 2000:
        truncated = truncated[:cutoff]
    return truncated.strip()

def search_supporting_documents(self, query_text: str | Any) -> str | None:
    if not isinstance(query_text, str):
        return None
    normalized = query_text.strip()
    if not normalized or not self.ensure_similarity_search():
        return None

    results = (
        self.similarity_search.search_with_reranking(
            normalized,
            candidate_k=self.rag_candidate_k,
            final_top_n=self.rag_top_n,
            use_reranking=self.rag_use_reranking,
        )
        if self.similarity_search
        else None
    )
    if not results:
        return None
    fragments: list[str] = []
    for index, record in enumerate(results, start=1):
        fragment = self.format_similarity_fragment(index, record)
        if fragment:
            fragments.append(fragment)
    if not fragments:
        return None
    return "\n".join(fragments)

async def repair_language_once(
    self,
    *,
    source_text: str,
    report_language: str,
) -> str:
    language_map = "en=English, it=Italian, de=German, fr=French, es=Spanish"
    repair_system = (
        "You rewrite clinical text into the requested language only. "
        "Do not add new clinical facts."
    )
    repair_user = (
        f"Target language code: {report_language}\n"
        f"Language map: {language_map}\n"
        "Rewrite the text entirely in the target language. "
        "Do not produce bilingual output. Keep drug names and direct quotes unchanged.\n\n"
        f"Text:\n{source_text}"
    )
    chat_kwargs: dict[str, Any] = {
        "model": self.llm_model,
        "messages": [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ],
    }
    if self.chat_supports_temperature:
        chat_kwargs["temperature"] = 0.0
    else:
        chat_kwargs["options"] = {"temperature": 0.0}
    repaired = await self.llm_client.chat(**chat_kwargs)
    return self.coerce_chat_text(repaired).strip()

async def request_drug_analysis(
    self,
    *,
    drug_name: str,
    canonical_name: str,
    origins: list[str],
    extraction_metadata: list[dict[str, Any]],
    livertox_status: str,
    excerpt: str,
    rag_documents: str | None,
    clinical_context: str,
    suspension: DrugSuspensionContext,
    visit_date: date | None,
    pattern_summary: str,
    metadata: dict[str, Any] | None,
    rucam: DrugRucamAssessment | None,
    knowledge_prompt: str = "No supplemental knowledge prompt available.",
    report_language: str = "en",
) -> str:
    start_details = self.format_start_prompt(suspension)
    suspension_details = self.format_suspension_prompt(suspension)
    timeline_note = (
        suspension.note
        or "No explicit timeline notes were available from extraction metadata."
    )
    visit_date_anchor = self.format_visit_date_anchor(visit_date)
    score, metadata_block = self.prepare_metadata_prompt(metadata)
    rag_documents = rag_documents or "No additional documents provided."
    origin_block = ", ".join(origins) if origins else "unknown"
    metadata_items = [
        f"- {json.dumps(item, ensure_ascii=False)}"
        for item in extraction_metadata
        if isinstance(item, dict) and item
    ]
    extraction_block = (
        "\n".join(metadata_items) if metadata_items else "- Not available"
    )
    rucam_block = self.format_rucam_prompt_block(rucam)
    user_prompt = LIVERTOX_CLINICAL_USER_PROMPT.format(
        drug_name=self.escape_braces(drug_name.strip() or drug_name),
        report_language=self.escape_braces(report_language),
        canonical_name=self.escape_braces(canonical_name.strip() or canonical_name),
        origins=self.escape_braces(origin_block),
        extraction_metadata=self.escape_braces(extraction_block),
        livertox_status=self.escape_braces(livertox_status),
        excerpt=self.escape_braces(excerpt),
        documents=self.escape_braces(rag_documents),
        clinical_context=self.escape_braces(clinical_context),
        visit_date_anchor=self.escape_braces(visit_date_anchor),
        therapy_start_details=self.escape_braces(start_details),
        suspension_details=self.escape_braces(suspension_details),
        timeline_note=self.escape_braces(timeline_note),
        pattern_summary=self.escape_braces(pattern_summary),
        rucam_block=self.escape_braces(rucam_block),
        knowledge_prompt=self.escape_braces(knowledge_prompt),
        metadata_block=self.escape_braces(metadata_block),
        livertox_score=self.escape_braces(score),
    )
    messages = [
        {"role": "system", "content": LIVERTOX_CLINICAL_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt},
    ]
    chat_kwargs: dict[str, Any] = {
        "model": self.llm_model,
        "messages": messages,
    }
    if self.chat_supports_temperature:
        chat_kwargs["temperature"] = self.temperature
    else:
        chat_kwargs["options"] = {"temperature": self.temperature}
    raw_response: Any = None
    for attempt in range(1, self.analysis_retry_attempts + 1):
        try:
            # Ask the clinical model to synthesise findings for this drug
            raw_response = await self.llm_client.chat(**chat_kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= self.analysis_retry_attempts:
                raise RuntimeError(
                    f"LLM analysis failed for {drug_name}: {exc}"
                ) from exc
            delay = self.retry_backoff_seconds(attempt, exc=exc)
            logger.warning(
                "Retrying LLM analysis for '%s' after error (attempt %d/%d, delay %.2fs): %s",
                drug_name,
                attempt,
                self.analysis_retry_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    response_text = self.coerce_chat_text(raw_response).strip()
    if not self.is_materially_in_report_language(response_text, report_language):
        logger.warning(
            "Language mismatch detected for drug analysis '%s' (target=%s); applying one repair pass",
            drug_name,
            report_language,
        )
        repaired_text = await self.repair_language_once(
            source_text=response_text,
            report_language=report_language,
        )
        if repaired_text:
            response_text = repaired_text
    return response_text

def coerce_chat_text(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response.strip()
    if isinstance(raw_response, dict):
        for key in ("content", "text", "response"):
            value = raw_response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(raw_response, ensure_ascii=False)
    return str(raw_response).strip()

def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
    message = str(exc)
    match = RATE_LIMIT_WAIT_HINT_RE.search(message)
    if match is None:
        return None
    try:
        parsed = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    # Add a small safety margin to avoid retrying too early.
    return min(parsed + 0.25, 30.0)

def retry_backoff_seconds(
    self, attempt: int, *, exc: Exception | None = None
) -> float:
    if exc is not None:
        hinted_wait = self.extract_rate_limit_wait_hint_seconds(exc)
        if hinted_wait is not None:
            return hinted_wait
    normalized_attempt = max(int(attempt), 1)
    return min(8.0, 0.75 * (2 ** (normalized_attempt - 1)))

async def finalize_patient_report(
    self,
    entries: list[DrugClinicalAssessment],
    *,
    clinical_context: str | None,
    report_language: str,
) -> str | None:
    matched_entries: list[DrugClinicalAssessment] = []
    unresolved_entries: list[DrugClinicalAssessment] = []
    for entry in entries:
        if self.should_render_as_matched_drug(entry):
            matched_entries.append(entry)
            continue
        unresolved_entries.append(entry)

    matched_sections = [
        self.render_matched_drug_section(entry, report_language=report_language)
        for entry in matched_entries
    ]
    matched_sections = [section for section in matched_sections if section]
    unresolved_section = self.render_unresolved_mentions_section(
        unresolved_entries,
        report_language=report_language,
    )
    sections: list[str] = []
    if matched_sections:
        sections.append(
            self.render_drug_assessment_section(
                matched_entries,
                report_language,
            )
        )
    if unresolved_section:
        sections.append(unresolved_section)
    if not sections:
        return None

    combined_report = "\n\n---\n\n".join(sections)
    if matched_sections:
        conclusion = await self.generate_conclusion(
            clinical_context=clinical_context or "",
            multi_drug_report="\n\n---\n\n".join(matched_sections),
            report_language=report_language,
        )
        if conclusion:
            heading = report_heading("report_section_summary", report_language)
            combined_report = f"{combined_report}\n\n## {heading}\n\n{conclusion}"
    return combined_report

def should_render_as_matched_drug(entry: DrugClinicalAssessment) -> bool:
    status = (entry.match_status or "").strip().lower()
    return status in {"matched", "matched_with_excerpt", "matched_no_excerpt"}

async def generate_conclusion(
    self,
    *,
    clinical_context: str,
    multi_drug_report: str,
    report_language: str,
) -> str | None:
    report_body = multi_drug_report.strip()
    if not report_body:
        return None
    context_body = clinical_context.strip()
    if not context_body:
        context_body = "No clinical context was provided."
    user_prompt = LIVERTOX_CONCLUSION_USER_PROMPT.format(
        report_language=self.escape_braces(report_language),
        clinical_context=self.escape_braces(context_body),
        multi_drug_report=self.escape_braces(report_body),
    )
    messages = [
        {"role": "system", "content": LIVERTOX_CONCLUSION_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt},
    ]
    chat_kwargs: dict[str, Any] = {
        "model": self.llm_model,
        "messages": messages,
    }
    if self.chat_supports_temperature:
        chat_kwargs["temperature"] = self.temperature
    else:
        chat_kwargs["options"] = {"temperature": self.temperature}
    raw_response: Any = None
    for attempt in range(1, self.analysis_retry_attempts + 1):
        try:
            raw_response = await self.llm_client.chat(**chat_kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= self.analysis_retry_attempts:
                logger.error("Failed to generate clinical conclusion: %s", exc)
                return None
            delay = self.retry_backoff_seconds(attempt, exc=exc)
            logger.warning(
                "Retrying clinical conclusion generation after error (attempt %d/%d, delay %.2fs): %s",
                attempt,
                self.analysis_retry_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    conclusion = self.coerce_chat_text(raw_response).strip()
    if conclusion and not self.is_materially_in_report_language(
        conclusion, report_language
    ):
        logger.warning(
            "Language mismatch detected for global conclusion (target=%s); applying one repair pass",
            report_language,
        )
        repaired = await self.repair_language_once(
            source_text=conclusion,
            report_language=report_language,
        )
        if repaired:
            conclusion = repaired
    return conclusion or None

def bibliography_source_label(self) -> str:
    return get_text_normalization_snapshot().knowledge_source_references.get(
        "livertox", "LiverTox"
    )


def summarize_drug_source_context(entry: DrugEntry) -> str:
    source = (entry.source or "unknown").strip() if isinstance(entry.source, str) else "unknown"
    if source == "therapy":
        return "Current/past therapy section entry."
    if source == "anamnesis":
        return "Historical anamnesis section entry."
    return "Source section unavailable."


def assess_temporal_plausibility(
    entry: DrugEntry,
    lab_timeline: PatientLabTimeline | None,
) -> str:
    _ = lab_timeline
    if entry.therapy_start_date and (entry.suspension_status is not None or entry.suspension_date):
        return "Temporal sequence available for plausibility assessment."
    if entry.therapy_start_date:
        return "Therapy start is available; temporal assessment is partially supported."
    return "Temporal evidence is limited."


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
