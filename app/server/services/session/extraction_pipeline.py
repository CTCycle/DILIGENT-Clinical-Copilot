from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Protocol

from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalPipelineValidationError,
    HepatotoxicityPatternAssessment,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from domain.clinical.extras import HepatoxPreparedInputs
from services.retrieval.query import DILIQueryBuilder
from services.llm.runtime_config import LLMRuntimeConfig

###############################################################################
class ClinicalSessionExtractionOwner(Protocol):
    drugs_parser: Any
    disease_extractor: Any
    lab_extractor: Any
    pattern_analyzer: Any
    rucam_estimator: Any
    input_preparator: Any
    latest_lab_extraction_audit: Any

    note_stage_runtime: Callable[..., None]
    emit_progress: Callable[..., None]
    build_stage_progress_callback: Callable[..., Any]
    _resolve_runtime_timeout: Callable[..., float]
    run_stop_check: Callable[..., None]
    note_stage_elapsed: Callable[..., None]
    classify_extraction_failure: Callable[..., tuple[str, str]]
    append_warning_issue: Callable[..., None]
    note_stage_fallback: Callable[..., None]
    classify_structured_failure_kind: Callable[..., str]
    build_fallback_therapy_drugs: Callable[..., PatientDrugs]
    append_knowledge_base_unavailable_issue: Callable[..., None]

###############################################################################
class ClinicalSessionExtractionPipelineMixin:

    # -------------------------------------------------------------------------
    async def extract_therapy_drugs(
        self: ClinicalSessionExtractionOwner,
        *,
        cleaned_therapy_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDrugs:
        self.note_stage_runtime("therapy_extraction")
        self.emit_progress(progress_callback, stage="drugs.extracting", value=16.0)
        therapy_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="drugs.extracting",
            start_value=16.0,
            end_value=23.0,
        )
        start_time = time.perf_counter()
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(getattr(self.drugs_parser, "timeout_s", 1.0))
        )
        try:
            therapy_drugs = await asyncio.wait_for(
                self.drugs_parser.extract_drugs_from_therapy(
                    cleaned_therapy_text,
                    progress_callback=therapy_progress_callback,
                ),
                timeout=timeout_s,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            self.note_stage_elapsed("therapy_extraction", elapsed)
            logger.info("Therapy drugs extraction required %.4f seconds", elapsed)
            logger.info(
                "Detected %s drugs from therapy list", len(therapy_drugs.entries)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.note_stage_elapsed("therapy_extraction", elapsed)
            logger.warning(
                (
                    "Therapy drugs extraction failed after %.4f seconds; "
                    "falling back to line-based parsing: %s"
                ),
                elapsed,
                exc,
            )
            issue_code, issue_message = self.classify_extraction_failure(
                exc,
                fallback_code="therapy_extraction_fallback",
                fallback_message=(
                    "Therapy extraction via LLM was unavailable; "
                    "the analysis continued using the raw therapy list."
                ),
            )
            self.append_warning_issue(
                issues,
                code=issue_code,
                message=issue_message,
                field="drugs",
            )
            self.note_stage_fallback(
                "therapy_extraction",
                issue_code=issue_code,
                issue_message=issue_message,
                structured_failure_kind=self.classify_structured_failure_kind(exc),
            )
            therapy_drugs = self.build_fallback_therapy_drugs(cleaned_therapy_text)
        self.emit_progress(progress_callback, stage="drugs.extracting", value=23.0)
        self.run_stop_check(stop_check)
        return therapy_drugs

    # -------------------------------------------------------------------------
    async def extract_anamnesis_drugs(
        self: ClinicalSessionExtractionOwner,
        *,
        anamnesis_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDrugs:
        self.note_stage_runtime("anamnesis_extraction")
        self.emit_progress(progress_callback, stage="drugs.extracting", value=23.0)
        anamnesis_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="drugs.extracting",
            start_value=23.0,
            end_value=30.0,
        )
        start_time = time.perf_counter()
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(getattr(self.drugs_parser, "timeout_s", 1.0))
        )
        try:
            anamnesis_drugs = await asyncio.wait_for(
                self.drugs_parser.extract_drugs_from_anamnesis(
                    anamnesis_text,
                    progress_callback=anamnesis_progress_callback,
                ),
                timeout=timeout_s,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            self.note_stage_elapsed("anamnesis_extraction", elapsed)
            logger.info("Anamnesis drugs extraction required %.4f seconds", elapsed)
            logger.info(
                "Detected %s drugs from anamnesis", len(anamnesis_drugs.entries)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.note_stage_elapsed("anamnesis_extraction", elapsed)
            logger.warning(
                (
                    "Anamnesis drugs extraction failed after %.4f seconds "
                    "(timeout budget %.1fs); continuing without historical "
                    "drug mentions: %s"
                ),
                elapsed,
                timeout_s,
                exc,
            )
            issue_code, issue_message = self.classify_extraction_failure(
                exc,
                fallback_code="anamnesis_extraction_failed",
                fallback_message=(
                    "Drug extraction from anamnesis was unavailable; "
                    "the analysis continued without historical drug mentions."
                ),
            )
            self.append_warning_issue(
                issues,
                code=issue_code,
                message=issue_message,
                field="anamnesis",
            )
            self.note_stage_fallback(
                "anamnesis_extraction",
                issue_code=issue_code,
                issue_message=issue_message,
                structured_failure_kind=self.classify_structured_failure_kind(exc),
            )
            anamnesis_drugs = PatientDrugs(entries=[])
        self.emit_progress(progress_callback, stage="drugs.extracting", value=30.0)
        self.run_stop_check(stop_check)
        return anamnesis_drugs

    # -------------------------------------------------------------------------
    async def extract_disease_context(
        self: ClinicalSessionExtractionOwner,
        *,
        anamnesis_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDiseaseContext:
        self.note_stage_runtime("anamnesis_disease_extraction")
        self.emit_progress(progress_callback, stage="diseases.extracting", value=38.0)
        disease_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="diseases.extracting",
            start_value=38.0,
            end_value=46.0,
        )
        start_time = time.perf_counter()
        max_attempts = 1
        backoff_seconds = 0.0
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(self.disease_extractor.timeout_s)
        )
        for attempt in range(1, max_attempts + 1):
            try:
                disease_context = await asyncio.wait_for(
                    self.disease_extractor.extract_diseases_from_anamnesis(
                        anamnesis_text,
                        progress_callback=disease_progress_callback,
                    ),
                    timeout=timeout_s,
                )
                self.run_stop_check(stop_check)
                elapsed = time.perf_counter() - start_time
                self.note_stage_elapsed("anamnesis_disease_extraction", elapsed)
                logger.info(
                    "Anamnesis disease extraction required %.4f seconds", elapsed
                )
                logger.info(
                    "Detected %s diseases from anamnesis", len(disease_context.entries)
                )
                self.emit_progress(
                    progress_callback, stage="diseases.extracting", value=46.0
                )
                self.run_stop_check(stop_check)
                return disease_context
            except TimeoutError:
                elapsed = time.perf_counter() - start_time
                if attempt < max_attempts:
                    self.emit_progress(
                        progress_callback,
                        stage="diseases.extracting",
                        value=38.0,
                        detail="diseases.extracting",
                    )
                    logger.warning(
                        (
                            "Anamnesis disease extraction timed out after %.4fs "
                            "(timeout budget %.1fs, attempt %d/%d). Retrying in %.1fs."
                        ),
                        elapsed,
                        timeout_s,
                        attempt,
                        max_attempts,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    self.run_stop_check(stop_check)
                    continue
                logger.warning(
                    (
                        "Anamnesis disease extraction timed out after %.4f seconds "
                        "(timeout budget %.1fs); continuing without structured "
                        "disease timeline."
                    ),
                    elapsed,
                    timeout_s,
                )
                issue_code, issue_message = self.classify_extraction_failure(
                    TimeoutError("Anamnesis disease extraction timed out"),
                    fallback_code="anamnesis_disease_extraction_timeout",
                    fallback_message=(
                        "Disease extraction from anamnesis timed out; "
                        "the analysis continued without structured disease timeline."
                    ),
                )
                self.append_warning_issue(
                    issues,
                    code=issue_code,
                    message=issue_message,
                    field="anamnesis",
                )
                self.note_stage_elapsed("anamnesis_disease_extraction", elapsed)
                self.note_stage_fallback(
                    "anamnesis_disease_extraction",
                    issue_code=issue_code,
                    issue_message=issue_message,
                    structured_failure_kind="llm_timeout",
                )
                disease_context = PatientDiseaseContext(entries=[])
                self.emit_progress(
                    progress_callback, stage="diseases.extracting", value=46.0
                )
                self.run_stop_check(stop_check)
                return disease_context
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                self.note_stage_elapsed("anamnesis_disease_extraction", elapsed)
                logger.warning(
                    (
                        "Anamnesis disease extraction failed after %.4f seconds "
                        "(timeout budget %.1fs); continuing without structured "
                        "disease timeline: %s"
                    ),
                    elapsed,
                    timeout_s,
                    exc,
                )
                issue_code, issue_message = self.classify_extraction_failure(
                    exc,
                    fallback_code="anamnesis_disease_extraction_failed",
                    fallback_message=(
                        "Disease extraction from anamnesis was unavailable; "
                        "the analysis continued without structured disease timeline."
                    ),
                )
                self.append_warning_issue(
                    issues,
                    code=issue_code,
                    message=issue_message,
                    field="anamnesis",
                )
                self.note_stage_fallback(
                    "anamnesis_disease_extraction",
                    issue_code=issue_code,
                    issue_message=issue_message,
                    structured_failure_kind=self.classify_structured_failure_kind(exc),
                )
                disease_context = PatientDiseaseContext(entries=[])
                self.emit_progress(
                    progress_callback, stage="diseases.extracting", value=46.0
                )
                self.run_stop_check(stop_check)
                return disease_context
        return PatientDiseaseContext(entries=[])

    # -------------------------------------------------------------------------
    async def extract_lab_timeline(
        self: ClinicalSessionExtractionOwner,
        *,
        payload: PatientData,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        self.note_stage_runtime("anamnesis_lab_extraction")
        self.emit_progress(progress_callback, stage="labs.extracting", value=46.0)
        lab_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="labs.extracting",
            start_value=46.0,
            end_value=54.0,
        )
        start_time = time.perf_counter()
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(getattr(self.lab_extractor, "timeout_s", 1.0))
        )
        try:
            if hasattr(self.lab_extractor, "extract_from_payload_with_audit"):
                lab_audit = await asyncio.wait_for(
                    self.lab_extractor.extract_from_payload_with_audit(
                        payload,
                        progress_callback=lab_progress_callback,
                    ),
                    timeout=timeout_s,
                )
                self.latest_lab_extraction_audit = lab_audit
                lab_timeline = lab_audit["lab_timeline"]
                onset_context = lab_audit["onset_context"]
            else:
                lab_timeline, onset_context = await asyncio.wait_for(
                    self.lab_extractor.extract_from_payload(
                        payload,
                        progress_callback=lab_progress_callback,
                    ),
                    timeout=timeout_s,
                )
                self.latest_lab_extraction_audit = None
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            self.note_stage_elapsed("anamnesis_lab_extraction", elapsed)
            logger.info("Anamnesis lab extraction required %.4f seconds", elapsed)
            logger.info("Detected %s timeline lab entries", len(lab_timeline.entries))
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.note_stage_elapsed("anamnesis_lab_extraction", elapsed)
            logger.warning(
                (
                    "Anamnesis lab extraction failed after %.4f seconds "
                    "(timeout budget %.1fs); continuing with deterministic "
                    "lab timeline fallback: %s"
                ),
                elapsed,
                timeout_s,
                exc,
            )
            issue_code, issue_message = self.classify_extraction_failure(
                exc,
                fallback_code="anamnesis_lab_extraction_failed",
                fallback_message=(
                    "Longitudinal lab extraction from anamnesis was unavailable; "
                    "the analysis continued with deterministic lab parsing fallback."
                ),
            )
            self.append_warning_issue(
                issues,
                code=issue_code,
                message=issue_message,
                field="anamnesis",
            )
            self.note_stage_fallback(
                "anamnesis_lab_extraction",
                issue_code=issue_code,
                issue_message=issue_message,
                structured_failure_kind=self.classify_structured_failure_kind(exc),
            )
            # Deterministic fallback: recover lab markers directly from text
            # so pattern estimation can still proceed when LLM extraction fails.
            fallback_entries = []
            primary_labs_text = self.lab_extractor.clean_text(
                payload.laboratory_analysis
            )
            supplemental_anamnesis_text = self.lab_extractor.clean_text(
                payload.anamnesis
            )
            fallback_entries.extend(
                self.lab_extractor.extract_entries_from_text(
                    text=primary_labs_text,
                    source="laboratory_analysis",
                    visit_date=payload.visit_date,
                )
            )
            fallback_entries.extend(
                self.lab_extractor.extract_entries_from_text(
                    text=supplemental_anamnesis_text,
                    source="anamnesis",
                    visit_date=payload.visit_date,
                )
            )
            normalized_entries = []
            seen_keys: set[tuple[str, str, str, str]] = set()
            for entry in fallback_entries:
                prepared = self.lab_extractor.normalize_entry(
                    entry, visit_date=payload.visit_date
                )
                if prepared is None:
                    continue
                key = self.lab_extractor.dedupe_key(prepared)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                normalized_entries.append(prepared)
            normalized_entries.sort(key=self.lab_extractor.lab_entry_sort_key)
            lab_timeline = PatientLabTimeline(entries=normalized_entries)
            onset_context = None
            self.latest_lab_extraction_audit = None
        self.emit_progress(progress_callback, stage="labs.extracting", value=54.0)
        self.run_stop_check(stop_check)
        return lab_timeline, onset_context

    # -------------------------------------------------------------------------
    def assess_pattern(
        self: ClinicalSessionExtractionOwner,
        *,
        lab_timeline: PatientLabTimeline,
        validation_bundle: Any,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> HepatotoxicityPatternAssessment:
        try:
            pattern_assessment = self.pattern_analyzer.assess_payload(lab_timeline)
        except ClinicalPipelineValidationError as exc:
            localized = [
                PipelineIssue(
                    severity=item.severity,
                    code=item.code,
                    message=(
                        validation_bundle.insufficient_labs
                        if item.code == "missing_hepatotoxicity_inputs"
                        else item.message
                    ),
                    field=item.field,
                    line_index=item.line_index,
                    raw_line=item.raw_line,
                )
                for item in exc.issues
            ]
            raise ClinicalPipelineValidationError(
                issues=localized,
                message=localized[0].message if localized else exc.args[0],
            ) from exc
        issues.extend(pattern_assessment.issues)
        pattern_score = pattern_assessment.score
        logger.info(
            "Patient hepatotoxicity pattern classified as %s (R=%.3f, status=%s)",
            pattern_score.classification,
            pattern_score.r_score
            if pattern_score.r_score is not None
            else float("nan"),
            pattern_assessment.status,
        )
        self.emit_progress(
            progress_callback, stage="hepatotoxicity_pattern", value=54.0
        )
        self.run_stop_check(stop_check)
        return pattern_assessment

    # -------------------------------------------------------------------------
    def estimate_rucam(
        self: ClinicalSessionExtractionOwner,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score,
        report_language: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientRucamAssessmentBundle:
        self.emit_progress(progress_callback, stage="rucam_estimation", value=54.0)
        start_time = time.perf_counter()
        try:
            rucam_bundle = self.rucam_estimator.estimate(
                payload=payload,
                analysis_drugs=analysis_drugs,
                anamnesis_drugs=anamnesis_drugs,
                disease_context=disease_context,
                lab_timeline=lab_timeline,
                onset_context=onset_context,
                pattern_score=pattern_score,
                resolved_drugs=None,
                report_language=report_language,
            )
            elapsed = time.perf_counter() - start_time
            logger.info("RUCAM estimation required %.4f seconds", elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                "RUCAM estimation failed after %.4f seconds; continuing without RUCAM: %s",
                elapsed,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="rucam_estimation_failed",
                message=(
                    "RUCAM estimation was unavailable; the analysis continued without "
                    "per-drug estimated RUCAM."
                ),
            )
            rucam_bundle = PatientRucamAssessmentBundle(entries=[])
        self.emit_progress(progress_callback, stage="rucam_estimation", value=68.0)
        self.run_stop_check(stop_check)
        return rucam_bundle

    # -------------------------------------------------------------------------
    def build_rag_query(
        self: ClinicalSessionExtractionOwner,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        structured_context: str,
        pattern_score,
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> dict[str, str] | None:
        self.emit_progress(progress_callback, stage="rag_query_building", value=75.0)
        rag_query: dict[str, str] | None = None
        if payload.use_rag:
            query_builder = DILIQueryBuilder(analysis_drugs)
            logger.info("RAG retrieval enabled for clinical consultation")
            rag_query = query_builder.build_dili_queries(
                clinical_context=structured_context,
                pattern_classification=pattern_score.classification,
                r_score=pattern_score.r_score,
            )
        self.emit_progress(progress_callback, stage="rag_query_building", value=82.0)
        self.run_stop_check(stop_check)
        return rag_query

    # -------------------------------------------------------------------------
    async def run_livertox_lookup(
        self: ClinicalSessionExtractionOwner,
        *,
        all_detected_drugs: PatientDrugs,
        structured_context: str,
        pattern_score,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
        use_rag: bool,
    ) -> HepatoxPreparedInputs | None:
        detail = "livertox_lookup.rag" if use_rag else "livertox_lookup.no_rag"
        self.emit_progress(
            progress_callback,
            stage="livertox_lookup",
            value=82.0,
            detail=detail,
        )
        livertox_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage=detail,
            start_value=82.0,
            end_value=88.0,
        )
        runtime = get_server_settings().runtime
        lookup_timeout_s = min(
            float(runtime.livertox_llm_timeout),
            float(runtime.cloud_llm_timeout_cap)
            if LLMRuntimeConfig.is_cloud_enabled()
            else float(runtime.local_llm_timeout_cap),
            120.0,
        )
        try:
            prepared_inputs = await asyncio.wait_for(
                self.input_preparator.prepare_inputs(
                    all_detected_drugs,
                    clinical_context=structured_context,
                    pattern_score=pattern_score,
                    progress_callback=livertox_progress_callback,
                    identity_resolution_client=getattr(
                        self.drugs_parser, "client", None
                    ),
                    identity_resolution_model=getattr(self.drugs_parser, "model", None),
                    identity_resolution_temperature=float(
                        getattr(self.drugs_parser, "temperature", 0.0)
                    ),
                ),
                timeout=max(float(runtime.minimum_llm_timeout), lookup_timeout_s),
            )
        except TimeoutError:
            logger.warning(
                "LiverTox input preparation timed out after %.1f seconds; continuing without prepared evidence",
                lookup_timeout_s,
            )
            self.append_warning_issue(
                issues,
                code="livertox_lookup_timeout",
                message=(
                    "LiverTox evidence preparation timed out; the analysis continued "
                    "without prepared LiverTox evidence."
                ),
            )
            prepared_inputs = None
        self.run_stop_check(stop_check)
        if prepared_inputs is None and all_detected_drugs.entries:
            self.append_knowledge_base_unavailable_issue(issues)
        self.emit_progress(
            progress_callback,
            stage="livertox_lookup",
            value=88.0,
            detail=detail,
        )
        return prepared_inputs

    # -------------------------------------------------------------------------
    def reestimate_rucam_with_livertox(
        self: ClinicalSessionExtractionOwner,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score,
        report_language: str,
        prepared_inputs,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
    ) -> PatientRucamAssessmentBundle:
        try:
            return self.rucam_estimator.estimate(
                payload=payload,
                analysis_drugs=analysis_drugs,
                anamnesis_drugs=anamnesis_drugs,
                disease_context=disease_context,
                lab_timeline=lab_timeline,
                onset_context=onset_context,
                pattern_score=pattern_score,
                resolved_drugs=prepared_inputs.resolved_drugs
                if prepared_inputs
                else None,
                report_language=report_language,
            )
        except Exception as exc:
            logger.warning("RUCAM re-estimation with LiverTox metadata failed: %s", exc)
            self.append_warning_issue(
                issues,
                code="rucam_reestimate_failed",
                message=(
                    "RUCAM refinement with matched LiverTox metadata failed; "
                    "using preliminary estimates."
                ),
            )
            return rucam_bundle
