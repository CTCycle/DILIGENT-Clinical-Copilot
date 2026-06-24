from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from common.utils.logger import logger
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

###############################################################################
class ClinicalSessionExtractionPipelineMixin:

    # -------------------------------------------------------------------------
    async def extract_therapy_drugs(
        self,
        *,
        cleaned_therapy_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDrugs:
        self.emit_progress(progress_callback, stage="therapy_extraction", value=16.0)
        therapy_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="therapy_extraction",
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
            logger.info("Therapy drugs extraction required %.4f seconds", elapsed)
            logger.info(
                "Detected %s drugs from therapy list", len(therapy_drugs.entries)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                (
                    "Therapy drugs extraction failed after %.4f seconds; "
                    "falling back to line-based parsing: %s"
                ),
                elapsed,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="therapy_extraction_fallback",
                message=(
                    "Therapy extraction via LLM was unavailable; "
                    "the analysis continued using the raw therapy list."
                ),
                field="drugs",
            )
            therapy_drugs = self.build_fallback_therapy_drugs(cleaned_therapy_text)
        self.emit_progress(progress_callback, stage="therapy_extraction", value=23.0)
        self.run_stop_check(stop_check)
        return therapy_drugs

    # -------------------------------------------------------------------------
    async def extract_anamnesis_drugs(
        self,
        *,
        anamnesis_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDrugs:
        self.emit_progress(progress_callback, stage="anamnesis_extraction", value=23.0)
        anamnesis_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_extraction",
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
            logger.info("Anamnesis drugs extraction required %.4f seconds", elapsed)
            logger.info(
                "Detected %s drugs from anamnesis", len(anamnesis_drugs.entries)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
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
            self.append_warning_issue(
                issues,
                code="anamnesis_extraction_failed",
                message=(
                    "Drug extraction from anamnesis was unavailable; "
                    "the analysis continued without historical drug mentions."
                ),
                field="anamnesis",
            )
            anamnesis_drugs = PatientDrugs(entries=[])
        self.emit_progress(progress_callback, stage="anamnesis_extraction", value=30.0)
        self.run_stop_check(stop_check)
        return anamnesis_drugs

    # -------------------------------------------------------------------------
    async def extract_disease_context(
        self,
        *,
        anamnesis_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDiseaseContext:
        self.emit_progress(
            progress_callback, stage="anamnesis_disease_extraction", value=38.0
        )
        disease_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_disease_extraction",
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
                logger.info(
                    "Anamnesis disease extraction required %.4f seconds", elapsed
                )
                logger.info(
                    "Detected %s diseases from anamnesis", len(disease_context.entries)
                )
                self.emit_progress(
                    progress_callback, stage="anamnesis_disease_extraction", value=46.0
                )
                self.run_stop_check(stop_check)
                return disease_context
            except TimeoutError:
                elapsed = time.perf_counter() - start_time
                if attempt < max_attempts:
                    self.emit_progress(
                        progress_callback,
                        stage="anamnesis_disease_extraction",
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
                self.append_warning_issue(
                    issues,
                    code="anamnesis_disease_extraction_timeout",
                    message=(
                        "Disease extraction from anamnesis timed out; "
                        "the analysis continued without structured disease timeline."
                    ),
                    field="anamnesis",
                )
                disease_context = PatientDiseaseContext(entries=[])
                self.emit_progress(
                    progress_callback, stage="anamnesis_disease_extraction", value=46.0
                )
                self.run_stop_check(stop_check)
                return disease_context
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
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
                self.append_warning_issue(
                    issues,
                    code="anamnesis_disease_extraction_failed",
                    message=(
                        "Disease extraction from anamnesis was unavailable; "
                        "the analysis continued without structured disease timeline."
                    ),
                    field="anamnesis",
                )
                disease_context = PatientDiseaseContext(entries=[])
                self.emit_progress(
                    progress_callback, stage="anamnesis_disease_extraction", value=46.0
                )
                self.run_stop_check(stop_check)
                return disease_context
        return PatientDiseaseContext(entries=[])

    # -------------------------------------------------------------------------
    async def extract_lab_timeline(
        self,
        *,
        payload: PatientData,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        self.emit_progress(
            progress_callback, stage="anamnesis_lab_extraction", value=46.0
        )
        lab_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_lab_extraction",
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
            logger.info("Anamnesis lab extraction required %.4f seconds", elapsed)
            logger.info("Detected %s timeline lab entries", len(lab_timeline.entries))
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
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
            self.append_warning_issue(
                issues,
                code="anamnesis_lab_extraction_failed",
                message=(
                    "Longitudinal lab extraction from anamnesis was unavailable; "
                    "the analysis continued with deterministic lab parsing fallback."
                ),
                field="anamnesis",
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
        self.emit_progress(
            progress_callback, stage="anamnesis_lab_extraction", value=54.0
        )
        self.run_stop_check(stop_check)
        return lab_timeline, onset_context

    # -------------------------------------------------------------------------
    def assess_pattern(
        self,
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
        self,
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
        self,
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
        self,
        *,
        all_detected_drugs: PatientDrugs,
        structured_context: str,
        pattern_score,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> HepatoxPreparedInputs | None:
        self.emit_progress(progress_callback, stage="livertox_lookup", value=82.0)
        livertox_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="livertox_lookup",
            start_value=82.0,
            end_value=88.0,
        )
        prepared_inputs = await self.input_preparator.prepare_inputs(
            all_detected_drugs,
            clinical_context=structured_context,
            pattern_score=pattern_score,
            progress_callback=livertox_progress_callback,
            identity_resolution_client=getattr(self.drugs_parser, "client", None),
            identity_resolution_model=getattr(self.drugs_parser, "model", None),
            identity_resolution_temperature=float(
                getattr(self.drugs_parser, "temperature", 0.0)
            ),
        )
        self.run_stop_check(stop_check)
        if prepared_inputs is None and all_detected_drugs.entries:
            self.append_knowledge_base_unavailable_issue(issues)
        self.emit_progress(progress_callback, stage="livertox_lookup", value=88.0)
        return prepared_inputs

    # -------------------------------------------------------------------------
    def reestimate_rucam_with_livertox(
        self,
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
