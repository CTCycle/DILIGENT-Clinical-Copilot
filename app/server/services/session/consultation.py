from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    PatientData,
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.hepatox_core import HepatoxConsultation
from services.clinical.job_progress import ClinicalConsultationProgressCallback
from services.llm.cloud import LLMError

###############################################################################
class ClinicalSessionConsultationMixin:

    # -------------------------------------------------------------------------
    async def run_consultation(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        prepared_inputs,
        consultation_context: str | None = None,
        report_language: str,
        rag_query: dict[str, str] | None,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[HepatoxConsultation, str | None]:
        clinical_session, final_report, _ = await self._run_consultation_internal(
            payload=payload,
            analysis_drugs=analysis_drugs,
            prepared_inputs=prepared_inputs,
            consultation_context=consultation_context,
            report_language=report_language,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        return clinical_session, final_report

    # -------------------------------------------------------------------------
    async def run_revision_consultation(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        prepared_inputs,
        consultation_context: str | None = None,
        consultation_context_metadata: dict[str, Any] | None = None,
        report_language: str,
        rag_query: dict[str, str] | None,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[HepatoxConsultation, str | None, dict[str, Any]]:
        return await self._run_revision_consultation_internal(
            payload=payload,
            analysis_drugs=analysis_drugs,
            prepared_inputs=prepared_inputs,
            consultation_context=consultation_context,
            report_language=report_language,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
            consultation_context_metadata=consultation_context_metadata,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_consultation_fallback_report(
        *,
        analysis_drugs: PatientDrugs,
        report_language: str,
        is_revision: bool = False,
    ) -> str:
        drug_names = [
            (entry.name or "").strip()
            for entry in analysis_drugs.entries
            if (entry.name or "").strip()
        ]
        unique_drugs: list[str] = []
        seen_drugs: set[str] = set()
        for name in drug_names:
            key = name.casefold()
            if key in seen_drugs:
                continue
            seen_drugs.add(key)
            unique_drugs.append(name)
            if len(unique_drugs) >= 8:
                break
        if report_language.lower().startswith("it"):
            if unique_drugs:
                if is_revision:
                    return (
                        "Report di revisione generato in modalità di fallback per indisponibilità della sintesi clinica. "
                        f"Farmaci selezionati per la revisione: {', '.join(unique_drugs)}. "
                        "Confrontare manualmente il risultato con il report precedente e con le evidenze strutturate aggiornate."
                    )
                return (
                    "Report finale generato in modalità di fallback per indisponibilità del motore clinico. "
                    f"Farmaci sospetti identificati nel testo: {', '.join(unique_drugs)}. "
                    "Rivedere manualmente la valutazione clinica e la conclusione specialistica originale."
                )
            if is_revision:
                return (
                    "Report di revisione generato in modalità di fallback per indisponibilità della sintesi clinica. "
                    "Non sono stati identificati farmaci di revisione affidabili; è necessaria revisione specialistica manuale."
                )
            return (
                "Report finale generato in modalità di fallback per indisponibilità del motore clinico. "
                "Non sono stati identificati farmaci sospetti affidabili; è necessaria revisione manuale."
            )
        if unique_drugs:
            if is_revision:
                return (
                    "Revision report generated in fallback mode because revision clinical synthesis was unavailable. "
                    f"Drugs selected for revision: {', '.join(unique_drugs)}. "
                    "Manual comparison against the previous report and revised structured evidence is required."
                )
            return (
                "Final report generated in fallback mode because clinical synthesis was unavailable. "
                f"Suspected drugs detected from source text: {', '.join(unique_drugs)}. "
                "Manual review against the original specialist assessment is required."
            )
        if is_revision:
            return (
                "Revision report generated in fallback mode because revision clinical synthesis was unavailable. "
                "No reliable revision-target drugs were detected; manual specialist review is required."
            )
        return (
            "Final report generated in fallback mode because clinical synthesis was unavailable. "
            "No reliable suspected drugs were detected; manual specialist review is required."
        )

    # -------------------------------------------------------------------------
    async def _run_consultation_internal(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        prepared_inputs,
        consultation_context: str | None,
        report_language: str,
        rag_query: dict[str, str] | None,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[HepatoxConsultation, str | None, dict[str, Any]]:
        clinical_session = self.hepatox_consultation_cls(
            analysis_drugs,
            patient_name=payload.name,
        )
        effective_inputs = prepared_inputs
        if (
            prepared_inputs is not None
            and consultation_context is not None
            and prepared_inputs.clinical_context != consultation_context
        ):
            effective_inputs = replace(
                prepared_inputs,
                clinical_context=consultation_context,
            )
        final_report: str | None = None
        start_time = time.perf_counter()
        consultation_timeout_s = self._resolve_consultation_timeout()
        try:
            consultation_progress_callback = ClinicalConsultationProgressCallback(
                progress_callback=progress_callback,
            )
            drug_assessment = await asyncio.wait_for(
                clinical_session.run_analysis(
                    prepared_inputs=effective_inputs,
                    visit_date=payload.visit_date,
                    report_language=report_language,
                    rag_query=rag_query,
                    rucam_bundle=rucam_bundle,
                    progress_callback=consultation_progress_callback,
                ),
                timeout=consultation_timeout_s,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info("Hepato-toxicity consultation required %.4f seconds", elapsed)
            if isinstance(drug_assessment, dict):
                raw_final_report = drug_assessment.get("final_report")
                if isinstance(raw_final_report, str):
                    final_report = raw_final_report.strip()
                elif raw_final_report is None:
                    final_report = None
                else:
                    final_report = str(raw_final_report).strip()
            issues.extend(getattr(clinical_session, "pipeline_issues", []))
        except TimeoutError as exc:
            self.append_warning_issue(
                issues,
                code="clinical_llm_timeout",
                message=(
                    "Clinical LLM analysis timed out; report generated without "
                    "per-drug synthesis."
                ),
            )
            logger.warning(
                "Clinical LLM timeout for patient '%s' after %.1fs: %s",
                payload.name or "unknown",
                consultation_timeout_s,
                exc,
            )
        except LLMError as exc:
            self.append_warning_issue(
                issues,
                code="clinical_llm_unavailable",
                message=(
                    "Clinical LLM analysis is unavailable; report generated without "
                    "per-drug synthesis."
                ),
            )
            logger.warning(
                "Clinical LLM unavailable for patient '%s': %s",
                payload.name or "unknown",
                exc,
            )
        used_fallback_report = not bool(str(final_report or "").strip())
        if used_fallback_report:
            final_report = self._build_consultation_fallback_report(
                analysis_drugs=analysis_drugs,
                report_language=report_language,
                is_revision=False,
            )
        payload_metadata = {
            "analysis_entrypoint": "run_analysis",
            "used_fallback_report": used_fallback_report,
            "consultation_model": getattr(clinical_session, "llm_model", None),
            "analysis_drug_names": [
                entry.name for entry in analysis_drugs.entries if entry.name
            ],
            "consultation_context_length": len(str(consultation_context or "").strip()),
        }
        return clinical_session, final_report, payload_metadata

    # -------------------------------------------------------------------------
    async def _run_revision_consultation_internal(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        prepared_inputs,
        consultation_context: str | None,
        report_language: str,
        rag_query: dict[str, str] | None,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
        consultation_context_metadata: dict[str, Any] | None,
    ) -> tuple[HepatoxConsultation, str | None, dict[str, Any]]:
        clinical_session = self.hepatox_consultation_cls(
            analysis_drugs,
            patient_name=payload.name,
        )
        effective_inputs = prepared_inputs
        if (
            prepared_inputs is not None
            and consultation_context is not None
            and prepared_inputs.clinical_context != consultation_context
        ):
            effective_inputs = replace(
                prepared_inputs,
                clinical_context=consultation_context,
            )
        final_report: str | None = None
        start_time = time.perf_counter()
        consultation_timeout_s = self._resolve_consultation_timeout()
        try:
            consultation_progress_callback = ClinicalConsultationProgressCallback(
                progress_callback=progress_callback,
            )
            drug_assessment = await asyncio.wait_for(
                clinical_session.run_revision_analysis(
                    prepared_inputs=effective_inputs,
                    visit_date=payload.visit_date,
                    report_language=report_language,
                    rag_query=rag_query,
                    rucam_bundle=rucam_bundle,
                    progress_callback=consultation_progress_callback,
                ),
                timeout=consultation_timeout_s,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Hepato-toxicity revision consultation required %.4f seconds", elapsed
            )
            if isinstance(drug_assessment, dict):
                raw_final_report = drug_assessment.get("final_report")
                if isinstance(raw_final_report, str):
                    final_report = raw_final_report.strip()
                elif raw_final_report is None:
                    final_report = None
                else:
                    final_report = str(raw_final_report).strip()
                revision_consultation_metadata = drug_assessment.get(
                    "revision_consultation_metadata"
                )
                if isinstance(revision_consultation_metadata, dict):
                    consultation_context_metadata = {
                        **(consultation_context_metadata or {}),
                        **revision_consultation_metadata,
                    }
            issues.extend(getattr(clinical_session, "pipeline_issues", []))
        except TimeoutError as exc:
            self.append_warning_issue(
                issues,
                code="clinical_llm_timeout",
                message=(
                    "Clinical LLM analysis timed out; report generated without "
                    "per-drug synthesis."
                ),
            )
            logger.warning(
                "Clinical LLM timeout for patient '%s' after %.1fs: %s",
                payload.name or "unknown",
                consultation_timeout_s,
                exc,
            )
        except LLMError as exc:
            self.append_warning_issue(
                issues,
                code="clinical_llm_unavailable",
                message=(
                    "Clinical LLM analysis is unavailable; report generated without "
                    "per-drug synthesis."
                ),
            )
            logger.warning(
                "Clinical LLM unavailable for patient '%s': %s",
                payload.name or "unknown",
                exc,
            )
        used_fallback_report = not bool(str(final_report or "").strip())
        if used_fallback_report:
            final_report = self._build_consultation_fallback_report(
                analysis_drugs=analysis_drugs,
                report_language=report_language,
                is_revision=True,
            )
        payload_metadata = {
            "analysis_entrypoint": "run_revision_analysis",
            "used_fallback_report": used_fallback_report,
            "consultation_model": getattr(clinical_session, "llm_model", None),
            "analysis_drug_names": [
                entry.name for entry in analysis_drugs.entries if entry.name
            ],
            "consultation_context_length": len(str(consultation_context or "").strip()),
        }
        if isinstance(consultation_context_metadata, dict):
            payload_metadata["source_version_id"] = consultation_context_metadata.get(
                "source_version_id"
            )
            payload_metadata["revision_version_id"] = consultation_context_metadata.get(
                "revision_version_id"
            )
            payload_metadata["pipeline_run_id"] = consultation_context_metadata.get(
                "pipeline_run_id"
            )
            payload_metadata["drug_analysis_entrypoint"] = (
                consultation_context_metadata.get("drug_analysis_entrypoint")
            )
            payload_metadata["report_finalization_entrypoint"] = (
                consultation_context_metadata.get("report_finalization_entrypoint")
            )
            payload_metadata["conclusion_entrypoint"] = (
                consultation_context_metadata.get("conclusion_entrypoint")
            )
            payload_metadata["synthesis_mode"] = consultation_context_metadata.get(
                "synthesis_mode"
            )
        return clinical_session, final_report, payload_metadata

    # -------------------------------------------------------------------------
    @classmethod
    def _resolve_consultation_timeout(cls) -> float:
        configured = float(get_server_settings().runtime.clinical_llm_timeout)
        return cls._resolve_runtime_timeout(base_timeout_s=configured)

    # -------------------------------------------------------------------------
    def apply_persisted_runtime_configuration(self) -> None:
        self.model_config_service.ensure_defaults()
