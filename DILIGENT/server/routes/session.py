from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from DILIGENT.server.schemas.clinical import (
    PatientData,
)
from DILIGENT.server.utils.configurations import LLMRuntimeConfig
from DILIGENT.server.utils.logger import logger
from DILIGENT.server.utils.repository.serializer import DataSerializer
from DILIGENT.server.utils.services.clinical.hepatox import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from DILIGENT.server.utils.services.clinical.parser import DrugsParser
from DILIGENT.server.utils.services.retrieval.query import DILIQueryBuilder

drugs_parser = DrugsParser()
pattern_analyzer = HepatotoxicityPatternAnalyzer()
router = APIRouter(tags=["session"])
serializer = DataSerializer()


###############################################################################
class NarrativeBuilder:
    
    # -------------------------------------------------------------------------
    @staticmethod
    def build_bullet_list(content: str | None) -> list[str]:
        lines: list[str] = []
        if content:
            for entry in content.splitlines():
                stripped = entry.strip()
                if stripped:
                    lines.append(f"- {stripped}")
        if not lines:
            lines.append("- No data provided.")
        return lines

    # -------------------------------------------------------------------------
    @staticmethod
    def compact_spacing(content: str) -> str:
        cleaned: list[str] = []
        previous_blank = False
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line:
                if previous_blank:
                    continue
                previous_blank = True
                cleaned.append("")
                continue
            previous_blank = False
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    # -------------------------------------------------------------------------
    @staticmethod
    def build_patient_narrative(
        *,
        patient_label: str,
        visit_label: str,
        anamnesis: str | None,
        drugs_text: str | None,
        pattern_score,
        pattern_strings: dict[str, str],
        detected_drugs: list[str],
        final_report: str | None,
    ) -> str:
        classification = getattr(pattern_score, "classification", "Not available")
        alt_multiple = pattern_strings.get("alt_multiple", "Not available")
        alp_multiple = pattern_strings.get("alp_multiple", "Not available")
        r_score = pattern_strings.get("r_score", "Not available")
        drug_summary = ", ".join(detected_drugs) if detected_drugs else "None detected"

        sections: list[str] = []

        header_section = [
            "# Clinical Visit Summary",
            "",
            f"- **Patient:** {patient_label}",
            f"- **Visit date:** {visit_label}",
        ]
        sections.append("\n".join(header_section))

        anamnesis_content = anamnesis if anamnesis else "_No anamnesis provided._"
        sections.append("\n".join(["## Anamnesis and Exams", "", anamnesis_content]))

        pattern_section = [
            "## Hepato-toxicity Pattern",
            "",
            f"- **Classification:** {classification}",
            f"- **ALT multiple:** {alt_multiple}",
            f"- **ALP multiple:** {alp_multiple}",
            f"- **R-score:** {r_score}",
        ]
        sections.append("\n".join(pattern_section))

        therapy_section = ["## Pharmacological Therapy", ""]
        therapy_section.extend(NarrativeBuilder.build_bullet_list(drugs_text))
        therapy_section.extend(
            [
                "",
                f"**Detected drugs ({len(detected_drugs)}):** {drug_summary}",
            ]
        )
        sections.append("\n".join(therapy_section))

        clinical_report_section = ["## Clinical Report", ""]
        clinical_report_section.append(
            final_report.strip() if final_report else "No clinical report generated."
        )
        sections.append("\n".join(clinical_report_section))

        return NarrativeBuilder.compact_spacing("\n\n".join(sections))


###############################################################################
class ClinicalSessionEndpoint:
    def __init__(
        self,
        *,
        router: APIRouter,
        drugs_parser: DrugsParser,
        pattern_analyzer: HepatotoxicityPatternAnalyzer,
        serializer: DataSerializer,
    ) -> None:
        self.router = router
        self.drugs_parser = drugs_parser
        self.pattern_analyzer = pattern_analyzer
        self.serializer = serializer

        self.router.add_api_route(
            "/clinical",
            self.start_clinical_session,
            methods=["POST"],
            response_model=None,
            status_code=status.HTTP_202_ACCEPTED,
            response_class=PlainTextResponse,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def serialize_validation_errors(
        errors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for error in errors:
            ctx = error.get("ctx")
            if isinstance(ctx, dict) and "error" in ctx:
                serialized.append({**error, "ctx": {**ctx, "error": str(ctx["error"])}})
                continue
            serialized.append(error)
        return serialized

    # -------------------------------------------------------------------------
    def apply_runtime_overrides(
        self,
        *,
        use_cloud_services: bool | None,
        llm_provider: str | None,
        cloud_model: str | None,
        parsing_model: str | None,
        clinical_model: str | None,
        ollama_temperature: float | None,
        ollama_reasoning: bool | None,
    ) -> None:
        if use_cloud_services is not None:
            LLMRuntimeConfig.set_use_cloud_services(use_cloud_services)
        if llm_provider is not None:
            LLMRuntimeConfig.set_llm_provider(llm_provider)
        if cloud_model is not None:
            LLMRuntimeConfig.set_cloud_model(cloud_model)
        if parsing_model is not None:
            LLMRuntimeConfig.set_parsing_model(parsing_model)
        if clinical_model is not None:
            LLMRuntimeConfig.set_clinical_model(clinical_model)
        if ollama_temperature is not None:
            LLMRuntimeConfig.set_ollama_temperature(ollama_temperature)
        if ollama_reasoning is not None:
            LLMRuntimeConfig.set_ollama_reasoning(ollama_reasoning)

        parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model("parser")
        clinical_provider, clinical_model_resolved = LLMRuntimeConfig.resolve_provider_and_model("clinical")
        logger.info(
            "Resolved LLM runtime for request: cloud=%s provider=%s cloud_model=%s parsing_provider=%s parsing_model=%s clinical_provider=%s clinical_model=%s temperature=%.2f reasoning=%s",
            LLMRuntimeConfig.is_cloud_enabled(),
            LLMRuntimeConfig.get_llm_provider(),
            LLMRuntimeConfig.get_cloud_model(),
            parser_provider,
            parser_model,
            clinical_provider,
            clinical_model_resolved,
            LLMRuntimeConfig.get_ollama_temperature(),
            LLMRuntimeConfig.is_ollama_reasoning_enabled(),
        )

    # -------------------------------------------------------------------------
    async def process_single_patient(self, payload: PatientData) -> str:
        logger.info(
            "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
            payload.name,
        )

        global_start_time = time.perf_counter()

        pattern_score = self.pattern_analyzer.calculate_hepatotoxicity_pattern(payload)
        logger.info(
            "Patient hepatotoxicity pattern classified as %s (R=%.3f)",
            pattern_score.classification,
            pattern_score.r_score if pattern_score.r_score is not None else float("nan"),
        )

        start_time = time.perf_counter()
        drug_data = await self.drugs_parser.extract_drug_list(payload.drugs or "")
        elapsed = time.perf_counter() - start_time
        logger.info("Drugs extraction required %.4f seconds", elapsed)
        logger.info("Detected %s drugs", len(drug_data.entries))

        rag_query: dict[str, str] | None = None
        if payload.use_rag:
            query_builder = DILIQueryBuilder(drug_data)
            logger.info("RAG retrieval enabled for clinical consultation")
            rag_query = query_builder.build_dili_queries(
                clinical_context=payload.anamnesis or "",
                pattern_classification=pattern_score.classification,
                r_score=pattern_score.r_score,
            )

        clinical_session = HepatoxConsultation(drug_data, patient_name=payload.name)
        drug_assessment = await clinical_session.run_analysis(
            clinical_context=payload.anamnesis,
            visit_date=payload.visit_date,
            pattern_score=pattern_score,
            rag_query=rag_query,
        )
        elapsed = time.perf_counter() - start_time
        logger.info("Hepato-toxicity consultation required %.4f seconds", elapsed)

        final_report: str | None = None
        if isinstance(drug_assessment, dict):
            final_report = drug_assessment.get("final_report", "").strip()

        patient_label = payload.name or "Unknown patient"
        visit_label = (
            payload.visit_date.strftime("%d %B %Y")
            if payload.visit_date
            else "Not provided"
        )

        global_elapsed = time.perf_counter() - global_start_time
        logger.info(
            "Total time for Drug Induced Liver Injury (DILI) assessment is %.4f seconds",
            global_elapsed,
        )

        detected_drugs = [entry.name for entry in drug_data.entries if entry.name]
        pattern_strings = self.pattern_analyzer.stringify_scores(pattern_score)
        await asyncio.to_thread(
            self.serializer.save_clinical_session,
            {
                "patient_name": payload.name,
                "session_timestamp": datetime.now(),
                "alt_value": payload.alt,
                "alt_upper_limit": payload.alt_max,
                "alp_value": payload.alp,
                "alp_upper_limit": payload.alp_max,
                "hepatic_pattern": pattern_score.classification,
                "anamnesis": payload.anamnesis,
                "drugs": payload.drugs,
                "parsing_model": getattr(self.drugs_parser, "model", None),
                "clinical_model": getattr(clinical_session, "llm_model", None),
                "total_duration": global_elapsed,
                "final_report": final_report,
            },
        )

        narrative = NarrativeBuilder.build_patient_narrative(
            patient_label=patient_label,
            visit_label=visit_label,
            anamnesis=payload.anamnesis,
            drugs_text=payload.drugs,
            pattern_score=pattern_score,
            pattern_strings=pattern_strings,
            detected_drugs=detected_drugs,
            final_report=final_report,
        )

        return narrative

    # -------------------------------------------------------------------------
    async def start_clinical_session(
        self,
        name: str | None = Body(default=None),
        visit_date: date | dict[str, int] | str | None = Body(default=None),
        anamnesis: str | None = Body(default=None),
        has_hepatic_diseases: bool = Body(default=False),
        use_rag: bool = Body(default=False),
        drugs: str | None = Body(default=None),
        alt: str | None = Body(default=None),
        alt_max: str | None = Body(default=None),
        alp: str | None = Body(default=None),
        alp_max: str | None = Body(default=None),
        use_cloud_services: bool | None = Body(default=None),
        llm_provider: str | None = Body(default=None),
        cloud_model: str | None = Body(default=None),
        parsing_model: str | None = Body(default=None),
        clinical_model: str | None = Body(default=None),
        ollama_temperature: float | None = Body(default=None),
        ollama_reasoning: bool | None = Body(default=None),
        ) -> PlainTextResponse:
        self.apply_runtime_overrides(
            use_cloud_services=use_cloud_services,
            llm_provider=llm_provider,
            cloud_model=cloud_model,
            parsing_model=parsing_model,
            clinical_model=clinical_model,
            ollama_temperature=ollama_temperature,
            ollama_reasoning=ollama_reasoning,
        )
        try:
            payload_data: dict[str, Any] = {
                "name": name,
                "visit_date": visit_date,
                "anamnesis": anamnesis,
                "has_hepatic_diseases": has_hepatic_diseases,
                "use_rag": use_rag,
                "drugs": drugs,
                "alt": alt,
                "alt_max": alt_max,
                "alp": alp,
                "alp_max": alp_max,
            }
            payload = PatientData.model_validate(payload_data)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=self.serialize_validation_errors(exc.errors()),
            ) from exc

        single_result = await self.process_single_patient(payload)
        return PlainTextResponse(content=single_result)


endpoint = ClinicalSessionEndpoint(
    router=router,
    drugs_parser=drugs_parser,
    pattern_analyzer=pattern_analyzer,
    serializer=serializer,
)
