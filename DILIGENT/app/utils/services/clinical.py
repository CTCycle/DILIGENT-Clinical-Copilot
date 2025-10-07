from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import date, datetime
from typing import Any

from DILIGENT.app.api.models.prompts import (
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
)
from DILIGENT.app.api.models.providers import initialize_llm_client
from DILIGENT.app.api.schemas.clinical import (
    DrugEntry,
    DrugClinicalAssessment,
    DrugSuspensionContext,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugClinicalReport,
    PatientDrugs,
)
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DRUG_SUSPENSION_EXCLUSION_DAYS,
)
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.serializer import DataSerializer
from DILIGENT.app.utils.services.livertox import LiverToxMatch, LiverToxMatcher


###############################################################################
class HepatotoxicityPatternAnalyzer:
    # -------------------------------------------------------------------------
    def analyze(self, payload: PatientData) -> HepatotoxicityPatternScore:
        alt_value = self._parse_marker_value(payload.alt)
        alt_max_value = self._parse_marker_value(payload.alt_max)
        alp_value = self._parse_marker_value(payload.alp)
        alp_max_value = self._parse_marker_value(payload.alp_max)

        alt_multiple = self._safe_ratio(alt_value, alt_max_value)
        alp_multiple = self._safe_ratio(alp_value, alp_max_value)

        r_score: float | None = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = "indeterminate"
        if r_score is not None:
            if r_score > 5:
                classification = "hepatocellular"
            elif r_score < 2:
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
    def _parse_marker_value(self, raw: str | None) -> float | None:
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
    def _safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class LiverToxConsultation:
    MAX_EXCERPT_LENGTH = 4000

    # -------------------------------------------------------------------------
    def __init__(
        self, drugs: PatientDrugs, *, timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS
    ) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.serializer = DataSerializer()
        self.livertox_df = None
        self.master_list_df = None
        self.matcher: LiverToxMatcher | None = None
        self.llm_client = initialize_llm_client(purpose="agent", timeout_s=timeout_s)
        _provider, model_candidate = ClientRuntimeConfig.resolve_provider_and_model(
            "agent"
        )
        self.llm_model = model_candidate or ClientRuntimeConfig.get_agent_model()
        try:
            chat_signature = inspect.signature(self.llm_client.chat)
        except (TypeError, ValueError):
            chat_signature = None
        self._chat_supports_temperature = (
            chat_signature is not None and "temperature" in chat_signature.parameters
        )
        self.temperature = 0.2

    # -------------------------------------------------------------------------
    async def run_analysis(
        self,
        *,
        anamnesis: str | None = None,
        visit_date: date | None = None,
        diseases: list[str] | None = None,
        hepatic_diseases: list[str] | None = None,
    ) -> dict[str, Any] | None:
        logger.info("Toxicity analysis stage 1/3: validating inputs")
        patient_drugs = self._collect_patient_drugs()
        if not patient_drugs:
            logger.info("No drugs detected for toxicity analysis")
            return None
        if not self._ensure_livertox_loaded():
            return None

        if self.matcher is None:
            return PatientDrugClinicalReport(entries=[], final_report=None).model_dump()

        logger.info("Toxicity analysis stage 2/3: matching drugs to LiverTox records")
        matches = await self.matcher.match_drug_names(patient_drugs)

        logger.info(
            "Toxicity analysis stage 3/3: performing clinical assessment for matched drugs"
        )
        resolved = self._resolve_matches(patient_drugs, matches)
        report = await self.compile_clinical_assessment(
            resolved,
            anamnesis=anamnesis,
            visit_date=visit_date,
            diseases=diseases or [],
            hepatic_diseases=hepatic_diseases or [],
        )
        return report.model_dump()

    # -------------------------------------------------------------------------
    def _ensure_livertox_loaded(self) -> bool:
        if self.matcher is not None:
            return True
        try:
            dataset = self.serializer.get_livertox_records()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading LiverTox monographs from database: %s", exc)
            self.matcher = None
            return False
        if dataset is None or dataset.empty:
            logger.warning(
                "LiverTox monograph table is empty; toxicity essay cannot run"
            )
            self.matcher = None
            return False
        self.livertox_df = dataset
        try:
            master_list = self.serializer.get_livertox_master_list()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading LiverTox master list from database: %s", exc)
            master_list = None
        self.master_list_df = master_list
        self.matcher = LiverToxMatcher(dataset, master_list)
        return True

    # -------------------------------------------------------------------------
    def _collect_patient_drugs(self) -> list[str]:
        return [entry.name for entry in self.drugs.entries if entry.name]

    # -------------------------------------------------------------------------
    def _resolve_matches(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        if self.matcher is None:
            return []
        return self.matcher.build_patient_mapping(patient_drugs, matches)

    # -------------------------------------------------------------------------
    async def compile_clinical_assessment(
        self,
        resolved_entries: list[dict[str, Any]],
        *,
        anamnesis: str | None,
        visit_date: date | None,
        diseases: list[str],
        hepatic_diseases: list[str],
    ) -> PatientDrugClinicalReport:
        normalized_anamnesis = (anamnesis or "").strip()
        if not normalized_anamnesis:
            normalized_anamnesis = "No anamnesis information was provided."
        normalized_diseases = self._normalize_list(diseases)
        normalized_hepatic = self._normalize_list(hepatic_diseases)
        disease_summary = ", ".join(normalized_diseases) if normalized_diseases else "None reported"
        hepatic_summary = (
            ", ".join(normalized_hepatic) if normalized_hepatic else "None reported"
        )

        entries: list[DrugClinicalAssessment] = []
        llm_tasks: list[Any] = []
        llm_indices: list[int] = []

        for idx, drug_entry in enumerate(self.drugs.entries):
            resolved = resolved_entries[idx] if idx < len(resolved_entries) else {}
            matched_row = (
                resolved.get("matched_livertox_row")
                if isinstance(resolved, dict)
                else None
            )
            if isinstance(resolved, dict):
                candidate_excerpts = resolved.get("extracted_excerpts", [])
                if isinstance(candidate_excerpts, list):
                    raw_excerpts = candidate_excerpts
                elif candidate_excerpts is None:
                    raw_excerpts = []
                else:
                    raw_excerpts = [str(candidate_excerpts)]
            else:
                raw_excerpts = []
            cleaned_excerpts = [
                str(excerpt).strip()
                for excerpt in raw_excerpts
                if isinstance(excerpt, str) and excerpt.strip()
            ]
            suspension = self._evaluate_suspension(drug_entry, visit_date)
            entry = DrugClinicalAssessment(
                drug_name=drug_entry.name,
                matched_livertox_row=matched_row if isinstance(matched_row, dict) else None,
                extracted_excerpts=cleaned_excerpts,
                suspension=suspension,
            )
            entries.append(entry)

            if suspension.excluded:
                entry.paragraph = self._build_excluded_paragraph(drug_entry.name, suspension)
                continue

            excerpt = self._select_excerpt(cleaned_excerpts)
            if excerpt is None:
                entry.paragraph = self._build_missing_excerpt_paragraph(drug_entry.name)
                continue

            llm_indices.append(idx)
            llm_tasks.append(
                self._request_drug_analysis(
                    drug_name=drug_entry.name,
                    excerpt=excerpt,
                    anamnesis=normalized_anamnesis,
                    diseases=disease_summary,
                    hepatic_diseases=hepatic_summary,
                    suspension=suspension,
                )
            )

        if llm_tasks:
            responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
            for idx, outcome in zip(llm_indices, responses):
                entry = entries[idx]
                if isinstance(outcome, Exception):
                    logger.error(
                        "Clinical analysis for drug '%s' failed: %s",
                        entry.drug_name,
                        outcome,
                    )
                    entry.paragraph = self._build_error_paragraph(entry.drug_name)
                else:
                    entry.paragraph = outcome

        final_report = self._compose_final_report(entries)
        return PatientDrugClinicalReport(entries=entries, final_report=final_report)

    # -------------------------------------------------------------------------
    def _select_excerpt(self, excerpts: list[str]) -> str | None:
        if not excerpts:
            return None
        combined = "\n\n".join(chunk.strip() for chunk in excerpts if chunk.strip())
        if not combined:
            return None
        if len(combined) <= self.MAX_EXCERPT_LENGTH:
            return combined
        truncated = combined[: self.MAX_EXCERPT_LENGTH]
        cutoff = truncated.rfind("\n")
        if cutoff > 2000:
            truncated = truncated[:cutoff]
        return truncated.strip()

    # -------------------------------------------------------------------------
    def _evaluate_suspension(
        self, entry: DrugEntry, visit_date: date | None
    ) -> DrugSuspensionContext:
        suspended = bool(entry.suspension_status)
        parsed_date = self._parse_suspension_date(entry.suspension_date, visit_date)
        excluded = False
        note: str | None = None
        if not suspended:
            return DrugSuspensionContext(
                suspended=False,
                suspension_date=None,
                excluded=False,
                note=None,
            )
        if parsed_date is None:
            note = "Suspension reported without a reliable date; drug kept in analysis."
        elif visit_date is None:
            note = (
                f"Suspended on {parsed_date.isoformat()}, but visit date missing; drug kept in analysis."
            )
        else:
            delta = (visit_date - parsed_date).days
            if delta >= DRUG_SUSPENSION_EXCLUSION_DAYS:
                excluded = True
                note = (
                    f"Suspended on {parsed_date.isoformat()} (≥{DRUG_SUSPENSION_EXCLUSION_DAYS} days before the visit)."
                )
            elif delta < 0:
                note = (
                    f"Suspension date {parsed_date.isoformat()} is after the visit; treat as ongoing exposure."
                )
            else:
                note = (
                    f"Suspended on {parsed_date.isoformat()} ({delta} days before the visit); still clinically relevant."
                )
        return DrugSuspensionContext(
            suspended=suspended,
            suspension_date=parsed_date,
            excluded=excluded,
            note=note,
        )

    # -------------------------------------------------------------------------
    def _parse_suspension_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        if raw_date is None:
            return None
        text = str(raw_date).strip()
        if not text:
            return None
        normalized = text.replace("/", "-").replace(".", "-").replace(",", "-")
        tokens = [token for token in normalized.split("-") if token]
        candidates: list[str] = []
        if visit_date is not None and len(tokens) == 2:
            day, month = tokens
            candidates.extend(
                [
                    f"{day.zfill(2)}-{month.zfill(2)}-{visit_date.year}",
                    f"{month.zfill(2)}-{day.zfill(2)}-{visit_date.year}",
                    f"{visit_date.year}-{month.zfill(2)}-{day.zfill(2)}",
                ]
            )
        candidates.append("-".join(tokens))
        candidates.append(text)
        candidates.append(normalized)
        checked: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in checked:
                continue
            checked.add(candidate)
            parsed = self._try_parse_date(candidate)
            if parsed is not None:
                return parsed
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _try_parse_date(value: str) -> date | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        iso_candidate = cleaned.replace(".", "-").replace("/", "-")
        try:
            return date.fromisoformat(iso_candidate)
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    def _format_suspension_prompt(self, suspension: DrugSuspensionContext) -> str:
        if not suspension.suspended:
            return "Active therapy; no suspension reported."
        if suspension.suspension_date is None:
            return (
                "Reported as suspended without a reliable date; residual exposure must be considered."
            )
        base = f"Suspended on {suspension.suspension_date.isoformat()}."
        if suspension.excluded:
            return (
                f"{base} This occurred well before the visit (≥{DRUG_SUSPENSION_EXCLUSION_DAYS} days)."
            )
        return f"{base} Suspension was close to the visit, so lingering effects are possible."

    # -------------------------------------------------------------------------
    async def _request_drug_analysis(
        self,
        *,
        drug_name: str,
        excerpt: str,
        anamnesis: str,
        diseases: str,
        hepatic_diseases: str,
        suspension: DrugSuspensionContext,
    ) -> str:
        suspension_details = self._format_suspension_prompt(suspension)
        user_prompt = LIVERTOX_CLINICAL_USER_PROMPT.format(
            drug_name=self._escape_braces(drug_name.strip() or drug_name),
            excerpt=self._escape_braces(excerpt),
            anamnesis=self._escape_braces(anamnesis),
            diseases=self._escape_braces(diseases),
            hepatic_diseases=self._escape_braces(hepatic_diseases),
            suspension_details=self._escape_braces(suspension_details),
        )
        messages = [
            {"role": "system", "content": LIVERTOX_CLINICAL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
        ]
        chat_kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
        }
        if self._chat_supports_temperature:
            chat_kwargs["temperature"] = self.temperature
        else:
            chat_kwargs["options"] = {"temperature": self.temperature}
        try:
            raw_response = await self.llm_client.chat(**chat_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM analysis failed for {drug_name}: {exc}") from exc
        return self._coerce_chat_text(raw_response)

    # -------------------------------------------------------------------------
    @staticmethod
    def _escape_braces(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_chat_text(raw_response: Any) -> str:
        if isinstance(raw_response, str):
            return raw_response.strip()
        if isinstance(raw_response, dict):
            for key in ("content", "text", "response"):
                value = raw_response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(raw_response, ensure_ascii=False)
        return str(raw_response).strip()

    # -------------------------------------------------------------------------
    def _compose_final_report(
        self, entries: list[DrugClinicalAssessment]
    ) -> str | None:
        if not entries:
            return None
        blocks: list[str] = []
        for entry in entries:
            summary = entry.paragraph or "No analysis available."
            block = f"{entry.drug_name.strip()}\nConclusions: {summary.strip()}"
            blocks.append(block.strip())
        return "\n\n".join(blocks) if blocks else None

    # -------------------------------------------------------------------------
    def _build_excluded_paragraph(
        self, drug_name: str, suspension: DrugSuspensionContext
    ) -> str:
        if suspension.suspension_date is not None:
            return (
                f"The therapy was suspended on {suspension.suspension_date.isoformat()}, "
                f"at least {DRUG_SUSPENSION_EXCLUSION_DAYS} days before the visit; the drug was excluded from this DILI assessment."
            )
        return (
            "The therapy was reported as suspended well before the visit and was excluded from the current DILI assessment."
        )

    # -------------------------------------------------------------------------
    def _build_missing_excerpt_paragraph(self, drug_name: str) -> str:
        return (
            "No LiverTox excerpt was available for this drug, so hepatotoxic involvement could not be evaluated."
        )

    # -------------------------------------------------------------------------
    def _build_error_paragraph(self, drug_name: str) -> str:
        return (
            "Automated analysis was unavailable due to a technical issue; manual review is recommended."
        )

    # -------------------------------------------------------------------------
    def _normalize_list(self, values: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for value in values:
            if value is None:
                continue
            cleaned = str(value).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in unique:
                unique[key] = cleaned
        return list(unique.values())
