from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_core import ErrorDetails

from common.utils.types import coerce_bool_or_unknown
from domain.clinical.entities import (
    ClinicalLabEntry,
    DiseaseContextEntry,
    DrugEntry,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PipelineIssue,
)
from services.text.normalization import normalize_drug_query_name


###############################################################################
class ClinicalSessionFormattingMixin:
    @staticmethod
    def serialize_validation_errors(
        errors: Sequence[ErrorDetails],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for error in errors:
            error_dict: dict[str, Any] = dict(error)
            ctx = error_dict.get("ctx")
            if isinstance(ctx, dict) and "error" in ctx:
                serialized.append(
                    {**error_dict, "ctx": {**ctx, "error": str(ctx["error"])}}
                )
                continue
            serialized.append(error_dict)
        return serialized

    @staticmethod
    def serialize_pipeline_issues(
        issues: Sequence[PipelineIssue],
    ) -> list[dict[str, Any]]:
        return [issue.model_dump() for issue in issues]

    @staticmethod
    def merge_drugs_for_analysis(
        therapy_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
    ) -> PatientDrugs:
        merged_entries: list[DrugEntry] = []
        seen_keys: set[str] = set()
        ordered = [*therapy_drugs.entries, *anamnesis_drugs.entries]
        for entry in ordered:
            raw_name = (entry.name or "").strip()
            if not raw_name:
                continue
            lookup_key = normalize_drug_query_name(raw_name)
            if not lookup_key or lookup_key in seen_keys:
                continue
            seen_keys.add(lookup_key)
            merged_entries.append(entry)
        return PatientDrugs(entries=merged_entries)

    @staticmethod
    def build_fallback_therapy_drugs(raw_text: str | None) -> PatientDrugs:
        if not raw_text:
            return PatientDrugs(entries=[])

        candidates = raw_text.replace(";", "\n").splitlines()
        entries: list[DrugEntry] = []
        seen_keys: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip().lstrip("-*• ").strip()
            if not normalized:
                continue
            lookup_key = normalize_drug_query_name(normalized)
            if not lookup_key or lookup_key in seen_keys:
                continue
            seen_keys.add(lookup_key)
            entries.append(
                DrugEntry(
                    name=normalized,
                    dosage=None,
                    administration_mode=None,
                    route=None,
                    administration_pattern=None,
                    suspension_status=None,
                    suspension_date=None,
                    therapy_start_status=None,
                    therapy_start_date=None,
                    source="therapy",
                    historical_flag=False,
                )
            )
        return PatientDrugs(entries=entries)

    @staticmethod
    def format_structured_diseases(disease_context: PatientDiseaseContext) -> list[str]:
        if not disease_context.entries:
            return ["- None detected."]
        lines: list[str] = []
        for entry in disease_context.entries:
            if not isinstance(entry, DiseaseContextEntry):
                continue
            occurrence = entry.occurrence_time or "unknown"
            chronic = coerce_bool_or_unknown(entry.chronic)
            hepatic_related = coerce_bool_or_unknown(entry.hepatic_related)
            evidence = entry.evidence or "Not reported."
            lines.append(
                f"- {entry.name} | occurrence: {occurrence} | chronic: {chronic} | hepatic-related: {hepatic_related} | evidence: {evidence}"
            )
        return lines or ["- None detected."]

    @staticmethod
    def format_lab_timeline(lab_timeline: PatientLabTimeline) -> list[str]:
        if not lab_timeline.entries:
            return ["- None extracted."]
        lines: list[str] = []
        for entry in lab_timeline.entries:
            if not isinstance(entry, ClinicalLabEntry):
                continue
            date_token = entry.sample_date or entry.relative_time or "unknown_time"
            value_token = (
                entry.value if entry.value is not None else (entry.value_text or "n/a")
            )
            uln_token = (
                entry.upper_limit_normal
                if entry.upper_limit_normal is not None
                else (entry.upper_limit_text or "n/a")
            )
            lines.append(
                f"- {date_token} | {entry.marker_name}: {value_token} (ULN: {uln_token}) | source: {entry.source}"
            )
        return lines or ["- None extracted."]

    @staticmethod
    def format_onset_context(
        onset_context: LiverInjuryOnsetContext | None,
    ) -> list[str]:
        if onset_context is None:
            return ["- Onset anchor unavailable."]
        return [
            f"- Onset date: {onset_context.onset_date or 'Not available'}",
            f"- Onset basis: {onset_context.onset_basis}",
            f"- Evidence: {onset_context.evidence or 'Not reported.'}",
        ]

    @staticmethod
    def build_structured_clinical_context(
        payload: PatientData,
        *,
        therapy_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score: Any,
    ) -> str:
        therapy_mentions = [
            entry.name.strip()
            for entry in therapy_drugs.entries
            if isinstance(entry.name, str) and entry.name.strip()
        ]
        anamnesis_mentions = [
            entry.name.strip()
            for entry in anamnesis_drugs.entries
            if isinstance(entry.name, str) and entry.name.strip()
        ]
        lines: list[str] = [
            "# Clinical Context",
            f"Anamnesis: {(payload.anamnesis or '').strip() or 'Not provided.'}",
            "",
            "# Laboratory Analysis (Raw)",
            (payload.laboratory_analysis or "").strip() or "Not provided.",
            "",
            "# Therapy List (Raw)",
            (payload.drugs or "").strip() or "Not provided.",
            "",
            "# Detected Drugs",
            f"- Therapy: {', '.join(therapy_mentions) if therapy_mentions else 'None'}",
            f"- Anamnesis: {', '.join(anamnesis_mentions) if anamnesis_mentions else 'None'}",
            "",
            "# Structured Disease Timeline (from Anamnesis)",
            *ClinicalSessionFormattingMixin.format_structured_diseases(disease_context),
            "",
            "# Longitudinal Laboratory Timeline",
            *ClinicalSessionFormattingMixin.format_lab_timeline(lab_timeline),
            "",
            "# Estimated Liver Injury Onset Anchor",
            *ClinicalSessionFormattingMixin.format_onset_context(onset_context),
            "",
            "# Visit Date Anchor",
            (
                f"- Visit date: {payload.visit_date.isoformat()}"
                if payload.visit_date
                else "- Visit date: Not provided."
            ),
            "",
            "# Hepatotoxicity Pattern",
            f"- Classification: {getattr(pattern_score, 'classification', 'indeterminate')}",
            f"- R score: {getattr(pattern_score, 'r_score', None)}",
        ]
        return "\n".join(lines).strip()

