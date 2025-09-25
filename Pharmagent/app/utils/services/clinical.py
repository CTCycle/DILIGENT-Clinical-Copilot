from __future__ import annotations

import re
from typing import Any

from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugs,
)
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.livertox import LiverToxMatcher


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
class DrugToxicityEssay:

    # -------------------------------------------------------------------------
    def __init__(self, drugs: PatientDrugs, *, timeout_s: float = 300.0) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.serializer = DataSerializer()
        self.llm_client = initialize_llm_client(purpose="agent", timeout_s=timeout_s)
        self.livertox_df = None
        self.matcher: LiverToxMatcher | None = None

    # -------------------------------------------------------------------------
    async def run_analysis(self) -> dict[str, dict[str, Any]]:
        patient_drugs = [entry.name for entry in self.drugs.entries if entry.name]
        if not patient_drugs:
            return {}
        self._ensure_livertox_loaded()
        if self.matcher is None:
            return self._empty_result(patient_drugs)
        matches = await self.matcher.match_drug_names(patient_drugs)
        return self.matcher.build_patient_mapping(matches)

    # -------------------------------------------------------------------------
    def _ensure_livertox_loaded(self) -> None:
        if self.matcher is not None:
            return
        try:
            dataset = self.serializer.get_livertox_records()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading LiverTox monographs from database: %s", exc)
            self.matcher = None
            return
        if dataset is None or dataset.empty:
            logger.warning("LiverTox monograph table is empty; toxicity essay cannot run")
            self.matcher = None
            return
        self.livertox_df = dataset
        self.matcher = LiverToxMatcher(dataset, llm_client=self.llm_client)

    # -------------------------------------------------------------------------
    def _empty_result(self, patient_drugs: list[str]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "matched_livertox_row": None,
                "extracted_excerpts": [],
            }
            for name in patient_drugs
        }

