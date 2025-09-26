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
from Pharmagent.app.utils.services.livertox import LiverToxMatcher, LiverToxMatch
from Pharmagent.app.constants import DEFAULT_LLM_TIMEOUT_SECONDS


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
    # -------------------------------------------------------------------------
    def __init__(
        self, drugs: PatientDrugs, *, timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS
    ) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.serializer = DataSerializer()
        self.llm_client = initialize_llm_client(purpose="parser", timeout_s=timeout_s)
        self.livertox_df = None
        self.matcher: LiverToxMatcher | None = None

    # -------------------------------------------------------------------------
    async def run_analysis(self) -> list[dict[str, Any]] | None:
        logger.info("Toxicity analysis stage 1/3: validating inputs")
        patient_drugs = self._collect_patient_drugs()
        if not patient_drugs:
            logger.info("No drugs detected for toxicity analysis")
            return None
        if not self._ensure_livertox_loaded():
            return None

        logger.info("Toxicity analysis stage 2/3: matching drugs to LiverTox records")
        matches = await self._match_livertox_entries(patient_drugs)

        logger.info("Toxicity analysis stage 3/3: compiling matched LiverTox excerpts")
        return self._resolve_matches(patient_drugs, matches)

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
        self.matcher = LiverToxMatcher(dataset, llm_client=self.llm_client)
        return True

    # -------------------------------------------------------------------------
    def _collect_patient_drugs(self) -> list[str]:
        return [entry.name for entry in self.drugs.entries if entry.name]

    # -------------------------------------------------------------------------
    async def _match_livertox_entries(
        self,
        patient_drugs: list[str],
    ) -> list[LiverToxMatch | None]:
        if self.matcher is None:
            return []
        return await self.matcher.match_drug_names(
            patient_drugs,
            livertox_drugs=self.matcher.lookup_entries,
        )

    # -------------------------------------------------------------------------
    def _resolve_matches(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        if self.matcher is None:
            return []
        return self.matcher.build_patient_mapping(patient_drugs, matches)
