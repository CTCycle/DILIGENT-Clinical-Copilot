from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import httpx
from Pharmagent.app.api.models.prompts import (
    HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
    HEPATOTOXICITY_ANALYSIS_USER_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    DrugHepatotoxicityAnalysis,
    DrugToxicityFindings,
    HepatotoxicityPatternScore,
    LiverToxMatchInfo,
    PatientData,
    PatientDrugToxicityBundle,
    PatientDrugs,
)
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.logger import logger


###############################################################################
@dataclass(slots=True)
class LiverToxMatch:
    nbk_id: str
    matched_name: str
    confidence: float
    reason: str


###############################################################################
@dataclass(slots=True)
class CandidateSummary:
    nbk_id: str
    title: str
    synonyms: set[str]


###############################################################################
@dataclass(slots=True)
class RxNormConcept:
    rxcui: str
    preferred_name: str | None
    synonyms: set[str]
    ingredients: set[str]
    tty: str | None


###############################################################################
@dataclass(slots=True)
class NameCandidate:
    origin: str
    name: str
    priority: int


###############################################################################
class HepatotoxicityPatternAnalyzer:
    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    def _safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class DrugToxicityEssay:
    
    def __init__(self, drugs: PatientDrugs, *, timeout_s: float = 300.0) -> None:
        pass

    def run_analysis(self) -> DrugToxicityFindings:
        pass
        