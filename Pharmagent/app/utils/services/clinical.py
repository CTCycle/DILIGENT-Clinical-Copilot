from __future__ import annotations

import asyncio
import re
from pathlib import Path

from Pharmagent.app.api.models.prompts import (
    HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
    HEPATOTOXICITY_ANALYSIS_USER_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    DrugHepatotoxicityAnalysis,
    DrugToxicityFindings,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugToxicityBundle,
    PatientDrugs,
)
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.constants import DOCS_PATH
from Pharmagent.app.logger import logger


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
    def __init__(self, drugs: PatientDrugs, *, timeout_s: float = 300.0) -> None:
        self.drugs = drugs
        self.timeout_s = float(timeout_s)
        self.client = initialize_llm_client(purpose="agent", timeout_s=self.timeout_s)
        self.model = ClientRuntimeConfig.get_agent_model()
        self.livertox_root = Path(DOCS_PATH) / "livertox"
        self.max_prompt_chars = 6000

    # -------------------------------------------------------------------------
    async def run(self) -> PatientDrugToxicityBundle:
        results: list[DrugHepatotoxicityAnalysis] = []
        for entry in self.drugs.entries:
            drug_name = entry.name.strip()
            if not drug_name:
                continue
            analysis = await self._process_drug(drug_name)
            results.append(analysis)
        return PatientDrugToxicityBundle(entries=results)

    # -------------------------------------------------------------------------
    async def _process_drug(self, drug_name: str) -> DrugHepatotoxicityAnalysis:
        source_text = await self._gather_livertox_text(drug_name)
        if not source_text:
            message = "No LiverTox monograph available for this drug."
            logger.warning("LiverTox data unavailable for drug '%s'", drug_name)
            return DrugHepatotoxicityAnalysis(
                drug_name=drug_name, source_text=None, analysis=None, error=message
            )

        prompt_text = self._prepare_prompt_text(source_text)
        user_prompt = HEPATOTOXICITY_ANALYSIS_USER_PROMPT.format(
            drug_name=drug_name,
            source_text=prompt_text,
        )

        try:
            findings = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=DrugToxicityFindings,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "LLM hepatotoxicity analysis failed for '%s': %s", drug_name, exc
            )
            return DrugHepatotoxicityAnalysis(
                drug_name=drug_name,
                source_text=prompt_text,
                analysis=None,
                error=str(exc),
            )

        return DrugHepatotoxicityAnalysis(
            drug_name=drug_name,
            source_text=prompt_text,
            analysis=findings,
            error=None,
        )

    # -------------------------------------------------------------------------
    async def _gather_livertox_text(self, drug_name: str) -> str | None:
        if not self.livertox_root.exists() or not self.livertox_root.is_dir():
            return None

        slug = self._slugify(drug_name)
        candidates = [
            self.livertox_root / f"{slug}.txt",
            self.livertox_root / f"{slug}.md",
            self.livertox_root / f"{slug}.html",
        ]

        for path in candidates:
            if path.exists():
                return await asyncio.to_thread(self._read_text_file, path)

        # Fallback: perform a case-insensitive search for files containing the name
        try:
            for path in self.livertox_root.glob("**/*"):
                if not path.is_file():
                    continue
                if slug in path.stem.lower():
                    return await asyncio.to_thread(self._read_text_file, path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed scanning LiverTox directory: %s", exc)
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _read_text_file(path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text.strip()

    # -------------------------------------------------------------------------
    def _prepare_prompt_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= self.max_prompt_chars:
            return normalized
        return normalized[: self.max_prompt_chars]

    # -------------------------------------------------------------------------
    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


