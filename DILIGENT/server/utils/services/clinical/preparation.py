from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from DILIGENT.server.schemas.clinical import (
    HepatotoxicityPatternScore,
    PatientDrugs,
)
from DILIGENT.server.utils.logger import logger
from DILIGENT.server.utils.repository.serializer import DataSerializer
from DILIGENT.server.utils.services.clinical.matches import LiverToxMatcher


@dataclass(slots=True)
class HepatoxPreparedInputs:
    resolved_drugs: dict[str, dict[str, Any]]
    pattern_prompt: str
    clinical_context: str


###############################################################################
class ClinicalKnowledgePreparation:
    def __init__(self) -> None:
        self.serializer = DataSerializer()
        self.livertox_matcher: LiverToxMatcher | None = None

    # -------------------------------------------------------------------------
    async def prepare_inputs(
        self,
        drugs: PatientDrugs,
        *,
        clinical_context: str | None,
        pattern_score: HepatotoxicityPatternScore | None,
    ) -> HepatoxPreparedInputs | None:
        patient_drugs = [entry.name for entry in drugs.entries if entry.name]
        if not patient_drugs:
            logger.info("No drugs detected for input preparation")
            return None
        if not await self.ensure_livertox_matcher() or self.livertox_matcher is None:
            return None

        matches = await asyncio.to_thread(
            self.livertox_matcher.match_drug_names,
            patient_drugs,
        )
        livertox_information = await asyncio.to_thread(
            self.livertox_matcher.build_drugs_to_excerpt_mapping,
            patient_drugs,
            matches,
        )

        resolved_drugs = self.normalize_livertox_mapping(livertox_information)
        pattern_prompt = self.build_pattern_prompt(pattern_score)
        normalized_context = (clinical_context or "").strip()

        return HepatoxPreparedInputs(
            resolved_drugs=resolved_drugs,
            pattern_prompt=pattern_prompt,
            clinical_context=normalized_context,
        )

    # -------------------------------------------------------------------------
    async def ensure_livertox_matcher(self) -> bool:
        if self.livertox_matcher is not None:
            return True
        try:
            dataset = await asyncio.to_thread(self.serializer.get_livertox_records)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading LiverTox monographs from database: %s", exc)
            self.livertox_matcher = None
            return False
        if dataset is None or dataset.empty:
            logger.warning(
                "LiverTox monograph table is empty; toxicity essay cannot run"
            )
            self.livertox_matcher = None
            return False
        catalog_stream = self.serializer.stream_drugs_catalog()
        try:
            self.livertox_matcher = await asyncio.to_thread(
                LiverToxMatcher,
                dataset,
                drugs_catalog_df=catalog_stream,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed preparing LiverTox matcher: %s", exc)
            self.livertox_matcher = None
            return False
        return True

    # -------------------------------------------------------------------------
    def normalize_livertox_mapping(self, data: Any) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, str):
                    continue
                stripped_key = key.strip()
                if not stripped_key:
                    continue
                if not isinstance(value, dict):
                    continue
                normalized[stripped_key] = {
                    "drug_name": stripped_key,
                    "matched_livertox_row": value.get("matched_livertox_row"),
                    "extracted_excerpts": self.normalize_excerpt_list(
                        value.get("extracted_excerpts")
                    ),
                }
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                drug_name = item.get("drug_name")
                if not isinstance(drug_name, str):
                    continue
                stripped_name = drug_name.strip()
                if not stripped_name:
                    continue
                normalized[stripped_name] = {
                    "drug_name": stripped_name,
                    "matched_livertox_row": item.get("matched_livertox_row"),
                    "extracted_excerpts": self.normalize_excerpt_list(
                        item.get("extracted_excerpts")
                    ),
                }
        return normalized

    # -------------------------------------------------------------------------
    def normalize_excerpt_list(self, excerpts: Any) -> list[str]:
        if isinstance(excerpts, list):
            normalized: list[str] = []
            for entry in excerpts:
                if isinstance(entry, str):
                    stripped = entry.strip()
                elif entry is None:
                    stripped = ""
                else:
                    stripped = str(entry).strip()
                if stripped:
                    normalized.append(stripped)
            return normalized
        if isinstance(excerpts, str):
            stripped = excerpts.strip()
            return [stripped] if stripped else []
        return []

    # -------------------------------------------------------------------------
    @staticmethod
    def build_pattern_prompt(
        pattern_score: HepatotoxicityPatternScore | None,
    ) -> str:
        if pattern_score is None:
            return "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."
        classification = pattern_score.classification.replace("_", " ")
        segments: list[str] = [
            f"Observed liver injury pattern: {classification.capitalize()}.",
        ]
        if pattern_score.r_score is not None:
            segments.append(f"R ratio ≈ {pattern_score.r_score:.2f}.")
        if pattern_score.alt_multiple is not None:
            segments.append(
                f"ALT is about {pattern_score.alt_multiple:.2f} × the upper reference limit."
            )
        if pattern_score.alp_multiple is not None:
            segments.append(
                f"ALP is about {pattern_score.alp_multiple:.2f} × the upper reference limit."
            )
        segments.append(
            "Treat drugs whose known hepatotoxicity pattern matches this classification as stronger causal candidates, and downgrade mismatches."
        )
        return " ".join(segments)
