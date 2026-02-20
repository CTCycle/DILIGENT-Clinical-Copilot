from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from DILIGENT.server.entities.clinical import (
    DrugEntry,
    HepatotoxicityPatternScore,
    PatientDrugs,
)
from DILIGENT.common.utils.logger import logger
from DILIGENT.server.repositories.serialization.data import DataSerializer
from DILIGENT.server.services.clinical.matches import LiverToxMatcher
from DILIGENT.server.services.text.normalization import (
    canonicalize_drug_query,
    normalize_drug_query_name,
)


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
        drug_candidates = self.build_drug_candidates(drugs)
        if not drug_candidates:
            logger.info("No drugs detected for input preparation")
            return None
        if not await self.ensure_livertox_matcher() or self.livertox_matcher is None:
            return None

        patient_drugs = [candidate["canonical_name"] for candidate in drug_candidates]
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
        self.attach_candidate_metadata(resolved_drugs, drug_candidates)
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
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                mapping_payload = self.normalize_livertox_item(
                    drug_name=key,
                    payload=value,
                )
                if mapping_payload is None:
                    continue
                normalized[mapping_payload["lookup_key"]] = mapping_payload
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                drug_name = item.get("drug_name")
                if not isinstance(drug_name, str):
                    continue
                mapping_payload = self.normalize_livertox_item(
                    drug_name=drug_name,
                    payload=item,
                )
                if mapping_payload is None:
                    continue
                normalized[mapping_payload["lookup_key"]] = mapping_payload
        return normalized

    # -------------------------------------------------------------------------
    def normalize_livertox_item(
        self,
        *,
        drug_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        canonical_name = canonicalize_drug_query(
            payload.get("canonical_drug_name") or drug_name
        )
        normalized_name = normalize_drug_query_name(
            payload.get("normalized_drug_name") or canonical_name or drug_name
        )
        lookup_key = normalized_name.strip()
        if not lookup_key:
            return None
        return {
            "lookup_key": lookup_key,
            "drug_name": (drug_name or "").strip(),
            "canonical_name": canonical_name or (drug_name or "").strip().lower(),
            "normalized_name": normalized_name,
            "matched_livertox_row": payload.get("matched_livertox_row"),
            "extracted_excerpts": self.normalize_excerpt_list(
                payload.get("extracted_excerpts")
            ),
            "match_confidence": payload.get("match_confidence"),
            "match_reason": payload.get("match_reason"),
            "match_status": payload.get("match_status"),
            "match_notes": payload.get("match_notes", []),
            "match_candidates": payload.get("match_candidates", []),
            "missing_livertox": bool(payload.get("missing_livertox")),
            "ambiguous_match": bool(payload.get("ambiguous_match")),
            "origins": [],
            "raw_mentions": [],
            "extraction_metadata": [],
        }

    # -------------------------------------------------------------------------
    def build_drug_candidates(self, drugs: PatientDrugs) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        by_key: dict[str, dict[str, Any]] = {}
        for entry in drugs.entries:
            candidate = self.normalize_drug_entry(entry)
            if candidate is None:
                continue
            key = candidate["lookup_key"]
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                ordered.append(candidate)
                continue
            for origin in candidate["origins"]:
                if origin not in existing["origins"]:
                    existing["origins"].append(origin)
            for raw_mention in candidate["raw_mentions"]:
                if raw_mention not in existing["raw_mentions"]:
                    existing["raw_mentions"].append(raw_mention)
            if candidate["extraction_metadata"]:
                existing["extraction_metadata"].extend(candidate["extraction_metadata"])
        for candidate in ordered:
            candidate["origins"] = sorted(
                dict.fromkeys(candidate["origins"]),
                key=lambda item: (item != "therapy", item),
            )
            candidate["raw_mentions"] = list(dict.fromkeys(candidate["raw_mentions"]))
            candidate["extraction_metadata"] = self.compact_entry_metadata(
                candidate["extraction_metadata"]
            )
        return ordered

    # -------------------------------------------------------------------------
    def normalize_drug_entry(self, entry: DrugEntry) -> dict[str, Any] | None:
        raw_name = (entry.name or "").strip()
        if not raw_name:
            return None
        canonical_name = canonicalize_drug_query(raw_name)
        normalized_name = normalize_drug_query_name(canonical_name or raw_name)
        lookup_key = normalized_name.strip()
        if not lookup_key:
            return None
        origin = entry.source if entry.source in {"therapy", "anamnesis"} else "therapy"
        metadata: dict[str, Any] = {}
        for field_name in (
            "dosage",
            "administration_mode",
            "route",
            "administration_pattern",
            "suspension_status",
            "suspension_date",
            "therapy_start_status",
            "therapy_start_date",
            "temporal_classification",
            "historical_flag",
        ):
            value = getattr(entry, field_name, None)
            if value is None or value == []:
                continue
            metadata[field_name] = value
        return {
            "lookup_key": lookup_key,
            "canonical_name": canonical_name or raw_name.lower(),
            "normalized_name": normalized_name,
            "origins": [origin],
            "raw_mentions": [raw_name],
            "extraction_metadata": [metadata] if metadata else [],
        }

    # -------------------------------------------------------------------------
    def compact_entry_metadata(self, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in metadata:
            if not item:
                continue
            key = str(sorted(item.items()))
            if key in seen:
                continue
            seen.add(key)
            compacted.append(item)
        return compacted

    # -------------------------------------------------------------------------
    def attach_candidate_metadata(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> None:
        for candidate in candidates:
            lookup_key = candidate.get("lookup_key")
            if not isinstance(lookup_key, str):
                continue
            payload = resolved_drugs.get(lookup_key)
            if payload is None:
                payload = {
                    "lookup_key": lookup_key,
                    "drug_name": candidate.get("canonical_name", ""),
                    "canonical_name": candidate.get("canonical_name", ""),
                    "normalized_name": candidate.get("normalized_name", ""),
                    "matched_livertox_row": None,
                    "extracted_excerpts": [],
                    "match_confidence": None,
                    "match_reason": "no_match",
                    "match_status": "missing",
                    "match_notes": ["No LiverTox match for candidate."],
                    "match_candidates": [],
                    "missing_livertox": True,
                    "ambiguous_match": False,
                    "origins": [],
                    "raw_mentions": [],
                    "extraction_metadata": [],
                }
                resolved_drugs[lookup_key] = payload
            payload.setdefault("origins", [])
            payload.setdefault("raw_mentions", [])
            payload.setdefault("extraction_metadata", [])
            for origin in candidate.get("origins", []):
                if origin not in payload["origins"]:
                    payload["origins"].append(origin)
            for raw_mention in candidate.get("raw_mentions", []):
                if raw_mention not in payload["raw_mentions"]:
                    payload["raw_mentions"].append(raw_mention)
            payload["extraction_metadata"] = self.compact_entry_metadata(
                payload.get("extraction_metadata", [])
                + candidate.get("extraction_metadata", [])
            )

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

