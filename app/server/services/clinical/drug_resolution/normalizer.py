from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from domain.clinical.entities import DrugEntry, PatientDrugs
from services.clinical.drug_identity import DrugIdentityResolver
from services.text.normalization import canonicalize_drug_query, normalize_drug_query_name


@dataclass(slots=True)
class NormalizedDrugMention:
    extracted_name: str
    canonical_name: str
    normalized_name: str
    source: Literal["therapy", "anamnesis", "unknown"]
    raw_mentions: list[str]
    origins: list[str]
    extraction_metadata: list[dict[str, Any]]
    regimen_group_id: str | None = None
    is_regimen_parent: bool = False
    regimen_components: list[str] = field(default_factory=list)


class DrugMentionNormalizer:
    """Normalize extracted drug entries and split regimen components once."""

    def normalize_entries(self, drugs: PatientDrugs) -> list[NormalizedDrugMention]:
        mentions: list[NormalizedDrugMention] = []
        seen: set[tuple[str, str | None, bool]] = set()
        for entry in drugs.entries:
            base = self._normalize_entry(entry)
            if base is None:
                continue
            for mention in self._expand_components(base):
                key = (
                    mention.normalized_name,
                    mention.regimen_group_id,
                    mention.is_regimen_parent,
                )
                if key in seen:
                    self._merge_mentions(mentions, mention, key)
                    continue
                seen.add(key)
                mentions.append(mention)
        return mentions

    def _normalize_entry(self, entry: DrugEntry) -> NormalizedDrugMention | None:
        raw_name = (entry.name or "").strip()
        if not raw_name:
            return None
        canonical_name = canonicalize_drug_query(raw_name) or raw_name.lower()
        normalized_name = normalize_drug_query_name(canonical_name or raw_name)
        if not normalized_name:
            return None
        source = entry.source if entry.source in {"therapy", "anamnesis"} else "unknown"
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
        return NormalizedDrugMention(
            extracted_name=raw_name,
            canonical_name=canonical_name,
            normalized_name=normalized_name,
            source=source,
            raw_mentions=[raw_name],
            origins=[source],
            extraction_metadata=[metadata] if metadata else [],
        )

    def _expand_components(
        self, mention: NormalizedDrugMention
    ) -> list[NormalizedDrugMention]:
        components = DrugIdentityResolver.split_components(mention.extracted_name)
        canonical_components = [
            canonical
            for component in components
            if (canonical := canonicalize_drug_query(component))
        ]
        canonical_components = list(dict.fromkeys(canonical_components))
        if len(canonical_components) <= 1:
            return [mention]
        regimen_group_id = "|".join(sorted(canonical_components))
        parent = replace(
            mention,
            regimen_group_id=regimen_group_id,
            is_regimen_parent=True,
            regimen_components=canonical_components[:],
        )
        expanded = [parent]
        for component in canonical_components:
            normalized_component = normalize_drug_query_name(component)
            if not normalized_component:
                continue
            expanded.append(
                NormalizedDrugMention(
                    extracted_name=mention.extracted_name,
                    canonical_name=component,
                    normalized_name=normalized_component,
                    source=mention.source,
                    raw_mentions=mention.raw_mentions[:],
                    origins=mention.origins[:],
                    extraction_metadata=mention.extraction_metadata[:],
                    regimen_group_id=regimen_group_id,
                    is_regimen_parent=False,
                    regimen_components=canonical_components[:],
                )
            )
        return expanded

    @staticmethod
    def _merge_mentions(
        mentions: list[NormalizedDrugMention],
        incoming: NormalizedDrugMention,
        key: tuple[str, str | None, bool],
    ) -> None:
        for mention in mentions:
            current_key = (
                mention.normalized_name,
                mention.regimen_group_id,
                mention.is_regimen_parent,
            )
            if current_key != key:
                continue
            mention.raw_mentions = list(dict.fromkeys(mention.raw_mentions + incoming.raw_mentions))
            mention.origins = list(dict.fromkeys(mention.origins + incoming.origins))
            mention.extraction_metadata.extend(incoming.extraction_metadata)
            return
