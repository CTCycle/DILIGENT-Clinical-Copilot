from __future__ import annotations

from typing import Any

from common.utils.logger import logger
from domain.clinical.matching import LiverToxMatch, MonographRecord

# Sentinel used to distinguish cache hits from None-valued cache entries.
CACHE_MISS = object()

###############################################################################
class DrugMatcher:
    """Drug matching workflow — multi-stage pipeline, result creation, diagnostics."""

    # -------------------------------------------------------------------------
    def __init__(self, lookup: Any) -> None:
        self.lookup = lookup

    # -------------------------------------------------------------------------
    def match_drug_names(self, patient_drugs: list[str]) -> list[LiverToxMatch]:
        results: list[LiverToxMatch] = []
        for raw_name in patient_drugs:
            canonical_query = self.lookup.canonicalize_query(raw_name)
            normalized_query = self.lookup.normalize_name(canonical_query or raw_name)
            if not normalized_query:
                results.append(
                    self.lookup.create_missing_result(
                        raw_name=raw_name,
                        canonical_query=canonical_query,
                        normalized_query=normalized_query,
                        reason="invalid_query",
                        notes=["Unable to normalize query."],
                    )
                )
                continue

            cached = self.lookup.match_cache.get(normalized_query, CACHE_MISS)
            if cached is not CACHE_MISS:
                results.append(
                    self.lookup.clone_cached_match(cached, raw_name, canonical_query)
                )
                continue

            alias_entries = self.lookup.resolve_alias_candidates(
                raw_name,
                normalized_query,
                include_catalog=False,
            )
            match = self.lookup.match_query(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                alias_entries=alias_entries,
            )

            if match.status in {"missing", "ambiguous"}:
                catalog_alias_entries = self.lookup.resolve_alias_candidates(
                    raw_name,
                    normalized_query,
                    include_catalog=True,
                )
                if catalog_alias_entries:
                    retry = self.lookup.match_query(
                        raw_name=raw_name,
                        canonical_query=canonical_query,
                        normalized_query=normalized_query,
                        alias_entries=catalog_alias_entries,
                    )
                    if match.status == "missing" or retry.status == "matched":
                        match = retry

            self.lookup.match_cache.put(normalized_query, match)
            results.append(match)
            if match.status == "matched":
                logger.info(
                    "Matched '%s' to '%s' via %s (confidence=%s)",
                    raw_name,
                    match.matched_name,
                    match.reason,
                    f"{match.confidence:.2f}" if match.confidence is not None else "NA",
                )
            elif match.status == "ambiguous":
                logger.warning(
                    "Ambiguous match for '%s': %s",
                    raw_name,
                    ", ".join(match.candidate_names),
                )
            else:
                alias_count = len(alias_entries) + len(
                    self.lookup.resolve_alias_candidates(
                        raw_name, normalized_query, include_catalog=True
                    )
                )
                if alias_count > 0:
                    logger.error(
                        "Silent miss: '%s' has %d alias candidate(s) but no match resolved",
                        raw_name,
                        alias_count,
                    )
                else:
                    logger.warning("No LiverTox match for '%s'", raw_name)
        return results

    # -------------------------------------------------------------------------
    def match_query(
        self,
        *,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        alias_entries: list[tuple[str, bool]],
    ) -> LiverToxMatch:
        if not alias_entries:
            return self.lookup.create_missing_result(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                reason="no_alias_candidates",
                notes=["No alias candidates available."],
            )

        source_backed_aliases = self.lookup.resolve_source_backed_query_variants(
            normalized_query
        )
        local_aliases = [
            alias for alias, from_catalog in alias_entries if not from_catalog
        ]
        stage1_keys = self.lookup.build_unique_keys(
            [canonical_query, *source_backed_aliases, *local_aliases],
            self.lookup.canonicalize_query,
        )
        stage1 = self.lookup.resolve_stage_matches(
            stage1_keys, self.lookup.match_primary_all
        )
        stage1_result = self.lookup.finalize_stage_result(
            stage_name="exact_canonical",
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            stage_matches=stage1,
        )
        if stage1_result is not None:
            return stage1_result

        stage2_keys = self.lookup.build_unique_keys(
            [*source_backed_aliases, *(alias for alias, _ in alias_entries)],
            self.lookup.canonicalize_query,
        )
        stage2 = self.lookup.resolve_stage_matches(
            stage2_keys, self.lookup.match_alias_exact_all
        )
        stage2_result = self.lookup.finalize_stage_result(
            stage_name="exact_alias",
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            stage_matches=stage2,
        )
        if stage2_result is not None:
            return stage2_result

        stage3_keys = self.lookup.build_unique_keys(
            [
                normalized_query,
                *source_backed_aliases,
                *(alias for alias, _ in alias_entries),
            ],
            self.lookup.normalize_name,
        )
        stage3 = self.lookup.resolve_stage_matches(
            stage3_keys, self.lookup.match_normalized_all
        )
        stage3_result = self.lookup.finalize_stage_result(
            stage_name="normalized_exact",
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            stage_matches=stage3,
        )
        if stage3_result is not None:
            return stage3_result

        spelling = self.lookup.match_authoritative_spelling_candidates(normalized_query)
        if not spelling:
            alias_spelling: list[tuple[MonographRecord, float, list[str]]] = []
            for alias, _from_catalog in alias_entries:
                normalized_alias = self.lookup.normalize_name(alias)
                if not normalized_alias or normalized_alias == normalized_query:
                    continue
                alias_spelling.extend(
                    self.lookup.match_authoritative_spelling_candidates(
                        normalized_alias
                    )
                )
            spelling = self.lookup.dedupe_stage_matches(alias_spelling)
        if len(spelling) == 1:
            record, confidence, notes = spelling[0]
            return self.lookup.create_matched_result(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                record=record,
                confidence=confidence,
                reason="spelling_correction",
                notes=notes,
            )
        if len(spelling) > 1:
            return self.lookup.create_ambiguous_result(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                reason="ambiguous_spelling_correction",
                stage_matches=spelling,
            )

        return self.lookup.create_missing_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            reason="no_match",
            notes=["No exact, alias, normalized, or unique spelling-correction match."],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def clone_cached_match(
        match: LiverToxMatch,
        raw_name: str,
        canonical_query: str,
    ) -> LiverToxMatch:
        return LiverToxMatch(
            status=match.status,
            query_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=match.normalized_query,
            nbk_id=match.nbk_id,
            matched_name=match.matched_name,
            confidence=match.confidence,
            reason=match.reason,
            notes=list(match.notes),
            candidate_names=list(match.candidate_names),
            rejected_candidate_names=list(match.rejected_candidate_names),
            record=match.record,
        )

    # -------------------------------------------------------------------------
    def resolve_stage_matches(
        self,
        keys: list[str],
        resolver: Any,
    ) -> list[tuple[MonographRecord, float, list[str]]]:
        merged: dict[str, tuple[MonographRecord, float, list[str]]] = {}
        for key in keys:
            for record, confidence, notes in resolver(key):
                record_key = self.lookup.record_identity_key(record)
                existing = merged.get(record_key)
                if existing is None or confidence > existing[1]:
                    merged[record_key] = (
                        record,
                        confidence,
                        list(dict.fromkeys(notes)),
                    )
                    continue
                if existing is not None:
                    combined = list(dict.fromkeys(existing[2] + notes))
                    merged[record_key] = (existing[0], existing[1], combined)
        ordered = list(merged.values())
        ordered.sort(key=lambda item: self.lookup.result_sort_key(item[0], item[1]))
        return ordered

    # -------------------------------------------------------------------------
    def finalize_stage_result(
        self,
        *,
        stage_name: str,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
    ) -> LiverToxMatch | None:
        if not stage_matches:
            return None
        preferred_combo = self.lookup.preferred_combo_name(
            raw_name, canonical_query, normalized_query
        )
        ranked = self.lookup.rank_stage_matches(
            stage_matches=stage_matches,
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
        )
        if len(ranked) == 1:
            record, confidence, notes = ranked[0]
            return self.lookup.create_matched_result(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                record=record,
                confidence=confidence,
                reason=stage_name,
                notes=notes,
            )
        if self.lookup.has_strict_rank_winner(
            stage_matches=ranked,
            normalized_query=normalized_query,
            preferred_combo=preferred_combo,
        ):
            best_record, best_confidence, best_notes = ranked[0]
            rejected = [record.drug_name for record, _, _ in ranked[1:]]
            combined_notes = list(
                dict.fromkeys([*best_notes, "deterministic_disambiguation_applied"])
            )
            return self.lookup.create_matched_result(
                raw_name=raw_name,
                canonical_query=canonical_query,
                normalized_query=normalized_query,
                record=best_record,
                confidence=best_confidence,
                reason=f"{stage_name}_ranked",
                notes=combined_notes,
                rejected_candidate_names=rejected,
            )
        return self.lookup.create_ambiguous_result(
            raw_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            reason=f"ambiguous_{stage_name}",
            stage_matches=ranked,
        )

    # -------------------------------------------------------------------------
    def diagnose_missing_drug(self, drug_name: str) -> dict[str, Any]:
        normalized = self.lookup.normalize_name(drug_name)
        data = self.lookup.require_data()
        diagnosis = {
            "original_name": drug_name,
            "normalized_name": normalized,
            "in_primary_index": normalized in data.primary_index,
            "in_synonym_index": normalized in data.synonym_index,
            "in_catalog_index": normalized in self.lookup.catalog_global_index,
            "catalog_entries": [],
            "alias_candidates": [],
            "token_matches": [],
        }
        if normalized in self.lookup.catalog_global_index:
            entry, is_synonym, original = self.lookup.catalog_global_index[normalized]
            diagnosis["catalog_entries"].append(
                {
                    "is_synonym": is_synonym,
                    "original": original,
                    "base_name": entry.get("base_name"),
                }
            )
        alias_entries = self.lookup.resolve_alias_candidates(drug_name, normalized)
        diagnosis["alias_candidates"] = [
            {"alias": alias, "from_catalog": from_catalog}
            for alias, from_catalog in alias_entries[:10]
        ]
        for token in self.lookup.tokenize(normalized):
            if token in data.token_index:
                diagnosis["token_matches"].append(
                    {"token": token, "record_count": len(data.token_index[token])}
                )
        return diagnosis

    # -------------------------------------------------------------------------
    def create_matched_result(
        self,
        *,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str],
        rejected_candidate_names: list[str] | None = None,
    ) -> LiverToxMatch:
        normalized_confidence = round(
            min(max(confidence, self.lookup.MIN_CONFIDENCE), 1.0), 2
        )
        cleaned_notes = list(dict.fromkeys(note for note in notes if note))
        return LiverToxMatch(
            status="matched",
            query_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            nbk_id=record.nbk_id,
            matched_name=record.drug_name,
            confidence=normalized_confidence,
            reason=reason,
            notes=cleaned_notes,
            candidate_names=[record.drug_name],
            rejected_candidate_names=list(
                dict.fromkeys(rejected_candidate_names or [])
            ),
            record=record,
        )

    # -------------------------------------------------------------------------
    def create_missing_result(
        self,
        *,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        reason: str,
        notes: list[str],
    ) -> LiverToxMatch:
        return LiverToxMatch(
            status="missing",
            query_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            nbk_id=None,
            matched_name=None,
            confidence=None,
            reason=reason,
            notes=list(dict.fromkeys(note for note in notes if note)),
            candidate_names=[],
            rejected_candidate_names=[],
            record=None,
        )

    # -------------------------------------------------------------------------
    def create_ambiguous_result(
        self,
        *,
        raw_name: str,
        canonical_query: str,
        normalized_query: str,
        reason: str,
        stage_matches: list[tuple[MonographRecord, float, list[str]]],
    ) -> LiverToxMatch:
        candidate_names = sorted(
            dict.fromkeys(record.drug_name for record, _, _ in stage_matches),
            key=str.casefold,
        )
        notes: list[str] = []
        for _, _, entry_notes in stage_matches:
            notes.extend(entry_notes)
        return LiverToxMatch(
            status="ambiguous",
            query_name=raw_name,
            canonical_query=canonical_query,
            normalized_query=normalized_query,
            nbk_id=None,
            matched_name=None,
            confidence=None,
            reason=reason,
            notes=list(dict.fromkeys(note for note in notes if note)),
            candidate_names=candidate_names,
            rejected_candidate_names=[],
            record=None,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def result_sort_key(
        record: MonographRecord,
        confidence: float,
    ) -> tuple[float, str, str, str]:
        return (
            -float(confidence),
            record.drug_name.casefold(),
            record.monograph_key or "",
            record.stable_key,
        )

    # -------------------------------------------------------------------------
    def create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        return self.lookup.create_matched_result(
            raw_name=record.drug_name,
            canonical_query=self.lookup.canonicalize_query(record.drug_name),
            normalized_query=self.lookup.normalize_name(record.drug_name),
            record=record,
            confidence=confidence,
            reason=reason,
            notes=list(notes or []),
        )
