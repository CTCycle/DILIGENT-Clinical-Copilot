from __future__ import annotations

from domain.clinical.drug_resolution import LiverToxResolutionCandidate
from services.clinical.drug_resolution.normalizer import NormalizedDrugMention
from services.clinical.matches_core import LiverToxMatcher


class LiverToxCandidateResolver:
    def __init__(self, matcher: LiverToxMatcher) -> None:
        self.matcher = matcher

    def build_candidates(
        self,
        mention: NormalizedDrugMention,
        rxnav_names: list[str],
    ) -> list[LiverToxResolutionCandidate]:
        queries = list(
            dict.fromkeys(
                [
                    mention.canonical_name,
                    mention.normalized_name,
                    *[name for name in rxnav_names if name],
                ]
            )
        )
        candidates: list[LiverToxResolutionCandidate] = []
        for query in queries:
            exact_alias_matches = self._exact_alias_matches(query)
            if len(exact_alias_matches) > 1:
                for record, confidence, _notes in exact_alias_matches:
                    candidates.append(
                        LiverToxResolutionCandidate(
                            nbk_id=record.nbk_id,
                            drug_name=record.drug_name,
                            normalized_name=record.normalized_name,
                            monograph_key=record.monograph_key or record.stable_key,
                            has_excerpt=bool(record.excerpt),
                            confidence=confidence,
                            reason="ambiguous_exact_livertox_alias",
                            rejected_reason="ambiguous_livertox_candidate",
                        )
                    )
                continue
            matches = self.matcher.match_drug_names([query])
            if not matches:
                continue
            match = matches[0]
            if match.status == "matched" and match.record is not None:
                candidates.append(
                    LiverToxResolutionCandidate(
                        nbk_id=match.nbk_id,
                        drug_name=match.record.drug_name,
                        normalized_name=match.record.normalized_name,
                        monograph_key=match.record.monograph_key or match.record.stable_key,
                        has_excerpt=bool(match.record.excerpt),
                        confidence=match.confidence,
                        reason=match.reason,
                    )
                )
            elif match.status == "ambiguous":
                for candidate_name in match.candidate_names:
                    candidates.append(
                        LiverToxResolutionCandidate(
                            nbk_id=None,
                            drug_name=candidate_name,
                            normalized_name=self.matcher.lookup.normalize_name(candidate_name),
                            monograph_key=None,
                            has_excerpt=False,
                            confidence=match.confidence,
                            reason=match.reason,
                            rejected_reason="ambiguous_livertox_candidate",
                        )
                    )
        return self._dedupe(candidates)

    def _exact_alias_matches(self, query: str) -> list:
        lookup = getattr(self.matcher, "lookup", None)
        if lookup is None or not hasattr(lookup, "match_alias_exact_all"):
            return []
        canonical = lookup.canonicalize_query(query)
        if not canonical:
            return []
        return lookup.match_alias_exact_all(canonical)

    @staticmethod
    def _dedupe(
        candidates: list[LiverToxResolutionCandidate],
    ) -> list[LiverToxResolutionCandidate]:
        by_key: dict[str, LiverToxResolutionCandidate] = {}
        for candidate in candidates:
            key = candidate.monograph_key or candidate.normalized_name
            existing = by_key.get(key)
            if existing is None or (candidate.confidence or 0) > (existing.confidence or 0):
                by_key[key] = candidate
        return sorted(
            by_key.values(),
            key=lambda item: (item.rejected_reason is not None, -(item.confidence or 0), item.drug_name),
        )
