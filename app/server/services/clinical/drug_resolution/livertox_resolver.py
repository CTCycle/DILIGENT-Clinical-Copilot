from __future__ import annotations

from domain.clinical.drug_resolution import LiverToxResolutionCandidate
from services.clinical.drug_identity import DrugIdentityResolver
from services.clinical.drug_resolution.normalizer import NormalizedDrugMention
from services.clinical.matches_core import LiverToxMatcher
from services.text.normalization import canonicalize_drug_query, normalize_drug_query_name


###############################################################################
class LiverToxCandidateResolver:

    # -------------------------------------------------------------------------
    def __init__(self, matcher: LiverToxMatcher) -> None:
        self.matcher = matcher
        self.identity_resolver = DrugIdentityResolver(matcher)

    # -------------------------------------------------------------------------
    def build_candidates(
        self,
        mention: NormalizedDrugMention,
        rxnav_names: list[str],
    ) -> list[LiverToxResolutionCandidate]:
        queries = self._candidate_queries(mention, rxnav_names)
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

    # -------------------------------------------------------------------------
    def _candidate_queries(
        self,
        mention: NormalizedDrugMention,
        rxnav_names: list[str],
    ) -> list[str]:
        queries: list[str] = []
        seed_values = [
            mention.extracted_name,
            mention.canonical_name,
            mention.normalized_name,
            *mention.raw_mentions,
            *rxnav_names,
        ]
        for value in seed_values:
            self._add_query_variants(queries, value)
            if len(self._exact_alias_matches(canonicalize_drug_query(value))) > 1:
                continue
            for identity in self.identity_resolver.resolve(value or ""):
                self._add_query_variants(queries, identity.canonical_candidate)
        return list(dict.fromkeys(query for query in queries if query))

    # -------------------------------------------------------------------------
    def _add_query_variants(self, queries: list[str], value: str | None) -> None:
        canonical = canonicalize_drug_query(value)
        if not canonical:
            return
        queries.append(canonical)
        for reduced in self._query_reductions(canonical):
            queries.append(reduced)

    # -------------------------------------------------------------------------
    @staticmethod
    def _query_reductions(value: str) -> list[str]:
        reductions: list[str] = []
        tokens = normalize_drug_query_name(value).split()
        while len(tokens) > 1:
            tokens = tokens[:-1]
            reduced = canonicalize_drug_query(" ".join(tokens))
            if reduced:
                reductions.append(reduced)
        return reductions

    # -------------------------------------------------------------------------
    def _exact_alias_matches(self, query: str) -> list:
        lookup = getattr(self.matcher, "lookup", None)
        if lookup is None or not hasattr(lookup, "match_alias_exact_all"):
            return []
        canonical = lookup.canonicalize_query(query)
        if not canonical:
            return []
        return lookup.match_alias_exact_all(canonical)

    # -------------------------------------------------------------------------
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
