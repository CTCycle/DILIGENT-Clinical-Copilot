# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from configurations.startup import get_server_settings
from repositories.terminology_lookup_repository import TerminologyLookupRepository
from services.terminology.external import RxNavRuntimeClient
from services.terminology.index import get_terminology_index_cache
from services.terminology.resolver import ResolverSettings, TerminologyResolver


# -----------------------------------------------------------------------------
def build_resolver_settings() -> ResolverSettings:
    matcher = get_server_settings().drugs_matcher
    return ResolverSettings(
        fuzzy_min_query_length=matcher.fuzzy_min_query_length,
        fuzzy_shortlist_size=matcher.fuzzy_shortlist_size,
        fuzzy_short_name_length=matcher.fuzzy_short_name_length,
        fuzzy_short_max_distance=matcher.fuzzy_short_max_distance,
        fuzzy_long_max_distance=matcher.fuzzy_long_max_distance,
        fuzzy_min_margin=matcher.fuzzy_min_margin,
        resolution_cache_limit=matcher.resolution_cache_limit,
        external_fallback_enabled=matcher.external_fallback_enabled,
        external_max_requests_per_name=matcher.external_max_requests_per_name,
        external_negative_cache_days=matcher.external_negative_cache_days,
    )


# -----------------------------------------------------------------------------
def build_terminology_resolver(
    lookup: TerminologyLookupRepository,
    *,
    allow_external: bool = True,
) -> TerminologyResolver:
    settings = build_resolver_settings()
    external = (
        RxNavRuntimeClient(timeout=float(get_server_settings().runtime.rxnav_request_timeout))
        if allow_external and settings.external_fallback_enabled
        else None
    )
    return TerminologyResolver(
        lookup,
        get_terminology_index_cache(),
        settings=settings,
        external=external,
    )
