# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Mapping

from common.paths import ARCHIVES_PATH
from configurations.startup import get_server_settings
from services.updater.terminology.snapshot import TerminologySource
from services.updater.terminology.sources.chembl import ChemblSource
from services.updater.terminology.sources.curated import CuratedAliasSource
from services.updater.terminology.sources.drugcentral import (
    DrugCentralConnection,
    DrugCentralSource,
)
from services.updater.terminology.sources.rxnorm import RxNormPrescribableSource
from services.updater.terminology.sources.swissmedic import SwissmedicSource

TERMINOLOGY_SOURCE_LICENSES = {
    "rxnorm": (RxNormPrescribableSource.license, RxNormPrescribableSource.attribution),
    "drugcentral": (DrugCentralSource.license, DrugCentralSource.attribution),
    "chembl": (ChemblSource.license, ChemblSource.attribution),
    "swissmedic": (SwissmedicSource.license, SwissmedicSource.attribution),
    "curated": (CuratedAliasSource.license, CuratedAliasSource.attribution),
}


# -----------------------------------------------------------------------------
def _flag(overrides: Mapping[str, object], key: str, default: bool) -> bool:
    value = overrides.get(key)
    return value if isinstance(value, bool) else default


# -----------------------------------------------------------------------------
def enabled_terminology_sources() -> dict[str, bool]:
    runtime = get_server_settings().runtime
    return {
        "rxnorm": True,
        "drugcentral": runtime.drugcentral_enabled,
        "chembl": runtime.chembl_enabled,
        "swissmedic": runtime.swissmedic_enabled,
        "curated": True,
    }


# -----------------------------------------------------------------------------
def build_rxnorm_sources() -> list[TerminologySource]:
    timeout = float(get_server_settings().runtime.terminology_download_timeout)
    return [
        RxNormPrescribableSource(ARCHIVES_PATH, timeout=timeout),
        CuratedAliasSource(),
    ]


# -----------------------------------------------------------------------------
def build_international_sources(overrides: Mapping[str, object]) -> list[TerminologySource]:
    runtime = get_server_settings().runtime
    timeout = float(runtime.terminology_download_timeout)
    sources: list[TerminologySource] = []
    if _flag(overrides, "drugcentral_enabled", runtime.drugcentral_enabled):
        sources.append(
            DrugCentralSource(
                DrugCentralConnection(
                    host=runtime.drugcentral_host,
                    port=runtime.drugcentral_port,
                    database=runtime.drugcentral_database,
                    user=runtime.drugcentral_user,
                )
            )
        )
    if _flag(overrides, "chembl_enabled", runtime.chembl_enabled):
        sources.append(ChemblSource(max_phase=runtime.chembl_max_phase))
    if _flag(overrides, "swissmedic_enabled", runtime.swissmedic_enabled):
        sources.append(SwissmedicSource(ARCHIVES_PATH, timeout=timeout))
    sources.append(CuratedAliasSource())
    return sources
