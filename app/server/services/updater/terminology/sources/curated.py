# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Curated aliases from the drug_matching reference catalog.

``terminology_aliases`` entries map an alias to an existing concept by exact
canonical name and are loaded as confirmed mappings. ``error_prone_abbreviations``
are never resolved automatically: they are stored as rejected mappings so the
resolver reports their possible meanings as ambiguous candidates.
"""

from __future__ import annotations

import hashlib

from services.catalogs.runtime import get_reference_catalog_snapshot
from services.updater.terminology.snapshot import (
    ProgressFn,
    StagedNamedAlias,
    StopFn,
    TerminologySnapshot,
)

LICENSE = "Project-maintained curated list"
CATALOG_DOMAIN = "drug_matching"

###############################################################################
class CuratedAliasSource:
    key = "curated"
    license = LICENSE
    attribution = "DILIGENT curated aliases"
    precedence = 0

    # -------------------------------------------------------------------------
    def _named_aliases(self) -> list[StagedNamedAlias]:
        snapshot = get_reference_catalog_snapshot()
        aliases = [
            StagedNamedAlias(entry.value, "error_prone_abbreviation", ())
            for entry in snapshot.entries(CATALOG_DOMAIN, "error_prone_abbreviations")
            if entry.active and entry.value.strip()
        ]
        aliases.extend(
            StagedNamedAlias(
                entry.value,
                str(entry.metadata.get("alias_kind") or "synonym"),
                (entry.key,),
            )
            for entry in snapshot.entries(CATALOG_DOMAIN, "terminology_aliases")
            if entry.active and entry.value.strip() and entry.key.strip()
        )
        return aliases

    # -------------------------------------------------------------------------
    def probe_version(self) -> str | None:
        digest = hashlib.sha256()
        for item in self._named_aliases():
            digest.update(f"{item.alias}|{item.alias_kind}|{'/'.join(item.target_names)}\n".encode())
        return "sha256:" + digest.hexdigest()[:12]

    # -------------------------------------------------------------------------
    def fetch(self, progress: ProgressFn, should_stop: StopFn) -> TerminologySnapshot:
        del should_stop
        progress(50.0, "Staging curated aliases")
        snapshot = TerminologySnapshot(source=self.key, version=self.probe_version(), license=self.license)
        snapshot.named_aliases.extend(self._named_aliases())
        snapshot.stats.update(named_aliases=len(snapshot.named_aliases))
        return snapshot
