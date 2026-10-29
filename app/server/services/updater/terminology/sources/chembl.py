# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""ChEMBL approved-drug nomenclature (CC BY-SA 3.0) via the public REST API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from services.updater.terminology.download import USER_AGENT, request_json
from services.updater.terminology.snapshot import (
    ProgressFn,
    StagedConcept,
    StopFn,
    TerminologySnapshot,
    TerminologySourceUnavailable,
)

API_ROOT = "https://www.ebi.ac.uk/chembl/api/data"
LICENSE = "CC BY-SA 3.0 (ChEMBL, EMBL-EBI)"
ATTRIBUTION = "ChEMBL (EMBL-EBI), CC BY-SA 3.0"
PAGE_SIZE = 1000
PAGE_DELAY_SECONDS = 0.5
FIELDS = "molecule_chembl_id,pref_name,molecule_synonyms,molecule_hierarchy,atc_classifications"
SYNONYM_KINDS = {
    "INN": "inn",
    "BAN": "ban",
    "USAN": "usan",
    "JAN": "jan",
    "TRADE_NAME": "trade_name",
    "RESEARCH_CODE": "research_code",
}

###############################################################################
class ChemblSource:
    key = "chembl"
    license = LICENSE
    attribution = ATTRIBUTION
    precedence = 3

    # -------------------------------------------------------------------------
    def __init__(self, *, max_phase: int = 4, timeout: float = 60.0) -> None:
        self.max_phase = max_phase
        self.timeout = timeout

    # -------------------------------------------------------------------------
    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    # -------------------------------------------------------------------------
    def probe_version(self) -> str | None:
        with self._client() as client:
            status = request_json(client, f"{API_ROOT}/status.json")
        return str(status.get("chembl_db_version") or "") or None

    # -------------------------------------------------------------------------
    def fetch(self, progress: ProgressFn, should_stop: StopFn) -> TerminologySnapshot:
        molecules: list[dict[str, Any]] = []
        with self._client() as client:
            status = request_json(client, f"{API_ROOT}/status.json")
            offset = 0
            total: int | None = None
            while total is None or offset < total:
                if should_stop():
                    raise TerminologySourceUnavailable("ChEMBL import cancelled")
                page = request_json(
                    client,
                    f"{API_ROOT}/molecule.json",
                    params={
                        "max_phase": self.max_phase,
                        "limit": PAGE_SIZE,
                        "offset": offset,
                        "only": FIELDS,
                    },
                )
                batch = page.get("molecules") or []
                total = int((page.get("page_meta") or {}).get("total_count") or 0)
                molecules.extend(batch)
                offset += PAGE_SIZE
                if not batch:
                    break
                progress(5.0 + 60.0 * min(1.0, offset / max(total, 1)), "Downloading ChEMBL molecules")
                time.sleep(PAGE_DELAY_SECONDS)
        if total and len(molecules) < total:
            raise TerminologySourceUnavailable(
                f"ChEMBL download incomplete ({len(molecules)} of {total})"
            )
        version = str(status.get("chembl_db_version") or "") or None
        return self.build_snapshot(molecules, version=version)

    # -------------------------------------------------------------------------
    def build_snapshot(
        self,
        molecules: list[dict[str, Any]],
        *,
        version: str | None,
    ) -> TerminologySnapshot:
        snapshot = TerminologySnapshot(source=self.key, version=version, license=self.license)
        known = {str(item.get("molecule_chembl_id")) for item in molecules}
        for item in molecules:
            chembl_id = str(item.get("molecule_chembl_id") or "").strip()
            name = str(item.get("pref_name") or "").strip()
            if not chembl_id or not name:
                continue
            hierarchy = item.get("molecule_hierarchy") or {}
            parent = str(hierarchy.get("parent_chembl_id") or chembl_id)
            synonyms = item.get("molecule_synonyms") or []
            inn = next(
                (
                    str(entry.get("molecule_synonym"))
                    for entry in synonyms
                    if entry.get("syn_type") == "INN" and entry.get("molecule_synonym")
                ),
                None,
            )
            snapshot.concepts[chembl_id] = StagedConcept(
                key=chembl_id,
                name=name.lower(),
                concept_type="ingredient" if parent == chembl_id else "precise_ingredient",
                identifiers={"chembl": chembl_id},
                preferred_inn=inn.lower() if inn else None,
                atc_codes=tuple(sorted({str(code) for code in item.get("atc_classifications") or []})),
                parent_key=parent if parent != chembl_id and parent in known else None,
            )
            snapshot.add_alias(chembl_id, name, "canonical", source_record_id=chembl_id)
            for entry in synonyms:
                kind = SYNONYM_KINDS.get(str(entry.get("syn_type")), "synonym")
                snapshot.add_alias(
                    chembl_id,
                    str(entry.get("molecule_synonym") or ""),
                    kind,
                    term_type=str(entry.get("syn_type") or "") or None,
                    source_record_id=chembl_id,
                )
        snapshot.stats.update(concepts=len(snapshot.concepts), aliases=len(snapshot.aliases))
        return snapshot
