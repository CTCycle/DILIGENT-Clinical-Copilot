# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""DrugCentral active-ingredient terminology (CC BY-SA 4.0).

Read-only SQL against a DrugCentral PostgreSQL database: by default the
public instance published by the DrugCentral team, or a local restore of the
official dump. Structures are active moieties. ``struct2parent`` is not used:
it links prodrugs and metabolites to other moieties (for example droxidopa to
norepinephrine), which must never be treated as interchangeable salt forms.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg

from services.updater.terminology.snapshot import (
    ProgressFn,
    StagedConcept,
    StopFn,
    TerminologySnapshot,
    TerminologySourceUnavailable,
)

LICENSE = "CC BY-SA 4.0 (DrugCentral, University of New Mexico)"
ATTRIBUTION = "DrugCentral (drugcentral.org), CC BY-SA 4.0"
PUBLIC_HOST = "unmtid-dbs.net"
# Credentials published by DrugCentral for its public read-only instance.
PUBLIC_USER = "drugman"
PUBLIC_PASSWORD = "dosage"
IDENTIFIER_SYSTEMS = {"UNII": "unii", "ChEMBL_ID": "chembl", "INN_ID": "inn_id"}

###############################################################################
@dataclass(frozen=True)
class DrugCentralConnection:
    host: str = PUBLIC_HOST
    port: int = 5433
    database: str = "drugcentral"
    user: str = PUBLIC_USER
    connect_timeout: int = 20
    statement_timeout_ms: int = 120_000

###############################################################################
class DrugCentralSource:
    key = "drugcentral"
    license = LICENSE
    attribution = ATTRIBUTION
    precedence = 2

    # -------------------------------------------------------------------------
    def __init__(self, connection: DrugCentralConnection) -> None:
        self.connection = connection

    # -------------------------------------------------------------------------
    def _connect(self) -> psycopg.Connection[Any]:
        settings = self.connection
        password = PUBLIC_PASSWORD if settings.host == PUBLIC_HOST else None
        try:
            conn = psycopg.connect(
                host=settings.host,
                port=settings.port,
                dbname=settings.database,
                user=settings.user,
                password=password,
                connect_timeout=settings.connect_timeout,
                options=f"-c statement_timeout={settings.statement_timeout_ms}",
            )
        except psycopg.Error as exc:
            raise TerminologySourceUnavailable(f"DrugCentral is unreachable: {exc}") from exc
        conn.read_only = True
        return conn

    # -------------------------------------------------------------------------
    def probe_version(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT version, dtime FROM dbversion").fetchone()
        return f"{row[0]} ({row[1]:%Y-%m-%d})" if row else None

    # -------------------------------------------------------------------------
    def fetch(self, progress: ProgressFn, should_stop: StopFn) -> TerminologySnapshot:
        progress(5.0, "Querying DrugCentral")
        try:
            with self._connect() as conn:
                version_row = conn.execute("SELECT version, dtime FROM dbversion").fetchone()
                structures = conn.execute(
                    "SELECT id, name, inchikey FROM structures WHERE name IS NOT NULL"
                ).fetchall()
                synonyms = conn.execute(
                    "SELECT id, name FROM synonyms WHERE id IS NOT NULL AND name IS NOT NULL"
                ).fetchall()
                identifiers = conn.execute(
                    "SELECT struct_id, id_type, identifier FROM identifier "
                    "WHERE id_type IN ('UNII', 'ChEMBL_ID', 'INN_ID') "
                    "AND parent_match IS NOT TRUE"
                ).fetchall()
                rxnorm = conn.execute("SELECT struct_id, rxcui FROM struct_rxnorm").fetchall()
                atc = conn.execute("SELECT struct_id, atc_code FROM struct2atc").fetchall()
        except psycopg.Error as exc:
            raise TerminologySourceUnavailable(f"DrugCentral query failed: {exc}") from exc
        if should_stop():
            raise TerminologySourceUnavailable("DrugCentral import cancelled")
        progress(40.0, "Staging DrugCentral concepts")
        version = f"{version_row[0]} ({version_row[1]:%Y-%m-%d})" if version_row else None
        return self.build_snapshot(
            version=version,
            structures=structures,
            synonyms=synonyms,
            identifiers=identifiers,
            rxnorm=rxnorm,
            atc=atc,
        )

    # -------------------------------------------------------------------------
    def build_snapshot(
        self,
        *,
        version: str | None,
        structures: list[Any],
        synonyms: list[Any],
        identifiers: list[Any],
        rxnorm: list[Any],
        atc: list[Any],
    ) -> TerminologySnapshot:
        snapshot = TerminologySnapshot(source=self.key, version=version, license=self.license)
        rxcui_by_struct = {str(struct_id): str(rxcui) for struct_id, rxcui in rxnorm}
        atc_by_struct: dict[str, list[str]] = defaultdict(list)
        for struct_id, code in atc:
            if code:
                atc_by_struct[str(struct_id)].append(str(code).strip())
        ids_by_struct: dict[str, dict[str, str]] = defaultdict(dict)
        for struct_id, id_type, value in identifiers:
            system = IDENTIFIER_SYSTEMS.get(str(id_type))
            if system and value and system not in ids_by_struct[str(struct_id)]:
                ids_by_struct[str(struct_id)][system] = str(value).strip()
        for struct_id, name, inchikey in structures:
            key = str(struct_id)
            identifiers_for = dict(ids_by_struct.get(key, {}))
            identifiers_for["drugcentral"] = key
            if inchikey:
                identifiers_for["inchikey"] = str(inchikey).strip()
            snapshot.concepts[key] = StagedConcept(
                key=key,
                name=str(name).strip(),
                concept_type="ingredient",
                rxcui=rxcui_by_struct.get(key),
                identifiers=identifiers_for,
                preferred_inn=str(name).strip() if "inn_id" in identifiers_for else None,
                atc_codes=tuple(sorted(set(atc_by_struct.get(key, ())))),
            )
            snapshot.add_alias(
                key,
                str(name),
                "inn" if "inn_id" in identifiers_for else "canonical",
                source_record_id=key,
            )
        for struct_id, synonym in synonyms:
            snapshot.add_alias(str(struct_id), str(synonym), "synonym", source_record_id=str(struct_id))
        snapshot.stats.update(concepts=len(snapshot.concepts), aliases=len(snapshot.aliases))
        return snapshot
