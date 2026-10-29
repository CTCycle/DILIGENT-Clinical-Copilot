# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Persistence for terminology ingestion: reconciliation, provenance, links."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.orm import Session

from configurations.startup import get_server_settings
from repositories.context import RepositoryContext
from repositories.database.upsert import dialect_insert
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugIdentifier,
    DrugRxnormCode,
    LiverToxMonograph,
)
from repositories.schemas.terminology import (
    DrugAtcCode,
    DrugConceptRelation,
    DrugLiverToxLink,
    MedicinalProduct,
    MedicinalProductIngredient,
    TerminologyConflict,
    TerminologyImportRun,
)
from services.terminology.normalization import fold
from services.updater.terminology.snapshot import (
    IDENTITY_SYSTEMS,
    StagedConcept,
    StagedProduct,
    TerminologySnapshot,
)

SQL_PARAMETER_BUDGET = 900
RESOLVABLE_STATUSES = ("verified", "confirmed")
# Lower value wins when sources disagree on what an alias denotes.
SOURCE_PRECEDENCE = {
    "curated": 0,
    "rxnorm": 1,
    "rxnav_runtime": 2,
    "drugcentral": 3,
    "chembl": 4,
    "swissmedic": 5,
    "livertox": 6,
}
BRAND_KINDS = frozenset({"brand", "trade_name"})
# LiverTox "ingredient" aliases name components of a monograph and its "brand"
# aliases are not composition-aware; neither establishes drug identity.
NON_IDENTITY_ALIASES = (("livertox", "ingredient"), ("livertox", "brand"))
PREFERRED_NAME_KINDS = frozenset({"canonical", "inn", "usan", "ban", "jan"})
PRODUCT_SOURCES = frozenset({"rxnorm", "swissmedic"})
TERMINOLOGY_LINK_SOURCES = ("curated", "rxnorm", "rxnav_runtime", "drugcentral", "chembl")
_COMBINATION_RE = re.compile(r"\s*(?:-|/|\+|\band\b)\s*", re.IGNORECASE)

# -----------------------------------------------------------------------------
def chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    safe = max(1, size)
    for start in range(0, len(values), safe):
        yield values[start : start + safe]


# -----------------------------------------------------------------------------
def identity_alias_filter() -> Any:
    return and_(
        *(
            ~((DrugAlias.source == source) & (DrugAlias.alias_kind == kind))
            for source, kind in NON_IDENTITY_ALIASES
        )
    )


###############################################################################
@dataclass
class PersistStats:
    concepts_matched: int = 0
    concepts_created: int = 0
    concepts_skipped: int = 0
    aliases: int = 0
    identifiers: int = 0
    atc_codes: int = 0
    relations: int = 0
    products: int = 0
    products_verified: int = 0
    products_unverified: int = 0
    brand_aliases: int = 0
    brand_family_aliases: int = 0
    conflicts: int = 0
    details: dict[str, int] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, int]:
        values = {
            key: value
            for key, value in self.__dict__.items()
            if key != "details" and isinstance(value, int)
        }
        values.update(self.details)
        return values


###############################################################################
class _ReconciliationState:
    """In-memory view of identity anchors, loaded once per source transaction."""

    # -------------------------------------------------------------------------
    def __init__(self, db_session: Session, source: str) -> None:
        self.rxcui_to_drug = {
            str(rxcui): int(drug_id)
            for rxcui, drug_id in db_session.execute(
                select(DrugRxnormCode.rxcui, DrugRxnormCode.drug_id)
            ).all()
        }
        # Anchors include the importing source's previous rows so concept
        # identity stays stable across refreshes; owners exclude them because
        # those rows are deleted and re-created by the refresh.
        self.identifier_anchor: dict[tuple[str, str], int] = {}
        self.identifier_owner: dict[tuple[str, str], int] = {}
        self.drug_identifiers: dict[int, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for system, value, drug_id, owner_source in db_session.execute(
            select(
                DrugIdentifier.identifier_system,
                DrugIdentifier.identifier_value,
                DrugIdentifier.drug_id,
                DrugIdentifier.source,
            )
        ).all():
            key = (str(system), str(value))
            self.identifier_anchor[key] = int(drug_id)
            if owner_source != source:
                self.identifier_owner[key] = int(drug_id)
            self.drug_identifiers[int(drug_id)][str(system)].add(str(value))
        self.drug_rxcuis: dict[int, set[str]] = defaultdict(set)
        for rxcui, drug_id in self.rxcui_to_drug.items():
            self.drug_rxcuis[drug_id].add(rxcui)
        self.name_to_drug: dict[str, int] = {}
        self.drug_types: dict[int, str] = {}
        for drug_id, name_norm, concept_type in db_session.execute(
            select(Drug.id, Drug.canonical_name_norm, Drug.concept_type)
        ).all():
            self.name_to_drug[str(name_norm)] = int(drug_id)
            self.drug_types[int(drug_id)] = str(concept_type)
        self.parents: dict[int, int] = {}
        self.components: dict[int, set[int]] = defaultdict(set)
        for drug_id, related_id, relation_type in db_session.execute(
            select(
                DrugConceptRelation.drug_id,
                DrugConceptRelation.related_drug_id,
                DrugConceptRelation.relation_type,
            )
        ).all():
            if relation_type == "form_of":
                self.parents[int(drug_id)] = int(related_id)
            else:
                self.components[int(drug_id)].add(int(related_id))
        self.multi_by_components: dict[frozenset[int], int] = {
            frozenset(members): drug_id
            for drug_id, members in self.components.items()
            if len(members) > 1
        }

    # -------------------------------------------------------------------------
    def moiety(self, drug_id: int) -> int:
        return self.parents.get(drug_id, drug_id)


###############################################################################
class TerminologyRepository:

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    @staticmethod
    def _batch_size(columns: int) -> int:
        configured = int(get_server_settings().database.insert_batch_size)
        return max(1, min(configured, SQL_PARAMETER_BUDGET // max(1, columns)))

    # -------------------------------------------------------------------------
    def start_run(self, *, source: str, mode: str, license_text: str) -> int:
        with self.session_factory() as db_session:
            run = TerminologyImportRun(
                source=source, mode=mode, status="running", license=license_text
            )
            db_session.add(run)
            db_session.commit()
            return int(run.id)

    # -------------------------------------------------------------------------
    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        version: str | None,
        stats: dict[str, Any] | None,
        error: str | None = None,
    ) -> None:
        with self.session_factory() as db_session:
            db_session.execute(
                update(TerminologyImportRun)
                .where(TerminologyImportRun.id == run_id)
                .values(
                    status=status,
                    source_version=version,
                    stats_json=stats,
                    error=error,
                    finished_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )
            db_session.commit()

    # -------------------------------------------------------------------------
    def last_completed_version(self, source: str) -> str | None:
        with self.session_factory() as db_session:
            return db_session.execute(
                select(TerminologyImportRun.source_version)
                .where(
                    TerminologyImportRun.source == source,
                    TerminologyImportRun.status == "completed",
                )
                .order_by(TerminologyImportRun.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def persist_snapshot(self, snapshot: TerminologySnapshot, run_id: int) -> PersistStats:
        """Replace one source's terminology in a single transaction."""
        stats = PersistStats()
        with self.session_factory() as db_session:
            try:
                state = _ReconciliationState(db_session, snapshot.source)
                self._delete_source_rows(db_session, snapshot.source)
                mapping = self._reconcile_concepts(db_session, snapshot, state, run_id, stats)
                self._write_concept_details(db_session, snapshot, state, mapping, run_id, stats)
                self._write_aliases(
                    db_session,
                    [
                        (mapping[alias.concept_key], alias)
                        for alias in snapshot.aliases
                        if alias.concept_key in mapping
                    ],
                    source=snapshot.source,
                    run_id=run_id,
                    status="confirmed" if snapshot.source == "curated" else "verified",
                    stats=stats,
                )
                if snapshot.named_aliases:
                    self._write_named_aliases(db_session, snapshot, run_id, stats)
                if snapshot.products:
                    self._write_products(db_session, snapshot, state, mapping, run_id, stats)
                stats.conflicts += self.reconcile_alias_conflicts(db_session)
                self.rebuild_livertox_links(db_session)
                db_session.commit()
            except Exception:
                db_session.rollback()
                raise
        return stats

    # -------------------------------------------------------------------------
    def _delete_source_rows(self, db_session: Session, source: str) -> None:
        alias_sources = [source, "derived"] if source == "rxnorm" else [source]
        db_session.execute(delete(DrugAlias).where(DrugAlias.source.in_(alias_sources)))
        db_session.execute(delete(DrugIdentifier).where(DrugIdentifier.source == source))
        db_session.execute(delete(DrugConceptRelation).where(DrugConceptRelation.source == source))
        db_session.execute(delete(DrugAtcCode).where(DrugAtcCode.source == source))
        product_ids = select(MedicinalProduct.id).where(MedicinalProduct.source == source)
        db_session.execute(
            delete(MedicinalProductIngredient).where(
                MedicinalProductIngredient.product_id.in_(product_ids)
            )
        )
        db_session.execute(delete(MedicinalProduct).where(MedicinalProduct.source == source))
        db_session.execute(
            delete(TerminologyConflict).where(
                TerminologyConflict.source == source,
                TerminologyConflict.status == "open",
            )
        )
        if source == "rxnorm":
            db_session.execute(delete(DrugRxnormCode))
            db_session.execute(
                update(Drug).where(Drug.rxnav_last_update.is_not(None)).values(rxnav_last_update=None)
            )

    # -------------------------------------------------------------------------
    def _conflict(
        self,
        db_session: Session,
        *,
        source: str,
        run_id: int,
        conflict_type: str,
        subject: str,
        existing: int | None,
        proposed: int | None,
        detail: str,
    ) -> None:
        db_session.add(
            TerminologyConflict(
                source=source,
                import_run_id=run_id,
                conflict_type=conflict_type,
                subject=subject[:500],
                existing_drug_id=existing,
                proposed_drug_id=proposed,
                detail=detail[:2000],
            )
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _concept_order(snapshot: TerminologySnapshot) -> list[StagedConcept]:
        def rank(concept: StagedConcept) -> int:
            if concept.concept_type == "multi_ingredient":
                return 2
            return 1 if concept.parent_key else 0

        return sorted(snapshot.concepts.values(), key=rank)

    # -------------------------------------------------------------------------
    def _reconcile_concepts(
        self,
        db_session: Session,
        snapshot: TerminologySnapshot,
        state: _ReconciliationState,
        run_id: int,
        stats: PersistStats,
    ) -> dict[str, int]:
        mapping: dict[str, int] = {}
        authoritative_types = snapshot.source == "rxnorm"
        for concept in self._concept_order(snapshot):
            drug_id = self._match_concept(db_session, snapshot, concept, state, mapping, run_id)
            if drug_id is False:
                stats.concepts_skipped += 1
                stats.conflicts += 1
                continue
            if drug_id is None:
                drug_id = self._create_concept(db_session, concept, state)
                stats.concepts_created += 1
            else:
                stats.concepts_matched += 1
                values: dict[str, Any] = {}
                if authoritative_types and state.drug_types.get(drug_id) != concept.concept_type:
                    values["concept_type"] = concept.concept_type
                    state.drug_types[drug_id] = concept.concept_type
                if concept.preferred_inn:
                    values["preferred_inn"] = func.coalesce(Drug.preferred_inn, concept.preferred_inn)
                if values:
                    db_session.execute(update(Drug).where(Drug.id == drug_id).values(**values))
            mapping[concept.key] = drug_id
            if concept.rxcui and snapshot.source == "rxnorm":
                state.rxcui_to_drug[concept.rxcui] = drug_id
        return mapping

    # -------------------------------------------------------------------------
    def _match_concept(
        self,
        db_session: Session,
        snapshot: TerminologySnapshot,
        concept: StagedConcept,
        state: _ReconciliationState,
        mapping: dict[str, int],
        run_id: int,
    ) -> int | None | bool:
        """Return the matching drug id, None to create, or False on conflict."""
        if concept.concept_type == "multi_ingredient" and concept.component_keys:
            members = [mapping.get(key) for key in concept.component_keys]
            if all(member is not None for member in members):
                existing = state.multi_by_components.get(
                    frozenset(int(member) for member in members if member is not None)
                )
                if existing is not None:
                    return existing
        anchors: set[int] = set()
        rxcui_hit = state.rxcui_to_drug.get(concept.rxcui) if concept.rxcui else None
        if rxcui_hit is not None:
            if (
                snapshot.source != "rxnorm"
                and concept.concept_type == "ingredient"
                and state.drug_types.get(rxcui_hit) == "precise_ingredient"
                and rxcui_hit in state.parents
            ):
                rxcui_hit = state.parents[rxcui_hit]
            anchors.add(rxcui_hit)
        for system in IDENTITY_SYSTEMS:
            value = concept.identifiers.get(system)
            hit = state.identifier_anchor.get((system, value)) if value else None
            if hit is not None:
                anchors.add(hit)
        if len(anchors) > 1:
            self._conflict(
                db_session,
                source=snapshot.source,
                run_id=run_id,
                conflict_type="identifier_mismatch",
                subject=concept.name,
                existing=rxcui_hit if rxcui_hit is not None else min(anchors),
                proposed=max(anchors),
                detail=f"Identifiers of {snapshot.source}:{concept.key} point to drugs {sorted(anchors)}",
            )
            return rxcui_hit if rxcui_hit is not None else False
        name_hit = state.name_to_drug.get(fold(concept.name))
        if anchors:
            anchor = next(iter(anchors))
            if (
                name_hit is not None
                and name_hit != anchor
                and state.moiety(name_hit) != state.moiety(anchor)
            ):
                # A cross-walk identifier points at a different, unrelated
                # concept than the exact name: trust the name, record why.
                self._conflict(
                    db_session,
                    source=snapshot.source,
                    run_id=run_id,
                    conflict_type="identifier_mismatch",
                    subject=concept.name,
                    existing=anchor,
                    proposed=name_hit,
                    detail=f"{snapshot.source}:{concept.key} identifiers point to another concept than its name",
                )
                return False if self._contradicts(concept, name_hit, state) else name_hit
            return anchor
        if name_hit is None:
            return None
        if self._contradicts(concept, name_hit, state):
            self._conflict(
                db_session,
                source=snapshot.source,
                run_id=run_id,
                conflict_type="name_identity_mismatch",
                subject=concept.name,
                existing=name_hit,
                proposed=None,
                detail=f"{snapshot.source}:{concept.key} shares a name but not identifiers",
            )
            return False
        return name_hit

    # -------------------------------------------------------------------------
    @staticmethod
    def _contradicts(concept: StagedConcept, drug_id: int, state: _ReconciliationState) -> bool:
        """True when the concept carries an identity identifier that differs
        from one already held by ``drug_id``. RxCUIs are not compared here:
        known RxCUIs are anchors, and unknown ones say nothing about identity."""
        existing = state.drug_identifiers.get(drug_id, {})
        for system in IDENTITY_SYSTEMS:
            value = concept.identifiers.get(system)
            if value and existing.get(system) and value not in existing[system]:
                return True
        return False

    # -------------------------------------------------------------------------
    @staticmethod
    def _create_concept(db_session: Session, concept: StagedConcept, state: _ReconciliationState) -> int:
        drug = Drug(
            canonical_name=concept.name,
            canonical_name_norm=fold(concept.name),
            concept_type=concept.concept_type,
            preferred_inn=concept.preferred_inn,
        )
        db_session.add(drug)
        db_session.flush()
        drug_id = int(drug.id)
        state.name_to_drug[drug.canonical_name_norm] = drug_id
        state.drug_types[drug_id] = concept.concept_type
        return drug_id

    # -------------------------------------------------------------------------
    def _write_concept_details(
        self,
        db_session: Session,
        snapshot: TerminologySnapshot,
        state: _ReconciliationState,
        mapping: dict[str, int],
        run_id: int,
        stats: PersistStats,
    ) -> None:
        source = snapshot.source
        rxcui_rows: list[dict[str, Any]] = []
        identifier_rows: list[dict[str, Any]] = []
        atc_rows: dict[tuple[int, str], dict[str, Any]] = {}
        relation_rows: dict[tuple[int, int, str], dict[str, Any]] = {}
        for key, concept in snapshot.concepts.items():
            drug_id = mapping.get(key)
            if drug_id is None:
                continue
            if source == "rxnorm" and concept.rxcui:
                rxcui_rows.append({"drug_id": drug_id, "rxcui": concept.rxcui})
            for system, value in concept.identifiers.items():
                owner = state.identifier_owner.get((system, value))
                if owner is not None and owner != drug_id:
                    self._conflict(
                        db_session,
                        source=source,
                        run_id=run_id,
                        conflict_type="identifier_mismatch",
                        subject=f"{system}:{value}",
                        existing=owner,
                        proposed=drug_id,
                        detail=f"{source}:{key} assigns an identifier owned by another concept",
                    )
                    stats.conflicts += 1
                    continue
                if owner is None:
                    state.identifier_owner[(system, value)] = drug_id
                    state.drug_identifiers[drug_id][system].add(value)
                    identifier_rows.append(
                        {
                            "drug_id": drug_id,
                            "identifier_system": system,
                            "identifier_value": value,
                            "source": source,
                            "import_run_id": run_id,
                        }
                    )
            for code in concept.atc_codes:
                atc_rows[(drug_id, code)] = {
                    "drug_id": drug_id,
                    "atc_code": code,
                    "source": source,
                    "import_run_id": run_id,
                }
            parent_id = mapping.get(concept.parent_key) if concept.parent_key else None
            if parent_id is not None and parent_id != drug_id:
                child, parent = self._form_of_edge(source, drug_id, parent_id, state)
                if child is not None and parent is not None:
                    relation_rows[(child, parent, "form_of")] = {
                        "drug_id": child,
                        "related_drug_id": parent,
                        "relation_type": "form_of",
                        "position": None,
                        "source": source,
                        "import_run_id": run_id,
                    }
                    state.parents[child] = parent
            component_ids = [mapping.get(item) for item in concept.component_keys]
            if concept.component_keys and all(item is not None for item in component_ids):
                for position, component_id in enumerate(component_ids):
                    if component_id is None:
                        continue
                    relation_rows[(drug_id, component_id, "has_component")] = {
                        "drug_id": drug_id,
                        "related_drug_id": component_id,
                        "relation_type": "has_component",
                        "position": position,
                        "source": source,
                        "import_run_id": run_id,
                    }
                members = frozenset(int(item) for item in component_ids if item is not None)
                state.components[drug_id] = set(members)
                state.multi_by_components.setdefault(members, drug_id)
        self._insert_ignore(db_session, DrugRxnormCode, rxcui_rows, ["rxcui"])
        self._insert_ignore(
            db_session, DrugIdentifier, identifier_rows, ["identifier_system", "identifier_value"]
        )
        self._insert_ignore(
            db_session, DrugAtcCode, list(atc_rows.values()), ["drug_id", "atc_code", "source"]
        )
        self._insert_ignore(
            db_session,
            DrugConceptRelation,
            list(relation_rows.values()),
            ["drug_id", "related_drug_id", "relation_type", "source"],
        )
        stats.identifiers += len(identifier_rows)
        stats.atc_codes += len(atc_rows)
        stats.relations += len(relation_rows)

    # -------------------------------------------------------------------------
    @staticmethod
    def _form_of_edge(
        source: str,
        child: int,
        parent: int,
        state: _ReconciliationState,
    ) -> tuple[int | None, int | None]:
        """Keep one salt/parent hierarchy rooted at the authoritative moiety.

        RxNorm defines the hierarchy. A lower-precedence source may not give
        an RxNorm ingredient, or a concept that already has a parent, another
        parent; its parent form is attached under the authoritative moiety.
        """
        if source == "rxnorm":
            return child, parent
        anchored = child in state.parents or (
            state.drug_types.get(child) == "ingredient" and child in state.drug_rxcuis
        )
        if not anchored:
            return child, parent
        root = state.parents.get(child, child)
        parent_free = parent not in state.parents and parent not in state.drug_rxcuis
        if parent != root and parent_free:
            return parent, root
        return None, None

    # -------------------------------------------------------------------------
    def _insert_ignore(
        self,
        db_session: Session,
        model: Any,
        rows: list[dict[str, Any]],
        conflict_columns: list[str],
    ) -> None:
        if not rows:
            return
        for batch in chunks(rows, self._batch_size(len(rows[0]))):
            statement = dialect_insert(db_session, model).values(batch)
            db_session.execute(statement.on_conflict_do_nothing(index_elements=conflict_columns))

    # -------------------------------------------------------------------------
    def _write_aliases(
        self,
        db_session: Session,
        items: Iterable[tuple[int, Any]],
        *,
        source: str,
        run_id: int,
        status: str,
        stats: PersistStats,
    ) -> None:
        rows: dict[tuple[int, str, str], dict[str, Any]] = {}
        for drug_id, alias in items:
            alias_norm = fold(alias.alias)
            if len(alias_norm) < 2:
                continue
            rows.setdefault(
                (drug_id, alias_norm, alias.alias_kind),
                {
                    "drug_id": drug_id,
                    "alias": alias.alias[:500],
                    "alias_norm": alias_norm,
                    "alias_kind": alias.alias_kind,
                    "source": source,
                    "term_type": alias.term_type,
                    "language": alias.language,
                    "country": alias.country,
                    "validation_status": status,
                    "import_run_id": run_id,
                    "source_record_id": alias.source_record_id,
                },
            )
        self._insert_ignore(
            db_session,
            DrugAlias,
            list(rows.values()),
            ["drug_id", "alias_norm", "alias_kind", "source"],
        )
        stats.aliases += len(rows)

    # -------------------------------------------------------------------------
    def _write_named_aliases(
        self,
        db_session: Session,
        snapshot: TerminologySnapshot,
        run_id: int,
        stats: PersistStats,
    ) -> None:
        db_session.flush()
        index = self.resolvable_alias_index(
            db_session,
            (fold(name) for item in snapshot.named_aliases for name in item.target_names),
        )
        resolved: list[tuple[int, Any]] = []
        blocked: list[tuple[int, Any]] = []
        blocked_items = [item for item in snapshot.named_aliases if item.alias_kind == "error_prone_abbreviation"]
        if blocked_items:
            blocked_index = self.resolvable_alias_index(db_session, (fold(item.alias) for item in blocked_items))
            for item in blocked_items:
                for drug_id in blocked_index.get(fold(item.alias), set()):
                    blocked.append((drug_id, item))
        unresolved = 0
        for item in snapshot.named_aliases:
            if item.alias_kind == "error_prone_abbreviation":
                continue
            target = self._unique_hit(index, [fold(name) for name in item.target_names])
            if target is None:
                unresolved += 1
                continue
            resolved.append((target, item))
        status = "confirmed" if snapshot.source == "curated" else "verified"
        self._write_aliases(
            db_session, resolved, source=snapshot.source, run_id=run_id, status=status, stats=stats
        )
        self._write_aliases(
            db_session, blocked, source=snapshot.source, run_id=run_id, status="rejected", stats=stats
        )
        stats.details["named_aliases_unresolved"] = unresolved
        stats.details["blocked_alias_mappings"] = len(blocked)

    # -------------------------------------------------------------------------
    @staticmethod
    def _unique_hit(index: dict[str, set[int]], names: list[str]) -> int | None:
        for name in names:
            hits = index.get(name)
            if hits:
                return next(iter(hits)) if len(hits) == 1 else None
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def resolvable_alias_index(db_session: Session, norms: Iterable[str]) -> dict[str, set[int]]:
        """Exact alias norm -> drug ids among verified or confirmed aliases."""
        wanted = sorted({norm for norm in norms if norm})
        index: dict[str, set[int]] = defaultdict(set)
        rejected: set[tuple[str, int]] = set()
        for batch in chunks(wanted, SQL_PARAMETER_BUDGET):
            for alias_norm, drug_id, status in db_session.execute(
                select(DrugAlias.alias_norm, DrugAlias.drug_id, DrugAlias.validation_status).where(
                    DrugAlias.alias_norm.in_(batch),
                    DrugAlias.validation_status.in_((*RESOLVABLE_STATUSES, "rejected")),
                    identity_alias_filter(),
                )
            ).all():
                if status == "rejected":
                    rejected.add((str(alias_norm), int(drug_id)))
                else:
                    index[str(alias_norm)].add(int(drug_id))
        for alias_norm, drug_id in rejected:
            index[alias_norm].discard(drug_id)
        return index

    # -------------------------------------------------------------------------
    def _write_products(
        self,
        db_session: Session,
        snapshot: TerminologySnapshot,
        state: _ReconciliationState,
        mapping: dict[str, int],
        run_id: int,
        stats: PersistStats,
    ) -> None:
        source = snapshot.source
        lookup_norms = {
            name for product in snapshot.products for item in product.ingredients for name in item.lookup_names
        }
        alias_index = self.resolvable_alias_index(db_session, lookup_norms)
        atc_by_drug: dict[int, set[str]] = defaultdict(set)
        for drug_id, code in db_session.execute(select(DrugAtcCode.drug_id, DrugAtcCode.atc_code)).all():
            atc_by_drug[int(drug_id)].add(str(code))
        names = {
            int(drug_id): str(name)
            for drug_id, name in db_session.execute(select(Drug.id, Drug.canonical_name)).all()
        }
        brand_targets: dict[str, set[int]] = defaultdict(set)
        brand_display: dict[str, str] = {}
        family_targets: dict[str, set[int | None]] = defaultdict(set)
        product_rows: list[dict[str, Any]] = []
        ingredient_rows: dict[str, list[dict[str, Any]]] = {}
        for product in snapshot.products:
            ingredient_ids = [
                self._resolve_ingredient(item, mapping, alias_index, state) for item in product.ingredients
            ]
            product_drug: int | None = None
            if all(item is not None for item in ingredient_ids):
                resolved = [int(item) for item in ingredient_ids if item is not None]
                if len(resolved) == 1 and self._atc_contradicts(product, resolved[0], atc_by_drug, state):
                    self._conflict(
                        db_session,
                        source=source,
                        run_id=run_id,
                        conflict_type="composition_mismatch",
                        subject=product.name,
                        existing=resolved[0],
                        proposed=None,
                        detail=f"ATC {product.atc_code} does not match the resolved ingredient",
                    )
                    stats.conflicts += 1
                else:
                    product_drug = self._product_concept(
                        db_session, resolved, state, names, source, run_id, stats
                    )
            status = "verified" if product_drug is not None else "unverified"
            brand_norm = fold(product.brand_name) if product.brand_name else None
            product_rows.append(
                {
                    "name": product.name[:1000],
                    "name_norm": fold(product.name),
                    "brand_name_norm": brand_norm,
                    "country": product.country,
                    "dose_form": product.dose_form,
                    "route": product.route,
                    "composition_status": status,
                    "drug_id": product_drug,
                    "source": source,
                    "source_product_id": product.product_id,
                    "import_run_id": run_id,
                }
            )
            ingredient_rows[product.product_id] = [
                {
                    "position": position,
                    "drug_id": ingredient_ids[position],
                    "ingredient_name": item.ingredient_name[:500],
                    "strength_value": item.strength_value,
                    "strength_unit": item.strength_unit,
                }
                for position, item in enumerate(product.ingredients)
            ]
            stats.products += 1
            if status == "verified":
                stats.products_verified += 1
            else:
                stats.products_unverified += 1
            if snapshot.brand_families and brand_norm and product.brand_name:
                brand_display.setdefault(brand_norm, product.brand_name)
                if product_drug is not None:
                    brand_targets[brand_norm].add(product_drug)
                family = brand_norm.split()[0]
                if len(family) >= 5:
                    brand_display.setdefault(family, product.brand_name.split()[0])
                    family_targets[family].add(
                        state.moiety(product_drug) if product_drug is not None else None
                    )
        self._insert_ignore(
            db_session, MedicinalProduct, product_rows, ["source", "source_product_id"]
        )
        product_ids = {
            str(source_product_id): int(product_id)
            for product_id, source_product_id in db_session.execute(
                select(MedicinalProduct.id, MedicinalProduct.source_product_id).where(
                    MedicinalProduct.source == source
                )
            ).all()
        }
        component_rows = [
            {**row, "product_id": product_ids[source_product_id]}
            for source_product_id, rows in ingredient_rows.items()
            if source_product_id in product_ids
            for row in rows
        ]
        self._insert_ignore(
            db_session, MedicinalProductIngredient, component_rows, ["product_id", "position"]
        )
        if not snapshot.brand_families:
            return
        country = snapshot.products[0].country if snapshot.products else None
        brand_aliases: list[tuple[int, Any]] = [
            (drug_id, _BrandAlias(brand_display[brand_norm], "brand", country=country))
            for brand_norm, targets in brand_targets.items()
            for drug_id in targets
        ]
        family_count = 0
        for family, targets in family_targets.items():
            if family in brand_display and family in brand_targets:
                continue
            if len(targets) != 1 or None in targets:
                continue
            drug_id = next(iter(targets))
            if drug_id is None:
                continue
            brand_aliases.append(
                (drug_id, _BrandAlias(brand_display[family], "brand", country=country, family=True))
            )
            family_count += 1
        before = stats.aliases
        self._write_aliases(
            db_session, brand_aliases, source=source, run_id=run_id, status="verified", stats=stats
        )
        stats.brand_aliases += stats.aliases - before
        stats.brand_family_aliases += family_count

    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_ingredient(
        item: Any,
        mapping: dict[str, int],
        alias_index: dict[str, set[int]],
        state: _ReconciliationState,
    ) -> int | None:
        if item.concept_key is not None:
            return mapping.get(item.concept_key)
        for name in item.lookup_names:
            hits = alias_index.get(name)
            if not hits:
                continue
            if len(hits) == 1:
                return next(iter(hits))
            moieties = {state.moiety(hit) for hit in hits}
            return next(iter(moieties)) if len(moieties) == 1 else None
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _atc_contradicts(
        product: StagedProduct,
        drug_id: int,
        atc_by_drug: dict[int, set[str]],
        state: _ReconciliationState,
    ) -> bool:
        if not product.atc_code:
            return False
        codes = atc_by_drug.get(drug_id, set()) | atc_by_drug.get(state.moiety(drug_id), set())
        return bool(codes) and product.atc_code not in codes

    # -------------------------------------------------------------------------
    def _product_concept(
        self,
        db_session: Session,
        ingredient_ids: list[int],
        state: _ReconciliationState,
        names: dict[int, str],
        source: str,
        run_id: int,
        stats: PersistStats,
    ) -> int:
        if len(ingredient_ids) == 1:
            return ingredient_ids[0]
        members = frozenset(state.moiety(drug_id) for drug_id in ingredient_ids)
        if len(members) == 1:
            return next(iter(members))
        existing = state.multi_by_components.get(members)
        if existing is not None:
            return existing
        name = " / ".join(sorted(names.get(member, str(member)) for member in members))
        name_hit = state.name_to_drug.get(fold(name))
        if name_hit is not None:
            drug_id = name_hit
        else:
            drug = Drug(canonical_name=name, canonical_name_norm=fold(name), concept_type="multi_ingredient")
            db_session.add(drug)
            db_session.flush()
            drug_id = int(drug.id)
            state.name_to_drug[drug.canonical_name_norm] = drug_id
            state.drug_types[drug_id] = "multi_ingredient"
            names[drug_id] = name
            stats.concepts_created += 1
        rows = [
            {
                "drug_id": drug_id,
                "related_drug_id": member,
                "relation_type": "has_component",
                "position": position,
                "source": source,
                "import_run_id": run_id,
            }
            for position, member in enumerate(sorted(members))
        ]
        self._insert_ignore(
            db_session, DrugConceptRelation, rows, ["drug_id", "related_drug_id", "relation_type", "source"]
        )
        state.components[drug_id] = set(members)
        state.multi_by_components[members] = drug_id
        stats.relations += len(rows)
        return drug_id

    # -------------------------------------------------------------------------
    def reconcile_alias_conflicts(self, db_session: Session) -> int:
        """Demote lower-precedence alias mappings that contradict a higher one.

        Brand names used by different countries for different products are a
        legitimate ambiguity and stay resolvable as candidates.
        """
        db_session.flush()
        ambiguous_norms = [
            str(row[0])
            for row in db_session.execute(
                select(DrugAlias.alias_norm)
                .where(DrugAlias.validation_status.in_(("verified", "confirmed")), identity_alias_filter())
                .group_by(DrugAlias.alias_norm)
                .having(func.count(func.distinct(DrugAlias.drug_id)) > 1)
            ).all()
        ]
        demote_ids: list[int] = []
        conflicts = 0
        moiety = self._moiety_map(db_session)
        reported: set[tuple[str, int]] = set()
        for batch in chunks(ambiguous_norms, SQL_PARAMETER_BUDGET):
            rows = db_session.execute(
                select(
                    DrugAlias.id,
                    DrugAlias.alias_norm,
                    DrugAlias.drug_id,
                    DrugAlias.source,
                    DrugAlias.alias_kind,
                    DrugAlias.country,
                    DrugAlias.validation_status,
                ).where(
                    DrugAlias.alias_norm.in_(batch),
                    DrugAlias.validation_status.in_(("verified", "confirmed")),
                    identity_alias_filter(),
                )
            ).all()
            grouped: dict[str, list[Any]] = defaultdict(list)
            for row in rows:
                grouped[str(row.alias_norm)].append(row)
            for alias_norm, entries in grouped.items():
                winners = self._winning_moieties(entries, moiety)
                losers = [entry for entry in entries if moiety.get(entry.drug_id, entry.drug_id) not in winners]
                if not losers or self._legitimate_brand_ambiguity(entries):
                    continue
                winner = min(winners)
                for entry in losers:
                    if entry.validation_status == "confirmed":
                        continue
                    demote_ids.append(int(entry.id))
                    if (alias_norm, int(entry.drug_id)) in reported:
                        continue
                    reported.add((alias_norm, int(entry.drug_id)))
                    db_session.add(
                        TerminologyConflict(
                            source=str(entry.source),
                            conflict_type="name_identity_mismatch",
                            subject=alias_norm,
                            existing_drug_id=winner,
                            proposed_drug_id=int(entry.drug_id),
                            detail=(
                                f"{entry.source} maps '{alias_norm}' to a different concept "
                                "than a higher-precedence source; kept as provisional"
                            ),
                        )
                    )
                    conflicts += 1
        for batch in chunks(demote_ids, SQL_PARAMETER_BUDGET):
            db_session.execute(
                update(DrugAlias).where(DrugAlias.id.in_(batch)).values(validation_status="provisional")
            )
        return conflicts

    # -------------------------------------------------------------------------
    @staticmethod
    def _moiety_map(db_session: Session) -> dict[int, int]:
        """Map salt/ester concepts to their parent so related forms agree."""
        return {
            int(drug_id): int(parent)
            for drug_id, parent in db_session.execute(
                select(DrugConceptRelation.drug_id, DrugConceptRelation.related_drug_id).where(
                    DrugConceptRelation.relation_type == "form_of"
                )
            ).all()
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _winning_moieties(entries: list[Any], moiety: dict[int, int]) -> set[int]:
        """Highest-precedence moieties; ties broken by independent source support."""
        best_rank: dict[int, tuple[int, int]] = {}
        support: dict[int, set[str]] = defaultdict(set)
        for entry in entries:
            key = moiety.get(int(entry.drug_id), int(entry.drug_id))
            rank = TerminologyRepository._alias_rank(entry)
            best_rank[key] = min(rank, best_rank.get(key, rank))
            support[key].add(str(entry.source))
        top = min(best_rank.values())
        winners = {key for key, rank in best_rank.items() if rank == top}
        if len(winners) > 1:
            most = max(len(support[key]) for key in winners)
            leaders = {key for key in winners if len(support[key]) == most}
            if len(leaders) == 1:
                return leaders
        return winners

    # -------------------------------------------------------------------------
    @staticmethod
    def _alias_rank(entry: Any) -> tuple[int, int]:
        """Rank an alias claim: confirmed, then preferred names and product
        brands (composition-backed), then plain synonyms; ties by source."""
        if entry.validation_status == "confirmed":
            return (-1, 0)
        source_rank = SOURCE_PRECEDENCE.get(str(entry.source), 9)
        if entry.alias_kind in PREFERRED_NAME_KINDS:
            return (0, source_rank)
        if entry.alias_kind in BRAND_KINDS and entry.source in PRODUCT_SOURCES:
            return (0, source_rank)
        return (1, source_rank)

    # -------------------------------------------------------------------------
    @staticmethod
    def _legitimate_brand_ambiguity(entries: list[Any]) -> bool:
        if not all(entry.alias_kind in BRAND_KINDS for entry in entries):
            return False
        countries_by_drug: dict[int, set[str | None]] = defaultdict(set)
        for entry in entries:
            countries_by_drug[int(entry.drug_id)].add(entry.country)
        if any(None in countries for countries in countries_by_drug.values()):
            return False
        seen: set[str | None] = set()
        for countries in countries_by_drug.values():
            if countries & seen:
                return False
            seen |= countries
        return True

    # -------------------------------------------------------------------------
    def rebuild_livertox_links(self, db_session: Session | None = None) -> int:
        if db_session is None:
            with self.session_factory() as owned:
                count = self.rebuild_livertox_links(owned)
                owned.commit()
                return count
        db_session.flush()
        db_session.execute(delete(DrugLiverToxLink))
        monographs = db_session.execute(
            select(LiverToxMonograph.id, LiverToxMonograph.drug_id, Drug.canonical_name, Drug.canonical_name_norm)
            .join(Drug, Drug.id == LiverToxMonograph.drug_id)
        ).all()
        links: dict[tuple[int, int], str] = {}
        for monograph_id, drug_id, _name, _norm in monographs:
            links[(int(drug_id), int(monograph_id))] = "monograph_owner"
        names = {str(row.canonical_name_norm) for row in monographs}
        parts_by_monograph: dict[int, list[str]] = {}
        for monograph_id, _drug_id, name, _norm in monographs:
            parts = [fold(part) for part in _COMBINATION_RE.split(str(name)) if fold(part)]
            if len(parts) > 1:
                parts_by_monograph[int(monograph_id)] = parts
                names.update(parts)
        index = self._terminology_alias_index(db_session, names)
        relations = db_session.execute(
            select(DrugConceptRelation.drug_id, DrugConceptRelation.related_drug_id, DrugConceptRelation.relation_type)
        ).all()
        parents = {int(a): int(b) for a, b, kind in relations if kind == "form_of"}
        components: dict[int, set[int]] = defaultdict(set)
        for drug_id, related, kind in relations:
            if kind == "has_component":
                components[int(drug_id)].add(int(related))
        multi_by_members = {frozenset(members): drug_id for drug_id, members in components.items() if len(members) > 1}
        for monograph_id, drug_id, _name, norm in monographs:
            targets = index.get(str(norm), set()) - {int(drug_id)}
            if len(targets) == 1:
                links.setdefault((next(iter(targets)), int(monograph_id)), "verified_alias")
            parts = parts_by_monograph.get(int(monograph_id))
            if parts:
                members: set[int] = set()
                for part in parts:
                    hits = index.get(part, set())
                    if len(hits) != 1:
                        members = set()
                        break
                    hit = next(iter(hits))
                    members.add(parents.get(hit, hit))
                multi = multi_by_members.get(frozenset(members)) if len(members) > 1 else None
                if multi is not None:
                    links.setdefault((multi, int(monograph_id)), "component_set")
        linked_drugs: dict[int, list[int]] = defaultdict(list)
        for (drug_id, monograph_id), _kind in links.items():
            linked_drugs[drug_id].append(monograph_id)
        for child, parent in parents.items():
            if child in linked_drugs:
                continue
            for monograph_id in linked_drugs.get(parent, []):
                links.setdefault((child, monograph_id), "form_of_parent")
        children: dict[int, list[int]] = defaultdict(list)
        for child, parent in parents.items():
            if child in linked_drugs:
                children[parent].append(child)
        for parent, linked_children in children.items():
            if parent in linked_drugs or len(linked_children) != 1:
                continue
            for monograph_id in linked_drugs[linked_children[0]]:
                links.setdefault((parent, monograph_id), "form_of_child")
        rows = [
            {"drug_id": drug_id, "monograph_id": monograph_id, "link_type": kind}
            for (drug_id, monograph_id), kind in links.items()
        ]
        self._insert_ignore(db_session, DrugLiverToxLink, rows, ["drug_id", "monograph_id"])
        return len(rows)

    # -------------------------------------------------------------------------
    @staticmethod
    def _terminology_alias_index(db_session: Session, norms: Iterable[str]) -> dict[str, set[int]]:
        wanted = sorted({norm for norm in norms if norm})
        index: dict[str, set[int]] = defaultdict(set)
        for batch in chunks(wanted, SQL_PARAMETER_BUDGET):
            for alias_norm, drug_id in db_session.execute(
                select(DrugAlias.alias_norm, DrugAlias.drug_id).where(
                    DrugAlias.alias_norm.in_(batch),
                    DrugAlias.validation_status.in_(RESOLVABLE_STATUSES),
                    DrugAlias.source.in_(TERMINOLOGY_LINK_SOURCES),
                )
            ).all():
                index[str(alias_norm)].add(int(drug_id))
        return index


###############################################################################
@dataclass
class _BrandAlias:
    alias: str
    alias_kind: str
    country: str | None = None
    family: bool = False
    term_type: str | None = None
    language: str | None = None

    # -------------------------------------------------------------------------
    @property
    def source_record_id(self) -> str | None:
        return "brand_family" if self.family else None
