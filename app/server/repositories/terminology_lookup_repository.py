# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Read-mostly terminology queries used by runtime drug resolution."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from repositories.context import RepositoryContext
from repositories.database.upsert import dialect_insert
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    KbMatchCache,
    LiverToxMonograph,
)
from repositories.schemas.terminology import (
    DrugConceptRelation,
    DrugLiverToxLink,
    MedicinalProduct,
    MedicinalProductIngredient,
    TerminologyImportRun,
)
from repositories.terminology_repository import (
    RESOLVABLE_STATUSES,
    SQL_PARAMETER_BUDGET,
    chunks,
    identity_alias_filter,
)

EXACT_ONLY_KINDS = ("abbreviation", "research_code")
FUZZY_MIN_ALIAS_LENGTH = 5
RUNTIME_SOURCE = "rxnav_runtime"

###############################################################################
@dataclass(frozen=True)
class AliasHit:
    alias_norm: str
    drug_id: int
    alias: str
    alias_kind: str
    source: str
    country: str | None
    validation_status: str
    source_version: str | None

###############################################################################
@dataclass
class ConceptInfo:
    drug_id: int
    name: str
    concept_type: str
    parent_id: int | None = None
    component_ids: list[int] = field(default_factory=list)

###############################################################################
@dataclass(frozen=True)
class LiverToxLinkInfo:
    drug_id: int
    monograph_id: int
    monograph_key: str
    nbk_id: str | None
    drug_name: str
    link_type: str

###############################################################################
@dataclass(frozen=True)
class UnverifiedProduct:
    name: str
    country: str | None
    ingredient_names: tuple[str, ...]

###############################################################################
class TerminologyLookupRepository:

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    def revision(self) -> int | None:
        with self.session_factory() as db_session:
            return db_session.execute(
                select(func.max(TerminologyImportRun.id)).where(
                    TerminologyImportRun.status == "completed"
                )
            ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def exact_hits(self, norms: Iterable[str]) -> list[AliasHit]:
        wanted = [norm for norm in dict.fromkeys(norms) if norm]
        if not wanted:
            return []
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(
                    DrugAlias.alias_norm,
                    DrugAlias.drug_id,
                    DrugAlias.alias,
                    DrugAlias.alias_kind,
                    DrugAlias.source,
                    DrugAlias.country,
                    DrugAlias.validation_status,
                    TerminologyImportRun.source_version,
                )
                .outerjoin(TerminologyImportRun, TerminologyImportRun.id == DrugAlias.import_run_id)
                .where(
                    DrugAlias.alias_norm.in_(wanted),
                    DrugAlias.validation_status.in_((*RESOLVABLE_STATUSES, "rejected")),
                    identity_alias_filter(),
                )
            ).all()
        rejected = {(str(row[0]), int(row[1])) for row in rows if row[6] == "rejected"}
        return [
            AliasHit(str(a), int(b), str(c), str(d), str(e), f, str(g), h)
            for a, b, c, d, e, f, g, h in rows
            if g != "rejected" and (str(a), int(b)) not in rejected
        ]

    # -------------------------------------------------------------------------
    def blocked_hits(self, norms: Iterable[str]) -> list[AliasHit]:
        """Concepts an error-prone abbreviation could denote (never auto-resolved)."""
        wanted = [norm for norm in dict.fromkeys(norms) if norm]
        if not wanted:
            return []
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(
                    DrugAlias.alias_norm,
                    DrugAlias.drug_id,
                    DrugAlias.alias,
                    DrugAlias.alias_kind,
                    DrugAlias.source,
                    DrugAlias.country,
                    DrugAlias.validation_status,
                ).where(
                    DrugAlias.alias_norm.in_(wanted),
                    DrugAlias.alias_kind == "error_prone_abbreviation",
                    DrugAlias.validation_status == "rejected",
                )
            ).all()
        return [AliasHit(str(a), int(b), str(c), str(d), str(e), f, str(g), None) for a, b, c, d, e, f, g in rows]

    # -------------------------------------------------------------------------
    def concepts(self, drug_ids: Iterable[int]) -> dict[int, ConceptInfo]:
        pending = {int(item) for item in drug_ids}
        result: dict[int, ConceptInfo] = {}
        with self.session_factory() as db_session:
            for _depth in range(2):
                missing = sorted(pending - set(result))
                if not missing:
                    break
                loaded = self._load_concepts(db_session, missing)
                result.update(loaded)
                pending = {
                    related
                    for info in loaded.values()
                    for related in [*info.component_ids, *([info.parent_id] if info.parent_id else [])]
                }
        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def _load_concepts(db_session: Session, drug_ids: list[int]) -> dict[int, ConceptInfo]:
        result: dict[int, ConceptInfo] = {}
        for batch in chunks(drug_ids, SQL_PARAMETER_BUDGET):
            for drug_id, name, concept_type in db_session.execute(
                select(Drug.id, Drug.canonical_name, Drug.concept_type).where(Drug.id.in_(batch))
            ).all():
                result[int(drug_id)] = ConceptInfo(int(drug_id), str(name), str(concept_type))
            relations = db_session.execute(
                select(
                    DrugConceptRelation.drug_id,
                    DrugConceptRelation.related_drug_id,
                    DrugConceptRelation.relation_type,
                    DrugConceptRelation.position,
                )
                .where(DrugConceptRelation.drug_id.in_(batch))
                .order_by(DrugConceptRelation.position)
            ).all()
            for drug_id, related, relation_type, _position in relations:
                info = result.get(int(drug_id))
                if info is None:
                    continue
                if relation_type == "form_of":
                    info.parent_id = info.parent_id or int(related)
                elif int(related) not in info.component_ids:
                    info.component_ids.append(int(related))
        return result

    # -------------------------------------------------------------------------
    def multi_concept_for(self, component_ids: Iterable[int]) -> int | None:
        members = sorted({int(item) for item in component_ids})
        if len(members) < 2:
            return None
        with self.session_factory() as db_session:
            candidates = db_session.execute(
                select(DrugConceptRelation.drug_id)
                .where(
                    DrugConceptRelation.relation_type == "has_component",
                    DrugConceptRelation.related_drug_id.in_(members),
                )
                .group_by(DrugConceptRelation.drug_id)
                .having(func.count(func.distinct(DrugConceptRelation.related_drug_id)) == len(members))
            ).scalars().all()
            for candidate in candidates:
                total = db_session.execute(
                    select(func.count(func.distinct(DrugConceptRelation.related_drug_id))).where(
                        DrugConceptRelation.drug_id == candidate,
                        DrugConceptRelation.relation_type == "has_component",
                    )
                ).scalar_one()
                if int(total) == len(members):
                    return int(candidate)
        return None

    # -------------------------------------------------------------------------
    def unverified_products(self, norms: Iterable[str]) -> list[UnverifiedProduct]:
        wanted = [norm for norm in dict.fromkeys(norms) if norm]
        if not wanted:
            return []
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(MedicinalProduct.id, MedicinalProduct.name, MedicinalProduct.country)
                .where(
                    MedicinalProduct.brand_name_norm.in_(wanted),
                    MedicinalProduct.composition_status == "unverified",
                )
                .limit(10)
            ).all()
            products: list[UnverifiedProduct] = []
            for product_id, name, country in rows:
                ingredients = db_session.execute(
                    select(MedicinalProductIngredient.ingredient_name)
                    .where(MedicinalProductIngredient.product_id == product_id)
                    .order_by(MedicinalProductIngredient.position)
                ).scalars().all()
                products.append(UnverifiedProduct(str(name), country, tuple(str(i) for i in ingredients)))
        return products

    # -------------------------------------------------------------------------
    def strength_supported(self, drug_ids: Iterable[int], value: float, unit: str) -> bool:
        ids = sorted({int(item) for item in drug_ids})
        if not ids:
            return False
        with self.session_factory() as db_session:
            found = db_session.execute(
                select(MedicinalProductIngredient.id)
                .where(
                    MedicinalProductIngredient.drug_id.in_(ids),
                    func.abs(MedicinalProductIngredient.strength_value - value) < 1e-6,
                    func.lower(MedicinalProductIngredient.strength_unit) == unit.lower(),
                )
                .limit(1)
            ).first()
        return found is not None

    # -------------------------------------------------------------------------
    def fuzzy_vocabulary(self) -> Iterator[str]:
        with self.session_factory() as db_session:
            result = db_session.execute(
                select(DrugAlias.alias_norm)
                .where(
                    DrugAlias.validation_status.in_(RESOLVABLE_STATUSES),
                    DrugAlias.alias_kind.not_in(EXACT_ONLY_KINDS),
                    func.length(DrugAlias.alias_norm) >= FUZZY_MIN_ALIAS_LENGTH,
                    identity_alias_filter(),
                )
                .distinct()
                .execution_options(yield_per=5000)
            )
            for (alias_norm,) in result:
                yield str(alias_norm)

    # -------------------------------------------------------------------------
    def livertox_links(self, drug_ids: Iterable[int]) -> dict[int, list[LiverToxLinkInfo]]:
        ids = sorted({int(item) for item in drug_ids})
        links: dict[int, list[LiverToxLinkInfo]] = defaultdict(list)
        if not ids:
            return links
        with self.session_factory() as db_session:
            for batch in chunks(ids, SQL_PARAMETER_BUDGET):
                rows = db_session.execute(
                    select(
                        DrugLiverToxLink.drug_id,
                        LiverToxMonograph.id,
                        LiverToxMonograph.monograph_key,
                        LiverToxMonograph.nbk_id,
                        Drug.canonical_name,
                        DrugLiverToxLink.link_type,
                    )
                    .join(LiverToxMonograph, LiverToxMonograph.id == DrugLiverToxLink.monograph_id)
                    .join(Drug, Drug.id == LiverToxMonograph.drug_id)
                    .where(DrugLiverToxLink.drug_id.in_(batch))
                    .order_by(DrugLiverToxLink.drug_id, LiverToxMonograph.id)
                ).all()
                for drug_id, monograph_id, key, nbk_id, name, link_type in rows:
                    links[int(drug_id)].append(
                        LiverToxLinkInfo(int(drug_id), int(monograph_id), str(key), nbk_id, str(name), str(link_type))
                    )
        return links

    # -------------------------------------------------------------------------
    def drugs_for_rxcuis(self, rxcuis: Iterable[str]) -> dict[str, int]:
        wanted = sorted({str(item) for item in rxcuis if item})
        if not wanted:
            return {}
        with self.session_factory() as db_session:
            return {
                str(rxcui): int(drug_id)
                for rxcui, drug_id in db_session.execute(
                    select(DrugRxnormCode.rxcui, DrugRxnormCode.drug_id).where(
                        DrugRxnormCode.rxcui.in_(wanted)
                    )
                ).all()
            }

    # -------------------------------------------------------------------------
    def rxcuis_for(self, drug_ids: Iterable[int]) -> dict[int, str]:
        ids = sorted({int(item) for item in drug_ids})
        if not ids:
            return {}
        with self.session_factory() as db_session:
            result: dict[int, str] = {}
            for drug_id, rxcui in db_session.execute(
                select(DrugRxnormCode.drug_id, DrugRxnormCode.rxcui)
                .where(DrugRxnormCode.drug_id.in_(ids))
                .order_by(DrugRxnormCode.id)
            ).all():
                result.setdefault(int(drug_id), str(rxcui))
            return result

    # -------------------------------------------------------------------------
    def record_alias(
        self,
        *,
        drug_id: int,
        alias: str,
        alias_norm: str,
        alias_kind: str,
        source: str,
        validation_status: str,
        source_record_id: str | None = None,
    ) -> None:
        """Insert-or-update one alias; safe under concurrent identical requests."""
        with self.session_factory() as db_session:
            statement = dialect_insert(db_session, DrugAlias).values(
                drug_id=drug_id,
                alias=alias[:500],
                alias_norm=alias_norm,
                alias_kind=alias_kind,
                source=source,
                validation_status=validation_status,
                source_record_id=source_record_id,
            )
            statement = statement.on_conflict_do_update(
                index_elements=["drug_id", "alias_norm", "alias_kind", "source"],
                set_={"validation_status": statement.excluded.validation_status},
            )
            db_session.execute(statement)
            db_session.commit()

    # -------------------------------------------------------------------------
    def cached_outcome(self, key: str, revision: int | None) -> dict[str, Any] | None:
        now = datetime.now(UTC).replace(tzinfo=None)
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(KbMatchCache.resolution_status, KbMatchCache.evidence_json).where(
                    KbMatchCache.normalized_drug_key == key,
                    KbMatchCache.source == "terminology",
                    KbMatchCache.invalidated_at.is_(None),
                    or_(KbMatchCache.expires_at.is_(None), KbMatchCache.expires_at > now),
                    or_(
                        KbMatchCache.terminology_revision.is_(None),
                        and_(
                            KbMatchCache.terminology_revision.is_not(None),
                            KbMatchCache.terminology_revision == revision,
                        ),
                    ),
                )
            ).first()
        if row is None:
            return None
        try:
            evidence = json.loads(row[1]) if row[1] else {}
        except ValueError:
            evidence = {}
        return {"resolution_status": row[0], **evidence}

    # -------------------------------------------------------------------------
    def store_outcome(
        self,
        *,
        key: str,
        raw_name: str,
        status: str,
        method: str,
        revision: int | None,
        ttl_days: int,
        evidence: dict[str, Any],
    ) -> None:
        expires = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=max(1, ttl_days))
        values = {
            "raw_drug_name": raw_name[:500],
            "raw_drug_name_norm": key,
            "normalized_drug_key": key,
            "source": "terminology",
            "confidence": 0.0,
            "resolution_status": status,
            "method": method,
            "terminology_revision": revision,
            "expires_at": expires,
            "evidence_json": json.dumps(evidence, default=str),
            "invalidated_at": None,
            "invalidation_reason": None,
        }
        with self.session_factory() as db_session:
            statement = dialect_insert(db_session, KbMatchCache).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=["normalized_drug_key", "source"],
                set_={key_: statement.excluded[key_] for key_ in values if key_ not in {"source", "normalized_drug_key"}},
            )
            db_session.execute(statement)
            db_session.commit()
