from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import delete, func, or_, select

from common.utils.text_utils import normalize_drug_name
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.schemas.dilirank import DiliRankRecord
from repositories.schemas.knowledge import Drug, DrugAlias, DrugIdentifier

DILIRANK_IDENTIFIER_SYSTEM = "dilirank_ltkb"
DILIRANK_TRUSTED_ALIAS_SOURCES = ("livertox", "rxnorm")

###############################################################################
class DiliRankRepository:

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    def replace_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        with self.session_factory() as db_session:
            try:
                canonical_index: dict[str, int] = {
                    str(name): int(drug_id)
                    for name, drug_id in db_session.execute(
                        select(Drug.canonical_name_norm, Drug.id)
                    ).all()
                    if str(name or "").strip()
                }
                alias_index: dict[str, set[int]] = defaultdict(set)
                for alias_norm, drug_id in db_session.execute(
                    select(DrugAlias.alias_norm, DrugAlias.drug_id).where(
                        DrugAlias.source.in_(DILIRANK_TRUSTED_ALIAS_SOURCES)
                    )
                ).all():
                    normalized_alias = str(alias_norm or "").strip()
                    if normalized_alias:
                        alias_index[normalized_alias].add(int(drug_id))

                prepared: list[tuple[dict[str, Any], int | None]] = []
                unmatched: list[str] = []
                ambiguous: list[str] = []
                linked_count = 0
                for record in records:
                    compound_name = repository_values.normalize_string(
                        record.get("compound_name")
                    )
                    if compound_name is None:
                        continue
                    normalized = normalize_drug_name(compound_name)
                    drug_id: int | None = None
                    if normalized:
                        canonical_drug_id = canonical_index.get(normalized)
                        if canonical_drug_id is not None:
                            drug_id = canonical_drug_id
                        else:
                            alias_candidates = alias_index.get(normalized, set())
                            if len(alias_candidates) == 1:
                                drug_id = next(iter(alias_candidates))
                            elif len(alias_candidates) > 1:
                                ambiguous.append(compound_name)
                            else:
                                unmatched.append(compound_name)
                    else:
                        unmatched.append(compound_name)
                    if drug_id is not None:
                        linked_count += 1
                    prepared.append((record, drug_id))

                db_session.execute(delete(DiliRankRecord))
                db_session.execute(
                    delete(DrugIdentifier).where(
                        DrugIdentifier.identifier_system == DILIRANK_IDENTIFIER_SYSTEM
                    )
                )

                for record, drug_id in prepared:
                    ltkb_id = repository_values.normalize_string(record.get("ltkb_id"))
                    compound_name = repository_values.normalize_string(
                        record.get("compound_name")
                    )
                    dili_concern = repository_values.normalize_string(
                        record.get("dili_concern")
                    )
                    if ltkb_id is None or compound_name is None or dili_concern is None:
                        continue
                    db_session.add(
                        DiliRankRecord(
                            drug_id=drug_id,
                            ltkb_id=ltkb_id,
                            compound_name=compound_name,
                            compound_name_norm=normalize_drug_name(compound_name),
                            severity_class=repository_values.to_int(
                                record.get("severity_class")
                            ),
                            label_section=repository_values.normalize_string(
                                record.get("label_section")
                            ),
                            dili_concern=dili_concern,
                            comment=repository_values.normalize_string(
                                record.get("comment")
                            ),
                            source_url=repository_values.normalize_string(
                                record.get("source_url")
                            ),
                            source_last_modified=repository_values.normalize_string(
                                record.get("source_last_modified")
                            ),
                        )
                    )
                    if drug_id is not None:
                        db_session.add(
                            DrugIdentifier(
                                drug_id=drug_id,
                                identifier_system=DILIRANK_IDENTIFIER_SYSTEM,
                                identifier_value=ltkb_id,
                            )
                        )
                db_session.commit()
            except Exception:
                db_session.rollback()
                raise

        return {
            "source_records": len(records),
            "persisted_records": len(prepared),
            "linked_records": linked_count,
            "unmatched_records": len(unmatched),
            "ambiguous_records": len(ambiguous),
            "unmatched_compounds": sorted(set(unmatched))[:25],
            "ambiguous_compounds": sorted(set(ambiguous))[:25],
        }

    # -------------------------------------------------------------------------
    def list_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        query = select(DiliRankRecord, Drug).outerjoin(
            Drug, Drug.id == DiliRankRecord.drug_id
        )
        count_query = select(func.count()).select_from(DiliRankRecord)
        normalized_search = str(search or "").strip().casefold()
        if normalized_search:
            pattern = f"%{normalized_search.replace('%', r'\%').replace('_', r'\_')}%"
            condition = or_(
                func.lower(func.coalesce(Drug.canonical_name, "")).like(
                    pattern, escape="\\"
                ),
                func.lower(func.coalesce(DiliRankRecord.compound_name, "")).like(
                    pattern, escape="\\"
                ),
                func.lower(func.coalesce(DiliRankRecord.ltkb_id, "")).like(
                    pattern, escape="\\"
                ),
                func.lower(func.coalesce(DiliRankRecord.dili_concern, "")).like(
                    pattern, escape="\\"
                ),
            )
            query = query.where(condition)
            count_query = count_query.where(
                or_(
                    func.lower(func.coalesce(DiliRankRecord.compound_name, "")).like(
                        pattern, escape="\\"
                    ),
                    func.lower(func.coalesce(DiliRankRecord.ltkb_id, "")).like(
                        pattern, escape="\\"
                    ),
                    func.lower(func.coalesce(DiliRankRecord.dili_concern, "")).like(
                        pattern, escape="\\"
                    ),
                    DiliRankRecord.drug_id.in_(
                        select(Drug.id).where(
                            func.lower(func.coalesce(Drug.canonical_name, "")).like(
                                pattern, escape="\\"
                            )
                        )
                    ),
                )
            )
        with self.session_factory() as db_session:
            total = int(db_session.execute(count_query).scalar_one())
            rows = db_session.execute(
                query.order_by(
                    func.lower(DiliRankRecord.compound_name),
                    DiliRankRecord.ltkb_id,
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
        return [self._serialize(record, drug) for record, drug in rows], total

    # -------------------------------------------------------------------------
    def get_records_for_drug(self, drug_id: int) -> list[dict[str, Any]]:
        safe_drug_id = int(drug_id)
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(DiliRankRecord, Drug)
                .join(Drug, Drug.id == DiliRankRecord.drug_id)
                .where(DiliRankRecord.drug_id == safe_drug_id)
                .order_by(DiliRankRecord.ltkb_id.asc())
            ).all()
        return [self._serialize(record, drug) for record, drug in rows]

    # -------------------------------------------------------------------------
    def count_records(self) -> int:
        with self.session_factory() as db_session:
            return int(
                db_session.execute(
                    select(func.count()).select_from(DiliRankRecord)
                ).scalar_one()
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _serialize(record: DiliRankRecord, drug: Drug | None) -> dict[str, Any]:
        return {
            "drug_id": int(drug.id) if drug is not None else None,
            "drug_name": drug.canonical_name if drug is not None else None,
            "ltkb_id": record.ltkb_id,
            "compound_name": record.compound_name,
            "severity_class": record.severity_class,
            "label_section": record.label_section,
            "dili_concern": record.dili_concern,
            "comment": record.comment,
            "source_url": record.source_url,
            "source_last_modified": record.source_last_modified,
        }
