from __future__ import annotations

from sqlalchemy import Select, select

from DILIGENT.server.repositories.schemas.models import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)


###############################################################################
class DrugRepositoryQueries:

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_rxcui_mapping(rxcui: str) -> Select[tuple[DrugRxnormCode]]:
        return select(DrugRxnormCode).where(DrugRxnormCode.rxcui == rxcui)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_by_joined_rxcui(rxcui: str) -> Select[tuple[Drug]]:
        return select(Drug).join(DrugRxnormCode).where(DrugRxnormCode.rxcui == rxcui)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_by_rxnorm_rxcui(rxcui: str) -> Select[tuple[Drug]]:
        return select(Drug).where(Drug.rxnorm_rxcui == rxcui)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_by_canonical_name_norm(canonical_name_norm: str) -> Select[tuple[Drug]]:
        return select(Drug).where(Drug.canonical_name_norm == canonical_name_norm)

    # -------------------------------------------------------------------------
    @staticmethod
    def alias_by_norm(alias_norm: str) -> Select[tuple[DrugAlias]]:
        return select(DrugAlias).where(DrugAlias.alias_norm == alias_norm)

    # -------------------------------------------------------------------------
    @staticmethod
    def monograph_by_drug_id(drug_id: int) -> Select[tuple[LiverToxMonograph]]:
        return select(LiverToxMonograph).where(LiverToxMonograph.drug_id == drug_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def monograph_by_key(monograph_key: str) -> Select[tuple[LiverToxMonograph]]:
        return select(LiverToxMonograph).where(
            LiverToxMonograph.monograph_key == monograph_key
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def alias_for_drug(
        *,
        drug_id: int,
        alias_norm: str,
        alias_kind: str,
        source: str,
    ):
        return select(DrugAlias).where(
            DrugAlias.drug_id == drug_id,
            DrugAlias.alias_norm == alias_norm,
            DrugAlias.alias_kind == alias_kind,
            DrugAlias.source == source,
        )

