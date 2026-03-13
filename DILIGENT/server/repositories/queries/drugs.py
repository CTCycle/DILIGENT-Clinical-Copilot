from __future__ import annotations

from sqlalchemy import select

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
    def by_livertox_nbk_id(livertox_nbk_id: str):
        return select(Drug).where(Drug.livertox_nbk_id == livertox_nbk_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_rxcui_mapping(rxcui: str):
        return select(DrugRxnormCode).where(DrugRxnormCode.rxcui == rxcui)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_by_joined_rxcui(rxcui: str):
        return select(Drug).join(DrugRxnormCode).where(DrugRxnormCode.rxcui == rxcui)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_by_rxnorm_rxcui(rxcui: str):
        return select(Drug).where(Drug.rxnorm_rxcui == rxcui)

    # -------------------------------------------------------------------------
    @staticmethod
    def drug_by_canonical_name_norm(canonical_name_norm: str):
        return select(Drug).where(Drug.canonical_name_norm == canonical_name_norm)

    # -------------------------------------------------------------------------
    @staticmethod
    def alias_by_norm(alias_norm: str):
        return select(DrugAlias).where(DrugAlias.alias_norm == alias_norm)

    # -------------------------------------------------------------------------
    @staticmethod
    def monograph_by_drug_id(drug_id: int):
        return select(LiverToxMonograph).where(LiverToxMonograph.drug_id == drug_id)

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

