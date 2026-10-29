# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from sqlalchemy import Select, select

from repositories.schemas.knowledge import (
    LiverToxMonograph,
)

###############################################################################
class DrugRepositoryQueries:

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

