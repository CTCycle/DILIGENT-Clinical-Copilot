from __future__ import annotations

from sqlalchemy import Select, select

from repositories.schemas.models import ModelSelection, RuntimeSetting


###############################################################################
class ModelConfigRepositoryQueries:
    # -------------------------------------------------------------------------
    @staticmethod
    def select_all() -> Select[tuple[ModelSelection]]:
        return select(ModelSelection)

    # -------------------------------------------------------------------------
    @staticmethod
    def select_runtime_settings() -> Select[tuple[RuntimeSetting]]:
        return select(RuntimeSetting)
