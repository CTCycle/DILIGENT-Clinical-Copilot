from __future__ import annotations

from sqlalchemy import select

from DILIGENT.server.repositories.schemas.models import ModelSelection, RuntimeSetting


###############################################################################
class ModelConfigRepositoryQueries:
    # -------------------------------------------------------------------------
    @staticmethod
    def select_all():
        return select(ModelSelection)

    # -------------------------------------------------------------------------
    @staticmethod
    def select_runtime_settings():
        return select(RuntimeSetting)

