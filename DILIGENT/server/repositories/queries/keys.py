from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update

from DILIGENT.server.repositories.schemas.models import AccessKey, ResearchAccessKey

AccessKeyTable = type[AccessKey] | type[ResearchAccessKey]


###############################################################################
class AccessKeyRepositoryQueries:
    # -------------------------------------------------------------------------
    @staticmethod
    def list_for_provider(table: AccessKeyTable, provider: str):
        return (
            select(table)
            .where(table.provider == provider)
            .order_by(table.is_active.desc(), table.created_at.desc(), table.id.desc())
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def deactivate_provider_keys(table: AccessKeyTable, provider: str, *, now: datetime):
        return (
            update(table)
            .where(table.provider == provider)
            .values(is_active=False, updated_at=now)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def active_for_provider(table: AccessKeyTable, provider: str):
        return (
            select(table)
            .where(
                table.provider == provider,
                table.is_active.is_(True),
            )
            .order_by(table.updated_at.desc(), table.id.desc())
        )

