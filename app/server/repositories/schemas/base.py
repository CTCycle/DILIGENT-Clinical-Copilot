from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, event
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


###############################################################################
class UtcDateTime(TypeDecorator[datetime]):
    """Store UTC-normalized datetimes while returning aware Python values."""

    impl = DateTime
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(
        self, value: datetime | None, _dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    # -------------------------------------------------------------------------
    def process_result_value(
        self, value: datetime | None, _dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


###############################################################################
class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


###############################################################################
@event.listens_for(Base, "before_update", propagate=True)
def set_updated_at_before_update(_mapper: Any, _connection: Any, target: Any) -> None:
    """Keep update timestamps portable across SQLite and PostgreSQL."""
    if hasattr(target, "updated_at"):
        target.updated_at = datetime.now(UTC).replace(tzinfo=None)


JsonPayload = JSON
