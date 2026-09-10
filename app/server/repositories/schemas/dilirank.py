from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from repositories.schemas.base import Base

DRUGS_ID_FK = "drugs.id"

###############################################################################
class DiliRankRecord(Base):
    __tablename__ = "dilirank_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(DRUGS_ID_FK, ondelete="SET NULL"),
        nullable=True,
    )
    ltkb_id: Mapped[str] = mapped_column(String, nullable=False)
    compound_name: Mapped[str] = mapped_column(Text, nullable=False)
    compound_name_norm: Mapped[str] = mapped_column(String, nullable=False)
    severity_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label_section: Mapped[str | None] = mapped_column(Text, nullable=True)
    dili_concern: Mapped[str] = mapped_column(String, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_last_modified: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("ltkb_id", name="uq_dilirank_records_ltkb_id"),
        Index("ix_dilirank_records_drug_id", "drug_id"),
        Index("ix_dilirank_records_compound_name_norm", "compound_name_norm"),
        Index("ix_dilirank_records_dili_concern", "dili_concern"),
    )
