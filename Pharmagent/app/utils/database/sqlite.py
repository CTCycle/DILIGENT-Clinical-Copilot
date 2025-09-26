from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sqlalchemy import Column, Float, String, Text, UniqueConstraint, create_engine
import sqlalchemy
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import declarative_base, sessionmaker

from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.utils.singleton import singleton

Base = declarative_base()


###############################################################################
class Documents(Base):
    __tablename__ = "DOCUMENTS"
    id = Column(String, primary_key=True)
    text = Column(String)
    __table_args__ = (UniqueConstraint("id"),)


###############################################################################
class Patients(Base):
    __tablename__ = "PATIENTS"
    name = Column(String, primary_key=True)
    anamnesis = Column(String)
    symptoms = Column(String)
    ALT = Column(Float)
    ALT_max = Column(Float)
    ALP = Column(Float)
    ALP_max = Column(Float)
    additional_tests = Column(String)
    drugs = Column(String)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class LiverToxMonographs(Base):
    __tablename__ = "LIVERTOX_MONOGRAPHS"
    nbk_id = Column(String, primary_key=True)
    drug_name = Column(String, primary_key=True)
    excerpt = Column(Text)
    additional_names = Column(Text)
    synonyms = Column(Text)
    __table_args__ = (UniqueConstraint("nbk_id", "drug_name"),)


# [DATABASE]
###############################################################################
@singleton
class PharmagentDatabase:
    def __init__(self) -> None:
        self.db_path = os.path.join(DATA_PATH, "Pharmagent_database.db")
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 1000

    # -------------------------------------------------------------------------
    def initialize_database(self) -> None:
        Base.metadata.create_all(self.engine)

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if hasattr(cls, "__tablename__") and cls.__tablename__ == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def _upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
        table = table_cls.__table__
        session = self.Session()
        try:
            unique_cols = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = uc.columns.keys()
                    break
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")

            # Batch insertions for speed
            records = df.to_dict(orient="records")
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {
                    c: getattr(stmt.excluded, c)  # type: ignore
                    for c in batch[0]
                    if c not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self._upsert_dataframe(df, table_cls)

    # -----------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(f'SELECT COUNT(*) FROM "{table_name}"')
            )
            value = result.scalar() or 0
        return int(value)


# -----------------------------------------------------------------------------
database = PharmagentDatabase()
