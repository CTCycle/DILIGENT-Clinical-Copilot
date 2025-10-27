from __future__ import annotations

import os
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import declarative_base, sessionmaker

from DILIGENT.app.constants import DATA_PATH
from DILIGENT.app.utils.singleton import singleton

Base = declarative_base()


###############################################################################
class Documents(Base):
    __tablename__ = "DOCUMENTS"
    id = Column(String, primary_key=True)
    text = Column(String)
    __table_args__ = (UniqueConstraint("id"),)


###############################################################################
class ClinicalSession(Base):
    __tablename__ = "CLINICAL_SESSIONS"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_name = Column(String)
    session_timestamp = Column(DateTime)
    alt_value = Column(String)
    alt_upper_limit = Column(String)
    alp_value = Column(String)
    alp_upper_limit = Column(String)
    hepatic_pattern = Column(String)
    anamnesis = Column(Text)
    drugs = Column(Text)
    parsing_model = Column(String)
    clinical_model = Column(String)
    total_duration = Column(Float)
    final_report = Column(Text)


###############################################################################
class LiverToxData(Base):
    __tablename__ = "LIVERTOX_DATA"
    drug_name = Column(String, primary_key=True)
    ingredient = Column(String, primary_key=True)
    brand_name = Column(String, primary_key=True)
    nbk_id = Column(String)
    excerpt = Column(Text)
    synonyms = Column(Text)
    likelihood_score = Column(String)
    last_update = Column(String)
    reference_count = Column(String)
    year_approved = Column(String)
    agent_classification = Column(String)
    primary_classification = Column(String)
    secondary_classification = Column(String)
    include_in_livertox = Column(String)
    source_url = Column(String)
    source_last_modified = Column(String)
    __table_args__ = (UniqueConstraint("drug_name", "ingredient", "brand_name"),)


###############################################################################
class DrugCatalog(Base):
    __tablename__ = "DRUG_CATALOG"
    rxcui = Column(String, primary_key=True)
    preferred_name = Column(String)
    concept_type = Column(String)
    synonyms = Column(Text)
    brands = Column(Text)
    rxcui_parents = Column(Text)
    pubchem_cid = Column(String)
    inchikey = Column(String)
    unii = Column(String)
    cas = Column(String)
    xrefs = Column(Text)
    status = Column(String)
    updated_at = Column(DateTime)


###############################################################################
class FdaAdverseEvent(Base):
    __tablename__ = "FDA_ADVERSE_EVENTS"
    report_id = Column(String, primary_key=True)
    case_version = Column(String, primary_key=True)
    receipt_date = Column(String)
    occur_country = Column(String)
    patient_age = Column(String)
    patient_age_unit = Column(String)
    patient_sex = Column(String)
    reaction_terms = Column(Text)
    all_reactions = Column(Text)
    suspect_products = Column(Text)
    suspect_product_count = Column(Integer)
    serious = Column(Integer)
    seriousness_death = Column(Integer)
    seriousness_lifethreatening = Column(Integer)
    seriousness_hospitalization = Column(Integer)
    seriousness_disabling = Column(Integer)
    seriousness_congenital_anom = Column(Integer)
    seriousness_other = Column(Integer)
    __table_args__ = (UniqueConstraint("report_id", "case_version"),)


# [DATABASE]
###############################################################################
@singleton
class DILIGENTDatabase:
    def __init__(self) -> None:
        self.db_path = os.path.join(DATA_PATH, "database.db")
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
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
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

    # -----------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        query = f'SELECT * FROM "{table_name}"'
        with database.engine.connect() as connection:
            return pd.read_sql_query(query, connection)

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    # -----------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(f'SELECT COUNT(*) FROM "{table_name}"')
            )
            value = result.scalar() or 0
        return int(value)


# -----------------------------------------------------------------------------
database = DILIGENTDatabase()
