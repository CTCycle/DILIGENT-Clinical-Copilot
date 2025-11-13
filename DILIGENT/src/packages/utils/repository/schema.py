from __future__ import annotations

from sqlalchemy import (
    Column, 
    DateTime, 
    Float, Integer, 
    BigInteger, 
    String, 
    Text, 
    UniqueConstraint
)

from sqlalchemy.orm import declarative_base


Base = declarative_base()


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
    nbk_id = Column(String)
    excerpt = Column(Text)
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
    __table_args__ = (UniqueConstraint("drug_name"),)


###############################################################################
class DrugsCatalog(Base):
    __tablename__ = "DRUGS_CATALOG"
    rxcui = Column(String, primary_key=True)
    raw_name = Column(Text, primary_key=True)
    term_type = Column(String)
    name = Column(String)
    brand_names = Column(Text)
    synonyms = Column(Text)
    __table_args__ = (UniqueConstraint("rxcui", "raw_name"),)