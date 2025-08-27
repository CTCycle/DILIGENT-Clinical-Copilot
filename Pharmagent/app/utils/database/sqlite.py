import os
import pandas as pd

from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert

from Pharmagent.app.utils.singleton import singleton
from Pharmagent.app.constants import DATA_PATH

Base = declarative_base()


        
###############################################################################
class Documents(Base):
    __tablename__ = 'DOCUMENTS'
    id = Column(String, primary_key=True)
    text = Column(String)    
    __table_args__ = (
        UniqueConstraint('id'),
    )

###############################################################################
class Patients(Base):
    __tablename__ = 'PATIENTS'
    name = Column(String, primary_key=True)
    anamnesis = Column(String)
    blood_tests = Column(String)
    additional_tests = Column(String)
    drugs = Column(String)
    __table_args__ = (
        UniqueConstraint('name'),
    )

    

# [DATABASE]
###############################################################################
@singleton
class PharmagentDatabase:

    def __init__(self): 
        self.db_path = os.path.join(DATA_PATH, 'Pharmagent_database.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 5000
              
    #-------------------------------------------------------------------------       
    def initialize_database(self):
        Base.metadata.create_all(self.engine)     

    #-------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls):
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
            records = df.to_dict(orient='records')
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i:i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {c: getattr(stmt.excluded, c) for c in batch[0] if c not in unique_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols,
                    set_=update_cols
                )
                session.execute(stmt)
                session.commit()
            session.commit()
        finally:
            session.close()
   
    #-------------------------------------------------------------------------
    def save_documents(self, documents: pd.DataFrame):
        self.upsert_dataframe(documents, Documents)

    #-------------------------------------------------------------------------
    def save_patients_info(self, patients: pd.DataFrame):
        self.upsert_dataframe(patients, Patients)
       

 
    
#-----------------------------------------------------------------------------
database = PharmagentDatabase()   