import pandas as pd

from Pharmagent.app.utils.database.sqlite import database

from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self):        
        pass

    #--------------------------------------------------------------------------
    def save_patients_info(self, patients : pd.DataFrame) -> None:       
        database.save_patients_info(patients)
  
