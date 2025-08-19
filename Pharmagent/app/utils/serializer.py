import pandas as pd

from Pharmagent.app.utils.database.sqlite import PharmagentDatabase

from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self):        
        self.database = PharmagentDatabase() 

    #--------------------------------------------------------------------------
    def save_patients_info(self, patients : pd.DataFrame) -> None:       
        self.database.save_patients_info(patients)
  
