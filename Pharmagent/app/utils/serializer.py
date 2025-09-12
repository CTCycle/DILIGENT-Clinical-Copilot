from __future__ import annotations

import pandas as pd

from Pharmagent.app.utils.database.sqlite import database


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def save_patients_info(self, patients: pd.DataFrame) -> None:
        database.save_patients_info(patients)
