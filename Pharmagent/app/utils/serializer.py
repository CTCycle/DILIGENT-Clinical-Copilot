from __future__ import annotations

import pandas as pd
from typing import Any

from Pharmagent.app.utils.database.sqlite import database


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def save_patients_info(self, patients: dict[str, Any]) -> None:
        data = pd.DataFrame([patients])
        database.upsert_into_database(data, "PATIENTS")

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: list[dict[str, Any]]) -> None:
        if not records:            
            return
        columns = ["nbk_id", "drug_name", "excerpt"]
        data = pd.DataFrame(records)        
        database.save_into_database(data[columns], "LIVERTOX_MONOGRAPHS")
