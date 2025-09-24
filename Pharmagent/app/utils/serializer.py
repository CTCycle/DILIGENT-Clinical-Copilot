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
        prepared: list[dict[str, Any]] = []
        for record in records:
            nbk_id = record.get("nbk_id")
            drug_name = record.get("drug_name")
            if nbk_id is None or drug_name is None:
                continue
            excerpt = record.get("excerpt")
            prepared.append(
                {
                    "nbk_id": str(nbk_id),
                    "drug_name": str(drug_name),
                    "excerpt": str(excerpt) if excerpt is not None else None,
                }
            )
        database.replace_rows("LIVERTOX_MONOGRAPHS", prepared)
