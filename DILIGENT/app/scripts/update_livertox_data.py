from __future__ import annotations

import os

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater
from DILIGENT.app.utils.updater.rxnav import RxNavDrugCatalogBuilder

REDOWNLOAD = True
RXNAV_CATALOG_FILENAME = "rxnav_drug_catalog.jsonl"

###############################################################################
if __name__ == "__main__":
    catalog_path = os.path.join(SOURCES_PATH, RXNAV_CATALOG_FILENAME)
    builder = RxNavDrugCatalogBuilder()
    logger.info("Refreshing RxNav drug catalog from %s", builder.TERMS_URL)
    try:
        catalog_info = builder.build_catalog(catalog_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unable to refresh RxNav catalog: %s", exc)
        raise
    logger.info(
        "RxNav catalog saved to %s with %d entries",
        catalog_info.get("file_path"),
        catalog_info.get("count", 0),
    )

    database.initialize_database()
    updater = LiverToxUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)
