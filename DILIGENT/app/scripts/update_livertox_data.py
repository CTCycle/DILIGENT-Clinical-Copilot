from __future__ import annotations

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater
from DILIGENT.app.utils.updater.rxnav import RxNavDrugCatalogBuilder

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":
    database.initialize_database()
    catalog_path = os.path.join(SOURCES_PATH, RXNAV_CATALOG_FILENAME)
    builder = RxNavDrugCatalogBuilder()
    logger.info("Refreshing RxNav drug catalog from %s", builder.TERMS_URL)
    catalog_info = builder.update_drug_catalog(catalog_path)    
    logger.info(
        "RxNav catalog upserted into %s with %d entries",
        catalog_info.get("table_name"),
        catalog_info.get("count", 0),
    )

    # updater = LiverToxUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    # logger.info("Running LiverTox updater")
    # result = updater.update_from_livertox()
    #logger.info("LiverTox updater summary: %s", result)
