from __future__ import annotations

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.drug_catalog import DrugCatalogUpdater

REDOWNLOAD = False
RXNORM_ARCHIVE_PATH: str | None = None

###############################################################################
if __name__ == "__main__":
    database.initialize_database()
    updater = DrugCatalogUpdater(
        SOURCES_PATH, redownload=REDOWNLOAD, rxnorm_archive=RXNORM_ARCHIVE_PATH
    )
    logger.info("Running drug catalog updater")
    summary = updater.update_catalog()
    logger.info("Drug catalog updater summary: %s", summary)
    updater.close()
