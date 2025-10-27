from __future__ import annotations

from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.rxnav import DrugsCatalogUpdater

###############################################################################
if __name__ == "__main__":
    database.initialize_database()
    updater = DrugsCatalogUpdater()
    logger.info("Running drugs catalog updater")
    result = updater.update_catalog()
    logger.info("Drugs catalog updater summary: %s", result)
