from __future__ import annotations

from DILIGENT.src.packages.constants import SOURCES_PATH
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.database import database
from DILIGENT.src.packages.utils.updater.livertox import LiverToxUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":
    if database.requires_sqlite_initialization():
        logger.info("Database not found, creating instance and making all tables")
        database.initialize_database()
        logger.info("DILIGENT database has been initialized successfully.")
    updater = LiverToxUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)
