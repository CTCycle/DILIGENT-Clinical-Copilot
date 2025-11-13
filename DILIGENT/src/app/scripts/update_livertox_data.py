from __future__ import annotations

from DILIGENT.src.packages.constants import SOURCES_PATH
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.database import database
from DILIGENT.src.packages.utils.updater.livertox import LiverToxUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":
    database.initialize_database()
    updater = LiverToxUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)
