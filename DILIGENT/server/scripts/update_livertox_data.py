from __future__ import annotations

from DILIGENT.server.packages.constants import SOURCES_PATH
from DILIGENT.server.packages.logger import logger
from DILIGENT.server.packages.utils.updater.livertox import LiverToxUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":   
    updater = LiverToxUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)
