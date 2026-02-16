from __future__ import annotations

from DILIGENT.common.constants import SOURCES_PATH
from DILIGENT.common.utils.logger import logger
from DILIGENT.server.services.updater.livertox import LiverToxUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":   
    updater = LiverToxUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)

