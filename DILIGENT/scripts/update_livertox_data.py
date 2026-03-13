from __future__ import annotations

from DILIGENT.server.common.constants import ARCHIVES_PATH
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.services.updater.livertox import LiverToxUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":   
    updater = LiverToxUpdater(ARCHIVES_PATH, redownload=REDOWNLOAD)
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)

