from __future__ import annotations

from typing import Any

from Pharmagent.app.constants import SOURCES_PATH
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.services.livertox import LiverToxUpdater

REDOWNLOAD = True
CONVERT_TO_DATAFRAME = False

###############################################################################
if __name__ == "__main__":
    updater = LiverToxUpdater(
        SOURCES_PATH,
        redownload=REDOWNLOAD,
        convert_to_dataframe=CONVERT_TO_DATAFRAME,
    )
    try:
        result = updater.run()
    except Exception as exc:  # noqa: BLE001
        logger.error("LiverTox update failed: %s", exc)
        raise SystemExit(1) from exc
    

